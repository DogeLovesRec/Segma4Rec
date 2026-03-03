import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: text attends to speech and speech attends to text.
    Uses MultiheadAttention from PyTorch. Expects inputs as (B, L, D).
    Returns fused text and fused speech (pooled later by the caller).
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.text_to_speech_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.speech_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layernorm_t = nn.LayerNorm(embed_dim)
        self.layernorm_s = nn.LayerNorm(embed_dim)

    def forward(self, text_feats: torch.Tensor, speech_feats: torch.Tensor, text_mask: torch.Tensor = None, speech_mask: torch.Tensor = None):
        # text_feats: (B, Tt, D), speech_feats: (B, Ts, D)
        # MultiheadAttention with batch_first=True accepts key_padding_mask of shape (B, S)
        # text attends to speech
        attn_t, _ = self.text_to_speech_attn(query=text_feats, key=speech_feats, value=speech_feats, key_padding_mask=speech_mask)
        fused_text = self.layernorm_t(text_feats + attn_t)
        attn_s, _ = self.speech_to_text_attn(query=speech_feats, key=text_feats, value=text_feats, key_padding_mask=text_mask)
        fused_speech = self.layernorm_s(speech_feats + attn_s)
        return fused_text, fused_speech


def info_nce_loss(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.07, labels: torch.Tensor = None):
    """InfoNCE-style cross-modal loss.
    If labels is None, assumes one-to-one pairing (diagonal positives) and computes symmetric loss.
    If labels is provided (shape B,), positives are any pairs with equal label across modalities.
    Vectorized and skips anchors with no positives.
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    device = x.device
    B = x.size(0)
    logits = torch.matmul(x, y.t()) / temperature  # (B, B)

    if labels is None:
        # standard symmetric InfoNCE with diagonal positives
        labels_idx = torch.arange(B, device=device)
        loss_x = F.cross_entropy(logits, labels_idx)
        loss_y = F.cross_entropy(logits.t(), labels_idx)
        return (loss_x + loss_y) / 2.0

    # supervised cross-modal: labels indicates positive sets across modalities
    labels = labels.view(-1)
    assert labels.size(0) == B

    # compute row-wise logsumexp denominator
    logsumexp_all = torch.logsumexp(logits, dim=1)  # (B,)
    # compute numerator logsumexp over positives using label equality
    labels_x = labels.unsqueeze(1)  # (B,1)
    labels_y = labels.unsqueeze(0)  # (1,B)
    pos_mask = (labels_x == labels_y)  # (B,B)

    # For x->y direction
    masked_logits_pos = logits.masked_fill(~pos_mask, float('-inf'))  # -inf for non-positives
    logsumexp_pos = torch.logsumexp(masked_logits_pos, dim=1)  # (B,) may be -inf if no positives
    valid_pos = ~torch.isinf(logsumexp_pos)
    loss_x = torch.tensor(0.0, device=device)
    if valid_pos.sum() > 0:
        loss_x = - (logsumexp_pos[valid_pos] - logsumexp_all[valid_pos]).mean()

    # For y->x direction (transpose)
    logits_t = logits.t()
    logsumexp_all_t = torch.logsumexp(logits_t, dim=1)
    masked_logits_pos_t = logits_t.masked_fill(~pos_mask.t(), float('-inf'))
    logsumexp_pos_t = torch.logsumexp(masked_logits_pos_t, dim=1)
    valid_pos_t = ~torch.isinf(logsumexp_pos_t)
    loss_y = torch.tensor(0.0, device=device)
    if valid_pos_t.sum() > 0:
        loss_y = - (logsumexp_pos_t[valid_pos_t] - logsumexp_all_t[valid_pos_t]).mean()

    if (valid_pos.sum() + valid_pos_t.sum()) == 0:
        return torch.tensor(0.0, device=device)
    return (loss_x + loss_y) / 2.0


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """Vectorized supervised contrastive loss (Khosla et al.).
    embeddings: (B, D), labels: (B,)
    For each anchor i, positives are samples with same label (excluding i). Anchors with no positives are ignored.
    This implementation is fully vectorized and avoids explicit python loops.
    """
    device = embeddings.device
    embeddings = F.normalize(embeddings, dim=-1)
    labels = labels.view(-1)
    assert labels.size(0) == embeddings.size(0)
    B = embeddings.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=device)

    # similarity matrix
    sim = torch.matmul(embeddings, embeddings.t()) / temperature  # (B, B)

    # positive mask (exclude self)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = labels_eq & ~eye  # positives only, exclude diagonal

    # For denominator we need logsumexp over all except self
    # Mask out self by setting -inf so it doesn't contribute
    sim_masked = sim.masked_fill(eye, float('-inf'))
    log_den = torch.logsumexp(sim_masked, dim=1)  # (B,)

    # log probabilities for all pairs (i,j): log(exp(sim_ij) / sum_{k!=i} exp(sim_ik))
    log_prob = sim - log_den.unsqueeze(1)  # (B, B)

    # Sum log-probs over positives for each anchor
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)  # (B,)
    pos_count = pos_mask.sum(dim=1).float()  # (B,)

    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss_per_anchor = - pos_log_prob[valid] / pos_count[valid]
    loss = loss_per_anchor.mean()
    return loss


class MultiModalModel(nn.Module):
    """Multimodal model for speech and text.
    - Text encoder: ClinicalBERT (configurable)
    - Speech encoder: Wav2Vec2 (configurable)
    - Per-timestep projection, cross-attention fusion
    - Pooling to get final modality embeddings
    - Cross-modal contrastive loss (InfoNCE)
    - Supervised contrastive loss within each modality using severity labels
    """

    def __init__(self,
                 text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 speech_model_name: str = "facebook/wav2vec2-base-960h",
                 projection_dim: int = 256,
                 fusion_heads: int = 8,
                 temperature: float = 0.07,
                 freeze_encoders: bool = False,
                 use_lora: bool = False,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 lora_target_modules = None):
        super().__init__()
        # Foundation encoders
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.speech_encoder = Wav2Vec2Model.from_pretrained(speech_model_name)

        # Optionally apply LoRA adapters via PEFT to reduce finetuning cost
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                # sensible default target modules if not provided
                if lora_target_modules is None:
                    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "query", "key", "value", "out_proj"]
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLASSIFICATION if hasattr(TaskType, 'SEQ_CLASSIFICATION') else TaskType.TEXT_CLASSIFICATION,
                )
                self.text_encoder = get_peft_model(self.text_encoder, lora_config)
                self.speech_encoder = get_peft_model(self.speech_encoder, lora_config)
                print("Applied LoRA adapters to encoders via PEFT")
            except Exception as e:
                print("Warning: PEFT/LoRA not available or failed to attach adapters. Proceeding without LoRA. Error:", e)

        if freeze_encoders:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.speech_encoder.parameters():
                p.requires_grad = False

        text_hidden = self.text_encoder.config.hidden_size
        speech_hidden = self.speech_encoder.config.hidden_size
        self.proj_text_seq = nn.Linear(text_hidden, projection_dim)
        self.proj_speech_seq = nn.Linear(speech_hidden, projection_dim)

        # Cross-attention fusion operates on projected per-timestep features
        self.fusion = CrossAttentionFusion(embed_dim=projection_dim, num_heads=fusion_heads)

        # After fusion, pool and produce final embeddings for contrastive learning
        self.pool = nn.AdaptiveAvgPool1d(1)  # will be applied on (B, D, L)
        self.final_text_proj = nn.Linear(projection_dim, projection_dim)
        self.final_speech_proj = nn.Linear(projection_dim, projection_dim)

        self.temperature = temperature

        # Combined selection classifier: will take [text_emb, speech_emb, fused_emb]
        self.classifier_combined = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, 1)
        )

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # Returns sequence features (B, T, H)
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = outputs.last_hidden_state
        return seq

    def encode_speech(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        # input_values: (B, Ts)
        outputs = self.speech_encoder(input_values, attention_mask=attention_mask, return_dict=True)
        seq = outputs.last_hidden_state
        return seq

    def pool_sequence(self, seq: torch.Tensor):
        # seq: (B, L, D) -> (B, D)
        x = seq.transpose(1, 2)  # (B, D, L)
        pooled = self.pool(x).squeeze(-1)
        return pooled

    def forward(self, batch: dict, compute_losses: bool = True):
        """batch should contain:
        - input_ids, attention_mask (text)
        - input_values OR input_values_before/input_values_after and corresponding speech_attention_mask(s)
        - patient_ids: tensor (B,) used for cross-modal pairing
        - severity: tensor (B,) used for supervised contrastive within modality
        """
        input_ids = batch.get("input_ids")
        text_mask = batch.get("attention_mask")

        # Support single or two timepoints for speech
        input_values = batch.get("input_values")
        input_values_before = batch.get("input_values_before")
        input_values_after = batch.get("input_values_after")

        speech_mask = batch.get("speech_attention_mask")
        speech_mask_before = batch.get("speech_attention_mask_before")
        speech_mask_after = batch.get("speech_attention_mask_after")

        patient_ids = batch.get("patient_ids")
        severity = batch.get("severity")

        # Encode text
        text_seq = self.encode_text(input_ids, attention_mask=text_mask)  # (B, Tt, Ht)

        # Encode speech for available timepoints
        speech_seqs = []
        speech_masks = []
        seq_before = None
        seq_after = None
        if input_values is not None:
            seq = self.encode_speech(input_values, attention_mask=speech_mask)
            speech_seqs.append(seq)
            speech_masks.append(speech_mask)
        else:
            if input_values_before is not None:
                seq_before = self.encode_speech(input_values_before, attention_mask=speech_mask_before)
                speech_seqs.append(seq_before)
                speech_masks.append(speech_mask_before)
            if input_values_after is not None:
                seq_after = self.encode_speech(input_values_after, attention_mask=speech_mask_after)
                speech_seqs.append(seq_after)
                speech_masks.append(speech_mask_after)

        if len(speech_seqs) == 0:
            raise ValueError("No speech inputs found in batch. Provide input_values or input_values_before/after.")

        # Project per-timestep for each speech seq and concatenate along time dimension
        proj_speech_seqs = [self.proj_speech_seq(s) for s in speech_seqs]  # list of (B, Ts_i, D)
        speech_proj_seq = torch.cat(proj_speech_seqs, dim=1)  # (B, Ts_total, D)

        # Project text per-timestep
        text_proj_seq = self.proj_text_seq(text_seq)  # (B, Tt, D)

        # Prepare key_padding_mask for MultiheadAttention: True for positions that should be masked
        text_kpm = None
        speech_kpm = None
        if text_mask is not None:
            text_kpm = (text_mask == 0)

        if any(m is not None for m in [speech_mask, speech_mask_before, speech_mask_after]):
            # build combined speech padding mask matching concatenated sequence
            masks = []
            if speech_mask is not None:
                masks.append(speech_mask)
            else:
                if speech_mask_before is not None:
                    masks.append(speech_mask_before)
                if speech_mask_after is not None:
                    masks.append(speech_mask_after)
            # each mask is (B, Ts_i); concat along time dim
            speech_kpm = torch.cat([m for m in masks if m is not None], dim=1)
            speech_kpm = (speech_kpm == 0)

        # Cross-attention fusion between text and combined speech
        fused_text_seq, fused_speech_seq = self.fusion(text_proj_seq, speech_proj_seq, text_mask=text_kpm, speech_mask=speech_kpm)

        # Pool
        text_pooled = self.pool_sequence(fused_text_seq)  # (B, D)
        speech_pooled = self.pool_sequence(fused_speech_seq)  # (B, D)

        text_emb = F.normalize(self.final_text_proj(text_pooled), dim=-1)
        speech_emb = F.normalize(self.final_speech_proj(speech_pooled), dim=-1)
        # Also compute a fused pooled embedding from the fused sequences
        fused_text_pooled = self.pool_sequence(fused_text_seq)
        fused_speech_pooled = self.pool_sequence(fused_speech_seq)
        # simple fusion embedding: average of fused text/speech pooled representations
        fused_pooled = 0.5 * (fused_text_pooled + fused_speech_pooled)
        fused_emb = F.normalize(self.final_speech_proj(fused_pooled), dim=-1)

        out = {
            "text_emb": text_emb,
            "speech_emb": speech_emb,
            "text_pooled": text_pooled,
            "speech_pooled": speech_pooled,
        }

        if not compute_losses:
            return out

        losses = {}
        # Cross-modal contrastive (same patient should be close, different patients far)
        if patient_ids is not None:
            losses["cross_modal_contrastive"] = info_nce_loss(text_emb, speech_emb, temperature=self.temperature, labels=patient_ids)
        else:
            losses["cross_modal_contrastive"] = torch.tensor(0.0, device=text_emb.device)

        # Temporal contrastive between before/after if both available
        if seq_before is not None and seq_after is not None and patient_ids is not None:
            # compute pooled embeddings for each timepoint separately
            pooled_b = self.pool_sequence(self.proj_speech_seq(seq_before))
            pooled_a = self.pool_sequence(self.proj_speech_seq(seq_after))
            emb_b = F.normalize(self.final_speech_proj(pooled_b), dim=-1)
            emb_a = F.normalize(self.final_speech_proj(pooled_a), dim=-1)
            losses["temporal_contrastive"] = info_nce_loss(emb_b, emb_a, temperature=self.temperature, labels=patient_ids)
        else:
            losses["temporal_contrastive"] = torch.tensor(0.0, device=text_emb.device)

        # Supervised contrastive within modality by severity
        if severity is not None:
            losses["supcon_text"] = supervised_contrastive_loss(text_emb, severity, temperature=self.temperature)
            losses["supcon_speech"] = supervised_contrastive_loss(speech_emb, severity, temperature=self.temperature)
        else:
            losses["supcon_text"] = torch.tensor(0.0, device=text_emb.device)
            losses["supcon_speech"] = torch.tensor(0.0, device=text_emb.device)

        # Binary classification head: combine text_emb, speech_emb and fused_emb and predict single logit
        label = batch.get("label")
        # backward compatibility: if no explicit 'label' provided but 'severity' is binary (0/1), use it
        if label is None and severity is not None:
            uniq = torch.unique(severity)
            if uniq.numel() <= 2 and torch.all((uniq == 0) | (uniq == 1)):
                label = severity

        if label is not None:
            label = label.float().to(text_emb.device)
            combined_emb = torch.cat([text_emb, speech_emb, fused_emb], dim=-1)  # (B, 3D)
            logit = self.classifier_combined(combined_emb).squeeze(-1)
            loss_fn = nn.BCEWithLogitsLoss()
            try:
                losses["classification_loss"] = loss_fn(logit, label)
                losses["pred_logit"] = logit
            except Exception:
                # if shapes mismatch, skip classification loss
                losses["classification_loss"] = torch.tensor(0.0, device=text_emb.device)
                losses["pred_logit"] = torch.zeros(text_emb.size(0), device=text_emb.device)
        else:
            losses["classification_loss"] = torch.tensor(0.0, device=text_emb.device)
            losses["pred_logit"] = torch.zeros(text_emb.size(0), device=text_emb.device)

        out["losses"] = losses
        return out


if __name__ == "__main__":
    # Minimal usage example (random tensors). Replace with real tokenized text and speech signals.
    model = MultiModalModel()
    B = 4
    Tt = 32
    Ts = 16000  # example raw waveform length for wav2vec; in practice use proper preprocessing
    input_ids = torch.randint(0, 1000, (B, Tt))
    attention_mask = torch.ones(B, Tt, dtype=torch.long)
    input_values = torch.randn(B, Ts)
    speech_attention_mask = torch.ones(B, Ts, dtype=torch.long)
    patient_ids = torch.tensor([0, 1, 0, 2])
    severity = torch.tensor([1, 2, 1, 3])
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_values": input_values,
        "speech_attention_mask": speech_attention_mask,
        "patient_ids": patient_ids,
        "severity": severity,
    }
    out = model(batch)
    for k, v in out["losses"].items():
        print(k, v.item())
