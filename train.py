import os
import time
import argparse
import random
import json

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from data_loader import MultimodalDataset
from model import MultiModalModel
from utils import move_to_device, safe_save_checkpoint, safe_load_checkpoint, set_seed as utils_set_seed


def set_seed(seed: int):
    # prefer utils implementation but keep local wrapper
    utils_set_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # macOS MPS support
    if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device):
    # recursive move using utils.move_to_device
    return move_to_device(batch, device)


def save_checkpoint(state: dict, path: str):
    # atomic safe save
    safe_save_checkpoint(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, map_location=None):
    checkpoint = safe_load_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optim_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state"])
    if scheduler is not None and "sched_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["sched_state"])
    return checkpoint


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    it = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.use_amp and device.type == "cuda"):
            out = model(batch)
            losses = out["losses"]
            loss = cfg.alpha * losses.get("cross_modal_contrastive", torch.tensor(0.0, device=device))
            loss = loss + cfg.beta * (losses.get("supcon_text", torch.tensor(0.0, device=device)) + losses.get("supcon_speech", torch.tensor(0.0, device=device)))
            loss = loss + cfg.delta * losses.get("temporal_contrastive", torch.tensor(0.0, device=device))
            # classification loss (binary BCEWithLogits) if provided by model
            loss = loss + cfg.gamma * losses.get("classification_loss", torch.tensor(0.0, device=device))
        if cfg.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        it += 1
    return total_loss / max(1, it)


def validate(model, dataloader, device, cfg):
    model.eval()
    total = {"cross_modal_contrastive": 0.0, "supcon_text": 0.0, "supcon_speech": 0.0, "temporal_contrastive": 0.0, "classification_loss": 0.0}
    it = 0
    acc_sum = 0.0
    acc_count = 0
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            losses = out["losses"]
            for k in total.keys():
                total[k] += losses.get(k, torch.tensor(0.0, device=device)).item()
            # classification accuracy if labels present
            label = batch.get("label")
            if label is None and batch.get("severity") is not None:
                sev = batch.get("severity")
                uniq = torch.unique(sev)
                if uniq.numel() <= 2 and torch.all((uniq == 0) | (uniq == 1)):
                    label = sev
            if label is not None:
                # pred_logit is stored in losses by model
                pred_logit = losses.get("pred_logit")
                if pred_logit is not None:
                    probs = torch.sigmoid(pred_logit)
                    preds = (probs > 0.5).float()
                    lbl = label.float().to(device)
                    # Ensure shapes match
                    if preds.numel() == lbl.numel():
                        acc = (preds == lbl).float().mean().item()
                        acc_sum += acc
                        acc_count += 1
                        # store for ROC/AUC
                        all_logits.append(pred_logit.detach().cpu())
                        all_labels.append(lbl.detach().cpu())
            it += 1
    # average batch losses
    for k in total.keys():
        total[k] = total[k] / max(1, it)
    # average accuracy across batches that had labels
    if acc_count > 0:
        total["accuracy"] = acc_sum / acc_count
    else:
        total["accuracy"] = 0.0
    # compute AUROC on collected logits/labels if available
    if len(all_labels) > 0:
        y_true = torch.cat(all_labels).numpy()
        y_score = torch.cat(all_logits).numpy()
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = 0.0
    else:
        auc = 0.0
    total["auc"] = auc
    combined = cfg.alpha * total["cross_modal_contrastive"] + cfg.beta * (total["supcon_text"] + total["supcon_speech"]) + cfg.delta * total.get("temporal_contrastive", 0.0) + cfg.gamma * total.get("classification_loss", 0.0)
    total["combined"] = combined
    return total


def evaluate_test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            losses = out.get("losses", {})
            label = batch.get("label")
            if label is None and batch.get("severity") is not None:
                sev = batch.get("severity")
                uniq = torch.unique(sev)
                if uniq.numel() <= 2 and torch.all((uniq == 0) | (uniq == 1)):
                    label = sev
            pred_logit = losses.get("pred_logit")
            if label is not None and pred_logit is not None:
                all_labels.append(label.float().cpu())
                all_logits.append(pred_logit.detach().cpu())
    if len(all_labels) == 0:
        print("No binary labels found in test set; cannot compute classification metrics.")
        return {}
    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_logits).numpy()
    y_pred = (torch.tensor(y_score) > 0.5).int().numpy()
    # compute metrics
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = float((y_pred == y_true).sum()) / len(y_true)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "auc": auc, "f1": f1}
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def build_dataloaders(cfg):
    train_ds = MultimodalDataset(manifest_path=cfg.train_manifest, tokenizer_name=cfg.text_model_name, speech_feat_name=cfg.speech_model_name, target_sr=cfg.target_sr, max_audio_seconds=cfg.max_audio_seconds, cache_audio=cfg.cache_audio)
    val_ds = MultimodalDataset(manifest_path=cfg.val_manifest, tokenizer_name=cfg.text_model_name, speech_feat_name=cfg.speech_model_name, target_sr=cfg.target_sr, max_audio_seconds=cfg.max_audio_seconds, cache_audio=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=val_ds.collate_fn)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--text_model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--speech_model_name", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--max_audio_seconds", type=float, default=10.0)
    parser.add_argument("--cache_audio", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation AUC")
    parser.add_argument("--test_manifest", type=str, default=None, help="Optional test manifest CSV for final evaluation")
    # LoRA / PEFT options
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA via PEFT for encoder adapters")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default 0.0)
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated target module names for LoRA (optional)")

    args = parser.parse_args()
    cfg = args

    set_seed(cfg.seed)
    device = get_device()
    print("Using device:", device)

    train_loader, val_loader = build_dataloaders(cfg)

    # parse lora target modules if provided
    lora_targets = None
    if cfg.lora_target_modules is not None:
        lora_targets = [s.strip() for s in cfg.lora_target_modules.split(",") if s.strip()]

    model = MultiModalModel(text_model_name=cfg.text_model_name, speech_model_name=cfg.speech_model_name,
                            use_lora=cfg.use_lora, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, lora_target_modules=lora_targets)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")
    best_auc = 0.0
    patience_counter = 0
    if cfg.resume_checkpoint is not None:
        ckpt = load_checkpoint(cfg.resume_checkpoint, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg)
        val_metrics = validate(model, val_loader, device, cfg)
        val_auc = val_metrics.get("auc", 0.0)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.epochs} - train_loss: {train_loss:.4f} - val_combined: {val_metrics['combined']:.4f} - val_auc: {val_auc:.4f} - time: {epoch_time:.1f}s")
        # save
        ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch+1}.pt")
        save_checkpoint({
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        }, ckpt_path)
        # update best by AUC for early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_path = os.path.join(cfg.output_dir, "best_model.pt")
            save_checkpoint({
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
            }, best_path)
        else:
            patience_counter += 1
        # early stopping
        if patience_counter >= cfg.patience:
            print(f"Early stopping triggered (patience={cfg.patience}). Best val AUC={best_auc:.4f}")
            break

    # final save of config
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    # optional test evaluation
    if cfg.test_manifest is not None:
        test_ds = MultimodalDataset(manifest_path=cfg.test_manifest, tokenizer_name=cfg.text_model_name, speech_feat_name=cfg.speech_model_name, target_sr=cfg.target_sr, max_audio_seconds=cfg.max_audio_seconds, cache_audio=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=test_ds.collate_fn)
        # load best model
        best_ckpt = os.path.join(cfg.output_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            load_checkpoint(best_ckpt, model, map_location=device)
        evaluate_test(model, test_loader, device)


if __name__ == "__main__":
    main()
