import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor


class MultimodalDataset(Dataset):
    """Dataset that returns raw text, two waveforms (before/after), and labels.

    Expects a CSV/TSV with columns for text, audio_before_path, audio_after_path, patient id and severity.
    """

    def __init__(self,
                 manifest_path: str,
                 text_col: str = "text",
                 audio_before_col: str = "audio_before_path",
                 audio_after_col: str = "audio_after_path",
                 patient_col: str = "patient_id",
                 severity_col: str = "severity",
                 tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 speech_feat_name: str = "facebook/wav2vec2-base-960h",
                 target_sr: int = 16000,
                 max_audio_seconds: float = 10.0,
                 cache_audio: bool = False):
        self.df = pd.read_csv(manifest_path)
        self.text_col = text_col
        self.audio_before_col = audio_before_col
        self.audio_after_col = audio_after_col
        self.patient_col = patient_col
        self.severity_col = severity_col
        self.target_sr = target_sr
        self.max_samples = int(max_audio_seconds * target_sr) if max_audio_seconds is not None else None
        self.cache_audio = cache_audio

        # tokenizers / feature extractors used in collate
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(speech_feat_name)

        # optional cache for preprocessed waveforms
        self._audio_cache = {} if cache_audio else None

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str):
        # returns 1-D float tensor at target_sr
        if path is None or (isinstance(path, float) and pd.isna(path)):
            # return silence of fixed max_samples length to avoid inconsistent shapes
            if self.max_samples is None:
                return torch.zeros(1, dtype=torch.float)
            return torch.zeros(self.max_samples, dtype=torch.float)

        if self.cache_audio and path in self._audio_cache:
            return self._audio_cache[path]

        waveform, sr = torchaudio.load(path)  # waveform: (channels, samples)
        # convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        # truncate or pad
        if self.max_samples is not None:
            if waveform.numel() > self.max_samples:
                waveform = waveform[: self.max_samples]
            elif waveform.numel() < self.max_samples:
                pad = torch.zeros(self.max_samples - waveform.numel(), dtype=waveform.dtype)
                waveform = torch.cat([waveform, pad], dim=0)
        if self.cache_audio:
            self._audio_cache[path] = waveform
        return waveform

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = row.get(self.text_col, "")
        audio_before_path = row.get(self.audio_before_col, None)
        audio_after_path = row.get(self.audio_after_col, None)
        patient_id = row.get(self.patient_col, -1)
        severity = row.get(self.severity_col, -1)

        waveform_before = self._load_audio(audio_before_path)
        waveform_after = self._load_audio(audio_after_path)

        return {
            "text": str(text),
            "waveform_before": waveform_before,
            "waveform_after": waveform_after,
            "patient_id": int(patient_id) if not pd.isna(patient_id) else -1,
            "severity": int(severity) if not pd.isna(severity) else -1,
        }

    def collate_fn(self, samples):
        # samples: list of dicts from __getitem__
        texts = [s["text"] for s in samples]
        waveforms_before = [s["waveform_before"] for s in samples]
        waveforms_after = [s["waveform_after"] for s in samples]
        patient_ids = torch.tensor([s["patient_id"] for s in samples], dtype=torch.long)
        severity = torch.tensor([s["severity"] for s in samples], dtype=torch.long)
        # only produce a binary 'label' if severity is already binary (0/1)
        label = None
        try:
            uniq = torch.unique(severity)
            if uniq.numel() <= 2 and torch.all((uniq == 0) | (uniq == 1)):
                label = severity.clone()
        except Exception:
            label = None

        # Tokenize text
        tokenized = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Feature-extract / pad audio for both timepoints
        audio_list_before = [w.numpy() if torch.is_tensor(w) else w for w in waveforms_before]
        audio_list_after = [w.numpy() if torch.is_tensor(w) else w for w in waveforms_after]
        audio_batch_before = self.feat_ext(audio_list_before, sampling_rate=self.target_sr, padding=True, truncation=True, return_tensors="pt")
        audio_batch_after = self.feat_ext(audio_list_after, sampling_rate=self.target_sr, padding=True, truncation=True, return_tensors="pt")

        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized.get("attention_mask"),
            "input_values_before": audio_batch_before["input_values"],
            "speech_attention_mask_before": audio_batch_before.get("attention_mask"),
            "input_values_after": audio_batch_after["input_values"],
            "speech_attention_mask_after": audio_batch_after.get("attention_mask"),
            "patient_ids": patient_ids,
            "severity": severity,
            "label": label,
        }
        return batch


if __name__ == "__main__":
    # minimal usage example (replace manifest.csv with your file)
    manifest = "manifest.csv"
    if not os.path.exists(manifest):
        # create dummy manifest with before/after columns
        df = pd.DataFrame({
            "text": ["patient has cough", "no symptoms"],
            "audio_before_path": [None, None],
            "audio_after_path": [None, None],
            "patient_id": [0, 1],
            "severity": [1, 0],
        })
        df.to_csv(manifest, index=False)

    ds = MultimodalDataset(manifest_path=manifest, cache_audio=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    for batch in loader:
        # batch is already the dict expected by model.forward
        print({k: (v.shape if torch.is_tensor(v) else None) for k, v in batch.items()})
        break
