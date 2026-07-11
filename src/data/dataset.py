"""PyTorch Dataset for DTLN training.

Train split: infinite variety via on-the-fly random clean+noise+SNR mixing
(no extra disk, no repeated epochs of identical mixtures).
Val/test splits: load the FIXED pairs written by scripts/make_eval_set.py, so
validation loss/metrics are comparable across epochs and runs.
"""
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.mixing import fit_noise_to_length, is_silent, mix_at_snr

SNR_LEVELS_DB = [-5, 0, 5, 10, 15]


def _read_manifest_paths(manifest_path: Path, split: str) -> list[str]:
    with open(manifest_path, newline="") as f:
        return [row["path"] for row in csv.DictReader(f) if row["split"] == split]


def _load_random_segment(path: str, num_samples: int, rng: random.Random) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32")
    if len(audio) >= num_samples:
        start = rng.randint(0, len(audio) - num_samples)
        return audio[start:start + num_samples]
    return np.pad(audio, (0, num_samples - len(audio)))


class DNSTrainDataset(Dataset):
    """Random on-the-fly clean+noise+SNR mixing, resampled every access."""

    def __init__(self, processed_dir: str, segment_seconds: float = 4.0,
                 sample_rate: int = 16000, epoch_size: int = 10000, seed: int = 0):
        processed_dir = Path(processed_dir)
        self.clean_paths = _read_manifest_paths(processed_dir / "clean_manifest.csv", "train")
        self.noise_paths = _read_manifest_paths(processed_dir / "noise_manifest.csv", "train")
        if not self.clean_paths or not self.noise_paths:
            raise RuntimeError(
                f"No train-split clean/noise clips found under {processed_dir}. "
                "Run scripts/preprocess_dataset.py first."
            )
        self.segment_samples = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        self.seed = seed
        # Stateful, advances on every __getitem__ call. If num_workers > 0,
        # call reseed_for_worker() from a DataLoader worker_init_fn so each
        # worker process gets an independent stream instead of a forked copy
        # of this exact state (see src/train.py's worker_init_fn).
        self.rng = random.Random(seed)

    def reseed_for_worker(self, worker_seed: int):
        self.rng = random.Random(worker_seed)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        rng = self.rng
        clean_path = rng.choice(self.clean_paths)
        clean = _load_random_segment(clean_path, self.segment_samples, rng)

        for _ in range(5):  # reroll silent noise clips (see is_silent)
            noise_path = rng.choice(self.noise_paths)
            noise_raw, _ = sf.read(noise_path, dtype="float32")
            if not is_silent(noise_raw):
                break
        noise = fit_noise_to_length(noise_raw, self.segment_samples, rng)

        snr_db = rng.choice(SNR_LEVELS_DB)
        noisy = mix_at_snr(clean, noise, snr_db)

        return torch.from_numpy(noisy), torch.from_numpy(clean)


class DNSFixedPairDataset(Dataset):
    """Loads the fixed noisy/clean pairs written by scripts/make_eval_set.py."""

    def __init__(self, eval_dir: str):
        eval_dir = Path(eval_dir)
        pairs_csv = eval_dir / "pairs.csv"
        if not pairs_csv.exists():
            raise RuntimeError(
                f"{pairs_csv} not found. Run scripts/make_eval_set.py first."
            )
        with open(pairs_csv, newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        noisy, _ = sf.read(row["noisy_path"], dtype="float32")
        clean, _ = sf.read(row["clean_path"], dtype="float32")
        return torch.from_numpy(noisy), torch.from_numpy(clean)
