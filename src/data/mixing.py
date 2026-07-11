"""Shared noisy/clean mixing helpers used by both the fixed eval-set builder
(scripts/make_eval_set.py) and the on-the-fly training dataset (dataset.py)."""
import numpy as np

EPS = 1e-10
SILENCE_RMS_FLOOR = 1e-4  # a "noise" clip this quiet is effectively digital silence


def is_silent(audio: np.ndarray) -> bool:
    """True for near-zero-energy clips (e.g. silent padding in a noise dataset).
    Mixing such a clip in leaves noisy == clean, which is a degenerate (not
    genuinely noisy) eval/train example — see mix_at_snr callers, which reroll
    the noise pick when this is True."""
    return float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS_FLOOR


def fit_noise_to_length(noise: np.ndarray, length: int, rng) -> np.ndarray:
    """Randomly crop noise to `length`, tiling it first if it's too short."""
    if len(noise) >= length:
        start = rng.randint(0, len(noise) - length)
        return noise[start:start + length]
    reps = length // len(noise) + 1
    tiled = np.tile(noise, reps)
    start = rng.randint(0, len(tiled) - length)
    return tiled[start:start + length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so the mixture hits the target SNR (dB) relative to `clean`,
    then peak-normalize if the mixture would clip."""
    clean_rms = np.sqrt(np.mean(clean ** 2) + EPS)
    noise_rms = np.sqrt(np.mean(noise ** 2) + EPS)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    noisy = clean + scaled_noise
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy * (0.99 / peak)
    return noisy.astype(np.float32)
