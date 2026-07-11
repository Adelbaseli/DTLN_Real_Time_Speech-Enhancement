"""
Build FIXED noisy/clean validation and test pairs from the held-out ("val"/
"test") rows of the manifests produced by preprocess_dataset.py.

Fixed means: same seed, same SNR list, written to disk once — so PESQ/STOI
numbers from evaluate.py are reproducible across training runs instead of
being resampled randomly every time.

Usage:
    python scripts/make_eval_set.py [--processed-dir data/processed]
        [--eval-dir data/eval] [--n-val 30] [--n-test 50]
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.mixing import fit_noise_to_length, is_silent, mix_at_snr  # noqa: E402

SNR_LEVELS_DB = [-5, 0, 5, 10, 15]


def read_manifest(path: Path, split: str) -> list[str]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append(row["path"])
    return rows


def build_split(clean_paths, noise_paths, out_dir: Path, n_pairs: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    clean_paths = clean_paths[:]
    rng.shuffle(clean_paths)
    clean_paths = clean_paths[:n_pairs]

    pairs_csv = out_dir / "pairs.csv"
    with open(pairs_csv, "w", newline="") as pf:
        writer = csv.writer(pf)
        writer.writerow(["noisy_path", "clean_path", "snr_db", "clean_source", "noise_source"])
        for i, clean_path in enumerate(clean_paths):
            clean, sr = sf.read(clean_path, dtype="float32")
            for _ in range(5):  # reroll silent noise clips (see is_silent)
                noise_path = rng.choice(noise_paths)
                noise, noise_sr = sf.read(noise_path, dtype="float32")
                assert sr == noise_sr, f"sample rate mismatch: {sr} vs {noise_sr}"
                if not is_silent(noise):
                    break
            noise = fit_noise_to_length(noise, len(clean), rng)
            snr_db = rng.choice(SNR_LEVELS_DB)
            noisy = mix_at_snr(clean, noise, snr_db)

            clean_out = out_dir / f"{i:04d}_clean.wav"
            noisy_out = out_dir / f"{i:04d}_noisy.wav"
            sf.write(clean_out, clean, sr, subtype="PCM_16")
            sf.write(noisy_out, noisy, sr, subtype="PCM_16")
            writer.writerow([str(noisy_out), str(clean_out), snr_db, clean_path, noise_path])

    print(f"  wrote {len(clean_paths)} pairs to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    eval_dir = Path(args.eval_dir)

    clean_manifest = processed_dir / "clean_manifest.csv"
    noise_manifest = processed_dir / "noise_manifest.csv"

    for split, n_pairs in [("val", args.n_val), ("test", args.n_test)]:
        clean_paths = read_manifest(clean_manifest, split)
        noise_paths = read_manifest(noise_manifest, split)
        if not clean_paths or not noise_paths:
            print(f"  skipping {split}: no clean/noise rows found "
                  f"(clean={len(clean_paths)}, noise={len(noise_paths)})")
            continue
        print(f"Building {split} set ({n_pairs} pairs)...")
        build_split(clean_paths, noise_paths, eval_dir / split, n_pairs, args.seed)


if __name__ == "__main__":
    main()
