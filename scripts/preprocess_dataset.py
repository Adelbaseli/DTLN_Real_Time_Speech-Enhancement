"""
Turn the downloaded DNS-Challenge archives into a small, resampled, portfolio-
scale dataset:

  data/raw/*.tar.bz2 (48kHz, ~19GB total)
      -> stream-decompress, keep a random subset of clips (no full extraction
         to disk — skipped members are never written out)
      -> resample kept clips to 16kHz mono
      -> data/processed/{clean,noise}/*.wav + manifest CSVs with a
         deterministic train/val/test split at the file level.

This keeps peak disk usage to "raw archives + final small processed set"
instead of "raw archives + full 48kHz extraction + processed set".

Usage:
    python scripts/preprocess_dataset.py [--config configs/dns_subset.yaml]
        [--clean-hours 6] [--noise-hours 2] [--sample-rate 16000]
"""
import argparse
import csv
import hashlib
import io
import random
import tarfile
from pathlib import Path

import librosa
import soundfile as sf
import yaml

AUDIO_EXTS = (".wav", ".flac")
MIN_DURATION_SEC = 1.0


def split_for(name: str) -> str:
    """Deterministic 90/5/5 train/val/test split by filename hash (stable
    across reruns without keeping global state)."""
    digest = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    if digest < 90:
        return "train"
    if digest < 95:
        return "val"
    return "test"


def process_archive(archive_path: Path, out_dir: Path, kind: str, target_hours: float,
                     keep_prob: float, sample_rate: int, rng: random.Random, writer):
    target_seconds = target_hours * 3600
    kept_seconds = 0.0
    kept_count = 0
    seen_count = 0

    print(f"  streaming {archive_path.name} (target {target_hours:.1f}h of {kind})")
    with tarfile.open(archive_path, mode="r|bz2") as tar:
        for member in tar:
            if kept_seconds >= target_seconds:
                break
            if not member.isfile() or not member.name.lower().endswith(AUDIO_EXTS):
                continue
            seen_count += 1
            if rng.random() > keep_prob:
                continue

            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            raw_bytes = fileobj.read()
            try:
                audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            except Exception as exc:
                print(f"    skip (unreadable): {member.name}: {exc}")
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            duration = len(audio) / sr
            if duration < MIN_DURATION_SEC:
                continue

            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            stem = Path(member.name).stem.replace("/", "_")
            out_name = f"{archive_path.stem}__{stem}.wav"
            out_path = out_dir / out_name
            sf.write(out_path, audio, sample_rate, subtype="PCM_16")

            new_duration = len(audio) / sample_rate
            kept_seconds += new_duration
            kept_count += 1
            writer.writerow([str(out_path), f"{new_duration:.3f}", split_for(out_name),
                              archive_path.name, kind])

    print(f"    kept {kept_count}/{seen_count} candidate clips, "
          f"{kept_seconds / 3600:.2f}h")
    if kept_seconds < target_seconds * 0.5:
        print(f"    WARNING: kept far less than the {target_hours:.1f}h target — "
              f"consider raising --keep-prob and rerunning.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dns_subset.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--clean-hours", type=float, default=6.0)
    parser.add_argument("--noise-hours", type=float, default=2.0)
    parser.add_argument("--keep-prob", type=float, default=0.15,
                         help="Probability of keeping a candidate clip while streaming "
                              "each archive; raise this if a WARNING says too little "
                              "was kept.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(config["raw_dir"])
    processed_dir = Path(args.processed_dir)
    clean_out = processed_dir / "clean"
    noise_out = processed_dir / "noise"
    clean_out.mkdir(parents=True, exist_ok=True)
    noise_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    clean_manifest = processed_dir / "clean_manifest.csv"
    noise_manifest = processed_dir / "noise_manifest.csv"

    with open(clean_manifest, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["path", "duration_sec", "split", "source_archive", "kind"])
        # Split the clean-hours budget evenly across configured clean archives.
        per_archive_hours = args.clean_hours / max(len(config["clean_speech_archives"]), 1)
        for blob in config["clean_speech_archives"]:
            archive_path = raw_dir / Path(blob).name
            if not archive_path.exists():
                print(f"  missing {archive_path}, skipping (did you run download_dns_subset.py?)")
                continue
            process_archive(archive_path, clean_out, "clean", per_archive_hours,
                             args.keep_prob, args.sample_rate, rng, writer)

    with open(noise_manifest, "w", newline="") as nf:
        writer = csv.writer(nf)
        writer.writerow(["path", "duration_sec", "split", "source_archive", "kind"])
        per_archive_hours = args.noise_hours / max(len(config["noise_archives"]), 1)
        for blob in config["noise_archives"]:
            archive_path = raw_dir / Path(blob).name
            if not archive_path.exists():
                print(f"  missing {archive_path}, skipping (did you run download_dns_subset.py?)")
                continue
            process_archive(archive_path, noise_out, "noise", per_archive_hours,
                             args.keep_prob, args.sample_rate, rng, writer)

    print(f"\nWrote {clean_manifest} and {noise_manifest}")


if __name__ == "__main__":
    main()
