"""
Evaluate a trained DTLN checkpoint on the fixed test set (scripts/make_eval_set.py)
with PESQ, STOI, and SI-SDR improvement, broken down by input SNR bucket.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best.pt [--test-dir data/eval/test]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

from src.losses import si_sdr
from src.model import DTLN


@torch.no_grad()
def evaluate(checkpoint_path: str, test_dir: str, sample_rate: int, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DTLN(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pairs_csv = Path(test_dir) / "pairs.csv"
    with open(pairs_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    per_snr = defaultdict(lambda: {"pesq": [], "stoi": [], "sisdr_in": [], "sisdr_out": []})

    for row in rows:
        noisy, sr = sf.read(row["noisy_path"], dtype="float32")
        clean, _ = sf.read(row["clean_path"], dtype="float32")
        assert sr == sample_rate, f"expected {sample_rate}Hz, got {sr}Hz"

        noisy_t = torch.from_numpy(noisy).unsqueeze(0).to(device)
        clean_t = torch.from_numpy(clean).unsqueeze(0).to(device)
        enhanced_t = model(noisy_t)
        enhanced = enhanced_t.squeeze(0).cpu().numpy()

        snr = row["snr_db"]
        bucket = per_snr[snr]
        try:
            bucket["pesq"].append(pesq(sample_rate, clean, enhanced, "wb"))
        except Exception:
            pass
        bucket["stoi"].append(stoi(clean, enhanced, sample_rate, extended=False))
        bucket["sisdr_in"].append(si_sdr(noisy_t, clean_t).item())
        bucket["sisdr_out"].append(si_sdr(enhanced_t, clean_t).item())

    return per_snr


def render_report(per_snr: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        "| SNR (dB) | n | PESQ | STOI | SI-SDR in | SI-SDR out | SI-SDR improvement |",
        "|---|---|---|---|---|---|---|",
    ]
    all_pesq, all_stoi, all_in, all_out = [], [], [], []
    for snr in sorted(per_snr, key=lambda s: float(s)):
        b = per_snr[snr]
        n = len(b["stoi"])
        pesq_mean = np.mean(b["pesq"]) if b["pesq"] else float("nan")
        stoi_mean = np.mean(b["stoi"])
        in_mean = np.mean(b["sisdr_in"])
        out_mean = np.mean(b["sisdr_out"])
        lines.append(f"| {snr} | {n} | {pesq_mean:.3f} | {stoi_mean:.3f} | "
                      f"{in_mean:.2f} | {out_mean:.2f} | {out_mean - in_mean:+.2f} |")
        all_pesq.extend(b["pesq"])
        all_stoi.extend(b["stoi"])
        all_in.extend(b["sisdr_in"])
        all_out.extend(b["sisdr_out"])

    overall_pesq = np.mean(all_pesq) if all_pesq else float("nan")
    overall_stoi = np.mean(all_stoi)
    overall_in = np.mean(all_in)
    overall_out = np.mean(all_out)
    lines.append(f"| **overall** | {len(all_stoi)} | {overall_pesq:.3f} | {overall_stoi:.3f} | "
                  f"{overall_in:.2f} | {overall_out:.2f} | {overall_out - overall_in:+.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", default="data/eval/test")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output", default="results/eval_report.md")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--mlflow-experiment", default="dtln-speech-enhancement")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_snr = evaluate(args.checkpoint, args.test_dir, args.sample_rate, device)
    report = render_report(per_snr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(report)
    print(f"Wrote {args.output}")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("checkpoint", args.checkpoint)
        mlflow.log_artifact(args.output)
        all_pesq = [v for b in per_snr.values() for v in b["pesq"]]
        all_stoi = [v for b in per_snr.values() for v in b["stoi"]]
        all_in = [v for b in per_snr.values() for v in b["sisdr_in"]]
        all_out = [v for b in per_snr.values() for v in b["sisdr_out"]]
        mlflow.log_metrics({
            "test_pesq": float(np.mean(all_pesq)) if all_pesq else float("nan"),
            "test_stoi": float(np.mean(all_stoi)),
            "test_sisdr_improvement": float(np.mean(all_out) - np.mean(all_in)),
        })


if __name__ == "__main__":
    sys.exit(main())
