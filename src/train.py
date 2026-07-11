"""
DTLN training loop with MLflow tracking.

Usage:
    python -m src.train --config configs/train.yaml [--resume checkpoints/last.pt]
"""
import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from pesq import pesq
from pystoi import stoi
from torch.utils.data import DataLoader

from src.data.dataset import DNSFixedPairDataset, DNSTrainDataset
from src.losses import NegSISDRLoss
from src.model import DTLN


def flatten_config(config: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_config(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat


def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.reseed_for_worker(torch.initial_seed() % (2 ** 32))


@torch.no_grad()
def validate(model, loader, loss_fn, device, sample_rate):
    model.eval()
    losses, pesq_scores, stoi_scores = [], [], []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        enhanced = model(noisy)
        losses.append(loss_fn(enhanced, clean).item())

        enhanced_np = enhanced.cpu().numpy()
        clean_np = clean.cpu().numpy()
        for e, c in zip(enhanced_np, clean_np):
            try:
                pesq_scores.append(pesq(sample_rate, c, e, "wb"))
            except Exception:
                pass  # pesq can fail on degenerate/silent segments
            stoi_scores.append(stoi(c, e, sample_rate, extended=False))

    return {
        "val_loss": float(np.mean(losses)),
        "val_pesq": float(np.mean(pesq_scores)) if pesq_scores else float("nan"),
        "val_stoi": float(np.mean(stoi_scores)) if stoi_scores else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    data_cfg, model_cfg, train_cfg, mlflow_cfg = (
        config["data"], config["model"], config["train"], config["mlflow"]
    )

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = DNSTrainDataset(
        data_cfg["processed_dir"], segment_seconds=data_cfg["segment_seconds"],
        sample_rate=data_cfg["sample_rate"], epoch_size=data_cfg["epoch_size"],
        seed=train_cfg["seed"],
    )
    val_set = DNSFixedPairDataset(f"{data_cfg['eval_dir']}/val")

    train_loader = DataLoader(
        train_set, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    # batch_size=1: fixed val pairs are natural (variable-length) utterances,
    # not fixed-length training segments, so they can't be stacked into a batch.
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = DTLN(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    loss_fn = NegSISDRLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=train_cfg["amp"] and device.type == "cuda")

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]):
        mlflow.log_params(flatten_config(config))
        mlflow.log_artifact(args.config)

        for epoch in range(start_epoch, train_cfg["epochs"]):
            model.train()
            epoch_start = time.monotonic()
            train_losses = []

            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    enhanced = model(noisy)
                    loss = loss_fn(enhanced, clean)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(loss.item())

            metrics = validate(model, val_loader, loss_fn, device, data_cfg["sample_rate"])
            metrics["train_loss"] = float(np.mean(train_losses))
            metrics["epoch_seconds"] = time.monotonic() - epoch_start
            mlflow.log_metrics(metrics, step=epoch)
            print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            last_ckpt = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss, "model_config": model_cfg,
            }
            torch.save(last_ckpt, checkpoint_dir / "last.pt")

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                epochs_without_improvement = 0
                last_ckpt["best_val_loss"] = best_val_loss
                torch.save(last_ckpt, checkpoint_dir / "best.pt")
                mlflow.log_artifact(str(checkpoint_dir / "best.pt"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= train_cfg["patience"]:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val improvement for {train_cfg['patience']} epochs)")
                    break

    print(f"Best val loss: {best_val_loss:.4f}. Checkpoint at {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    sys.exit(main())
