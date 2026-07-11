#!/usr/bin/env bash
set -e
cd /home/adel/speech-enhancement-rt
source .venv/bin/activate

echo "=== [$(date)] Step 1/4: download_dns_subset.py ==="
python scripts/download_dns_subset.py

echo "=== [$(date)] Step 2/4: preprocess_dataset.py ==="
python scripts/preprocess_dataset.py

echo "=== [$(date)] Step 3/4: make_eval_set.py ==="
python scripts/make_eval_set.py

echo "=== [$(date)] Step 4/4: train.py ==="
python -m src.train --config configs/train.yaml

echo "=== [$(date)] PIPELINE COMPLETE ==="
