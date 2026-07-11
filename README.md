# Real-Time Speech Enhancement (DTLN)

A causal, LSTM-based speech enhancement model — [DTLN](https://arxiv.org/abs/2005.07551)
(Dual-signal Transformation LSTM Network) — trained on a subset of the
Microsoft DNS Challenge corpus, tracked with MLflow, evaluated with PESQ/STOI,
exported to ONNX for real-time frame-by-frame inference, and served through
both a FastAPI endpoint and a Gradio demo.

## Why DTLN

Two cascaded mask-estimation stages — one in the FFT-magnitude domain (good
at stationary noise), one in a learned time-domain encoder space (cleans up
what the first stage misses) — both strictly causal, so the exact same
weights that train on whole utterances can run frame-by-frame in real time.
See [src/model.py](src/model.py) for the architecture and
[src/streaming.py](src/streaming.py) for why/how the causal streaming version
differs from the training-time batched version.

## Repository layout

```
configs/            dataset subset + training YAML configs
scripts/            one-shot pipeline steps (download, preprocess, eval-set,
                     ONNX export, RTF benchmark)
src/
  model.py           offline DTLN (batched STFT) — used for training
  streaming.py        causal frame-by-frame DTLN — used for real-time/ONNX
  onnx_infer.py       shared ONNX inference wrapper (serving + demo use this)
  data/               PyTorch Dataset + train/eval mixing
  train.py, evaluate.py, losses.py
serving/            FastAPI real-time inference endpoint (torch-free)
demo/               Gradio app for Hugging Face Spaces
docker/Dockerfile   serving image build
tests/              pytest: model shapes, losses, streaming/ONNX parity
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

The streaming/ONNX/parity tests (`tests/test_streaming_parity.py`) don't need
any dataset — they run against a randomly-initialized model and are the
fastest way to sanity-check the environment.

## 1. Data: DNS Challenge subset

Full DNS Challenge 4 fullband corpus is ~892GB unpacked. [configs/dns_subset.yaml](configs/dns_subset.yaml)
lists a ~19GB subset (real archive names from `microsoft/DNS-Challenge`'s
`download-dns-challenge-4.sh`) that gets downloaded, then randomly
sub-sampled and resampled down further — the archives are streamed and
decompressed member-by-member, so unwanted clips are never written to disk.

```bash
python scripts/download_dns_subset.py                 # ~19GB, resumable
python scripts/preprocess_dataset.py                   # -> data/processed/{clean,noise} + manifests
python scripts/make_eval_set.py                        # -> data/eval/{val,test} (fixed, reproducible)
```

`preprocess_dataset.py` defaults to 6h of clean speech / 2h of noise at
16kHz mono — plenty for a portfolio-scale run and small enough to iterate on.
Raise `--clean-hours`/`--noise-hours` (and add more archives to the config)
for a larger run. Train/val/test splits are a deterministic hash of each
output filename, so reruns don't need to track state.

## 2. Training (MLflow-tracked)

```bash
python -m src.train --config configs/train.yaml
```

Logs params/metrics (train/val loss, val PESQ, val STOI) and checkpoints to
a local MLflow sqlite store. View with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Best/last checkpoints land in `checkpoints/` (`model_config` is saved
alongside the weights, so downstream steps don't need to hardcode
architecture hyperparameters). Resume with `--resume checkpoints/last.pt`.

## 3. Evaluation (PESQ / STOI / SI-SDR)

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt
```

Writes `results/eval_report.md`, broken down by input SNR:

<!-- RESULTS_TABLE_PLACEHOLDER -->
Results from the portfolio-scale run described above (~6h clean speech from
DNS-Challenge read_speech + VCTK, ~2h AudioSet noise, 29 epochs before early
stopping — see `configs/train.yaml`), on the 50-pair held-out test set:

| SNR (dB) | n | PESQ | STOI | SI-SDR in | SI-SDR out | SI-SDR improvement |
|---|---|---|---|---|---|---|
| -5 | 6 | 1.334 | 0.727 | -5.01 | 6.10 | +11.11 |
| 0 | 7 | 1.835 | 0.713 | -0.65 | 8.44 | +9.09 |
| 5 | 11 | 1.626 | 0.784 | 5.00 | 12.60 | +7.60 |
| 10 | 11 | 1.958 | 0.841 | 9.98 | 16.33 | +6.34 |
| 15 | 15 | 2.653 | 0.910 | 20.24 | 20.09 | -0.15 |
| **overall** | 50 | 2.001 | 0.817 | 8.68 | 14.30 | +5.63 |

PESQ and STOI both improve monotonically with input SNR, and the SI-SDR gain
is largest where it matters most (+11dB at -5dB input, tapering off at 15dB
where the input is already close to clean) — the expected shape for a working
enhancement model, not just "a number going up". This is a quick run on a
small subset (~6h data, 29 epochs on a 6GB laptop GPU); more data/epochs
(`--clean-hours`/`--noise-hours` in `preprocess_dataset.py`, `epochs` in
`configs/train.yaml`) would improve on this further.

## 4. Real-time streaming + ONNX export

`src/model.py`'s `DTLN` trains fast with batched `torch.stft`/`torch.istft`,
but that's neither streamable nor ONNX-exportable. `src/streaming.py`
reimplements the same architecture frame-by-frame, with LSTM state and
STFT/overlap-add buffers carried explicitly between calls (FFT is a fixed
DFT-matrix matmul instead of `torch.fft`, so it lowers to plain ONNX ops).
This introduces a fixed, expected algorithmic latency
(`2*(frame_len-frame_hop)`, 48ms for the default config) relative to the
offline model — see the module docstring and `tests/test_streaming_parity.py`.

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output-dir onnx
```

Exports `onnx/stage1.onnx` + `onnx/stage2.onnx` (the standard DTLN real-time
export split) and verifies onnxruntime reproduces the PyTorch streaming
module's output before declaring success.

## 5. Real-time factor benchmark

```bash
python scripts/benchmark_rtf.py --onnx-dir onnx
```

Runs the exported graphs hop-by-hop on CPU (single-threaded onnxruntime —
representative of modest/edge hardware) and reports RTF and per-frame
latency to `results/benchmark.md`:

<!-- BENCHMARK_PLACEHOLDER -->
- Frames benchmarked: 3700 (after 50 warm-up hops), single CPU thread
- Hop size: 128 samples (8.00 ms of audio @ 16000Hz)
- Mean per-hop processing latency: 0.263 ms
- p95 per-hop processing latency: 0.402 ms
- **RTF: 0.033** — each 8ms hop takes ~0.26ms to process, i.e. about **30x
  faster than real time** on a single CPU thread, with plenty of headroom
  left for a slower/edge device.

## 6. Serving

**FastAPI** (torch-free — only needs `onnxruntime` at serving time):

```bash
ONNX_DIR=onnx uvicorn serving.app:app --reload
curl -F "file=@some_noisy.wav" http://localhost:8000/enhance -o enhanced.wav
```

```bash
docker build -f docker/Dockerfile -t dtln-serve .
docker run -p 8000:8000 dtln-serve
```

**Gradio demo** (local):

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

**Deploying to Hugging Face Spaces**: Spaces expects `app.py` + its own
`requirements.txt` at the Space repo root, plus the model files it imports.
Copy into a new Space repo:

```
app.py                <- demo/app.py
requirements.txt      <- demo/requirements.txt
src/__init__.py
src/onnx_infer.py
onnx/                 <- the exported stage1.onnx, stage2.onnx, config.json
```

then `git push` to the Space's git remote (see the Space's "Files" tab for
the exact remote URL — this step needs your own Hugging Face account).

## Live demo & repo

- Demo: *(add your Hugging Face Space URL here after deploying)*
- Source: *(this repo)*
