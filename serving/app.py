"""
FastAPI real-time-inference serving endpoint for the exported DTLN ONNX model.

Runs the actual causal, frame-by-frame streaming graph (src/onnx_infer.py) —
the same code path benchmarked in scripts/benchmark_rtf.py — not the offline
training-time model. Deliberately has no torch/torchaudio dependency; see
serving/requirements.txt.

Usage:
    ONNX_DIR=onnx uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.onnx_infer import StreamingEnhancer, resample_audio  # noqa: E402

ONNX_DIR = os.environ.get("ONNX_DIR", "onnx")

enhancer: StreamingEnhancer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global enhancer
    enhancer = StreamingEnhancer(ONNX_DIR)
    yield


app = FastAPI(title="DTLN Real-Time Speech Enhancement", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    # Bare domain -> interactive API docs, so clicking the raw link (e.g. from
    # a CV) lands somewhere useful instead of FastAPI's default 404 JSON.
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    if enhancer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "sample_rate": enhancer.sample_rate,
        "algorithmic_latency_ms": 1000 * enhancer.latency / enhancer.sample_rate,
    }


@app.post("/enhance")
async def enhance(file: UploadFile):
    if enhancer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read audio file: {exc}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_audio(audio, sr, enhancer.sample_rate)

    enhanced = enhancer.enhance(audio)

    out_buffer = io.BytesIO()
    sf.write(out_buffer, enhanced, enhancer.sample_rate, format="WAV", subtype="PCM_16")
    out_buffer.seek(0)
    return StreamingResponse(out_buffer, media_type="audio/wav",
                              headers={"Content-Disposition": "attachment; filename=enhanced.wav"})
