"""
Gradio demo for Hugging Face Spaces: upload/record audio -> hear the DTLN-
enhanced version, and see a before/after spectrogram.

Runs the same causal ONNX streaming graph as serving/app.py (src/onnx_infer.py)
-- this is the real real-time inference path, not an offline shortcut.

To deploy: copy this file, requirements.txt, src/onnx_infer.py, src/__init__.py,
and the onnx/ directory to the root of a Hugging Face Space (Spaces expects
app.py at the repo root). See README.md's "Demo" section for exact steps.
"""
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.onnx_infer import StreamingEnhancer, resample_audio  # noqa: E402

ONNX_DIR = str(Path(__file__).resolve().parent.parent / "onnx")
enhancer = StreamingEnhancer(ONNX_DIR)


def spectrogram_db(audio: np.ndarray, sr: int):
    n_fft = 512
    hop = 128
    stft = np.array([
        np.fft.rfft(audio[i:i + n_fft] * np.hanning(n_fft), n=n_fft)
        for i in range(0, max(len(audio) - n_fft, 1), hop)
    ])
    mag_db = 20 * np.log10(np.abs(stft).T + 1e-6)
    return mag_db


def plot_comparison(noisy: np.ndarray, enhanced: np.ndarray, sr: int):
    noisy_db = spectrogram_db(noisy, sr)
    enhanced_db = spectrogram_db(enhanced, sr)
    vmin = np.percentile(np.concatenate([noisy_db, enhanced_db]), 2)
    vmax = np.percentile(np.concatenate([noisy_db, enhanced_db]), 98)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, data, title in zip(axes, [noisy_db, enhanced_db], ["Noisy input", "Enhanced output"]):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="magma",
                        vmin=vmin, vmax=vmax,
                        extent=[0, len(noisy) / sr, 0, sr / 2])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes, label="Magnitude (dB)", fraction=0.03, pad=0.02)
    return fig


def enhance_audio(audio):
    if audio is None:
        return None, None
    sr, data = audio
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak  # gr.Audio gives int16-range floats; normalize to [-1, 1]

    audio_16k = resample_audio(data, sr, enhancer.sample_rate)
    enhanced = enhancer.enhance(audio_16k)

    fig = plot_comparison(audio_16k, enhanced, enhancer.sample_rate)
    return (enhancer.sample_rate, enhanced), fig


demo = gr.Interface(
    fn=enhance_audio,
    inputs=gr.Audio(sources=["upload", "microphone"], label="Noisy audio"),
    outputs=[
        gr.Audio(label="Enhanced audio"),
        gr.Plot(label="Before / after spectrogram"),
    ],
    title="DTLN Real-Time Speech Enhancement",
    description=(
        "Causal LSTM-based speech enhancement (DTLN), running through the same "
        "ONNX streaming graph used for real-time inference "
        f"(~{1000 * enhancer.latency / enhancer.sample_rate:.0f}ms algorithmic latency). "
        "Upload or record noisy speech to hear it denoised."
    ),
)

if __name__ == "__main__":
    demo.launch()
