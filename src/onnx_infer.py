"""
Shared ONNX streaming-inference wrapper used by both the FastAPI service
(serving/app.py) and the Gradio demo (demo/app.py), so both call the exact
same real-time frame-by-frame code path — not a separate "offline" shortcut.
"""
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = math.gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


STAGE1_OUTPUT_NAMES = ["out_hop", "ana_buf_out", "ola_num_out", "ola_den_out",
                        "h1_out", "c1_out", "h2_out", "c2_out"]
STAGE2_OUTPUT_NAMES = ["out_hop", "enc_buf_out", "dec_buf_out",
                        "h1_out", "c1_out", "h2_out", "c2_out"]


class StreamingEnhancer:
    """Loads the two exported ONNX graphs once and runs the causal
    stage1 -> stage2 hop loop over a full waveform, hiding the per-hop state
    bookkeeping from callers."""

    def __init__(self, onnx_dir: str):
        onnx_dir = Path(onnx_dir)
        self.config = json.loads((onnx_dir / "config.json").read_text())
        self.frame_len = self.config["frame_len"]
        self.frame_hop = self.config["frame_hop"]
        self.hidden_dim = self.config["hidden_dim"]
        self.sample_rate = self.config["sample_rate"]
        self.latency = self.config["algorithmic_latency_samples"]

        opts = ort.SessionOptions()
        self.sess1 = ort.InferenceSession(str(onnx_dir / "stage1.onnx"), sess_options=opts,
                                           providers=["CPUExecutionProvider"])
        self.sess2 = ort.InferenceSession(str(onnx_dir / "stage2.onnx"), sess_options=opts,
                                           providers=["CPUExecutionProvider"])

    def _zero_state(self):
        h = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)
        c = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)
        return {
            "s1": {"ana_buf": np.zeros((1, self.frame_len), dtype=np.float32),
                   "ola_num": np.zeros((1, self.frame_len), dtype=np.float32),
                   "ola_den": np.zeros((1, self.frame_len), dtype=np.float32),
                   "h1": h.copy(), "c1": c.copy(), "h2": h.copy(), "c2": c.copy()},
            "s2": {"enc_buf": np.zeros((1, self.frame_len), dtype=np.float32),
                   "dec_buf": np.zeros((1, self.frame_len), dtype=np.float32),
                   "h1": h.copy(), "c1": c.copy(), "h2": h.copy(), "c2": c.copy()},
        }

    def enhance(self, waveform: np.ndarray) -> np.ndarray:
        """waveform: 1D float32 array at self.sample_rate. Returns an array of
        the same length, enhanced (causal, frame-hop at a time internally)."""
        waveform = waveform.astype(np.float32)
        original_len = len(waveform)
        hop = self.frame_hop

        # Pad so the length is a multiple of hop, plus `latency` extra samples
        # of silence at the end to flush the pipeline's algorithmic delay.
        padded_len = original_len + self.latency
        padded_len = ((padded_len + hop - 1) // hop) * hop
        padded = np.zeros(padded_len, dtype=np.float32)
        padded[:original_len] = waveform

        state = self._zero_state()
        n_hops = padded_len // hop
        out_chunks = []

        for i in range(n_hops):
            chunk = padded[i * hop:(i + 1) * hop][None, :]

            s1 = self.sess1.run(STAGE1_OUTPUT_NAMES, {
                "hop_in": chunk, "ana_buf": state["s1"]["ana_buf"],
                "ola_num": state["s1"]["ola_num"], "ola_den": state["s1"]["ola_den"],
                "h1": state["s1"]["h1"], "c1": state["s1"]["c1"],
                "h2": state["s1"]["h2"], "c2": state["s1"]["c2"],
            })
            (out_hop1, state["s1"]["ana_buf"], state["s1"]["ola_num"], state["s1"]["ola_den"],
             state["s1"]["h1"], state["s1"]["c1"], state["s1"]["h2"], state["s1"]["c2"]) = s1

            s2 = self.sess2.run(STAGE2_OUTPUT_NAMES, {
                "hop_in": out_hop1, "enc_buf": state["s2"]["enc_buf"],
                "dec_buf": state["s2"]["dec_buf"],
                "h1": state["s2"]["h1"], "c1": state["s2"]["c1"],
                "h2": state["s2"]["h2"], "c2": state["s2"]["c2"],
            })
            (out_hop2, state["s2"]["enc_buf"], state["s2"]["dec_buf"],
             state["s2"]["h1"], state["s2"]["c1"], state["s2"]["h2"], state["s2"]["c2"]) = s2

            out_chunks.append(out_hop2[0])

        output = np.concatenate(out_chunks)
        return output[self.latency:self.latency + original_len]
