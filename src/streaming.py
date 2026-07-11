"""
Causal, frame-by-frame DTLN for real-time inference and ONNX export.

`src/model.py`'s `DTLN` processes a whole utterance at once with batched
`torch.stft`/`torch.istft` — great for training throughput, but neither
streamable nor ONNX-exportable (LSTM state lives inside a single big call,
and `torch.fft` ops have patchy ONNX opset support). This module reimplements
the same architecture so it can run one `frame_hop`-sized chunk at a time
with LSTM state (and STFT/OLA buffers) carried explicitly between calls —
that's what actually runs in real time / gets exported to ONNX.

Design notes (see plan phase 4):
  - The analysis/synthesis FFT is a fixed (non-trainable) DFT-matrix matmul
    instead of torch.fft, so it lowers to plain MatMul in ONNX.
  - Stage-1 synthesis replicates torch.istft's overlap-add-with-window-power
    normalization (numerator/denominator accumulators) so it matches the
    offline model once the pipeline is warmed up.
  - Stage-2's Conv1d encoder / ConvTranspose1d decoder, when applied to a
    single frame_len-sized window, reduce exactly to a matmul against the
    conv weight — reused directly from the trained offline model.
  - Latency vs. the offline model: the offline model's `center=True` STFT is
    non-causal (each analysis frame is centered on its hop position, so it
    "sees" frame_len/2 samples of future context "for free"). A live stream
    can't do that, so each stage's causal window instead trails behind by
    frame_len-frame_hop samples relative to where the acausal one would
    place it. The two stages compound, giving a fixed algorithmic latency of
    `StreamingDTLN.algorithmic_latency_samples` (2*(frame_len-frame_hop),
    e.g. 768 samples = 48ms @16kHz) — i.e. streaming_output[n] corresponds
    to offline_output[n - algorithmic_latency_samples], not offline_output[n].
    This is standard for overlap-add streaming DSP (see
    tests/test_streaming_parity.py, which aligns for this shift before
    comparing), not a bug or an approximation.
  - Stage-1 and Stage-2 are separate nn.Modules on purpose: scripts/export_onnx.py
    exports them as two independent ONNX graphs, mirroring DTLN's standard
    real-time export split.
"""
import math

import torch
import torch.nn as nn

from src.model import DTLN, SeparationCore


class RealDFT(nn.Module):
    """Onesided real-input DFT of size n_fft, implemented as a fixed matmul
    (equivalent to the analysis half of torch.stft) so it's ONNX-friendly."""

    def __init__(self, n_fft: int):
        super().__init__()
        n_freq = n_fft // 2 + 1
        n = torch.arange(n_fft).unsqueeze(0).float()
        k = torch.arange(n_freq).unsqueeze(1).float()
        angle = 2 * math.pi * k * n / n_fft  # (n_freq, n_fft)
        self.register_buffer("cos_mat", torch.cos(angle))
        self.register_buffer("sin_mat", torch.sin(angle))

    def forward(self, frame: torch.Tensor):
        # frame: (..., n_fft) -> real, imag: (..., n_freq)
        real = frame @ self.cos_mat.T
        imag = -(frame @ self.sin_mat.T)
        return real, imag


class RealIDFT(nn.Module):
    """Inverse of RealDFT: reconstructs the full n_fft-length real frame from
    its onesided (real, imag) spectrum, again as a fixed matmul."""

    def __init__(self, n_fft: int):
        super().__init__()
        n_freq = n_fft // 2 + 1
        n = torch.arange(n_fft).unsqueeze(0).float()
        k = torch.arange(n_freq).unsqueeze(1).float()
        angle = 2 * math.pi * k * n / n_fft  # (n_freq, n_fft)

        weight = torch.full((n_freq, 1), 2.0 / n_fft)
        weight[0, 0] = 1.0 / n_fft
        if n_fft % 2 == 0:
            weight[-1, 0] = 1.0 / n_fft  # Nyquist bin, only present for even n_fft

        self.register_buffer("inv_cos", weight * torch.cos(angle))
        self.register_buffer("inv_sin", -weight * torch.sin(angle))

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # real, imag: (..., n_freq) -> frame: (..., n_fft)
        return real @ self.inv_cos + imag @ self.inv_sin


def zero_state(batch_size: int, hidden_dim: int, device=None):
    h = torch.zeros(1, batch_size, hidden_dim, device=device)
    c = torch.zeros(1, batch_size, hidden_dim, device=device)
    return h, c


class StreamingStage1(nn.Module):
    """Causal FFT-domain masking stage. One call = one frame_hop-sized chunk in,
    one frame_hop-sized chunk out."""

    def __init__(self, frame_len: int, frame_hop: int, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.eps = eps
        n_freq = frame_len // 2 + 1
        self.core = SeparationCore(n_freq, hidden_dim)
        self.dft = RealDFT(frame_len)
        self.idft = RealIDFT(frame_len)
        self.register_buffer("window", torch.hann_window(frame_len))

    def forward(self, hop_in, ana_buf, ola_num, ola_den, h1, c1, h2, c2):
        hop = self.frame_hop
        ana_buf = torch.cat([ana_buf[:, hop:], hop_in], dim=1)

        windowed = ana_buf * self.window
        real, imag = self.dft(windowed)
        mag = torch.sqrt(real ** 2 + imag ** 2 + self.eps).unsqueeze(1)  # (batch,1,n_freq)

        mask, (h1, c1), (h2, c2) = self.core(mag, (h1, c1), (h2, c2))
        mask = mask.squeeze(1)  # (batch, n_freq)

        est_real = mask * real
        est_imag = mask * imag
        synth = self.idft(est_real, est_imag) * self.window  # (batch, frame_len)

        ola_num = ola_num + synth
        ola_den = ola_den + self.window ** 2
        out_hop = ola_num[:, :hop] / (ola_den[:, :hop] + self.eps)

        pad = torch.zeros_like(ola_num[:, :hop])
        ola_num = torch.cat([ola_num[:, hop:], pad], dim=1)
        ola_den = torch.cat([ola_den[:, hop:], pad], dim=1)

        return out_hop, ana_buf, ola_num, ola_den, h1, c1, h2, c2


class StreamingStage2(nn.Module):
    """Causal learned-encoder masking stage. Conv1d/ConvTranspose1d applied to a
    single frame_len window reduce to a plain matmul against the conv weight."""

    def __init__(self, frame_len: int, frame_hop: int, encoder_dim: int, hidden_dim: int):
        super().__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.core = SeparationCore(encoder_dim, hidden_dim)
        # filled in by from_offline(): (encoder_dim, frame_len) matmul weights
        self.register_buffer("encoder_weight", torch.zeros(encoder_dim, frame_len))
        self.register_buffer("decoder_weight", torch.zeros(encoder_dim, frame_len))

    def forward(self, hop_in, enc_buf, dec_buf, h1, c1, h2, c2):
        hop = self.frame_hop
        enc_buf = torch.cat([enc_buf[:, hop:], hop_in], dim=1)

        encoded = enc_buf @ self.encoder_weight.T  # (batch, encoder_dim)
        mask, (h1, c1), (h2, c2) = self.core(encoded.unsqueeze(1), (h1, c1), (h2, c2))
        mask = mask.squeeze(1)

        masked = encoded * mask
        decoded_frame = masked @ self.decoder_weight  # (batch, frame_len)

        dec_buf = dec_buf + decoded_frame
        out_hop = dec_buf[:, :hop]
        pad = torch.zeros_like(out_hop)
        dec_buf = torch.cat([dec_buf[:, hop:], pad], dim=1)

        return out_hop, enc_buf, dec_buf, h1, c1, h2, c2


class StreamingDTLN(nn.Module):
    """Composes StreamingStage1 + StreamingStage2. Convenient for parity
    testing against the offline model; scripts/export_onnx.py exports
    `stage1` and `stage2` as two separate ONNX graphs, which is how this
    actually gets deployed for real-time inference."""

    def __init__(self, frame_len=512, frame_hop=128, encoder_dim=256, hidden_dim=128):
        super().__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.hidden_dim = hidden_dim
        self.stage1 = StreamingStage1(frame_len, frame_hop, hidden_dim)
        self.stage2 = StreamingStage2(frame_len, frame_hop, encoder_dim, hidden_dim)

    @classmethod
    def from_offline(cls, offline: DTLN) -> "StreamingDTLN":
        model = cls(offline.frame_len, offline.frame_hop, offline.encoder_dim,
                     offline.stage1.lstm1.hidden_size)
        model.stage1.core.load_state_dict(offline.stage1.state_dict())
        model.stage2.core.load_state_dict(offline.stage2.state_dict())
        model.stage2.encoder_weight.copy_(offline.encoder.weight.squeeze(1))
        model.stage2.decoder_weight.copy_(offline.decoder.weight.squeeze(1))
        return model

    @property
    def algorithmic_latency_samples(self) -> int:
        """Fixed extra delay of the causal stream vs. the offline model. Each
        stage's causal window trails frame_len-frame_hop samples behind where
        an acausal (center=True/non-padded) equivalent would place it, and
        the two stages compound, so total latency is 2*(frame_len-frame_hop)
        (e.g. 768 samples = 48ms @16kHz for the default 512/128 config). This
        is standard for overlap-add streaming DSP, not a bug — see
        tests/test_streaming_parity.py."""
        return 2 * (self.frame_len - self.frame_hop)

    def init_state(self, batch_size: int, device=None) -> dict:
        return {
            "ana_buf": torch.zeros(batch_size, self.frame_len, device=device),
            "ola_num": torch.zeros(batch_size, self.frame_len, device=device),
            "ola_den": torch.zeros(batch_size, self.frame_len, device=device),
            "enc_buf": torch.zeros(batch_size, self.frame_len, device=device),
            "dec_buf": torch.zeros(batch_size, self.frame_len, device=device),
            "s1_h1": zero_state(batch_size, self.hidden_dim, device),
            "s1_h2": zero_state(batch_size, self.hidden_dim, device),
            "s2_h1": zero_state(batch_size, self.hidden_dim, device),
            "s2_h2": zero_state(batch_size, self.hidden_dim, device),
        }

    @torch.no_grad()
    def forward_stream(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convenience full-utterance runner (loops the two stages hop by hop).
        waveform: (batch, num_samples). Not used by the exported ONNX graphs —
        those export stage1/stage2 forward() directly for per-hop inference."""
        batch, num_samples = waveform.shape
        hop = self.frame_hop
        state = self.init_state(batch, waveform.device)
        n_hops = num_samples // hop
        out_chunks = []

        for i in range(n_hops):
            chunk = waveform[:, i * hop:(i + 1) * hop]

            s1_out, state["ana_buf"], state["ola_num"], state["ola_den"], \
                s1_h, s1_c, s1_h2, s1_c2 = self.stage1(
                    chunk, state["ana_buf"], state["ola_num"], state["ola_den"],
                    *state["s1_h1"], *state["s1_h2"])
            state["s1_h1"], state["s1_h2"] = (s1_h, s1_c), (s1_h2, s1_c2)

            s2_out, state["enc_buf"], state["dec_buf"], \
                s2_h, s2_c, s2_h2, s2_c2 = self.stage2(
                    s1_out, state["enc_buf"], state["dec_buf"],
                    *state["s2_h1"], *state["s2_h2"])
            state["s2_h1"], state["s2_h2"] = (s2_h, s2_c), (s2_h2, s2_c2)

            out_chunks.append(s2_out)

        return torch.cat(out_chunks, dim=1)
