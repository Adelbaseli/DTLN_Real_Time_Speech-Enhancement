"""
Verifies the causal StreamingDTLN (src/streaming.py) reproduces the offline
batched DTLN (src/model.py) closely enough to trust it for ONNX export.

Exact bit-for-bit, zero-delay equality isn't expected or possible: the offline
model's `torch.stft(center=True)` is non-causal (frames see frame_len/2 samples
of "future" context for free), so the causal stream is a fixed
`algorithmic_latency_samples` behind it — see the module docstring in
src/streaming.py. This test aligns for that shift before comparing.
"""
import torch

from src.losses import si_sdr
from src.model import DTLN
from src.streaming import StreamingDTLN


def test_streaming_matches_offline_after_latency_shift():
    torch.manual_seed(0)
    model = DTLN(frame_len=512, frame_hop=128, encoder_dim=32, hidden_dim=16)
    model.eval()

    streaming_model = StreamingDTLN.from_offline(model)
    streaming_model.eval()

    num_hops = 100
    num_samples = num_hops * model.frame_hop
    waveform = torch.randn(1, num_samples) * 0.1

    with torch.no_grad():
        offline_out = model(waveform)
        streaming_out = streaming_model.forward_stream(waveform)

    assert streaming_out.shape[-1] == num_samples

    # streaming_out[n] corresponds to offline_out[n - latency]: drop a warm-up
    # margin from the front of both, then shift streaming back by `latency`.
    latency = streaming_model.algorithmic_latency_samples
    warmup = model.frame_len
    offline_trimmed = offline_out[:, warmup:num_samples - latency]
    streaming_trimmed = streaming_out[:, warmup + latency:]

    match_db = si_sdr(streaming_trimmed, offline_trimmed).item()
    assert match_db > 40, (
        f"streaming vs. offline SI-SDR match only {match_db:.1f}dB after "
        f"latency alignment (expected > 40dB) — streaming reconstruction is off."
    )


def test_streaming_state_is_batchable():
    torch.manual_seed(0)
    model = DTLN(frame_len=512, frame_hop=128, encoder_dim=32, hidden_dim=16)
    model.eval()
    streaming_model = StreamingDTLN.from_offline(model)

    waveform = torch.randn(3, 20 * model.frame_hop) * 0.1
    with torch.no_grad():
        out = streaming_model.forward_stream(waveform)
    assert out.shape == (3, 20 * model.frame_hop)
