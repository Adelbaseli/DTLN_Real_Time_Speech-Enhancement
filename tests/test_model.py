import torch

from src.model import DTLN


def test_forward_shape_matches_input():
    model = DTLN(frame_len=512, frame_hop=128, encoder_dim=32, hidden_dim=16)
    waveform = torch.randn(2, 16000)
    out = model(waveform)
    assert out.shape == waveform.shape


def test_forward_handles_short_input():
    model = DTLN(frame_len=512, frame_hop=128, encoder_dim=32, hidden_dim=16)
    waveform = torch.randn(1, 600)  # shorter than one frame_len
    out = model(waveform)
    assert out.shape == waveform.shape


def test_output_is_finite():
    model = DTLN(frame_len=512, frame_hop=128, encoder_dim=32, hidden_dim=16)
    waveform = torch.randn(1, 16000)
    out = model(waveform)
    assert torch.isfinite(out).all()
