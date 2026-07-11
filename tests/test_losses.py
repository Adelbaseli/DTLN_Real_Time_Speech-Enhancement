import torch

from src.losses import NegSISDRLoss, si_sdr


def test_si_sdr_perfect_reconstruction_is_high_and_finite():
    # Regression test: estimate == target exactly means zero residual noise
    # energy, which previously produced literal `inf` (eps was added after
    # the division instead of to the denominator) and silently passed a
    # `> 100` bound check. Real-world equivalent: a "noisy" eval pair whose
    # noise clip happened to be pure silence, so noisy == clean exactly.
    target = torch.randn(4, 4000)
    result = si_sdr(target, target)
    assert torch.isfinite(result).all()
    assert result.min().item() > 100


def test_si_sdr_scale_invariant():
    target = torch.randn(4, 4000)
    scaled = target * 3.7
    result = si_sdr(scaled, target)
    assert torch.isfinite(result).all()
    assert result.min().item() > 100  # a pure rescale is still "perfect"


def test_si_sdr_penalizes_noise():
    target = torch.randn(4, 4000)
    noisy = target + torch.randn(4, 4000) * 2.0
    assert si_sdr(noisy, target).mean().item() < si_sdr(target, target).mean().item()


def test_neg_si_sdr_loss_is_lower_for_better_estimate():
    loss_fn = NegSISDRLoss()
    target = torch.randn(4, 4000)
    good_estimate = target + torch.randn(4, 4000) * 0.01
    bad_estimate = torch.randn(4, 4000)
    assert loss_fn(good_estimate, target) < loss_fn(bad_estimate, target)
