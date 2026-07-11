"""SI-SDR loss — the standard training objective for DTLN-style speech
enhancement (Le Roux et al., "SDR - half-baked or well done?", 2019)."""
import torch


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, per-example. Shapes: (batch, samples)."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj_scale = (estimate * target).sum(dim=-1, keepdim=True) / target_energy
    target_proj = proj_scale * target

    noise = estimate - target_proj
    noise_energy = (noise ** 2).sum(dim=-1) + eps
    ratio = (target_proj ** 2).sum(dim=-1) / noise_energy
    return 10 * torch.log10(ratio + eps)


class NegSISDRLoss(torch.nn.Module):
    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -si_sdr(estimate, target).mean()
