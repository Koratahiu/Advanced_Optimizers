import torch


def apply_stochastic_sign(update: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
    """
    Applies the Stochastic Sign operator S_R(v).
    Uses uniform noise injection to compute the stochastic sign
    """
    R = update.abs().max().clamp_min(1e-12)

    if noise is None:
        noise = torch.rand_like(update) * 2.0 - 1.0
    return torch.sign(update / R + noise, out=update)
