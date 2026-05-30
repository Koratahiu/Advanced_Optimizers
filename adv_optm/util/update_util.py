import torch

import math

def _init_fisher_wd_scaler(group: dict, state: dict, p: torch.Tensor) -> torch.Tensor | None:
    if not group.get('fisher_wd', False):
        return

    state["wd_scaler"] = torch.tensor(1.0, device=p.device)

def _get_fisher_wd_scaler(group: dict, stored_scaler: torch.Tensor, p: torch.Tensor, denom: torch.Tensor, atan2: bool, eps: float | None = None) -> torch.Tensor | None:
    """
    Calculates the Fisher weight decay scaler.
    Maps the decay direction through the empirical Fisher information matrix
    and clips its RMS to ensure stability.
    From the paper:
    "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information"
    """
    if not group.get('fisher_wd', False):
        return None

    if atan2:
        wd_scaler = torch.atan2(stored_scaler, denom).mul_(4 / math.pi)
    else:
        scaled_denom = denom + eps if eps is not None else denom
        wd_scaler = 1.0 / (scaled_denom)

    # Reshape scaler if necessary to match parameter shape (for factored states)
    wd_scaler = wd_scaler.view(p.shape)

    if not atan2:
        gw_rms = torch.sqrt(torch.mean((p * wd_scaler) ** 2))
        clip_coef = torch.clamp(gw_rms / 1.0, min=1.0)
    else:
        clip_coef = 1
    return wd_scaler / clip_coef

