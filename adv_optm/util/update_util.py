import torch

import math

def _grams_update(mt: torch.Tensor, grad: torch.Tensor, inplace: bool=False):
    """
    Applies the update rule of "Gradient Descent with Adaptive Momentum Scaling"
    (https://arxiv.org/abs/2412.17107).
    """
    if inplace:
        return mt.abs_().mul_(grad.sign())
    return grad.sign().mul_(mt.abs())

def _cautious_update(mt: torch.Tensor, grad: torch.Tensor, inplace: bool=False):
    """
    Applies the update rule of "Cautious Optimizers: Improving Training with One
    Line of Code" (https://arxiv.org/abs/2411.16085).
    """
    mask = (mt * grad > 0).to(grad.dtype)
    mask.div_(mask.mean().clamp_min_(1e-3))
    if inplace:
        update_mt = mt.mul_(mask)
    else:
        update_mt = mt.mul(mask)
    del mask
    return update_mt

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

