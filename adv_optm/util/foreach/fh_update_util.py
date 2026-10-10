import torch

import math

def _foreach_get_fisher_wd_scaler(group: dict, params: list[torch.Tensor], denom: list[torch.Tensor], eps: float | None = None) -> tuple | None:
    if not group.get('fisher_wd', False):
        return None

    if eps is not None:
        scaled_denom = torch._foreach_add(denom, eps)
        wd_scaler = torch._foreach_reciprocal_(scaled_denom)
    else:
        wd_scaler = torch._foreach_reciprocal(denom) 

    # gw = params * wd_scaler
    gw = torch._foreach_mul(params, wd_scaler)
    # RMS(gw) = norm_2(gw) / sqrt(numel)
    norms = torch._foreach_norm(gw, 2)
    factors = [math.sqrt(p.numel()) for p in params]
    gw_rms = torch._foreach_div_(norms, factors)
    clip_coef = torch._foreach_clamp_min_(gw_rms, 1.0)

    return torch._foreach_div_(wd_scaler, clip_coef)

