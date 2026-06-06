import torch
import math

from . import param_update

def apply_stochastic_sign_(update: torch.Tensor, noise: torch.Tensor | None, is_vector: bool = False) -> torch.Tensor:
    """
    Applies the Iterative L-infinity Stochastic Sign operator.
    Uses uniform noise injection to compute the stochastic sign.
    """
    if is_vector or update.dim() < 2:
        return update.sign_()

    # Iterative L-infinity Sinkhorn algorithm for 2D+ matrices
    # Step 1: Row Max (every row max is 1.0, all values <= 1.0)
    R_row = torch.linalg.vector_norm(update, ord=float('inf'), dim=1, keepdim=True).clamp_min_(1e-12)
    update.div_(R_row)

    # Step 2: Col Max (every col max is 1.0 and every row max stays 1.0)
    R_col = torch.linalg.vector_norm(update, ord=float('inf'), dim=0, keepdim=True).clamp_min_(1e-12)
    update.div_(R_col)

    if noise is None:
        noise = param_update._get_random_noise_for_sso(update)

    # Final stochastic step: sign(v + U[-1, 1])
    return update.add_(noise).sign_()

def get_signsgd_wd_target(
    p: torch.Tensor,
    denom: torch.Tensor | None = None,
    stochastic_sign: bool = False,
    noise: torch.Tensor | None = None,
    is_vector: bool = False,
):
    """
    Computes a signed weight decay target.
    """
    if stochastic_sign:
        wd_target = apply_stochastic_sign_(p.clone(), noise, is_vector)
    else:
        wd_target = torch.sign(p)

    if denom is not None:
        wd_target.atan2_(denom)
        norm_lb = 1 / math.sqrt(p.numel())
        target_norm = torch.linalg.vector_norm(wd_target, ord=2).clamp_min_(norm_lb)
    else:
        target_norm = math.sqrt(p.numel())

    p_norm = torch.linalg.vector_norm(p, ord=2)

    # Scale the target so it match L2 wd strength
    wd_target.mul_(p_norm / target_norm)

    return wd_target.view_as(p)
