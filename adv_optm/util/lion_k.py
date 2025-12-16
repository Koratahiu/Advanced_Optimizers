import math

import torch

def _get_lion_k_update(raw_update: torch.Tensor, kappa_p: float, auto_scale: bool = True) -> torch.Tensor:
    """
    Calculates the update of lion-k proposed in the paper "Lion Secretly Solves Constrained
    Optimization, As Lyapunov Predicts". (https://arxiv.org/abs/2310.05898)

    Auto-Scale Logic:
    Standard Lion (p=1) produces a vector with a global L2 norm of √N.
    Spherical Lion (p=2) produces a vector with a global L2 norm of 1.

    To allow the same Learning Rate to be used for both, we scale p!=1 methods
    by √N so they match the magnitude of Standard Lion.
    """
    eps = 1e-12
    x = raw_update
    p = kappa_p
    N = x.numel()

    # Standard Lion (p=1) - sign update
    if p == 1.0:
        return x.sign_()

    # Spherical Lion (p=2) - rotation invariant
    if p == 2.0:
        # Normalize (L2=1)
        norm = x.norm(p=2).clamp_(min=eps)
        x.div_(norm)

        # Scale (L2=√N) if needed
        if auto_scale:
            x.mul_(math.sqrt(N))

        return x

    # General p case - hybrid optimizer
    # Calculate the 'Direction' Numerator: sign(x) * |x|^(p-1)
    num = x.sign() * x.abs().pow_(p - 1)

    if auto_scale:
        # Force global L2 norm to be √N
        l2_norm = num.norm(p=2).clamp_(min=eps)
        scale_factor = math.sqrt(N)
        # Result = Direction / Current_L2 * Target_L2
        return num.div_(l2_norm).mul_(scale_factor)

    else:
        # Mathematical definition without scaling
        # Denominator: ||x||_p^(p-1)
        norm_term = x.norm(p=p).pow_(p - 1).clamp_(min=eps)
        return num.div_(norm_term)
