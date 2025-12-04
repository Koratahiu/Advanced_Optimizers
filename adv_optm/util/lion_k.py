import torch

def _get_lion_k_update(self, raw_update: torch.Tensor, kappa_p: float) -> torch.Tensor:
    """
    Calculates Gradient of K(x) = ||x||_p based on paper "Lion Secretly Solves
    Constrained Optimization, As Lyapunov Predicts"
    (https://arxiv.org/abs/2310.05898)
    Formula: nabla K(x) = sign(x) * |x|^(p-1) / ||x||_p^(p-1)
    """
    # Standard Lion (p=1) - sign update
    if kappa_p == 1.0:
        return raw_update.sign_()

    # Epsilon to avoid division by zero
    eps = 1e-12

    # Optimization for Spherical Lion (p=2): x / ||x||_2
    if kappa_p == 2.0:
        norm = raw_update.norm(p=2).clamp_(min=eps)
        return raw_update.div_(norm)

    # General p case
    # Numerator: sign(x) * |x|^(p-1)
    # We can compute this as x * |x|^(p-2), but sign * abs^(p-1) is more robust for p < 2
    numerator = raw_update.sign() * raw_update.abs().pow_(kappa_p - 1)

    # Denominator: ||x||_p^(p-1)
    norm_term = raw_update.norm(p=kappa_p).pow_(kappa_p - 1).clamp_(min=eps)

    return numerator.div_(norm_term)