import torch
import math

def _orthogonalize_gradient(p: torch.Tensor, grad: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Projects the gradient `grad` to be orthogonal to the parameter `p`.
    Supports two modes: 'flattened' (vectorized) and 'iterative' (matrix-wise).
    """
    if mode == 'disabled':
        return grad
    elif mode == 'flattened':
        return flattened_ortho_project(p, grad)
    elif mode == 'iterative':
        return iterative_ortho_project(p, grad, iters=3)

def flattened_ortho_project(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Projects the flattened gradient `grad` to be orthogonal to the flattened parameter `p`.
    Modified from:
    https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/720d2444df12b851d6cb417ab08cf125c822b2ae/orthograd.py
    """
    original_shape = grad.shape
    original_dtype = grad.dtype
    w = p.view(-1).float()
    g = grad.view(-1).float()
    w_norm_sq = torch.dot(w, w).add_(1e-30)
    proj = torch.dot(w, g) / w_norm_sq
    g_orth = g.sub(w * proj)
    g_norm = g.norm(2)
    g_orth_norm = g_orth.norm(2).add_(1e-30)
    g_orth_scaled = g_orth * (g_norm / g_orth_norm)
    return g_orth_scaled.view(original_shape).to(original_dtype)


def iterative_ortho_project(p: torch.Tensor, grad: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """
    Applies iterative alternating orthogonal projection to a 2D matrix.
    Projects the grad to be orthogonal to the parameter matrix along
    rows and columns sequentially, alternating dimensions.
    Inspired from Sinkhorn algorithm, 2-3 iterations is enough to converge
    to cosine similarity of -1e4 to -1e-6 for every row/col (semi orthogonal).
    """
    # 1D Vector Case fallback to the standard OrthoGrad
    is_vector = p.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
    if is_vector:
        return _orthogonalize_gradient(p, grad)

    original_shape = grad.shape

    # 2D+ Matrix Case
    grad_2d = grad.view(grad.shape[0], -1)
    param_2d = p.view(p.shape[0], -1)

    m, n = grad_2d.shape

    # Dynamically determine the order based on aspect ratio
    row_first = m > n
    dim = 0 if row_first else 1

    p_norm_sq_dim = torch.sum(param_2d * param_2d, dim=dim, keepdim=True).add_(1e-30)
    p_norm_sq_adim = torch.sum(param_2d * param_2d, dim=1-dim, keepdim=True).add_(1e-30)

    for _ in range(iters):
        # First dimension
        grad_2d = _ortho_normed_dim(param_2d, grad_2d, p_norm_sq_dim, dim)
        # Second dimension
        grad_2d = _ortho_normed_dim(param_2d, grad_2d, p_norm_sq_adim, 1 - dim)

    return grad_2d.view(original_shape)


def _ortho_normed_dim(p_2d: torch.Tensor, grad_2d: torch.Tensor, p_norm_sq: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Projects the grad to be orthogonal to p along 'dim' and dynamically restores 
    the original magnitude of that dimension pre-projection.
    """
    # Record target magnitude before projection
    norm_lb = 1 / math.sqrt(grad_2d.shape[dim])
    target_norm = grad_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(norm_lb)

    # Project: g_orth = g - (p * <p, g> / ||p||^2)
    dot_prod = torch.sum(p_2d * grad_2d, dim=dim, keepdim=True)
    proj = dot_prod / p_norm_sq

    # In-place subtraction: grad_2d = grad_2d - (proj * p_2d)
    # Standard gamma is -1, but -1.01 proved to converge faster
    grad_2d.addcmul_(proj, p_2d, value=-1.01)

    # Magnitude Preservation
    g_orth_norm = grad_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(norm_lb)
    scale_factor = target_norm / g_orth_norm
    return grad_2d.mul_(scale_factor)