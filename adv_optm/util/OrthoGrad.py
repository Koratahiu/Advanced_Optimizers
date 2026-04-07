import torch

def _orthogonalize_gradient(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Projects the gradient `grad` to be orthogonal to the parameter `p`.
    Modified from:
    https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/720d2444df12b851d6cb417ab08cf125c822b2ae/orthograd.py
    """
    if getattr(p, '_is_oft', False) or getattr(p, '_is_lora_A', False):
        return _orthogonalize_gradient_granular(p, grad, dim=1)
    elif getattr(p, '_is_lora_B', False):
        return _orthogonalize_gradient_granular(p, grad, dim=0)

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

def _orthogonalize_gradient_granular(p: torch.Tensor, grad: torch.Tensor, dim: int = 1, eps: float = 1e-30) -> torch.Tensor:
    """
    Projects the gradient `grad` to be orthogonal to the parameter `p` row/col-wise,
    while preserving the original norm of the gradient for each row/col.
    """
    original_dtype = grad.dtype
    p_f32 = p.float()
    grad_f32 = grad.float()

    # Calculate the dot product <p, grad> for each row/col
    dot_prod = torch.sum(p_f32 * grad_f32, dim=dim, keepdim=True)

    # Calculate ||p||^2 for each row/col
    p_norm_sq = torch.sum(p_f32 * p_f32, dim=dim, keepdim=True).add_(eps)

    # Project: g_orth = g - (p * <p, g> / ||p||^2)
    proj = dot_prod / p_norm_sq
    grad_orth = grad_f32 - (proj * p_f32)

    # Magnitude Preservation
    g_norm = torch.norm(grad_f32, p=2, dim=dim, keepdim=True)
    g_orth_norm = torch.norm(grad_orth, p=2, dim=dim, keepdim=True).add_(eps)
    grad_orth_scaled = grad_orth * (g_norm / g_orth_norm)

    return grad_orth_scaled.to(original_dtype)
