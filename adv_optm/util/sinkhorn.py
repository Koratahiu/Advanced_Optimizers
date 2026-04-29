import math
import torch

def apply_sr_sinkhorn(update: torch.Tensor, p: torch.Tensor, ortho_project: bool, iters: int = 5) -> torch.Tensor:
    """
    Applies Square-Root Sinkhorn (SR-Sinkhorn) multi-normalization.
    As described in 'Gradient Multi-Normalization for Efficient LLM Training'.

    This technique normalizes a 2D matrix alternatively by its row-wise L2 norm 
    and column-wise L2 norm, driving it toward a fixed point that uniformly 
    distributes update magnitudes.
    """
    original_shape = update.shape
    original_dtype = update.dtype
    update = update.float()

    # 1D Vector Case
    if update.dim() == 1:
        if ortho_project:
            p_float = p.float()
            p_norm_sq = torch.dot(p_float, p_float).add_(1e-30)
            proj = torch.dot(p_float, update) / p_norm_sq
            update.sub_(p_float * proj) 
        norm = update.norm(p=2).clamp_min_(1e-12)
        return update.mul_(math.sqrt(update.numel()) / norm).view(original_shape).to(original_dtype)

    # 2D+ Matrix Case
    update_2d = update.view(update.shape[0], -1)

    m, n = update_2d.shape

    # Dynamically determine the order of normalization based on aspect ratio
    # Normalizing the longer dimension first aids stability.
    scale_cond = update_2d.shape[0] > update_2d.shape[1]
    dim = 0 if scale_cond else 1


    # Precompute scaling factors. 
    scale_first = math.sqrt(m if scale_cond else n)
    scale_second = math.sqrt(n if scale_cond else m)

    if ortho_project:
        # Pre-compute squares for the mathematical trick in ortho_normed
        target_norm_sq_first = scale_first ** 2
        target_norm_sq_second = scale_second ** 2
        param_2d = p.float().view(p.shape[0], -1)
        p_norm_sq_dim = torch.sum(param_2d * param_2d, dim=dim, keepdim=True).add_(1e-30)
        p_norm_sq_adim = torch.sum(param_2d * param_2d, dim=1-dim, keepdim=True).add_(1e-30)

    # In-place alternating Sinkhorn normalization steps
    for _ in range(iters):
        # First normalization step
        norm1 = update_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
        update_2d.mul_(scale_first / norm1)
        if ortho_project:
            update_2d = ortho_normed(param_2d, update_2d, p_norm_sq_dim, dim, scale_first, target_norm_sq_first)

        # Second normalization step
        norm2 = update_2d.norm(p=2, dim=1-dim, keepdim=True).clamp_min_(1e-12)
        update_2d.mul_(scale_second / norm2)
        if ortho_project:
            update_2d = ortho_normed(param_2d, update_2d, p_norm_sq_adim, 1-dim, scale_second, target_norm_sq_second)

    # Final step
    norm1 = update_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    update_2d.mul_(scale_first / norm1)
    if ortho_project:
        update_2d = ortho_normed(param_2d, update_2d, p_norm_sq_dim, dim, scale_first, target_norm_sq_first)

    return update_2d.view(original_shape).to(original_dtype)

def ortho_normed(p_2d, update_2d, p_norm_sq, dim, target_norm, target_norm_sq):
    """
    Projects the update to be orthogonal to p along 'dim' and restores the original norm.
    """
    # Project: g_orth = g - (p * <p, g> / ||p||^2)
    dot_prod = torch.sum(p_2d * update_2d, dim=dim, keepdim=True)
    proj = dot_prod / p_norm_sq

    # In-place subtraction: update_2d = update_2d - (proj * p_2d)
    update_2d.addcmul_(proj, p_2d, value=-1.0)

    # Magnitude Preservation via Pythagorean theorem 
    # ||g_orth||^2 = ||g||^2 - ||proj * p||^2
    proj_norm_sq = (dot_prod ** 2) / p_norm_sq
    g_orth_norm_sq = (target_norm_sq - proj_norm_sq).clamp_min_(1e-30)

    scale_factor = target_norm / torch.sqrt(g_orth_norm_sq)
    return update_2d.mul_(scale_factor)
