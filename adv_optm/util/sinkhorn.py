import math
import torch

def apply_sr_sinkhorn(update: torch.Tensor, iters: int = 5, p: torch.Tensor | None = None, ortho_project: bool = False) -> torch.Tensor:
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
        param_2d = p.float().view(p.shape[0], -1)
        p_norm_sq_dim = torch.sum(param_2d * param_2d, dim=dim, keepdim=True).add_(1e-30)
        p_norm_sq_adim = torch.sum(param_2d * param_2d, dim=1-dim, keepdim=True).add_(1e-30)

    # In-place alternating Sinkhorn normalization steps
    for _ in range(iters):
        # First normalization step
        # Stability floor: equivalent to a single-element vector norm lower bound (lb)
        norm1_lb = 1 / math.sqrt(update_2d.shape[dim])
        norm1 = update_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(norm1_lb)
        update_2d.mul_(scale_first / norm1)
        if ortho_project:
            update_2d = ortho_normed(param_2d, update_2d, p_norm_sq_dim, dim, scale_first)

        # Second normalization step
        norm2_lb = 1 / math.sqrt(update_2d.shape[1-dim])
        norm2 = update_2d.norm(p=2, dim=1-dim, keepdim=True).clamp_min_(norm2_lb)
        update_2d.mul_(scale_second / norm2)
        if ortho_project:
            update_2d = ortho_normed(param_2d, update_2d, p_norm_sq_adim, 1-dim, scale_second)

    return update_2d.view(original_shape).to(original_dtype)

def ortho_normed(p_2d, update_2d, p_norm_sq, dim, target_norm):
    """
    Projects the update to be orthogonal to p along 'dim' and restores the original norm.
    """
    # Project: g_orth = g - (p * <p, g> / ||p||^2)
    dot_prod = torch.sum(p_2d * update_2d, dim=dim, keepdim=True)
    proj = dot_prod / p_norm_sq

    # In-place subtraction: update_2d = update_2d - (proj * p_2d)
    update_2d.addcmul_(proj, p_2d, value=-1.0)

    # Magnitude Preservation
    norm_lb = 1 / math.sqrt(update_2d.shape[dim])
    g_orth_norm = update_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(norm_lb)
    scale_factor = target_norm / g_orth_norm
    return update_2d.mul_(scale_factor)

def _sinkhorn_sq_grad(
    vt_row: torch.Tensor,
    vt_col: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstructs the variance precondition from its rank-1 factors.
    Modified from:
    https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/adafactor.py
    """
    r_factor = (
        (vt_row / vt_row.mean(dim=-1).clamp_min_(1e-30))
        .sqrt_()
        .unsqueeze(-1)
    )
    c_factor = vt_col.unsqueeze(-2).sqrt()
    return torch.mul(r_factor, c_factor) 

def get_sinkhorn_wd_scaler(
    p: torch.Tensor, 
    row_denom: torch.Tensor | None = None, 
    col_denom: torch.Tensor | None = None
):
    """
    Computes a structural weight decay multiplier.
    Penalizes parameters belonging to dominant rows/columns more heavily, 
    while protecting parameters in under-utilized/noisy rows/columns from decay.
    """
    if p.ndim < 2:
        return 1.0 

    p_2d = p.view(p.shape[0], -1)

    # Lower bounds based on the effective 2D shapes
    row_lb = 1 / math.sqrt(p_2d.shape[1])
    col_lb = 1 / math.sqrt(p_2d.shape[0])

    # Get the norms
    row_norms = torch.linalg.vector_norm(p_2d, ord=2, dim=1, keepdim=True).clamp_min_(row_lb)
    col_norms = torch.linalg.vector_norm(p_2d, ord=2, dim=0, keepdim=True).clamp_min_(col_lb)

    # Compute the structural scaler
    row_factor = row_norms.sqrt_()
    col_factor = col_norms.sqrt_()

    if row_denom is not None and col_denom is not None:
        # Reshape denominators to ensure safe in-place broadcasting
        row_denom = row_denom.view(p_2d.shape[0], 1)
        col_denom = col_denom.view(1, p_2d.shape[1])

        # High denom (noise) -> smaller angle (protects weights)
        # Low denom (confident) -> larger angle (decays weights)
        row_factor.atan2_(row_denom)
        col_factor.atan2_(col_denom)

    # Outer product: merges the row and column confidences into a 2D matrix
    wd_scaler = row_factor * col_factor

    # Normalize the scaler so its mean is exactly 1.0
    wd_scaler.div_(wd_scaler.mean().clamp_min_(1e-12))

    return wd_scaler.view_as(p)
