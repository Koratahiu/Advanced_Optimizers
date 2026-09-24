import math
import torch

from . import scaled_optm

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
        row_denom = row_denom.sqrt().view(p_2d.shape[0], 1)
        col_denom = col_denom.sqrt().view(1, p_2d.shape[1])

        # High denom (noise) -> smaller angle (protects weights)
        # Low denom (confident) -> larger angle (decays weights)
        row_factor.atan2_(row_denom)
        col_factor.atan2_(col_denom)

    # Outer product: merges the row and column confidences into a 2D matrix
    wd_scaler = row_factor * col_factor

    # Normalize the scaler so its mean is exactly 1.0
    wd_scaler.div_(wd_scaler.mean().clamp_min_(1e-12))

    return wd_scaler.view_as(p)

def apply_oft_sinkhorn(
    update: torch.Tensor, 
    iters: int = 5, 
    p: torch.Tensor | None = None, 
    ortho_project: bool = False
):
    n_el = update.shape[-1]
    block_size = int((1 + math.sqrt(1 + 8 * n_el)) / 2)
    device, dtype = update.device, update.dtype
    rows, cols = scaled_optm.get_cached_structural_tensors(block_size, device)
    batch_size = update.shape[0]

    # Initialize matrices
    G = torch.zeros(batch_size, block_size, block_size, device=device, dtype=dtype)
    batch_idx = torch.arange(batch_size, device=device)[:, None]

    # Construct skew-symmetric gradient matrix G
    G = G.index_put((batch_idx, rows, cols), update)
    G = G - G.transpose(-2, -1)

    # If OrthoGrad is enabled, construct the skew-symmetric parameter matrix Q
    if ortho_project and p is not None:
        Q = torch.zeros_like(G)
        p_flat = p.view(batch_size, -1)
        Q = Q.index_put((batch_idx, rows, cols), p_flat)
        Q = Q - Q.transpose(-2, -1)
        # Compute the squared Frobenius norm for each block (batch-wise)
        Q_norm_sq = torch.sum(Q * Q, dim=(1, 2), keepdim=True).add_(1e-30)
        # Theoretical norm of a Sinkhorn-normalized matrix to restore magnitude
        target_norm = math.sqrt(block_size) 

    for _ in range(iters):
        norms = torch.linalg.vector_norm(G, ord=2, dim=2, keepdim=True).clamp_min_(1e-12)
        S = norms.sqrt_()
        G.div_(S * S.mT)

        # OrthoGrad step
        if ortho_project and p is not None:
            # Batch-wise Frobenius inner product
            dot_prod = torch.sum(Q * G, dim=(1, 2), keepdim=True)
            proj = dot_prod / Q_norm_sq
            # Subtract projection
            G.addcmul_(proj, Q, value=-1.0)
            # Restore magnitude
            g_orth_norm = torch.linalg.vector_norm(G, ord=2, dim=(1, 2), keepdim=True).clamp_min_(1e-12)
            G.mul_(target_norm / g_orth_norm)

    update.copy_(G[batch_idx, rows, cols])

    # Global normalization
    norm = torch.linalg.vector_norm(update).clamp_min_(1e-12)
    target_norm_global = math.sqrt(update.numel())
    update.mul_(target_norm_global / norm)

    return update

@torch.no_grad()
def apply_spectral_oft_sinkhorn(
    p: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    state: dict,
    iters: int = 5,
    ortho_project: bool = True
) -> torch.Tensor:
    """
    Applies Spectral Normalization directly on the skew-symmetric gradient.
    """
    n_el = p.shape[-1]
    block_size = int((1 + math.sqrt(1 + 8 * n_el)) / 2)
    device, dtype = p.device, p.dtype
    rows, cols = scaled_optm.get_cached_structural_tensors(block_size, device)

    orig_shape = p.shape

    # Align the scale of p with the forward pass
    scale_factor = getattr(p, '_oft_scale_factor', 1.0)
    batch_size = update.shape[0]

    # Initialize matrices
    G = torch.zeros(batch_size, block_size, block_size, device=device, dtype=dtype)
    batch_idx = torch.arange(batch_size, device=device)[:, None]

    # Construct skew-symmetric gradient matrix G
    G = G.index_put((batch_idx, rows, cols), update)
    G = G - G.transpose(-2, -1)

    # If OrthoGrad is enabled, construct the skew-symmetric parameter matrix Q
    if ortho_project and p is not None:
        Q = torch.zeros_like(G)
        p_flat = p.view(batch_size, -1)
        Q = Q.index_put((batch_idx, rows, cols), p_flat)
        Q = Q - Q.transpose(-2, -1)
        # Compute the squared Frobenius norm for each block (batch-wise)
        Q_norm_sq = torch.sum(Q * Q, dim=(1, 2), keepdim=True).add_(1e-30)
        # Theoretical norm of a Sinkhorn-normalized matrix to restore magnitude
        target_norm = math.sqrt(block_size) 

    for _ in range(iters):
        norms = torch.linalg.vector_norm(G, ord=2, dim=2, keepdim=True).clamp_min_(1e-12)
        S = norms.sqrt_()
        G.div_(S * S.mT)

        # OrthoGrad step
        if ortho_project and p is not None:
            # Batch-wise Frobenius inner product
            dot_prod = torch.sum(Q * G, dim=(1, 2), keepdim=True)
            proj = dot_prod / Q_norm_sq
            # Subtract projection
            G.addcmul_(proj, Q, value=-1.0)
            # Restore magnitude
            g_orth_norm = torch.linalg.vector_norm(G, ord=2, dim=(1, 2), keepdim=True).clamp_min_(1e-12)
            G.mul_(target_norm / g_orth_norm)

    update.copy_(G[batch_idx, rows, cols])

    # Spectral Normalization on G
    u_state = state['spectral_u'].unsqueeze(-1).to(dtype)
    v_state = state['spectral_v'].unsqueeze(-1).to(dtype)
    # Power Iteration step to estimate the largest singular value (sigma)
    # Update v (Right Singular Vector)
    v_raw = torch.bmm(G.mT, u_state)
    v_norm = torch.linalg.vector_norm(v_raw, dim=1, keepdim=True)
    candidate_v = v_raw / v_norm.clamp_min(1e-8)
    next_v = torch.where(v_norm >= 1e-6, candidate_v, v_state)
    # Update u (Left Singular Vector)
    u_raw = torch.bmm(G, next_v)
    u_norm = torch.linalg.vector_norm(u_raw, dim=1, keepdim=True)
    candidate_u = u_raw / u_norm.clamp_min(1e-8)
    next_u = torch.where(u_norm >= 1e-6, candidate_u, u_state)
    state['spectral_v'].copy_(next_v.squeeze(-1))
    state['spectral_u'].copy_(next_u.squeeze(-1))

    # Estimate sigma (The spectral norm) for each block
    sigma = torch.sum(next_u * u_raw, dim=1, keepdim=True)

    # Squeeze out the last dimension so shape becomes (batch_size, 1)
    sigma = sigma.squeeze(-1) 

    target_scale = 0.5 * scale_factor
    spectral_eps = 1.0 / (2.0 * math.sqrt(block_size))

    # Apply the clamp and scaling block-wise
    scale = lr * (target_scale / sigma.clamp_min(spectral_eps))

    return update.mul_(scale).view(orig_shape)
