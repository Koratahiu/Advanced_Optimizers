import math
import torch
from torch import Tensor

# foreach version of adv_optm\util\sinkhorn.py


def foreach_apply_sr_sinkhorn(
    updates: list[Tensor] | tuple[Tensor, ...],
    iters: int = 5,
    params: list[Tensor] | tuple[Tensor, ...] | None = None,
    ortho_project: bool = False,
) -> tuple[Tensor, ...]:
    """
    Applies Square-Root Sinkhorn (SR-Sinkhorn) multi-normalization to a list of
    update tensors using torch._foreach ops where available.

    As described in 'Gradient Multi-Normalization for Efficient LLM Training'.

    This technique normalizes each 2D matrix alternatively by its row-wise L2 norm
    and column-wise L2 norm, driving it toward a fixed point that uniformly
    distributes update magnitudes.
    """
    original_shapes = [u.shape for u in updates]
    original_dtypes = [u.dtype for u in updates]

    mat_updates = [u.view(u.shape[0], -1) for u in updates]
    mat_params = [
        p.view(p.shape[0], -1) if p is not None else None for p in (params if params is not None else [None] * len(updates))
    ]

    mat_results = _foreach_sinkhorn_matrix(mat_updates, mat_params, iters, ortho_project)

    return [r.view(s).to(d) for r, s, d in zip(mat_results, original_shapes, original_dtypes)]


def _foreach_sinkhorn_matrix(
    updates_2d: list[Tensor] | tuple[Tensor, ...],
    params_2d: list[Tensor] | tuple[Tensor, ...],
    iters: int,
    ortho_project: bool,
) -> tuple[Tensor, ...]:
    """
    Foreach implementation of SR-Sinkhorn for 2D matrices.
    """
    g = updates_2d
    w = params_2d

    # Determine per-tensor normalization order based on each tensor's aspect ratio
    # (mirrors the original per-tensor logic in apply_sr_sinkhorn)
    scale_firsts = []
    scale_seconds = []
    dims = []
    norm_lb_dims = []
    norm_lb_adims = []
    for g_i in g:
        m_i, n_i = g_i.shape
        scale_cond_i = m_i > n_i
        dim_i = 0 if scale_cond_i else 1
        dims.append(dim_i)
        scale_firsts.append(math.sqrt(m_i if scale_cond_i else n_i))
        scale_seconds.append(math.sqrt(n_i if scale_cond_i else m_i))
        norm_lb_dims.append(1 / math.sqrt(g_i.shape[dim_i]))
        norm_lb_adims.append(1 / math.sqrt(g_i.shape[1 - dim_i]))

    # Pre-compute per-tensor norm-squared along each dim for ortho projection
    if ortho_project and w[0] is not None:
        w_sq = torch._foreach_mul(w, w)
        p_norm_sq_dim = [
            torch.sum(p_sq, dim=dims[j], keepdim=True).add_(1e-30)
            for j, p_sq in enumerate(w_sq)
        ]
        p_norm_sq_adim = [
            torch.sum(p_sq, dim=1 - dims[j], keepdim=True).add_(1e-30)
            for j, p_sq in enumerate(w_sq)
        ]
        del w_sq
    else:
        p_norm_sq_dim = None
        p_norm_sq_adim = None

    for it in range(iters):
        # First normalization step (dim)
        norm1 = [
            torch.linalg.vector_norm(g_i, ord=2, dim=dims[it % len(g)], keepdim=True)
            for g_i in g
        ]
        torch._foreach_clamp_min_(norm1, [norm_lb_dims[j] for j in range(len(g))])
        torch._foreach_mul_(g, [scale_firsts[j] / norm1[j] for j in range(len(g))])

        if ortho_project and w[0] is not None:
            g = _foreach_ortho_project_matrix(g, w, p_norm_sq_dim, dims[it % len(g)], scale_firsts)

        # Second normalization step (1-dim)
        norm2 = [
            torch.linalg.vector_norm(g_i, ord=2, dim=1 - dims[it % len(g)], keepdim=True)
            for g_i in g
        ]
        torch._foreach_clamp_min_(norm2, [norm_lb_adims[j] for j in range(len(g))])
        torch._foreach_mul_(g, [scale_seconds[j] / norm2[j] for j in range(len(g))])

        if ortho_project and w[0] is not None:
            g = _foreach_ortho_project_matrix(g, w, p_norm_sq_adim, 1 - dims[it % len(g)], scale_seconds)

    return g


def _foreach_ortho_project_matrix(
    grads: list[Tensor] | tuple[Tensor, ...],
    params: list[Tensor] | tuple[Tensor, ...],
    p_norm_sq: list[Tensor] | tuple[Tensor, ...],
    dim: int,
    scale_factors: list[float],
) -> tuple[Tensor, ...]:
    """
    Projects each gradient to be orthogonal to the corresponding parameter
    along `dim` and restores the original norm.
    """
    target_norm = [torch.full_like(g_i[:, :1], scale_factors[i]) for i, g_i in enumerate(grads)]
    norm_lb = 1 / math.sqrt(grads[0].shape[dim])
    torch._foreach_clamp_min_(target_norm, 1e-8)

    # Project: g_orth = g - (p * <p, g> / ||p||^2)
    p_dot_g = torch._foreach_mul(params, grads)
    dots = [torch.sum(x_i, dim=dim, keepdim=True) for x_i in p_dot_g]
    projs = torch._foreach_div_(dots, p_norm_sq)
    torch._foreach_addcmul_(grads, projs, params, value=-1.0)

    # Magnitude Preservation
    g_orth_norms = [torch.linalg.vector_norm(g_i, dim=dim, keepdim=True).clamp_min_(norm_lb) for g_i in grads]
    scales = torch._foreach_div(target_norm, g_orth_norms)
    torch._foreach_mul_(grads, scales)
    return grads


def foreach_get_sinkhorn_wd_scaler(
    params: list[Tensor] | tuple[Tensor, ...],
    row_denom: list[Tensor] | tuple[Tensor, ...] | None = None,
    col_denom: list[Tensor] | tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, ...]:
    """
    Computes a structural weight decay multiplier for a list of parameters
    using foreach operations where available.
    Penalizes parameters belonging to dominant rows/columns more heavily,
    while protecting parameters in under-utilized/noisy rows/columns from decay.
    """
    results = []

    row_norms = [torch.linalg.vector_norm(p.view(p.shape[0], -1), ord=2, dim=1, keepdim=True) for p in params]
    col_norms = [torch.linalg.vector_norm(p.view(p.shape[0], -1), ord=2, dim=0, keepdim=True) for p in params]
    torch._foreach_clamp_min_(row_norms, 1e-8)
    torch._foreach_clamp_min_(col_norms, 1e-8)
    torch._foreach_sqrt_(row_norms)
    torch._foreach_sqrt_(col_norms)

    if row_denom:
        torch._foreach_sqrt_(row_denom)
        torch._foreach_sqrt_(col_denom)
        for i, rd in enumerate(row_denom):
            row_norms[i].atan2_(row_denom[i])
            col_norms[i].atan2_(col_denom[i])

    # Outer product: merges the row and column confidences into a 2D matrix
    wd_scaler = torch._foreach_mul(row_norms, col_norms)

    # Normalize the scaler so its mean is 1.0
    # TODO, workaround for torch._foreach_mean
    numels = [ws.numel() for ws in wd_scaler]
    sums = torch._foreach_norm(wd_scaler, ord=1)
    means = torch._foreach_div(sums, numels)
    torch._foreach_clamp_min_(means, 1e-8)
    torch._foreach_div_(wd_scaler, means)

    # Reshape back to original parameter shapes
    return [ws.reshape(p.shape) for ws, p in zip(wd_scaler, params)]
