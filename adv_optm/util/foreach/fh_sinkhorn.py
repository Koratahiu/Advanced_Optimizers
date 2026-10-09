import math
import torch

# foreach version of adv_optm\util\sinkhorn.py


def foreach_apply_sr_sinkhorn(
    updates: list[torch.Tensor],
    iters: int = 5,
    params: list[torch.Tensor] | None = None,
    ortho_project: bool = False,
) -> list[torch.Tensor]:
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

    # Cast to float for numerical stability
    updates = [u.float() for u in updates]

    mat_updates = [u.view(u.shape[0], -1) for u in updates]
    mat_params = [
        p.view(p.shape[0], -1) if p is not None else None for p in (params if params is not None else [None] * len(updates))
    ]

    mat_results = _foreach_sinkhorn_matrix(mat_updates, mat_params, iters, ortho_project)

    return [r.view(s).to(d) for r, s, d in zip(mat_results, original_shapes, original_dtypes)]


def _foreach_sinkhorn_matrix(
    updates_2d: list[torch.Tensor],
    params_2d: list[torch.Tensor | None],
    iters: int,
    ortho_project: bool,
) -> list[torch.Tensor]:
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
            g = _foreach_ortho_project_matrix(g, w, p_norm_sq_dim, dims[it % len(g)])

        # Second normalization step (1-dim)
        norm2 = [
            torch.linalg.vector_norm(g_i, ord=2, dim=1 - dims[it % len(g)], keepdim=True)
            for g_i in g
        ]
        torch._foreach_clamp_min_(norm2, [norm_lb_adims[j] for j in range(len(g))])
        torch._foreach_mul_(g, [scale_seconds[j] / norm2[j] for j in range(len(g))])

        if ortho_project and w[0] is not None:
            g = _foreach_ortho_project_matrix(g, w, p_norm_sq_adim, 1 - dims[it % len(g)])

    return g


def _foreach_ortho_project_matrix(
    grads: list[torch.Tensor],
    params: list[torch.Tensor],
    p_norm_sq: list[torch.Tensor],
    dim: int,
) -> list[torch.Tensor]:
    """
    Projects each gradient to be orthogonal to the corresponding parameter
    along `dim` and restores the original norm.
    """
    target_norm = [torch.linalg.vector_norm(g_i, dim=dim, keepdim=True) for g_i in grads]
    norm_lb = 1 / math.sqrt(grads[0].shape[dim])
    torch._foreach_clamp_min_(target_norm, 1e-8)

    # Project: g_orth = g - (p * <p, g> / ||p||^2)
    dots = [torch.sum(p_i * g_i, dim=dim, keepdim=True) for p_i, g_i in zip(params, grads)]
    projs = torch._foreach_div_(dots, p_norm_sq)
    torch._foreach_addcmul_(grads, projs, params, value=-1.0)

    # Magnitude Preservation
    g_orth_norms = [torch.linalg.vector_norm(g_i, dim=dim, keepdim=True).clamp_min_(norm_lb) for g_i in grads]
    scales = torch._foreach_div(target_norm, g_orth_norms)
    torch._foreach_mul_(grads, scales)
    return grads


def foreach_get_sinkhorn_wd_scaler(
    params: list[torch.Tensor],
    row_denom: list[torch.Tensor] | None = None,
    col_denom: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """
    Computes a structural weight decay multiplier for a list of parameters
    using foreach operations where available.
    Penalizes parameters belonging to dominant rows/columns more heavily,
    while protecting parameters in under-utilized/noisy rows/columns from decay.
    """
    results = []
    for p in params:
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

        if row_denom is not None:
            # Find corresponding denom indices (assumes same ordering as params)
            idx = params.index(p)
            rd = row_denom[idx].sqrt().view(p_2d.shape[0], 1)
            cd = col_denom[idx].sqrt().view(1, p_2d.shape[1]) if col_denom is not None else None

            # High denom (noise) -> smaller angle (protects weights)
            # Low denom (confident) -> larger angle (decays weights)
            row_factor.atan2_(rd)
            if cd is not None:
                col_factor.atan2_(cd)

        # Outer product: merges the row and column confidences into a 2D matrix
        wd_scaler = row_factor * col_factor

        # Normalize the scaler so its mean is exactly 1.0
        wd_scaler.div_(wd_scaler.mean().clamp_min_(1e-12))

        results.append(wd_scaler.view_as(p))

    return results
