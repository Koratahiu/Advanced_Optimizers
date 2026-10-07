import torch
import math


def foreach_iterative_ortho_project(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    inplace: bool = False,
    iters: int = 3,
) -> list[torch.Tensor]:
    """
    Applies iterative alternating orthogonal projection to 2D parameter/gradient matrices
    using torch._foreach ops. Projects along rows and columns sequentially, alternating dimensions.
    Vector tensors fall back to foreach_flattened_ortho_project.
    """
    vec_params, vec_grads = [], []
    mat_params, mat_grads = [], []
    for p, g in zip(params, grads):
        is_vector = p.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
        if is_vector:
            vec_params.append(p)
            vec_grads.append(g)
        else:
            mat_params.append(p.view(p.shape[0], -1))
            mat_grads.append(g.view(g.shape[0], -1))

    # Vector tensors: use flattened foreach version
    vec_results = foreach_flattened_ortho_project(vec_params, vec_grads, inplace) if vec_params else []

    # Matrix tensors: foreach iterative version
    if mat_params:
        mat_results = _foreach_iterative_ortho_project_matrix(mat_params, mat_grads, iters)
    else:
        mat_results = []

    # Reassemble in original order
    result = []
    vi = mi = 0
    for p, g in zip(params, grads):
        is_vector = p.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
        if is_vector:
            result.append(vec_results[vi])
            vi += 1
        else:
            result.append(mat_results[mi])
            mi += 1
    return result


def _foreach_iterative_ortho_project_matrix(
    params_2d: list[torch.Tensor],
    grads_2d: list[torch.Tensor],
    iters: int,
) -> list[torch.Tensor]:
    """
    Foreach implementation of iterative_ortho_project for 2D matrices.
    """
    g = grads_2d
    w = params_2d
    m, n = w[0].shape

    dim = 0 # row_first

    # Pre-compute per-tensor norm-squared along each dim
    w_sq = torch._foreach_mul(w, w)
    p_norm_sq_dim = [torch.sum(p_sq, dim=dim, keepdim=True).add_(1e-30) for p_sq in w_sq]
    p_norm_sq_adim = [torch.sum(p_sq, dim=1 - dim, keepdim=True).add_(1e-30) for p_sq in w_sq]
    del w_sq
    norm_lb_dim = 1 / math.sqrt(m)
    norm_lb_adim = 1 / math.sqrt(n)
    target_norm_dim = [torch.linalg.vector_norm(g_i, dim=dim, keepdim=True) for g_i in g]
    target_norm_adim = [torch.linalg.vector_norm(g_i, dim=1 - dim, keepdim=True) for g_i in g]
    torch._foreach_clamp_min_(target_norm_dim, 1e-8)
    torch._foreach_clamp_min_(target_norm_adim, 1e-8)

    for _ in range(iters):
        # First dimension (row)
        dots = [torch.sum(w_i * g_i, dim=dim, keepdim=True) for w_i, g_i in zip(w, g)]
        projs = torch._foreach_div_(dots, p_norm_sq_dim)
        torch._foreach_addcmul_(g, projs, w, value=-1.01)
        g_norms = [torch.linalg.vector_norm(g_i, dim=dim, keepdim=True).clamp_min_(norm_lb_dim) for g_i in g]
        torch._foreach_mul_(g, target_norm_dim)
        torch._foreach_div_(g, g_norms)

        # Second dimension (col)
        dots = [torch.sum(w_i * g_i, dim=1 - dim, keepdim=True) for w_i, g_i in zip(w, g)]
        projs = torch._foreach_div_(dots, p_norm_sq_adim)
        torch._foreach_addcmul_(g, projs, w, value=-1.01)
        g_norms = [torch.linalg.vector_norm(g_i, dim=1 - dim, keepdim=True).clamp_min_(norm_lb_adim) for g_i in g]
        torch._foreach_mul_(g, target_norm_adim)
        torch._foreach_div_(g, g_norms)

    return g


def foreach_flattened_ortho_project(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    inplace: bool = False,
) -> list[torch.Tensor]:
    """
    Orthogonally projects gradient(s) onto the tangent space of parameter(s)
    and rescales to preserve the original gradient norm using torch._foreach ops.
    """
    w = params
    g = grads
    # Compute dot(w_i, g_i)
    dots = [torch.dot(w_i.reshape(-1), g_i.reshape(-1)) for w_i, g_i in zip(w, g)]
    # Compute ||w||^2 + 1e-30 across all tensors in 1 multi-tensor launch
    w_norms = torch._foreach_norm(w, 2)
    w_norm_sq = torch._foreach_mul(w_norms, w_norms)
    torch._foreach_add_(w_norm_sq, 1e-30)
    # proj = dot(w, g) / (||w||^2 + 1e-30)
    projs = torch._foreach_div_(dots, w_norm_sq)
    # Compute ||g||
    g_norms = torch._foreach_norm(g, 2)
    # g_orth = g - w * proj
    w_projs = torch._foreach_mul(w, projs)
    if inplace:
        # Modify g in-place: g <- g - w * proj
        torch._foreach_sub_(g, w_projs)
        g_orth = g
    else:
        g_orth = torch._foreach_sub(g, w_projs)
    # Compute ||g_orth|| + 1e-30 using foreach norms
    g_orth_norms = torch._foreach_norm(g_orth, 2)
    torch._foreach_add_(g_orth_norms, 1e-30)
    # scale = ||g|| / (||g_orth|| + 1e-30)
    scales = torch._foreach_div(g_norms, g_orth_norms)
    # In-place rescale g_orth: g_orth_scaled = g_orth * scale
    torch._foreach_mul_(g_orth, scales)
    return g_orth
