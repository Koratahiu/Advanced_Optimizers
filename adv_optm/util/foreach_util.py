import torch

import torch
from typing import Sequence, Union, overload


def foreach_flattened_ortho_project(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    inplace: bool = False,
) -> Union[torch.Tensor, list[torch.Tensor]]:
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
    projs = torch._foreach_div(dots, w_norm_sq)

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
