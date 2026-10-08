import torch
from torch import Tensor

import math

from .. import scaled_optm

def foreach_scale_update(
    params: list[Tensor] | tuple[Tensor, ...],
    update: list[Tensor] | tuple[Tensor, ...],
    lr: float,
    u_state: list[Tensor] | tuple[Tensor, ...] | None = None,
    v_state: list[Tensor] | tuple[Tensor, ...] | None = None,
    target_scale: list | None = None,
) -> tuple[Tensor, ...]:
    """
    Applies adaptive scaling to the parameter update based on the parameter's
    role (DoRA, OFT, or LoRA/Full Finetuning).

    Args:
        params: The original parameter tensors.
        update: The computed gradient/update tensors to be scaled.
        lr: The learning rate.
        state: The state dict used for spectral normalization.

    Returns:
        The scaled update tensors.
    """
    vector_param = []
    oft_param = []
    spectral_param = []
    for p in params:
        if p.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False):
            vector_param.append(p)
        elif getattr(p, '_is_oft', False):
            oft_param.append(p)
        else:
            spectral_param.append(p)

    # DoRA Magnitude Scales (1D) or 1D Bias/Norm layers
    if vector_param:
        return foreach_max_abs_normalization(update, lr=lr)

    # OFT Block Parameters: shape (k, C(b,2))
    # Direct spectral normalization on the skew-symmetric blocks.
    if oft_param:
        return apply_foreach_spectral_riemannian_oft(params, update, u_state=u_state, v_state=v_state, lr=lr, target_scale=target_scale)

    # LoRA Factors or Full Finetuning weights
    # Scales update to maintain consistent spectral norm across different layer sizes and ranks.
    if spectral_param:
        return foreach_spectral_normalization(update, u_state=u_state, v_state=v_state, lr=lr, target_scale=target_scale)

@torch.no_grad()
def foreach_max_abs_normalization(update: list[Tensor] | tuple[Tensor, ...], lr: float) -> tuple[Tensor, ...]:
    """
    Performs foreach L-infinity (Max Absolute) normalization.
    """
    # ord=float('inf') computes the maximum absolute value
    norm = torch._foreach_norm(update, float('inf'))
    torch._foreach_clamp_min_(norm, 1e-8)
    torch._foreach_div_(update, norm)
    return torch._foreach_mul_(update, lr)

def _foreach_collect_spectral_vars(
    self,
    params: list[Tensor] | tuple[Tensor, ...],
) -> tuple[list[Tensor], list[Tensor]]:
    u_states = []
    v_states = []
    spectral_target = []
    for p in params:
        state = self.state[p]
        if 'spectral_u' in state:
            u_states.append(state['spectral_u'])
            v_states.append(state['spectral_v'])
        d_out = p.shape[0]
        d_in = p.numel() // d_out
        target_scale = 1 if getattr(p, '_is_lora_A', False) else math.sqrt(d_out / d_in)
        spectral_target.append(target_scale)
    return u_states, v_states, spectral_target


@torch.no_grad()
def foreach_spectral_normalization(
    update: list[Tensor] | tuple[Tensor, ...],
    u_state: list[Tensor] | tuple[Tensor, ...],
    v_state: list[Tensor] | tuple[Tensor, ...],
    lr: float,
    target_scale: float | list
) -> tuple[Tensor, ...]:
    """Applies Spectral Normalization via a single step of Power Iteration
    using foreach operations for batched tensor processing.
    """
    n = len(update)
    if n == 0:
        return tuple(update)

    d_outs = [u.shape[0] for u in update]
    d_ins = [u.numel() // d_out for u, d_out in zip(update, d_outs)]

    # Power Iteration: Update v (Right Singular Vector)
    v_raws = [
        torch.mv(update[i].view(d_outs[i], d_ins[i]).mT, u_state[i])
        for i in range(n)
    ]
    v_norms = list(torch._foreach_norm(v_raws))

    # Stability mask: mask = (sign(norm - 1e-6) + 1) / 2  (1.0 if >= 1e-6 else 0.0)
    diff_v = torch._foreach_sub(v_norms, 1e-6)
    torch._foreach_sign_(diff_v)
    mask_v = torch._foreach_div_(torch._foreach_add_(diff_v, 1.0), 2.0)

    # In-place normalization: candidate_v = v_raws / clamp_min(v_norms, 1e-8)
    torch._foreach_clamp_min_(v_norms, 1e-8)
    torch._foreach_div_(v_raws, v_norms)

    # In-place state update: v_state += mask_v * (candidate_v - v_state)
    # Reuses v_raws to avoid any intermediate buffer allocations
    torch._foreach_sub_(v_raws, v_state)
    torch._foreach_mul_(v_raws, mask_v)
    torch._foreach_add_(v_state, v_raws)

    # Power Iteration: Update u (Left Singular Vector)
    u_raws = [
        torch.mv(update[i].view(d_outs[i], d_ins[i]), v_state[i])
        for i in range(n)
    ]
    # u_norms is mathematically sigma = ||u_raw||_2 = u^T u_raw
    sigmas = torch._foreach_norm(u_raws)

    # Stability mask for u
    diff_u = torch._foreach_sub(sigmas, 1e-6)
    torch._foreach_sign_(diff_u)
    mask_u = torch._foreach_div_(torch._foreach_add_(diff_u, 1.0), 2.0)

    # In-place normalization: candidate_u = u_raws / clamp_min(sigmas, 1e-8)
    # Note: cloning sigmas is unnecessary because epsilons >> 1e-8 in step 4
    torch._foreach_clamp_min_(sigmas, 1e-8)
    torch._foreach_div_(u_raws, sigmas)

    # In-place state update: u_state += mask_u * (candidate_u - u_state)
    torch._foreach_sub_(u_raws, u_state)
    torch._foreach_mul_(u_raws, mask_u)
    torch._foreach_add_(u_state, u_raws)

    # Batched Spectral Scaling via Foreach Ops
    # Create per-tensor epsilons in a single batched allocation
    eps_buf = torch.tensor(
        [1.0 / (math.sqrt(d_out) + math.sqrt(d_in)) for d_out, d_in in zip(d_outs, d_ins)],
        device=update[0].device,
        dtype=update[0].dtype,
    )
    eps_tensors = eps_buf.unbind(0)

    # Replaces python loop with 3 fused multi-tensor kernels:
    # scale = (lr * target_scale) / max(sigma, eps)
    torch._foreach_maximum_(sigmas, eps_tensors)
    torch._foreach_reciprocal_(sigmas)
    torch._foreach_mul_(sigmas, lr * target_scale)

    # Apply scaling in-place to all update tensors
    torch._foreach_mul_(update, sigmas)
    return tuple(update)

@torch.no_grad()
def apply_foreach_spectral_riemannian_oft(
    params: list[Tensor] | tuple[Tensor, ...],
    update: list[Tensor] | tuple[Tensor, ...],
    u_state: list[Tensor] | tuple[Tensor, ...],
    v_state: list[Tensor] | tuple[Tensor, ...],
    lr: float,
    target_scale: float | list
) -> tuple[Tensor, ...]:
    """
    Applies Spectral Normalization directly on the skew-symmetric gradient
    using foreach operations for batched tensor processing.
    Groups params by block_size so each group can use full foreach ops.
    """
    from collections import defaultdict

    n = len(update)
    if n == 0:
        return tuple(update)

    # Group params by block_size (different OFT layers can have different blocks)
    groups = defaultdict(list)  # n_el -> list of indices
    for i, p in enumerate(params):
        groups[p.shape[-1]].append(i)

    results = [None] * n

    for n_el in groups:
        indices = groups[n_el]
        block_size = int((1 + math.sqrt(1 + 8 * n_el)) / 2)
        g_params = [params[i] for i in indices]
        g_updates = [update[i] for i in indices]
        device, dtype = g_params[0].device, g_params[0].dtype
        rows, cols = scaled_optm.get_cached_structural_tensors(block_size, device)
        scale_factor = getattr(g_params[0], '_oft_scale_factor', 1.0)
        g_n = len(indices)

        # Construct skew-symmetric G matrices
        G_list = []
        update_flat_list = []
        orig_shapes = []
        for p, upd in zip(g_params, g_updates):
            orig_shapes.append(p.shape)
            upd_flat = upd.view(-1, n_el)
            update_flat_list.append(upd_flat)
            batch_size = upd_flat.shape[0]
            G = torch.zeros(batch_size, block_size, block_size, device=device, dtype=dtype)
            batch_idx = torch.arange(batch_size, device=device)[:, None]
            G = G.index_put((batch_idx, rows, cols), upd_flat)
            G = G - G.transpose(-2, -1)
            G_list.append(G)

        # Index into u/v states for this group only
        g_u_state = [u_state[i] for i in indices]
        g_v_state = [v_state[i] for i in indices]

        # Unsqueeze u/v to 3D for bmm: (batch, block, 1)
        u_3d = [u.unsqueeze(-1) for u in g_u_state]
        v_3d = [v.unsqueeze(-1) for v in g_v_state]

        # Power Iteration - Update v (Right Singular Vector)
        v_raws = [torch.bmm(G.mT, u) for G, u in zip(G_list, u_3d)]
        v_norms = [torch.linalg.vector_norm(v, dim=1, keepdim=True) for v in v_raws]

        # Stability mask: mask = (sign(norm - 1e-6) + 1) / 2
        diff_v = torch._foreach_sub(v_norms, 1e-6)
        torch._foreach_sign_(diff_v)
        mask_v = torch._foreach_div_(torch._foreach_add_(diff_v, 1.0), 2.0)

        # In-place normalization: candidate_v = v_raws / clamp_min(v_norms, 1e-8)
        torch._foreach_clamp_min_(v_norms, 1e-8)
        v_raws = torch._foreach_div(v_raws, v_norms)

        # In-place state update: v_state += mask_v * (candidate_v - v_state)
        diff_v2 = torch._foreach_sub(v_raws, v_3d)
        scaled_v = torch._foreach_mul(diff_v2, mask_v)
        v_3d = torch._foreach_add(v_3d, scaled_v)
        # Write back to global state lists
        for idx, i in enumerate(indices):
            v_state[i] = v_3d[idx].squeeze(-1)

        # Power Iteration - Update u (Left Singular Vector)
        u_raws_pre = [torch.bmm(G, v) for G, v in zip(G_list, v_3d)]
        sigmas = [torch.linalg.vector_norm(u, dim=1, keepdim=True) for u in u_raws_pre]

        # Stability mask for u
        diff_u = torch._foreach_sub(sigmas, 1e-6)
        torch._foreach_sign_(diff_u)
        mask_u = torch._foreach_div_(torch._foreach_add_(diff_u, 1.0), 2.0)

        # Save pre-normalization u_raws for sigma computation
        u_raws_for_sigma = list(torch._foreach_clone(u_raws_pre))

        # In-place normalization: candidate_u = u_raws / clamp_min(sigmas, 1e-8)
        torch._foreach_clamp_min_(sigmas, 1e-8)
        u_raws_pre = torch._foreach_div(u_raws_pre, sigmas)

        # In-place state update: u_state += mask_u * (candidate_u - u_state)
        diff_u2 = torch._foreach_sub(u_raws_pre, u_3d)
        scaled_u = torch._foreach_mul(diff_u2, mask_u)
        u_3d = torch._foreach_add(u_3d, scaled_u)

        # Compute sigma (spectral norm) for each block BEFORE squeezing
        # sigma = sum(next_u * u_raw_pre, dim=1) -> shape (batch_size, 1)
        sigma_list = [torch.sum(u * ur, dim=1) for u, ur in zip(u_3d, u_raws_for_sigma)]

        # Squeeze back to 2D and write to global state lists
        for idx, i in enumerate(indices):
            u_state[i] = u_3d[idx].squeeze(-1)
            v_state[i] = v_3d[idx].squeeze(-1)

        # Scale updates using foreach operations
        target_scale = 0.5 * scale_factor
        spectral_eps = 1.0 / (2.0 * math.sqrt(block_size))

        # Create per-tensor epsilons in a single batched allocation
        eps_buf = torch.tensor(
            [spectral_eps] * g_n,
            device=device,
            dtype=sigma_list[0].dtype,
        )
        eps_tensors = eps_buf.unbind(0)

        # Apply clamp and scaling block-wise via fused foreach ops:
        # scale = lr * target_scale / max(sigma, eps)
        torch._foreach_maximum_(sigma_list, eps_tensors)
        torch._foreach_reciprocal_(sigma_list)
        torch._foreach_mul_(sigma_list, lr * target_scale)

        # Apply scaling in-place to all flat updates
        # sigma_list[i] has shape (batch_size, 1); update_flat_list[i] has shape (batch_size, n_el)
        scaled_updates = torch._foreach_mul(update_flat_list, sigma_list)

        # Reshape back to original shapes and place in results
        for idx, (i, su) in enumerate(zip(indices, scaled_updates)):
            results[i] = su.reshape(orig_shapes[idx])

    return tuple(results)