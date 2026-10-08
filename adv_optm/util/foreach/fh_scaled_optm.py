import torch
from torch import Tensor

import math

from .. import scaled_optm

_OFT_INDICES_CACHE = {}

def foreach_scale_update(
    params: list[Tensor] | tuple[Tensor, ...],
    update: list[Tensor] | tuple[Tensor, ...],
    lr: float,
    u_state: list[Tensor] | tuple[Tensor, ...] | None = None,
    v_state: list[Tensor] | tuple[Tensor, ...] | None = None,
    target_scale: list | None = None
) -> Tensor:
    """
    Applies adaptive scaling to the parameter update based on the parameter's
    role (DoRA, OFT, or LoRA/Full Finetuning).

    Args:
        p: The original parameter tensor.
        update: The computed gradient/update tensor to be scaled.
        lr: The learning rate.
        state: The state dict used for spectral normalization.

    Returns:
        The scaled update tensor.
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
        return apply_spectral_riemannian_oft(params, update, lr, state)

    # LoRA Factors or Full Finetuning weights
    # Scales update to maintain consistent spectral norm across different layer sizes and ranks.
    if spectral_param:
        d_out = update.shape[0]
        d_in = update.numel() // d_out
        target_scale = 1 if getattr(p, '_is_lora_A', False) else math.sqrt(d_out / d_in)
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
    params: list[Tensor] | tuple[Tensor, ...],
    state: dict,
) -> tuple[list[Tensor], list[Tensor]]:
    u_states = []
    v_states = []
    spectral_target = []
    for p in params:
        p_state = state[p]
        if 'spectral_u' in p_state:
            u_states.append(p_state['spectral_u'])
            v_states.append(p_state['spectral_v'])
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
    target_scale: float,
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
