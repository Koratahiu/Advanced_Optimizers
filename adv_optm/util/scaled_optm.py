import torch

from . import param_update

import math

def scale_update(
    p: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    state: dict | None = None,
    depth: int = 1,
) -> torch.Tensor:
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
    is_dora_scale = getattr(p, '_is_dora_scale', False)

    # DoRA Magnitude Scales (1D) or 1D Bias/Norm layers
    if p.ndim < 2 or is_dora_scale:
        return l2_normalization(update, dim=None, lr=lr)


    # LoRA Factors or Full Finetuning weights
    # Scales update to maintain consistent spectral norm across different layer sizes and ranks.
    if p.ndim >= 2:
        return spectral_normalization(update, state['spectral_u'], state['spectral_v'], lr, depth)


def scale_eps(eps: float | None, p: torch.Tensor) -> float:
    """
    Scales Adam eps to be scale-invariant.
    """
    if eps is None:
        return (1.0 / math.sqrt(p.numel()))
    else:
        return eps

def adjust_wds(wd: float, cwd: float, p: torch.Tensor) -> tuple[float, float]:
    """
    Adjusts standard weight decay and centered weight decay.
    """
    # DoRA Scale (Magnitude Vector)
    if getattr(p, '_is_dora_scale', False):
        return wd, cwd

    if getattr(p, '_is_oft', False):
        return wd, 0.0

    if p.ndim >= 2:
        is_lora = getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False)
        if is_lora:
            return wd, 0.0

        return wd, cwd

    else:
        # 1D Biases or generic 1D parameters
        # Centered WD safely regularizes the delta without collapsing base feature variance.
        return 0.0, cwd


def scale_wds(wd: float, cwd: float, p: torch.Tensor) -> tuple[float, float]:
    """
    Scales standard weight decay and centered weight decay based on the parameter's
    shape and type to maintain effective regularization strength.
    """
    is_lora = getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False)
    if is_lora:
        return wd, cwd

    if p.ndim >= 2:
        fan_in = p.numel() // p.shape[0]
        return wd / fan_in, cwd / fan_in

    # 1D tensors (like DoRA scale and Biases)
    return wd, cwd


@torch.no_grad()
def l2_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs L2 normalization on the update tensor."""
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-8)
    return update.mul_(lr / norm)


@torch.no_grad()
def rms_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs Root Mean Square normalization on the update tensor."""
    n = update.numel() if dim is None else update.shape[dim]
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-8)
    scale_n = math.sqrt(n)
    return update.mul_(lr * scale_n / norm)


def is_spectral(p: torch.Tensor) -> bool:
    """Determines if a parameter should undergo spectral normalization updates."""
    if p.ndim < 2:
        return False
    return getattr(p, 'is_hidden', True)

@torch.no_grad()
def init_spectral_norm(state: dict, p: torch.Tensor):
    """Initializes the singular vectors 'u' and 'v' for the Power Iteration method."""
    gen = param_update.get_generator(p.device)

    d_out = p.shape[0]
    d_in = p.numel() // d_out

    # Initialize v (Right singular vector)
    v = torch.randn(d_in, device=p.device, dtype=p.dtype, generator=gen)
    state['spectral_v'] = v.div_(v.norm().clamp_min_(1e-8))

    # Initialize u (Left singular vector)
    u = torch.randn(d_out, device=p.device, dtype=p.dtype, generator=gen)
    state['spectral_u'] = u.div_(u.norm().clamp_min_(1e-8))

@torch.no_grad()
def spectral_normalization(
    update: torch.Tensor, 
    u_state: torch.Tensor, 
    v_state: torch.Tensor, 
    lr: float,
    depth: int
) -> torch.Tensor:
    """
    Applies Spectral Normalization via a single step of Power Iteration.
    Implementation follows: "Scalable Optimization in the Modular Norm" (arXiv:2405.14813).
    """
    d_out = update.shape[0]
    d_in = update.numel() // d_out
    update = update.to(u_state.dtype)
    update_flat = update.view(d_out, d_in)

    # Target scale derived from the "Modular Norm" paper
    target_scale = math.sqrt(d_out / d_in) / depth

    # Power Iteration step to estimate the largest singular value (sigma)
    # Update v (Right Singular Vector)
    v_raw = torch.mv(update_flat.mT, u_state)
    v_norm = torch.linalg.vector_norm(v_raw)
    candidate_v = v_raw / v_norm.clamp_min(1e-8)
    # Stability: Only update the state if the norm is significant
    next_v = torch.where(v_norm >= 0.5, candidate_v, v_state)
    v_state.copy_(next_v)

    # Update u (Left Singular Vector)
    u_raw = torch.mv(update_flat, v_state)
    u_norm = torch.linalg.vector_norm(u_raw)
    candidate_u = u_raw / u_norm.clamp_min(1e-8)
    next_u = torch.where(u_norm >= 0.5, candidate_u, u_state)
    u_state.copy_(next_u)

    # Estimate sigma (The spectral norm)
    sigma = torch.linalg.vecdot(u_state, u_raw)

    norm_eps = 1 / (math.sqrt(d_in) + math.sqrt(d_out))

    # Rescale update
    scale = lr * (target_scale / sigma.add_(norm_eps))
    return update.mul_(scale)
