import torch

from . import param_update

import math

def scale_update(
    p: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    vector_state: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Applies adaptive scaling to the parameter update based on the parameter's
    role (DoRA, OFT, or LoRA/Full Finetuning).

    Args:
        p: The original parameter tensor.
        update: The computed gradient/update tensor to be scaled.
        lr: The learning rate.
        vector_state: The singular vector state used for spectral normalization.

    Returns:
        The scaled update tensor.
    """
    is_dora_scale = getattr(p, '_is_dora_scale', False)
    is_oft = getattr(p, '_is_oft', False)

    # DoRA Magnitude Scales (1D) or 1D Bias/Norm layers
    if is_dora_scale or p.ndim == 1:
        return l2_normalization(update, dim=None, lr=lr)

    # Orthogonal Fine-Tuning (OFT)
    # This guarantees O(1) update complexity scaling, independent of block sizes.
    if is_oft:
        n = update.shape[1]
        # Calculate block size (b)
        b = (1 + math.sqrt(1 + 8 * n)) / 2
        target_norm = math.sqrt(b / 8)
        scale = target_norm / math.sqrt(n)
        return rms_normalization(update, dim=1, lr=lr * scale)

    # LoRA Factors or Full Finetuning weights
    # Scales update to maintain consistent spectral norm across different layer sizes and ranks.
    if p.ndim >= 2:
        return spectral_normalization(update, vector_state, lr)

    return update.mul_(lr)


def scale_eps(group: dict, p) -> float:
    """
    Scales Adam eps to be scale-invariant.
    """
    if group.get('scaled_optm', False):
        adaptive_eps = (1.0 / group['n_layers']) * (1.0 / math.sqrt(p.numel()))
    else:
        adaptive_eps = group['eps']
    return adaptive_eps

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

    else:
        # 1D Biases or generic 1D parameters
        # Centered WD safely regularizes the delta without collapsing base feature variance.
        return 0.0, cwd


def scale_wds(wd: float, cwd: float, p: torch.Tensor) -> tuple[float, float]:
    """
    Scales standard weight decay and centered weight decay based on the parameter's
    shape and type to maintain effective regularization strength.
    """
    if p.ndim >= 2:
        fan_in = p.numel() // p.shape[0]
        return wd / fan_in, cwd / fan_in

    # 1D tensors (like DoRA scale and Biases)
    return wd, cwd


@torch.no_grad()
def l2_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs L2 normalization on the update tensor."""
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    return update.mul_(lr / norm)


@torch.no_grad()
def rms_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs Root Mean Square normalization on the update tensor."""
    n = update.numel() if dim is None else update.shape[dim]
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    scale_n = math.sqrt(n)
    return update.mul_(lr * scale_n / norm)


def is_spectral(p: torch.Tensor) -> bool:
    """Determines if a parameter should undergo spectral normalization updates."""
    if getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False):
        return True
    if getattr(p, '_is_oft', False) or getattr(p, '_is_dora_scale', False) or p.ndim == 1:
        return False
    return getattr(p, 'is_hidden', True)

@torch.no_grad()
def init_spectral_norm(group: dict, state: dict, p: torch.Tensor):
    """Initializes the singular vector 'v' for the Power Iteration method."""
    gen = param_update.get_generator(p.device)
    v = torch.randn(p.numel() // p.shape[0], device=p.device, dtype=p.dtype, generator=gen)
    state['spectral_v'] = v.div_(v.norm().clamp_min_(1e-12))

@torch.no_grad()
def spectral_normalization(update: torch.Tensor, vector_state: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Applies Spectral Normalization via a single step of Power Iteration.
    Implementation follows: "Scalable Optimization in the Modular Norm" (arXiv:2405.14813).
    """
    d_out = update.shape[0]
    d_in = update.numel() // d_out
    update = update.to(vector_state.dtype)
    update_flat = update.view(d_out, d_in)
    # Target scale derived from the "Modular Norm" paper
    target_scale = math.sqrt(d_out / d_in)
    # Power Iteration step to estimate the largest singular value (sigma)
    # u = Wv
    u = torch.mv(update_flat, vector_state)
    # v_new = W.T u
    v_new = torch.mv(update_flat.mT, u)

    v_norm = torch.linalg.vector_norm(v_new)

    # Stability: Only update the state if the norm is significant
    candidate_v = v_new / v_norm
    next_state = torch.where(v_norm >= 0.5, candidate_v, vector_state)
    vector_state.copy_(next_state.to(vector_state.dtype))

    Av = torch.mv(update_flat, vector_state)

    # Calculate sigma (the spectral norm)
    sigma = torch.linalg.vector_norm(Av)

    # Rescale update
    scale = lr * (target_scale / sigma.clamp_min_(1e-12))
    return update.mul_(scale)
