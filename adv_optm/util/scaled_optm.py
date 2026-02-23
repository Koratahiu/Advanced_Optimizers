import torch

from . import param_update

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
        return rms_normalization(update, dim=None, lr=lr)

    # Orthogonal Fine-Tuning (OFT)
    # Scales update to achieve block-size invariance O(sqrt(rank))
    elif is_oft:
        return l2_normalization(update, dim=1, lr=lr)

    # LoRA Factors or Full Finetuning weights
    # Scales update to maintain consistent spectral norm across different layer sizes and ranks.
    elif p.ndim >= 2:
        return spectral_normalization(update, vector_state, lr)

    return update.mul_(lr)

def scale_wd(wd: float, p: torch.Tensor) -> float:
    """
    Adjusts weight decay based on the parameter's shape and type to maintain 
    effective regularization strength across varying architectures.
    """
    is_lora_a = getattr(p, '_is_lora_A', False)
    is_dora_scale = getattr(p, '_is_dora_scale', False)

    if is_dora_scale:
        return 0.0  # No WD on DoRA scales

    if is_lora_a:
        # Normalize WD by the rank (in_features)
        return wd / p.shape[0]

    if p.ndim >= 2:
        # Normalize WD by the width
        return wd / p.shape[1:].numel()

    return 0.0


@torch.no_grad()
def l2_normalization(update: torch.Tensor, dim: int |None, lr: float) -> torch.Tensor:
    """Performs L2 normalization on the update tensor."""
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    return update.mul_(lr / norm)


@torch.no_grad()
def rms_normalization(update: torch.Tensor, dim: int |None, lr: float) -> torch.Tensor:
    """Performs Root Mean Square normalization on the update tensor."""
    n = update.numel() if dim is None else update.shape[dim]
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    return update.mul_(lr * (n**0.5) / norm)


def is_spectral(p: torch.Tensor) -> bool:
    """Determines if a parameter should undergo spectral normalization updates."""
    if getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False):
        return True
    if getattr(p, '_is_oft', False) or getattr(p, '_is_dora_scale', False):
        return False
    if p.ndim == 1:
        return False
    return getattr(p, 'is_hidden', True)

@torch.no_grad()
def init_spectral_norm(group: dict, state: dict, p: torch.Tensor):
    """Initializes the singular vector 'v' for the Power Iteration method."""
    device = p.device
    dtype = p.dtype
    d_in_flat = p.numel() // p.shape[0]
    gen = param_update.get_generator(device)
    v = torch.randn(d_in_flat, device=device, dtype=dtype, generator=gen)
    state['spectral_v'] = v.div_(v.norm().clamp_min_(1e-12))

@torch.no_grad()
def spectral_normalization(update: torch.Tensor, vector_state: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Applies Spectral Normalization via a single step of Power Iteration.
    Implementation follows: "Scalable Optimization in the Modular Norm" (arXiv:2405.14813).

    This ensures the update's spectral norm is scaled proportionally to the 
    ideal 'target_scale' derived from the weight dimensions.
    """
    original_shape = update.shape
    d_out = original_shape[0]
    d_in = original_shape[1:].numel()
    # Reshape for matrix operations if it's a higher-order tensor (e.g. Conv)
    update_flat = update.view(d_out, d_in)
    # Target scale derived from the "Modular Norm" paper
    target_scale = (d_out / d_in) ** 0.5
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
