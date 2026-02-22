import torch

from . import param_update

def scale_update(
        p: torch.Tensor,
        update: torch.Tensor,
        lr: float,
        vector_state: torch.Tensor | None = None):
    """
    """
    dora_scale = getattr(p, '_is_dora_scale', False)
    is_oft = getattr(p, '_is_oft', False)

    if dora_scale:
        # Hidden 1D norm
        return rms_normalization(update, dim=None, lr=lr)
    elif is_oft:
        # Achieve block-size invariance O(√rank)
        # OFT matrix [rank, n_elements]
        # where n_elements derived from the block size
        return l2_normalization(update, dim=0, lr=lr)

    elif p.ndim > 2:
        # Full Finetuning O(1)
        # Scales LoRA factors to have the same LR as full finetuning 
        return spectral_normalization(update, vector_state, lr)

    else: # p.ndim == 1
        return rms_normalization(update, dim=None, lr=lr)

def scale_wd(wd, p):
    """
    """
    lora_a = getattr(p, '_is_lora_A', False)
    dora_scale = getattr(p, '_is_dora_scale', False)
    if dora_scale:
        wd = 0
    elif lora_a:
        wd = wd/p.shape[0]
    elif p.ndim > 2:
        wd = wd/p.shape[1]
    else:
        wd = 0
    return wd


@torch.no_grad()
def l2_normalization(update: torch.Tensor, dim: float |None, lr: float):
    """
    """
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    return update.mul_(lr /norm)


@torch.no_grad()
def rms_normalization(update: torch.Tensor, dim: float |None, lr: float):
    """
    """
    n = update.numel()
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    return update.mul_(lr * (n**0.5)/norm)


def is_spectral(p):
    if getattr(p, '_is_lora_A', False):
        return True
    if getattr(p, '_is_lora_B', False):
        return True
    if getattr(p, '_is_oft', False):
        return False
    if getattr(p, '_is_dora_scale', False):
        return False
    if p.ndim == 1:
        return False
    if not getattr(p, 'is_hidden', True):
        return False
    return True

@torch.no_grad()
def init_spectral_norm(group, state, p):
    device = p.device
    dtype = p.dtype
    if group.get('spectral_normalization', False):
        gen = param_update.get_generator(device)
    d_in_flat = p.numel() // p.shape[0]
    state['spectral_v'] = torch.randn(d_in_flat, device=device, dtype=dtype, generator=gen)
    state['spectral_v'].div_(state['spectral_v'].norm())

@torch.no_grad()
def spectral_normalization(update: torch.Tensor, vector_state: torch.Tensor, lr):
    """
    From the paper:
    "Scalable Optimization in the Modular Norm" (https://arxiv.org/abs/2405.14813)
    Applies explicit Spectral Normalization.
    """
    shape = update.shape
    d_out, d_in = shape[0], shape[1]
    if len(shape) > 2:
        d_in = shape[1:].numel()
    target_scale = (d_out / d_in) ** 0.5
    u = torch.mv(update, vector_state)
    v_new = torch.mv(update.mT, u)
    v_norm = torch.linalg.vector_norm(v_new)
    candidate_v = v_new / v_norm
    next_state = torch.where(v_norm >= 0.5, candidate_v, vector_state)
    vector_state.copy_(next_state.to(vector_state.dtype))
    Av = torch.mv(update, vector_state)
    sigma = torch.linalg.vector_norm(Av)
    update.mul_(lr * (target_scale / sigma.clamp_min_(1e-12)))
    return update
