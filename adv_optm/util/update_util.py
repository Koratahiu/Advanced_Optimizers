import torch

import math

def _grams_update(mt: torch.Tensor, grad: torch.Tensor, inplace: bool=False):
    """
    Applies the update rule of "Gradient Descent with Adaptive Momentum Scaling"
    (https://arxiv.org/abs/2412.17107).
    """
    if inplace:
        return mt.abs_().mul_(grad.sign())
    return grad.sign().mul_(mt.abs())

def _cautious_update(mt: torch.Tensor, grad: torch.Tensor, inplace: bool=False):
    """
    Applies the update rule of "Cautious Optimizers: Improving Training with One
    Line of Code" (https://arxiv.org/abs/2411.16085).
    """
    mask = (mt * grad > 0).to(grad.dtype)
    mask.div_(mask.mean().clamp_min_(1e-3))
    if inplace:
        update_mt = mt.mul_(mask)
    else:
        update_mt = mt.mul(mask)
    del mask
    return update_mt

def _scale_sim_AdEMAMix_update(beta: float, current_step: int, alpha_grad: float, lr: float, scaled_optm: bool=False):
    if scaled_optm:
        return lr
    momentum_scale = (1 - beta ** current_step) / (1 - beta)
    total_scale = 1 / (momentum_scale + alpha_grad)
    lr = lr * total_scale
    return lr


def _init_fisher_wd_scaler(group: dict, state: dict, p: torch.Tensor) -> torch.Tensor | None:
    if not group.get('fisher_wd', False):
        return

    state["wd_scaler"] = torch.tensor(1.0, device=p.device)

def _get_fisher_wd_scaler(group: dict, stored_scaler: torch.Tensor, p: torch.Tensor, denom: torch.Tensor, atan2: bool) -> torch.Tensor | None:
    """
    Calculates the Fisher weight decay scaler.
    Maps the decay direction through the empirical Fisher information matrix
    and clips its RMS to ensure stability.
    From the paper:
    "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information"
    """
    if not group.get('fisher_wd', False):
        return None

    if atan2:
        wd_scaler = torch.atan2(stored_scaler, denom).mul_(4 / math.pi)
    else:
        eps = group.get('eps', 1e-8)
        wd_scaler = 1.0 / (denom + eps)

    # Reshape scaler if necessary to match parameter shape (for factored states)
    wd_scaler = wd_scaler.view(p.shape)

    gw_rms = torch.sqrt(torch.mean((p * wd_scaler) ** 2))
    clip_coef = torch.clamp(gw_rms / 1.0, min=1.0)
    return wd_scaler / clip_coef

def _get_l1_adaptive_lr(
    p: torch.Tensor,
    update: torch.Tensor,
    state: dict,
    group: dict,
    kappa_p: float,
    rescale: bool = False,
) -> torch.Tensor:
    """
    Calculates the L1 adaptive learning rate based on gradient heterogeneity.
    """
    if not group.get("l1_adaptive", False) and kappa_p != 1:
        return None

    update_view = update.view(p.shape)

    if rescale:
        momentum = group["momentum"]
        alpha_grad = group["alpha_grad"]

        # Calculate scale factor based on momentum/update magnitude
        scale_factor = _scale_sim_AdEMAMix_update(
            momentum, state["step"] + 1, alpha_grad, 1, False
        )
    else:
        scale_factor = 1

    # Determine dimension for mean calculation based on parameter type
    if getattr(p, '_is_oft', False) or getattr(p, '_is_lora_A', False):
        l1_dim = 1
    elif getattr(p, '_is_lora_B', False):
        l1_dim = 0
    else:
        update_abs = update_view.abs() * scale_factor
        if update_abs.ndim >= 2:
            orig_shape = update_abs.shape
            update_2d = update_abs.view(orig_shape[0], -1)
            mean_l1_norm_2d = torch.outer(update_2d.mean(dim=1), update_2d.mean(dim=0))
            return mean_l1_norm_2d.view(orig_shape)
        else:
            return update_abs.mean()

    mean_l1_norm = update_view.abs().mean(dim=l1_dim, keepdim=True) * scale_factor

    return mean_l1_norm
