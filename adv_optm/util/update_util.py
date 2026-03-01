import torch

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

def _get_l1_adaptive_lr(
    p: torch.Tensor,
    update: torch.Tensor,
    state: dict,
    group: dict,
    kappa_p: float
) -> torch.Tensor:
    """
    Calculates the L1 adaptive learning rate based on gradient heterogeneity.
    """
    if not group.get("l1_adaptive", False) and kappa_p != 1:
        return None

    momentum = group["momentum"]
    alpha_grad = group["alpha_grad"]
    update_view = update.view(p.shape)

    # Calculate scale factor based on momentum/update magnitude
    scale_factor = _scale_sim_AdEMAMix_update(
        momentum, state["step"] + 1, alpha_grad, 1, False
    )

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
