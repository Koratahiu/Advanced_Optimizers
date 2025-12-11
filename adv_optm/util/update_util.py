import torch

def _grams_update(mt: torch.tensor, grad: torch.tensor):
    """
    Applies the update rule of "Gradient Descent with Adaptive Momentum Scaling"
    (https://arxiv.org/abs/2412.17107).
    """
    return grad.sign().mul_(mt.abs())

def _cautious_update(mt: torch.tensor, grad: torch.tensor):
    """
    Applies the update rule of "Cautious Optimizers: Improving Training with One
    Line of Code" (https://arxiv.org/abs/2411.16085).
    """
    mask = (mt * grad > 0).to(grad.dtype)
    mask.div_(mask.mean().clamp_(min=1e-3))
    update_mt = mt.mul(mask)
    del mask
    return update_mt
