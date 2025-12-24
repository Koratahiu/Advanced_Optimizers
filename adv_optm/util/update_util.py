import torch

def _grams_update(mt: torch.Tensor, grad: torch.Tensor, inplace: bool=False):
    """
    Applies the update rule of "Gradient Descent with Adaptive Momentum Scaling"
    (https://arxiv.org/abs/2412.17107).
    """
    if inplace:
        return mt.mul_(grad.sign())
    return grad.sign().mul_(mt.abs())

def _cautious_update(mt: torch.Tensor, grad: torch.Tensor, inplace: bool=False):
    """
    Applies the update rule of "Cautious Optimizers: Improving Training with One
    Line of Code" (https://arxiv.org/abs/2411.16085).
    """
    mask = (mt * grad > 0).to(grad.dtype)
    mask.div_(mask.mean().clamp_(min=1e-3))
    if inplace:
        update_mt = mt.mul_(mask)
    else:
        update_mt = mt.mul(mask)
    del mask
    return update_mt
