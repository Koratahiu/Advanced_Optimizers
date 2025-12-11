import torch

def _grams_update(mt: torch.tensor, grad: torch.tensor):
    return grad.sign().mul_(mt.abs())

def _cautious_update(mt: torch.tensor, grad: torch.tensor):
    mask = (mt * grad > 0).to(grad.dtype)
    mask.div_(mask.mean().clamp_(min=1e-3))
    update_mt = mt.mul(mask)
    del mask
    return update_mt

def _atan2_update(update, denom, a=1.2732395):
        return update.atan2_(denom).mul_(a)