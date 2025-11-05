import torch
from torch import Tensor

from typing import Dict, Any

def copy_stochastic_(target: Tensor, source: Tensor):
    """
    Nerogar's implementation of stochastic rounding in the paper "Revisiting BFloat16 Training"
    (https://arxiv.org/abs/2010.06192).
    see:
    https://github.com/pytorch/pytorch/issues/120376
    https://github.com/Nerogar/OneTrainer/blob/daae18eaed8c0fa39289b2ff79cc2c1e08577fcb/modules/util/bf16_stochastic_rounding.py

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result

def add_stochastic_(input: Tensor, other: Tensor, alpha: float = 1.0):
    """
    adds other to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
    """
    result = other.clone() if other.dtype == torch.float32 else other.to(dtype=torch.float32)

    result.add_(input, alpha=alpha)
    copy_stochastic_(input, result)

def apply_parameter_update(
    self,
    p: torch.Tensor,
    group: Dict[str, Any],
    update: torch.Tensor,
    lr: float,
    wd:float | None,
) -> None:
    """
    Applies decoupled weight decay (standard or cautious) and the final
    parameter update to p.data in-place.

    Args:
        p: The parameter tensor whose data (p.data) will be updated.
        group: The parameter group dictionary (must contain "weight_decay").
        update: The pre-calculated update tensor (e.g., scaled gradient or momentum term).
        lr: The current learning rate.
    """
    wd = group["weight_decay"] if wd is None else wd
    if wd != 0:
        if group.get('cautious_wd', False):
            # Cautious Weight Decay applies decay only where update and parameter signs align
            mask = (update * p.data > 0).to(p.dtype)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                decay_amount = p.data.clone().mul_(-wd * lr)
                decay_amount.mul_(mask)
                add_stochastic_(p.data, decay_amount)
                del decay_amount, mask
            else:
                p.data.addcmul_(p.data, mask, value=-wd * lr)
        else:
            # Standard decoupled weight decay
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-wd * lr)
            else:
                p.data.add_(p.data, alpha=-wd * lr)

    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        add_stochastic_(p.data, -update)
    else:
        p.data.add_(-update)
    del update
