import torch
from torch import Tensor

from typing import Dict, Any


def apply_parameter_update(
    self,
    p: Tensor,
    group: Dict[str, Any],
    update: Tensor,
    lr: float | Tensor,
    wd: float | None = None,
) -> None:
    """
    Applies decoupled weight decay (standard or cautious) and the final
    parameter update to p.data in-place.

    Args:
        p: The parameter tensor whose data (p.data) will be updated.
        group: The parameter group dictionary (must contain "weight_decay").
        update: The pre-calculated update tensor (e.g., scaled gradient or momentum term).
        lr: The current learning rate.
        wd: Optional float value for weight decay, if another value other than group["weight_decay"] is needed.
    """
    wd = group["weight_decay"] if wd is None else wd
    cautious = group.get('cautious_wd', False)

    # Compute full update in float32 if using bfloat16 with stochastic rounding
    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        p_fp32 = p.data.float()
        update_fp32 = update.float()

        # Apply weight decay if needed
        if wd != 0:
            if cautious:
                # Cautious Weight Decay
                mask = (update_fp32 * p_fp32 > 0).float()
                p_fp32.addcmul_(p_fp32, mask, value=-wd * lr)
                del mask
            else:
                # Standard decoupled weight decay
                p_fp32.add_(p_fp32, alpha=-wd * lr)

        # Apply main update
        p_fp32.add_(-update_fp32)

        # Single stochastic rounding at the end
        copy_stochastic_(p.data, p_fp32)
        del p_fp32, update_fp32

    else:
        # Standard path for non-bfloat16 or without stochastic rounding
        if wd != 0:
            if cautious:
                # Cautious Weight Decay
                mask = (update * p.data > 0).to(p.dtype)
                p.data.addcmul_(p.data, mask, value=-wd * lr)
                del mask
            else:
                # Standard decoupled weight decay
                p.data.add_(p.data, alpha=-wd * lr)
        
        # Apply main update
        p.data.add_(-update)
    
    del update


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
