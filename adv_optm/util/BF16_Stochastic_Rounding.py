import torch
from torch import Tensor

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