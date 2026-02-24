import torch
from torch import Tensor
from torch.optim import Optimizer

from typing import Dict, Any

from .scaled_optm import scale_wd
from .centered_decay import dequantize_anchor

_generators: Dict[torch.device, torch.Generator] = {}


def _apply_weight_decay(
    p_calc: Tensor,
    update_calc: Tensor,
    p: Tensor,
    state: Dict[str, Any],
    group: Dict[str, Any],
    scaled_wd: float | Tensor
) -> None:
    """
    Apply decoupled weight decay 
    (Standard, Cautious, and/or Centered) to the parameter.
    """
    cautious = group.get('cautious_wd', False)
    centered = group.get('centered_wd', False)

    if centered and 'anchor_type' in state:
        return _apply_centered_flipped_wd(p_calc, update_calc, p, state, group, scaled_wd)


    if cautious:
        # Cautious Weight Decay: only decay if the update pushes in the same direction as the decay
        mask = (update_calc * p_calc >= 0).to(p_calc.dtype)
        p_calc.addcmul_(p_calc, mask, value=-scaled_wd)
        del mask
    else:
        # Standard decoupled weight decay
        p_calc.add_(p_calc, alpha=-scaled_wd)

def _apply_centered_flipped_wd(
    p_calc: Tensor,
    update_calc: Tensor,
    p: Tensor,
    state: Dict[str, Any],
    group: Dict[str, Any],
    scaled_wd: float | Tensor
) -> None:
    """
    Subclass of centered weight decay using (Anchor - P) math.
    Mathematically equivalent to: p_calc -= scaled_wd * (p_calc - anchor)
    """
    cautious = group.get('cautious_wd', False)

    # We use the anchor's memory to save an allocation
    anchor = dequantize_anchor(p, state).to(p_calc.dtype)
    decay_target = anchor.sub_(p_calc) 

    if cautious:
        # Flip the mask logic
        # Original: (update * (P - Anchor) >= 0)
        # Flipped:  (update * (Anchor - P) <= 0)
        mask = (update_calc * decay_target <= 0).to(p_calc.dtype)

        #  Add the target (since target is Anchor - P, adding it is 
        # equivalent to subtracting P - Anchor)
        p_calc.addcmul_(decay_target, mask, value=scaled_wd)
        del mask
    else:
        # Standard update: p_calc += scaled_wd * (Anchor - P)
        p_calc.add_(decay_target, alpha=scaled_wd)

    del anchor, decay_target


def apply_parameter_update(
    self,
    p: Tensor,
    group: Dict[str, Any],
    update: Tensor,
    lr: float | Tensor,
    wd: float | None = None,
    random_int_tensor: Tensor | None = None,
    decoupled: bool = False,
) -> None:
    """
    Applies decoupled weight decay (standard, cautious, centered) and the final
    parameter update to p in-place.

    Args:
        p: The parameter tensor whose data (p) will be updated.
        group: The parameter group dictionary (must contain "weight_decay").
        update: The pre-calculated update tensor (e.g., scaled gradient or momentum term).
        lr: The current learning rate.
        wd: Optional float value for weight decay, if another value other than group["weight_decay"] is needed.
        random_int_tensor: Optional pre-generated random tensor for stochastic
            rounding. Required for the `torch.compile` path.
        decoupled: Whenever to use the true decoupled weight decay.
    """
    wd = group["weight_decay"] if wd is None else wd

    if group.get('scaled_optm', False):
        decoupled = True
        wd = scale_wd(wd, p, group)

    if decoupled:
        scaled_wd = wd * (lr / self._init_lr)
    else:
        scaled_wd = wd * lr

    state = self.state[p]

    # Compute full update in float32 if using bfloat16 with stochastic rounding
    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        p_fp32 = p.float()
        update_fp32 = update.float()

        # Apply weight decay if needed
        if wd != 0:
            _apply_weight_decay(p_fp32, update_fp32, p, state, group, scaled_wd)

        # Apply main update
        p_fp32.add_(-update_fp32)

        # Single stochastic rounding at the end
        if random_int_tensor is not None:
            # Compiled path: use the pre-computed random tensor
            _copy_stochastic_core_(p, p_fp32, random_int_tensor)
            del random_int_tensor
        else:
            # Uncompiled path: generate randoms inside
            copy_stochastic_(p, p_fp32)
        del p_fp32, update_fp32

    else:
        # Standard path for non-bfloat16 or without stochastic rounding
        if wd != 0:
            _apply_weight_decay(p, update, p, state, group, scaled_wd)

        # Apply main update
        p.add_(-update)

    del update


def set_seed(device: torch.device):
    """
    Initializes or resets the deterministic generator for a specific device.
    This ensures that the sequence of random numbers used for stochastic
    rounding is reproducible.
    """
    global _generators
    if device not in _generators:
        _generators[device] = torch.Generator(device=device)
    _generators[device].manual_seed(42)


def get_generator(device: torch.device) -> torch.Generator:
    """
    Retrieves (and initializes if necessary) the deterministic generator 
    for the specified device.
    """
    if device not in _generators:
        set_seed(device)
    return _generators[device]


def _get_random_int_for_sr(source: Tensor) -> Tensor:
    """
    Generates a random int32 tensor for stochastic rounding.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    global _generators
    device = source.device

    if device not in _generators:
        set_seed(device)

    # TODO, this is a workaround until torch compile error
    # NotImplementedError: UserDefinedObjectVariable(generator) is fixed
    generator = _generators[device]

    # create a random 16 bit integer
    return torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=generator,
    )


def _copy_stochastic_core_(target: Tensor, source: Tensor, random_int_tensor: Tensor):
    """
    Core logic for stochastic rounding using a pre-computed random integer tensor.
    This version is designed to be torch.compile-friendly.
    """
    result = random_int_tensor
    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def copy_stochastic_(target: Tensor, source: Tensor):
    """
    Nerogar's implementation of stochastic rounding in the paper "Revisiting BFloat16 Training"
    (https://arxiv.org/abs/2010.06192). Made deterministic.
    This version is for uncompiled paths; it generates its own random numbers.
    see:
    https://github.com/pytorch/pytorch/issues/120376
    https://github.com/Nerogar/OneTrainer/blob/daae18eaed8c0fa39289b2ff79cc2c1e08577fcb/modules/util/bf16_stochastic_rounding.py

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    random_int_tensor = _get_random_int_for_sr(source)
    _copy_stochastic_core_(target, source, random_int_tensor)
    del random_int_tensor


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

def post_process_loaded_state(optimizer: Optimizer) -> None:
    """
    Fixes the dtypes of optimizer states after loading a state_dict.
    PyTorch's load_state_dict casts all states to the parameter's dtype,
    which breaks 8-bit/4-bit quantization and factorized float32 states.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, None)
            if not state:
                continue

            # Factorized states (SMMF) must remain float32 for stability
            is_factored = state.get('factored', False)
            target_dtype = torch.float32 if is_factored else p.dtype

            a_type = state.get('anchor_type', None)

            for key, val in state.items():
                if not isinstance(val, torch.Tensor):
                    continue

                # Handle Quantized Anchor States
                if key == 'anchor_data':
                    if a_type in ['int8', 'int4']:
                        if val.dtype != torch.uint8:
                            state[key] = val.to(torch.uint8)
                    elif a_type == 'float8':
                        if val.dtype != torch.float8_e4m3fn:
                            state[key] = val.to(torch.float8_e4m3fn)
                    elif a_type == 'full':
                        if val.dtype != p.dtype:
                            state[key] = val.to(p.dtype)

                elif key in ['anchor_scale', 'anchor_min']:
                    # Scales and mins should match the parameter dtype
                    if val.dtype != p.dtype:
                        state[key] = val.to(p.dtype)

                # Handle Quantized Factorization States (Sign tensors)
                elif key in ['sign', 'sign_slow']:
                    if val.dtype != torch.uint8:
                        state[key] = val.to(torch.uint8)

                # Handle standard Optimizer Floating Point states (m, v, etc.)
                elif val.is_floating_point():
                    if val.dtype != target_dtype:
                        state[key] = val.to(target_dtype)

                if state[key].device != p.device:
                    state[key] = state[key].to(p.device)