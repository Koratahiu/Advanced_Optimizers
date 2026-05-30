import torch
from torch import Tensor
from torch.optim import Optimizer

import torch.nn.functional as F

from typing import Dict, Any

from .scaled_optm import adjust_wds, scale_wds
from .centered_decay import dequantize_anchor

_generators: Dict[torch.device, torch.Generator] = {}


def _apply_weight_decay(
    p_calc: Tensor,
    update_calc: Tensor,
    p: Tensor,
    state: Dict[str, Any],
    group: Dict[str, Any],
    scaled_wd: float | Tensor | None,
    scaled_cwd: float | Tensor | None,
    wd_target: Tensor | None = None,
    cwd_target: Tensor | None = None,
) -> None:
    """
    Apply decoupled weight decay (Standard and/or Centered) independently.
    """
    cautious = group.get('cautious_wd', False)

    # Standard Weight Decay (pulls toward zero)
    if scaled_wd is not None:
        if wd_target is None:
            wd_target = p_calc
        # Cautious Weight Decay: only decay if the update pushes in the same direction as the decay
        if cautious:
            mask = (update_calc * p_calc >= 0).to(p_calc.dtype)
            if isinstance(scaled_wd, Tensor):
                p_calc.addcmul_(wd_target, mask * scaled_wd, value=-1.0)
            else:
                p_calc.addcmul_(wd_target, mask, value=-scaled_wd)
            del mask
        else:
            # Standard decoupled weight decay
            if isinstance(scaled_wd, Tensor):
                p_calc.addcmul_(wd_target, scaled_wd, value=-1.0)
            else:
                p_calc.add_(wd_target, alpha=-scaled_wd)

    # Centered Weight Decay (pulls toward anchor)
    if scaled_cwd is not None and 'anchor_type' in state:
        if cwd_target is not None:
            decay_target = cwd_target
        else:
            anchor = dequantize_anchor(p, state, group, p_calc.dtype)
            decay_target = p_calc.sub(anchor)
            del anchor

        if cautious:
            # Cautious Weight Decay: only decay if the update pushes in the same direction as the decay
            mask = (update_calc * decay_target >= 0).to(p_calc.dtype)
            if isinstance(scaled_cwd, Tensor):
                p_calc.addcmul_(decay_target, mask * scaled_cwd, value=-1.0)
            else:
                p_calc.addcmul_(decay_target, mask, value=-scaled_cwd)
            del mask
        else:
            # Standard decoupled weight decay
            if isinstance(scaled_cwd, Tensor):
                p_calc.addcmul_(decay_target, scaled_cwd, value=-1.0)
            else:
                p_calc.add_(decay_target, alpha=-scaled_cwd)

        if cwd_target is None:
            del decay_target

def apply_parameter_update(
    self,
    p: Tensor,
    group: Dict[str, Any],
    update: Tensor,
    lr: float | Tensor,
    wd: float | None = None,
    random_int_tensor: Tensor | None = None,
    decoupled: bool = False,
    wd_scaler: float | Tensor | None = None,
    wd_target: Tensor | None = None,
    cwd_target: Tensor | None = None,
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
        wd_scaler: A multiplier/tensor to scale the calculated wd/cwd magnitude (e.g. for Fisher Adam WD).
    """
    wd = group["weight_decay"] if wd is None else wd
    cwd = group.get("centered_wd", 0.0)
    wd, cwd = adjust_wds(wd, cwd, p)

    # Calculate global decay factor for decoupled vs standard
    decay_factor = (lr / self._init_lr) if (decoupled and self._init_lr > 0) else lr

    scaled_wd = (wd * decay_factor) if wd != 0 else None
    scaled_cwd = (cwd * decay_factor) if cwd != 0 else None

    if wd_scaler is not None:
        if scaled_wd is not None:
            scaled_wd = scaled_wd * wd_scaler
        if scaled_cwd is not None:
            scaled_cwd = scaled_cwd * wd_scaler

    state = self.state[p]

    # Compute full update in float32 if using bfloat16 with stochastic rounding
    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        p_fp32 = p.float()
        update_fp32 = update.float()

        wd_t = wd_target.float() if wd_target is not None else None
        cwd_t = cwd_target.float() if cwd_target is not None else None

        # Apply weight decay if needed
        if scaled_wd is not None or scaled_cwd is not None:
            _apply_weight_decay(p_fp32, update_fp32, p, state, group, scaled_wd, scaled_cwd, wd_t, cwd_t)

        # Apply main update
        p_fp32.add_(-update_fp32)

        # Single stochastic rounding at the end
        if random_int_tensor is not None:
            # Compiled path: use the pre-computed random tensor
            _copy_stochastic_core_(p, p_fp32, random_int_tensor, inplace=True)
            del random_int_tensor
        else:
            # Uncompiled path: generate randoms inside
            copy_stochastic_(p, p_fp32, inplace=True)
        del p_fp32, update_fp32

    else:
        # Standard path for non-bfloat16 or without stochastic rounding
        if scaled_wd is not None or scaled_cwd is not None:
            _apply_weight_decay(p, update, p, state, group, scaled_wd, scaled_cwd, wd_target, cwd_target)

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

def _copy_stochastic_core_(target: Tensor, source: Tensor, random_int_tensor: Tensor, inplace: bool = False):
    """
    Core logic for BF16 stochastic rounding using a pre-computed random integer tensor.
    This version is designed to be torch.compile-friendly.
    """
    result = random_int_tensor if inplace else random_int_tensor.clone()
    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def copy_stochastic_(target: Tensor, source: Tensor, inplace: bool = False):
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
    _copy_stochastic_core_(target, source, random_int_tensor, inplace)
    del random_int_tensor

def _get_random_int_for_fp8_sr(source: torch.Tensor) -> torch.Tensor:
    """
    Generates a random int32 tensor for FP8 stochastic rounding.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    device = source.device

    if device not in _generators:
        set_seed(device)

    # TODO: this is a workaround until torch compile error
    # NotImplementedError: UserDefinedObjectVariable(generator) is fixed
    generator = _generators[device]

    # FP8 e4m3fn always preserves exactly 3 mantissa bits from FP32 (23 bits).
    # We need uniform noise in [0, 2^20 - 1]
    return torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=1048576,  # 1 << 20
        generator=generator,
    )


def _copy_fp8_stochastic_core_(
    target: torch.Tensor, 
    source: torch.Tensor, 
    scale: torch.Tensor, 
    random_int_tensor: torch.Tensor
):
    """
    Core logic for FP8 (float8_e4m3fn) stochastic rounding using a pre-computed 
    random integer tensor.
    """
    # Scale the source to FP32
    buffer = (source * scale).to(torch.float32)

    # Extract magnitude and sign
    sign_x = torch.sign(buffer)
    buffer.abs_()

    # Create and apply the magic offset
    offset = (buffer < 0.015625).to(torch.float32).mul_(0.015625)
    buffer.add_(offset)

    # Apply Stochastic Rounding
    buffer_int = buffer.view(torch.int32)
    buffer_int.add_(random_int_tensor)
    buffer_int.bitwise_and_(-1048576)

    # Remove offset and reapply sign
    buffer = buffer_int.view(torch.float32)
    buffer.sub_(offset)
    buffer.mul_(sign_x)

    target.copy_(buffer.to(torch.float8_e4m3fn))


def copy_fp8_stochastic_(target: torch.Tensor, source: torch.Tensor, scale: torch.Tensor):
    """
    Stochastic rounding implementation for FP8 e4m3fn states.
    """
    random_int_tensor = _get_random_int_for_fp8_sr(source)
    _copy_fp8_stochastic_core_(target, source, scale, random_int_tensor)
    del random_int_tensor


def _get_random_int_for_8bit_sr(source: torch.Tensor, numel: int | None = None) -> torch.Tensor:
    """
    Generates a flat random int32 tensor for unit stochastic rounding.
    Values are in [0, 2^16 - 1]; they are later scaled to U[0, 1) inside
    the core function.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    device = source.device

    if device not in _generators:
        set_seed(device)

    # TODO: this is a workaround until torch compile error
    # NotImplementedError: UserDefinedObjectVariable(generator) is fixed
    generator = _generators[device]

    size = (numel,) if numel is not None else (((source.numel() + 2047) // 2048) * 2048,)
    return torch.randint(
        size=size,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=generator,
    )


# 
# ASYMMETRIC UINT8 PATH
# 

def _copy_int8_blockwise_stochastic_core_(
    target: torch.Tensor,
    source: torch.Tensor,
    scales: torch.Tensor,
    random_int_tensor: torch.Tensor | None,
    block_size: int = 2048,
    val_blocks: torch.Tensor | None = None,
) -> None:
    """
    Core logic for blockwise asymmetric uint8 stochastic rounding.
    """

    orig_shape = source.shape
    orig_numel = source.numel()

    n_blocks = scales.shape[0]
    pad_len = n_blocks * block_size - orig_numel

    if val_blocks is None:
        val_flat = source.reshape(-1).float()
        val_flat = F.pad(val_flat.reshape(1, -1), (0, pad_len), mode='replicate')
        val_blocks = val_flat.view(n_blocks, block_size)

    # Normalise to [0, 255] per block
    safe_scales = scales.float().clamp_min(1e-12).unsqueeze(1)  # (n_blocks, 1)
    normalised = (val_blocks) / safe_scales

    # Stochastic rounding: floor(x + u), u ~ U[0, 1) — unbiased for any sign
    if random_int_tensor is not None:
        noise = random_int_tensor.reshape(n_blocks, block_size).float().mul_(1.0 / (1 << 16))
        normalised = normalised + noise
        del noise

    quantised = normalised.floor_().clamp_(0, 255).to(torch.uint8)

    # Strip padding and restore original shape.
    target.copy_(quantised.view(-1)[:orig_numel].view(orig_shape))


def copy_int8_blockwise_stochastic_(
    target: torch.Tensor,
    source: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 2048,
) -> None:
    """
    Blockwise asymmetric uint8 stochastic rounding for uint8 optimizer states.
    """
    padded_numel = scales.shape[0] * block_size
    random_int_tensor = _get_random_int_for_8bit_sr(source, padded_numel)
    _copy_int8_blockwise_stochastic_core_(target, source, scales, random_int_tensor, block_size)
    del random_int_tensor


#
# SYMMETRIC INT8 PATH
#

def _copy_int8_sym_blockwise_stochastic_core_(
    target: torch.Tensor,
    source: torch.Tensor,
    scales: torch.Tensor,
    random_int_tensor: torch.Tensor | None,
    block_size: int = 2048,
    val_blocks: torch.Tensor | None = None,
) -> None:
    """
    Core logic for blockwise symmetric int8 stochastic rounding.
    """
    orig_shape = source.shape
    orig_numel = source.numel()

    n_blocks = scales.shape[0]
    pad_len = n_blocks * block_size - orig_numel

    if val_blocks is None:
        val_flat = source.reshape(-1).float()
        val_flat = F.pad(val_flat.reshape(1, -1), (0, pad_len), mode='replicate')
        val_blocks = val_flat.view(n_blocks, block_size)

    # Normalise to [-127, 127] per block
    safe_scales = scales.float().clamp_min(1e-12).unsqueeze(1)  # (n_blocks, 1)
    normalised = val_blocks / safe_scales

    # Stochastic rounding: floor(x + u), u ~ U[0, 1) — unbiased for any sign
    noise = random_int_tensor.reshape(n_blocks, block_size).float().mul_(1.0 / (1 << 16))
    normalised = normalised + noise
    del noise

    quantised = normalised.floor_().clamp_(-127, 127).to(torch.int8)

    # Strip padding and restore original shape.
    target.copy_(quantised.view(-1)[:orig_numel].view(orig_shape))


def copy_int8_sym_blockwise_stochastic_(
    target: torch.Tensor,
    source: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 2048,
) -> None:
    """
    Blockwise symmetric int8 stochastic rounding for int8 optimizer states.
    """
    padded_numel = scales.shape[0] * block_size
    random_int_tensor = _get_random_int_for_8bit_sr(source, padded_numel)
    _copy_int8_sym_blockwise_stochastic_core_(target, source, scales, random_int_tensor, block_size)
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
    from .state_util import fix_loaded_state_dtype

    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, None)
            if not state:
                continue

            fix_loaded_state_dtype(state, p, group)


def _get_random_noise_for_sso(source: torch.Tensor) -> torch.Tensor:
    """
    Generates a random noise tensor for Stochastic Sign operator.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    global _generators
    device = source.device
    if device not in _generators:
        set_seed(device)
    # TODO, this is a workaround until torch compile error
    # NotImplementedError: UserDefinedObjectVariable(generator) is fixed
    generator = _generators[device]
    # create a uniform noise tensor in [0, 1) for stochastic sign decisions
    return torch.rand(
        source.shape,
        device=source.device,
        dtype=source.dtype,
        generator=generator,
    ).mul_(2).sub_(1)


def _get_random_noise_for_low_rank_ortho(source: torch.Tensor, ortho_rank: int) -> torch.Tensor:
    """
    Generates a random noise tensor for low-rank orthogonalization.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    global _generators
    device = source.device
    if device not in _generators:
        set_seed(device)
    # TODO, this is a workaround until torch compile error
    # NotImplementedError: UserDefinedObjectVariable(generator) is fixed
    generator = _generators[device]
    source_flat = source.flatten(1)
    r = min(ortho_rank, source_flat.shape[0], source_flat.shape[1])
    return torch.randn(
        source_flat.shape[1],
        r,
        device=source_flat.device,
        dtype=source_flat.dtype,
        generator=generator
        )
