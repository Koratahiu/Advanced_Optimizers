import torch
import torch.nn.functional as F

from .param_update import (
    copy_stochastic_, _copy_stochastic_core_,
    copy_fp8_stochastic_, _copy_fp8_stochastic_core_,
    copy_int8_blockwise_stochastic_, _copy_int8_blockwise_stochastic_core_,
    copy_int8_sym_blockwise_stochastic_, _copy_int8_sym_blockwise_stochastic_core_,
)

_int8_sr_BLOCK_SIZE = 2048


def init_state_tensor(state: dict, key: str, shape: tuple, state_precision: str, device: torch.device, default_dtype: torch.dtype, non_neg: bool=False):
    """
    Initializes a generic optimizer state tensor based on the requested precision.
    """
    # Determine storage dtype based on precision selection
    if state_precision == 'fp32':
        store_dtype = torch.float32
    elif state_precision == 'bf16_sr':
        store_dtype = torch.bfloat16
    elif state_precision in ['fp8', 'fp8_sr']:
        store_dtype = torch.float8_e4m3fn
    elif state_precision == 'int8_sr':
        store_dtype = torch.uint8 if non_neg else torch.int8
    else:  # 'auto'
        store_dtype = default_dtype

    if store_dtype == getattr(torch, 'float8_e4m3fn', None):
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)
        state[f"{key}_scale"] = torch.tensor(1.0, device=device, dtype=torch.float32)
    elif store_dtype in (torch.uint8, torch.int8):
        numel = 1
        for s in shape:
            numel *= s
        n_blocks = (numel + _int8_sr_BLOCK_SIZE - 1) // _int8_sr_BLOCK_SIZE
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)
        state[f"{key}_scale"] = torch.ones(n_blocks, device=device, dtype=torch.float32)
    else:
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)


def get_state(state: dict, key: str, state_precision: str) -> torch.Tensor:
    """
    Retrieves and dequantizes the state tensor to float32.
    """
    tensor = state[key]
    if state_precision in ['fp8', 'fp8_sr']:
        scale = state[f"{key}_scale"]
        return tensor.float() / scale
    elif state_precision == 'int8_sr':
        scales = state[f"{key}_scale"] # (n_blocks,) fp32
        blocks, orig_shape, orig_numel = _prepare_int8_blocks(state[key], _int8_sr_BLOCK_SIZE)

        # dequantize: q * scale (for both int8 symmetric and uint8 non-negative)
        result = blocks * scales.unsqueeze(1)

        return result.view(-1)[:orig_numel].view(orig_shape)
    elif state_precision == 'bf16_sr':
        return tensor.float()
    else: # 'auto', 'fp32'.
        return tensor


def _prepare_int8_blocks(
    value: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, tuple, int]:
    """
    Pads and reshapes a float32 view of ``value`` into (n_blocks, block_size)
    blocks.
    """
    orig_shape = value.shape
    orig_numel = value.numel()
    pad_len = (block_size - (orig_numel % block_size)) % block_size
    val_flat = F.pad(value.reshape(1, -1), (0, pad_len), mode='replicate')
    return val_flat.view(-1, block_size).float(), orig_shape, orig_numel


def _compute_uint8_non_neg_block_stats(value: torch.Tensor, block_size: int, bits: int = 8,
                                    val_blocks: torch.Tensor | None = None):
    """
    Computes per-block scales for specialized non-negative blockwise quantization.
    """
    if val_blocks is None:
        val_blocks, _, _ = _prepare_int8_blocks(value, block_size)

    # Calc Stats: max value
    max_vals = val_blocks.amax(dim=1, keepdim=True)

    # Scale calculation (0 to 255)
    max_int = (1 << bits) - 1
    scales = max_vals.div_(float(max_int))

    return scales.squeeze(1)


def _compute_int8_sym_block_stats(value: torch.Tensor, block_size: int, bits: int = 8,
                                  val_blocks: torch.Tensor | None = None):
    """
    Computes per-block scales for symmetric blockwise quantization.
    """
    if val_blocks is None:
        val_blocks, _, _ = _prepare_int8_blocks(value, block_size)

    # Calc Stats: max absolute value
    abs_max_vals = val_blocks.abs().amax(dim=1, keepdim=True)

    # Scale calculation (-127 to 127)
    max_int = (1 << (bits - 1)) - 1
    scales = abs_max_vals.div_(float(max_int))

    return scales.squeeze(1)


def set_state(state: dict, key: str, value: torch.Tensor, state_precision: str, random_int_state_tensor: torch.Tensor | None, non_neg: bool=False):
    """
    Quantizes or packs the computed state value.
    """
    if state_precision == 'fp32':
        if state[key] is not value:
            state[key].copy_(value)

    elif state_precision == 'fp8_sr':
        amax = value.abs().max().clamp_min(1e-12)
        # Calculate amax
        scale = 448.0 / amax

        state[f"{key}_scale"].copy_(scale)

        # Quantize with bitwise Stochastic Rounding
        if random_int_state_tensor is None:
            copy_fp8_stochastic_(state[key], value, scale)
        else:
            _copy_fp8_stochastic_core_(state[key], value, scale, random_int_state_tensor)

    elif state_precision == 'bf16_sr':
        # Apply stochastic rounding for BF16 states
        if random_int_state_tensor is None:
            copy_stochastic_(state[key], value, False)
        else:
            _copy_stochastic_core_(state[key], value, random_int_state_tensor, False)

    elif state_precision == 'int8_sr':
        val_blocks, _, _ = _prepare_int8_blocks(value, _int8_sr_BLOCK_SIZE)

        if non_neg:
            scales = _compute_uint8_non_neg_block_stats(
                value,
                block_size=_int8_sr_BLOCK_SIZE, 
                bits=8, 
                val_blocks=val_blocks
            )

            state[f"{key}_scale"].copy_(scales)

            # Apply stochastic rounding
            if random_int_state_tensor is not None:
                _copy_int8_blockwise_stochastic_core_(state[key], value, scales,random_int_state_tensor, block_size=_int8_sr_BLOCK_SIZE, val_blocks=val_blocks)
            else:
                copy_int8_blockwise_stochastic_(state[key], value, scales,block_size=_int8_sr_BLOCK_SIZE)
        
        else: # int8_sr symmetric
            scales = _compute_int8_sym_block_stats(
                value,
                block_size=_int8_sr_BLOCK_SIZE, 
                bits=8, 
                val_blocks=val_blocks
            )

            state[f"{key}_scale"].copy_(scales)

            # Apply stochastic rounding
            if random_int_state_tensor is not None:
                _copy_int8_sym_blockwise_stochastic_core_(state[key], value, scales, random_int_state_tensor, block_size=_int8_sr_BLOCK_SIZE, val_blocks=val_blocks)
            else:
                copy_int8_sym_blockwise_stochastic_(state[key], value, scales, block_size=_int8_sr_BLOCK_SIZE)

        del val_blocks

    else:  # 'auto'
        if state[key] is not value:
            state[key].copy_(value)

def upcast_grad_for_precision(grad: torch.Tensor, state: dict, state_precision: str) -> torch.Tensor:
    """
    Upcasts the gradient to float32 if the optimizer state precision 
    or factorization requires higher precision for accumulation.
    """
    # Factored states (SMMF) always require FP32 for reconstruction/factorization logic
    if state.get('factored', False):
        return grad.float()

    # Low-precision storage modes benefit from FP32 accumulation to 
    # maintain accuracy before quantizing back down in set_state.
    if state_precision in ['bf16_sr', 'fp8_sr', 'int8_sr', 'factored']:
        return grad.float()

    return grad

def fix_loaded_state_dtype(state: dict, p: torch.Tensor, group: dict) -> None:
    """
    Fixes the dtypes of an optimizer state after loading a state_dict.
    Accounts for state_precision options and works around PyTorch's auto-casting bug.
    """
    mode = group.get('centered_wd_mode', 'full')
    is_factored = state.get('factored', False)

    # Retrieve the active precision mode
    actual_precision = group['actual_state_precision']
    numel = p.numel()
    is_full = (
        numel < 10000 or
        p.ndim == 1
    )

    # Determine the target dtype for general floating-point states based on state_precision
    if actual_precision == 'fp32' or is_full:
        base_dtype = torch.float32
    elif actual_precision == 'bf16_sr':
        base_dtype = torch.bfloat16
    elif actual_precision in ['fp8', 'fp8_sr'] and hasattr(torch, 'float8_e4m3fn'):
        base_dtype = torch.float8_e4m3fn
    elif actual_precision == 'int8_sr':
        base_dtype = torch.float32
    else:
        # Fallback ('auto').
        base_dtype = torch.float32 if is_factored else p.dtype

    # Deterministically check if this parameter skipped quantization
    numel = p.numel()
    is_skipped = (
        numel == 0 or
        (mode in ['int8', 'int4'] and numel < 10000) or
        p.ndim == 1 or
        getattr(p, '_is_dora_scale', False)
    )

    # Pre-define sets for known exact-match keys
    uint8_keys = {'sign', 'sign_slow', 'sign_buf'}
    fp32_keys = {'mu_m_nmf', 'mv_m_nmf', 'mu_v_nmf', 'mv_v_nmf', 'mu_m_slow_nmf', 'mv_m_slow_nmf'}

    for key, val in state.items():
        if not isinstance(val, torch.Tensor):
            continue

        # Handle Quantized Anchor States
        if key == 'anchor_data':
            if is_skipped or mode == 'full':
                if val.dtype != p.dtype:
                    state[key] = val.to(p.dtype)
            elif mode in ['int8', 'int4']:
                if val.dtype != torch.uint8:
                    state[key] = val.to(torch.uint8)
            elif mode == 'float8':
                if val.dtype != torch.float8_e4m3fn:
                    state[key] = val.to(torch.float8_e4m3fn)
            continue

        elif key in ['anchor_scale', 'anchor_min']:
            if val.dtype != p.dtype:
                state[key] = val.to(p.dtype)
            continue

        # Handle Quantized Factorization States (Sign tensors)
        if key in uint8_keys:
            if val.dtype != torch.uint8:
                state[key] = val.to(torch.uint8)
            continue

        # Handle Factorized Tensors, FP8 Scales, and blockwise INT8 scale
        if key in fp32_keys or (key.endswith('_scale') and key != 'anchor_scale'):
            if val.dtype != torch.float32:
                state[key] = val.to(torch.float32)
            continue

        # Handle Standard Floating Point Optimizer States
        if val.is_floating_point():
            # Apply base_dtype which accounts for `state_precision` and upcasting logic
            if val.dtype != base_dtype:
                state[key] = val.to(base_dtype)

        # Handle INT8 Stochastic-Rounded States specifically
        elif actual_precision == 'int8_sr' and val.dtype not in (torch.int8, torch.uint8):
            state[key] = val.to(torch.int8)

        # Ensure device match
        if state[key].device != p.device:
            state[key] = state[key].to(p.device)
