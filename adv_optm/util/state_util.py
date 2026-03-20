import torch
from .param_update import (
    copy_stochastic_, _copy_stochastic_core_,
    copy_fp8_stochastic_, _copy_fp8_stochastic_core_,
    copy_int8_stochastic_, _copy_int8_stochastic_core_,
)


def init_state_tensor(state: dict, key: str, shape: tuple, state_precision: str, device: torch.device, default_dtype: torch.dtype):
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
        store_dtype = torch.int8
    else:  # 'auto'
        store_dtype = default_dtype

    if store_dtype == getattr(torch, 'float8_e4m3fn', None):
        # FP8 initialization: we need to store a separate scale for dequantization
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)
        state[f"{key}_scale"] = torch.tensor(1.0, device=device, dtype=torch.float32)
    elif store_dtype == torch.int8:
        # INT8 initialization: we need to store a separate scale for dequantization
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)
        state[f"{key}_scale"] = torch.tensor(1.0, device=device, dtype=torch.float32)
    else:
        state[key] = torch.zeros(shape, device=device, dtype=store_dtype)


def get_state(state: dict, key: str, state_precision: str) -> torch.Tensor:
    """
    Retrieves the state tensor.
    """
    tensor = state[key]
    if state_precision in ['fp8', 'fp8_sr']:
        scale = state[f"{key}_scale"]
        return tensor.float() / scale
    elif state_precision == 'int8_sr':
        scale = state[f"{key}_scale"]
        return tensor.float() / scale
    elif state_precision == 'bf16_sr':
        return tensor.float()
    else: # 'auto', 'fp32'.
        return tensor


def set_state(state: dict, key: str, value: torch.Tensor, state_precision: str, random_int_state_tensor: torch.Tensor | None):
    """
    Quantizes or packs the computed state value.
    """
    if state_precision == 'fp32':
        if state[key] is not value:
            state[key].copy_(value)

    elif state_precision in ['fp8', 'fp8_sr']:
        # Calculate amax
        amax = value.abs().max().clamp_min(1e-12)
        # We find the largest power of 2 such that (amax * scale) <= 448
        raw_scale = 448.0 / amax
        log_scale = torch.floor(torch.log2(raw_scale))
        scale = 2.0 ** log_scale

        state[f"{key}_scale"].copy_(scale)

        if state_precision == 'fp8_sr':
            # Quantize with bitwise Stochastic Rounding
            if random_int_state_tensor is None:
                copy_fp8_stochastic_(state[key], value, scale)
            else:
                _copy_fp8_stochastic_core_(state[key], value, scale, random_int_state_tensor)
        else:
            # Standard Round-to-Nearest
            state[key].copy_((value * scale).clamp(min=-448, max=448).to(torch.float8_e4m3fn))

    elif state_precision == 'bf16_sr':
        # Apply stochastic rounding for BF16 states
        if random_int_state_tensor is None:
            copy_stochastic_(state[key], value, False)
        else:
            _copy_stochastic_core_(state[key], value, random_int_state_tensor, False)

    elif state_precision == 'int8_sr':
        # Calculate amax and derive a power-of-2 scale so that
        # (amax * scale) fits within the int8 range [-127, 127].
        amax = value.abs().max().clamp_min(1e-12)
        scale = 127.0 / amax

        state[f"{key}_scale"].copy_(scale)

        # Quantize with stochastic rounding
        if random_int_state_tensor is None:
            copy_int8_stochastic_(state[key], value, scale)
        else:
            _copy_int8_stochastic_core_(state[key], value, scale, random_int_state_tensor)

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
    if state_precision in ['bf16_sr', 'fp8', 'fp8_sr', 'int8_sr', 'factored']:
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
    actual_precision = state.get('actual_state_precision', group.get('state_precision', 'auto'))

    # Determine the target dtype for general floating-point states based on state_precision
    if actual_precision == 'fp32':
        base_dtype = torch.float32
    elif actual_precision == 'bf16_sr':
        base_dtype = torch.bfloat16
    elif actual_precision in ['fp8', 'fp8_sr'] and hasattr(torch, 'float8_e4m3fn'):
        base_dtype = torch.float8_e4m3fn
    elif actual_precision == 'int8_sr':
        base_dtype = torch.int8
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

        # Handle Factorized Tensors and FP8 Scales (Must be float32)
        if key in fp32_keys or (key.endswith('_scale') and key != 'anchor_scale'):
            if val.dtype != torch.float32:
                state[key] = val.to(torch.float32)
            continue

        # Handle Standard Floating Point Optimizer States
        if val.is_floating_point():
            # Apply base_dtype which accounts for `state_precision` and upcasting logic
            if val.dtype != base_dtype:
                state[key] = val.to(base_dtype)

        # Handle INT8 Stochastic-Rounded States (integer, not floating point)
        elif actual_precision == 'int8_sr' and val.dtype != torch.int8:
            state[key] = val.to(torch.int8)

        # Ensure device match
        if state[key].device != p.device:
            state[key] = state[key].to(p.device)