import torch
from .param_update import copy_stochastic_, _copy_stochastic_core_, copy_fp8_stochastic_, _copy_fp8_stochastic_core_


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
    else:  # 'auto'
        store_dtype = default_dtype

    if store_dtype == getattr(torch, 'float8_e4m3fn', None):
        # FP8 initialization: we need to store a separate scale for dequantization
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
    if state_precision in ['bf16_sr', 'fp8', 'fp8_sr', 'factored']:
        return grad.float()

    return grad