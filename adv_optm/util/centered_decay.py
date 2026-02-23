import torch
import torch.nn.functional as F

def is_wd_centered(p):
    """Determines if the weight decay is already centered."""
    if getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False):
        return True
    if getattr(p, '_is_oft', False):
        return True
    if getattr(p, '_is_dora_scale', False):
        return False
    return False

def quantize_blockwise(p, block_size, bits=8):
    """Helper to perform asymmetric block-wise quantization."""
    val_flat = p.flatten()
    numel = val_flat.numel()

    # Pad to multiple of block_size
    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        val_padded = F.pad(val_flat, (0, pad_len), mode='replicate')
    else:
        val_padded = val_flat

    # Block Reshape
    val_blocks = val_padded.view(-1, block_size).float()

    # Calc Stats
    min_vals, max_vals = torch.aminmax(val_blocks, dim=1, keepdim=True)

    # Scale calculation
    max_int = (1 << bits) - 1
    scales = (max_vals - min_vals).div_(float(max_int))
    scales.masked_fill_(scales == 0, 1.0)

    # Quantize: (val - min) / scale
    quantized = val_blocks.sub(min_vals).div_(scales).round_().clamp_(0, max_int).to(torch.uint8)

    return quantized, scales.squeeze(1), min_vals.squeeze(1)

def _init_anchor(p, state, mode):
    """Initializes the anchor state based on the selected mode."""
    numel = p.numel()

    # Skip empty/tiny tensors or 1D tensors (like biases/LayerNorms)
    if numel == 0 or (mode in ['int8', 'int4'] and numel < 256) or p.ndim == 1:
        state['anchor_data'] = p.detach().clone()
        state['anchor_type'] = 'full'
        return

    p_detached = p.detach()

    # Hoist shared state definitions
    state['anchor_orig_shape'] = p.shape
    state['anchor_numel'] = numel
    state['anchor_type'] = mode

    if mode == 'float8':
        state['anchor_data'] = p_detached.to(torch.float8_e4m3fn)
        return

    elif mode == 'int8':
        block_size = 128
        q_blocks, scales, mins = quantize_blockwise(p_detached, block_size, bits=8)

        state['anchor_data'] = q_blocks
        state['anchor_scale'] = scales.to(p.dtype)
        state['anchor_min'] = mins.to(p.dtype)
        state['anchor_block_size'] = block_size

    elif mode == 'int4':
        block_size = 32
        q_blocks, scales, mins = quantize_blockwise(p_detached, block_size, bits=4)

        q_flat = q_blocks.view(-1)

        # Vectorized packing: High bits | Low bits
        packed = (q_flat[0::2] << 4) | q_flat[1::2]

        state['anchor_data'] = packed
        state['anchor_scale'] = scales.to(p.dtype)
        state['anchor_min'] = mins.to(p.dtype)
        state['anchor_block_size'] = block_size

    elif mode == 'full':
        state['anchor_data'] = p_detached.clone()

def dequantize_anchor(p, state):
    """Restores the anchor to the original shape/dtype."""
    a_type = state.get('anchor_type', 'full')

    if a_type in ('full', 'float8'):
        return state['anchor_data'].to(p.dtype)

    # Dequantize Setup
    scales = state['anchor_scale']
    mins = state['anchor_min']
    block_size = state['anchor_block_size']
    orig_shape = state['anchor_orig_shape']
    orig_numel = state['anchor_numel']

    if a_type == 'int4':
        packed = state['anchor_data']

        # Vectorized unpacking using pre-allocation
        unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
        unpacked[0::2] = packed >> 4
        unpacked[1::2] = packed & 0x0F

        quantized_blocks = unpacked.view(-1, block_size)

    elif a_type == 'int8':
        quantized_blocks = state['anchor_data']
    else:
        raise ValueError(f"Unknown anchor type: {a_type}")

    # Core Dequantization: (q * scale) + min
    anchor_blocks = quantized_blocks.to(p.dtype) * scales.unsqueeze(1) + mins.unsqueeze(1)

    # Flatten, truncate any padding added during quantization, and reshape
    return anchor_blocks.view(-1)[:orig_numel].view(orig_shape)