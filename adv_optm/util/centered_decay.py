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
    quantized = val_blocks.sub_(min_vals).div_(scales).round_().clamp_(0, max_int).to(torch.uint8)

    return quantized, scales.squeeze(1), min_vals.squeeze(1)

def _init_anchor(p, state, group):
    """Initializes the anchor state."""
    if not group.get('centered_wd', False) or is_wd_centered(p):
        return

    # Prevent re-initialization
    if 'anchor_data' in state:
        return

    mode = group.get('centered_wd_mode', 'full')
    numel = p.numel()

    # Skip empty/tiny tensors or 1D tensors (store as full precision)
    if numel == 0 or (mode in ['int8', 'int4'] and numel < 10000) or p.ndim == 1 or getattr(p, '_is_dora_scale', False):
        state['anchor_data'] = p.clone()
        return

    if mode == 'float8':
        state['anchor_data'] = p.to(torch.float8_e4m3fn)
        return

    elif mode == 'int8':
        q_blocks, scales, mins = quantize_blockwise(p, block_size=128, bits=8)
        state['anchor_data'] = q_blocks
        state['anchor_scale'] = scales.to(p.dtype)
        state['anchor_min'] = mins.to(p.dtype)

    elif mode == 'int4':
        q_blocks, scales, mins = quantize_blockwise(p, block_size=32, bits=4)
        q_flat = q_blocks.view(-1)
        # Vectorized packing: High bits | Low bits
        packed = (q_flat[0::2] << 4) | q_flat[1::2]

        state['anchor_data'] = packed
        state['anchor_scale'] = scales.to(p.dtype)
        state['anchor_min'] = mins.to(p.dtype)

    elif mode == 'full':
        state['anchor_data'] = p.clone()

def dequantize_anchor(p, state, group):
    """Restores the anchor to the original shape/dtype."""
    anchor_data = state['anchor_data']

    # If it was saved as full precision or float8
    if anchor_data.dtype in (p.dtype, torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        return anchor_data.to(p.dtype)

    # Dequantize Setup
    mode = group.get('centered_wd_mode', 'full')
    scales = state['anchor_scale']
    mins = state['anchor_min']
    orig_shape = p.shape
    orig_numel = p.numel()

    if mode == 'int4' and anchor_data.dtype == torch.uint8:
        block_size = 32
        unpacked = torch.empty(anchor_data.numel() * 2, dtype=torch.uint8, device=anchor_data.device)
        unpacked[0::2] = anchor_data >> 4
        unpacked[1::2] = anchor_data & 0x0F
        quantized_blocks = unpacked.view(-1, block_size)

    elif mode == 'int8' and anchor_data.dtype == torch.uint8:
        block_size = 128
        quantized_blocks = anchor_data

    # Core Dequantization: (q * scale) + min
    anchor_blocks = quantized_blocks.to(p.dtype) * scales.unsqueeze(1) + mins.unsqueeze(1)

    # Flatten, truncate any padding added during quantization, and reshape
    return anchor_blocks.view(-1)[:orig_numel].view(orig_shape)