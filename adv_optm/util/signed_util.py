import torch

from . import param_update

def apply_stochastic_sign_(update: torch.Tensor, noise: torch.Tensor | None, is_vector: bool = False) -> torch.Tensor:
    """
    Applies the Stochastic Sign operator S_R(v).
    Uses uniform noise injection to compute the stochastic sign
    """
    if update.dim() >= 2 and not is_vector:
        update_abs = update.abs()
        # Calculate row and col maximums
        R_col = update_abs.amax(dim=0, keepdim=True) # Shape: (1, cols)
        R_row = update_abs.amax(dim=1, keepdim=True) # Shape: (rows, 1)
        R = torch.minimum(R_row, R_col)
    else:
        # Fallback for 1D tensors (e.g., biases, layernorm)
        # Block-wise scaling to protect against outliers
        block_size = 128
        numel = update.numel()

        if numel <= block_size:
            # Too small to chunk, just use global max
            R = update.abs().max()
        else:
            # Calculate how much padding we need to make it divisible by block_size
            remainder = numel % block_size

            # Flatten update to ensure 1D padding works correctly for different shapes like (3000, 1)
            flat_update = update.reshape(-1)

            if remainder != 0:
                pad_len = block_size - remainder
                # Pad with zeros so they don't affect the maximum
                padded_update = torch.nn.functional.pad(flat_update, (0, pad_len))
            else:
                padded_update = flat_update

            # Reshape into blocks and get max per block
            blocks = padded_update.view(-1, block_size)
            R_blocks = blocks.abs().max(dim=1, keepdim=True).values

            # Broadcast R_blocks back to the padded shape, slice off padding, and restore original shape
            R = R_blocks.expand_as(blocks).reshape(-1)[:numel].view_as(update)

    # Prevent division by zero
    R = R.clamp_min(1e-12)

    if noise is None:
        noise = param_update._get_random_noise_for_sso(update)

    # Chain inplace operations: torch.sign(update / R + noise)
    return update.div_(R).add_(noise).sign_()
