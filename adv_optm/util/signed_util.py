import torch

from . import param_update

def apply_stochastic_sign_(update: torch.Tensor, noise: torch.Tensor | None, is_vector: bool = False) -> torch.Tensor:
    """
    Applies the Iterative L-infinity Stochastic Sign operator.
    Uses uniform noise injection to compute the stochastic sign
    """
    if update.dim() >= 2 and not is_vector:
        # Iterative L-infinity Sinkhorn algorithm
        # This converges in just one iteration
        # Step 1: Row Max (every row max is 1.0, all values <= 1.0)
        R_row = torch.linalg.vector_norm(update, ord=float('inf'), dim=1, keepdim=True).clamp_min_(1e-12)
        update.div_(R_row)

        # Step 2: Col Max (every col max is 1.0 and every row max stays 1.0)
        R_col = torch.linalg.vector_norm(update, ord=float('inf'), dim=0, keepdim=True).clamp_min_(1e-12)
        update.div_(R_col)
    else:
        # Fallback for 1D tensors (e.g., biases, layernorm)
        # Block-wise scaling to protect against outliers
        block_size = 128
        numel = update.numel()

        if numel <= block_size:
            # Too small to chunk, just use global max
            R = update.abs().max().clamp_min_(1e-12)
            update.div_(R)
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
            R = R_blocks.expand_as(blocks).reshape(-1)[:numel].view_as(update).clamp_min(1e-12)
            update.div_(R)

    if noise is None:
        noise = param_update._get_random_noise_for_sso(update)

    # Final stochastic step: sign(v + U[-1, 1])
    return update.add_(noise).sign_()