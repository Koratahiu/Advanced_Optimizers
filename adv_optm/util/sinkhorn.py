import math
import torch

def apply_sr_sinkhorn(update: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """
    Applies Square-Root Sinkhorn (SR-Sinkhorn) multi-normalization.
    As described in 'Gradient Multi-Normalization for Efficient LLM Training'.

    This technique normalizes a 2D matrix alternatively by its row-wise L2 norm 
    and column-wise L2 norm, driving it toward a fixed point that uniformly 
    distributes update magnitudes.
    """
    original_shape = update.shape

    if update.dim() == 1:
            norm = update.norm(p=2).clamp_min_(1e-12)
            return update.mul_(math.sqrt(update.numel()) / norm)
    else:
        # Flatten >= 3D tensors into 2D matrices
        update_2d = update.view(update.shape[0], -1)

    m, n = update_2d.shape

    # Dynamically determine the order of normalization based on aspect ratio
    # Normalizing the longer dimension first aids stability.
    dim = 0 if m > n else 1

    # Precompute scaling factors. 
    scale_first = math.sqrt(m) if dim == 0 else math.sqrt(n)
    scale_second = math.sqrt(n) if dim == 0 else math.sqrt(m)

    # In-place alternating Sinkhorn normalization steps
    for _ in range(iters):
        # First normalization step
        norm1 = update_2d.norm(p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
        update_2d.mul_(scale_first / norm1)

        # Second normalization step
        norm2 = update_2d.norm(p=2, dim=1-dim, keepdim=True).clamp_min_(1e-12)
        update_2d.mul_(scale_second / norm2)

    return update_2d.view(original_shape)
