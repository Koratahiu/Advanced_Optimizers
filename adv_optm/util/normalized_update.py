import torch

from typing import Dict, Any

import torch

def spectral_norm_update(update: torch.Tensor, shape: torch.Size, is_hidden: bool, lr: float, L: int, vector_state: torch.Tensor | None = None):
    """
    Applies the modular norm scaling to the update tensor.
    paper:
    "Scalable Optimization in the Modular Norm"
    Allows scalable hyperparameters transfer
    Args:
        update: The gradient/update tensor (d_out, d_in, ...)
        shape: The shape of the weight tensor.
        is_hidden: Boolean, true if this is a hidden layer.
        lr: The learning rate.
        L: The number of hidden blocks (depth).
        vector_state: A buffer tensor of shape (d_in_total,) for power iteration.
    """

    # Determine Target Scale (Mass Logic)
    d_out = shape[0]
    d_in_total = update.numel() // d_out # Handles Conv2D (In * K * K)

    target_norm = (d_out / d_in_total) ** 0.5 

    # Hidden layers get 1/L of the budget
    if is_hidden:
        target_norm *= (1.0 / L)

    # Compute Current Norm (Sigma)
    if len(shape) == 1:
        # For biases / vectors: Use Euclidean / RMS norm
        sigma = torch.linalg.vector_norm(update)
    else:
        # For Matrices / Conv Kernels: Use Power Iteration
        # Flatten input dims: (Out, In, H, W) -> (Out, In*H*W)
        mat = update.view(d_out, -1) 

        # Power Iteration Step
        # u = A @ v_old
        u = torch.mv(mat, vector_state)
        # v_new = A.T @ u
        v_new = torch.mv(mat.T, u)

        # Normalize v
        v_norm = torch.linalg.vector_norm(v_new)

        # if v_norm >= 0.5:
        #    vector_state.copy_(v_new.div_(v_norm.clamp_min_(1e-12))).to(vector_state.dtype))
        candidate_v = v_new / v_norm
        next_state = torch.where(v_norm >= 0.5, candidate_v, vector_state)
        vector_state.copy_(next_state.to(vector_state.dtype))
        # Else: We keep the old vector_state (which is a random unit vector at init)

        # Estimate Sigma = || A @ v ||
        Av = torch.mv(mat, vector_state)
        sigma = torch.linalg.vector_norm(Av)

    #  Normalize and Apply LR
    scale_factor = lr * (target_norm / sigma.clamp_min_(1e-12))

    update.mul_(scale_factor)

    return update


def get_weight_decay_scaling(shape: torch.Size):
    """
    Assumes shape is (d_out, d_in).

    Returns:
        wd_scale: Weight decay scale
    """
    # Weight Decay -> 1/width
    d_in = shape[1]
    wd_scale = 1.0 / d_in

    return wd_scale


def get_adaptive_eps_scaling(shape: torch.Size, is_hidden: bool, L: int):
    """
    Returns:
        adaptive_eps: Epsilon for Adam-style denominator.
    """
    d_out = shape[0]
    d_in_total = shape.numel() // d_out 
    # Adaptive Denominator Epsilon
    # This ensures the Adam-style division doesn't explode or vanish.
    # Formula: (1/L) * (1 / sqrt(d_in * d_out))
    adaptive_eps = (1.0 / (d_in_total * d_out)**0.5)

    if is_hidden:
        adaptive_eps  *= (1/L)

    return adaptive_eps