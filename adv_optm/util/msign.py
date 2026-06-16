import torch

import math

@torch.no_grad()
def msign_ortho_precond_(p):
    """
    Applies an orthogonal preconditioner to a parameter tensor in-place using Matrix Sign.

    Originally proposed in the paper:
    "MSign: An Optimizer Preventing Training Instability in Large Language Models via Stable Rank Restoration"(arxiv:2602.01734)
    Modified to use CANS instead of SVD.

    Args:
        p (torch.Tensor): The parameter tensor to be preconditioned. Must be a dense 
            tensor. Vectors and biases should generally be excluded before calling this. 
            The operation modifies `p` in-place.

    Note:
        - The Frobenius norm of the matrix is strictly preserved.
        - High-dimensional tensors (e.g., Conv2D weights) are reshaped into 
          `(out_channels, in_channels * ...)` automatically before computation.
    """
    # Record the original Frobenius norm of the weight matrix
    orig_norm = torch.linalg.vector_norm(p, ord=2).clamp_min_(1e-12)
    # Reshape parameter to 2D for the matrix operation
    p_2d = p.view(p.shape[0], -1)
    # Approximate the matrix sign UV^T using CA Newton-Schulz
    p_sign = _cans_newton_schulz_iteration(
        p_2d,
        steps=15,
        eps=1e-7,
        cns_a_bound=None, # auto
    )
    # Calculate the Frobenius norm of the sign-projected matrix
    sign_norm = torch.linalg.vector_norm(p_sign, ord=2).clamp_min_(1e-12)
    # Restore original Frobenius norm scale and write back in-place
    p.copy_(p_sign.mul_(orig_norm / sign_norm).view_as(p))

@torch.no_grad()
def _cans_newton_schulz_iteration(
    G: torch.Tensor,
    steps: int = 15,
    eps: float = 1e-7,
    cns_a_bound: float | None = None,
) -> torch.Tensor:
    """
    Computes the matrix sign function using Chebyshev-Optimized Newton-Schulz (CANS) iterations.

    This function iteratively approximates the orthogonalized version of a matrix (i.e., UV^T from SVD) 
    using purely matrix multiplications. This is highly optimized for Tensor Cores on modern GPUs 
    and is significantly faster than computing an exact SVD.

    Args:
        G (torch.Tensor): The input matrix/tensor to be orthogonalized. Must be at least 2D.
        steps (int, optional): Number of Newton-Schulz iterations. Deep learning models 
            usually only require 5 or 6 steps for sufficient orthogonal preconditioner 
            precision due to quadratic convergence. Defaults to 6.
        eps (float, optional): A small epsilon value used to prevent division by zero 
            during the initial spectral normalization. Defaults to 1e-7.
        cns_a_bound (float | None, optional): The lower bound for singular values after 
            normalization. If None, it is automatically calculated based on the Marchenko-Pastur 
            theoretical minimum and baseline L2 bounds. Defaults to None.

    Returns:
        torch.Tensor: The approximate matrix sign of G (orthogonalized matrix), preserving 
        the original shape and device of G.
    """
    X = G

    # Transpose if needed
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Normalize spectral norm to at most 1
    X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp_min_(eps))

    if cns_a_bound is None:
        M = G.shape[-2]
        N = G.shape[-1]
        # baseline L2 norm bound (for square matrices)
        baseline_bound = 1.0 / math.sqrt(M * N)
        # Marchenko-Pastur theoretical minimum (for rectangular matrices)
        mp_bound = abs(1.0 / math.sqrt(N) - 1.0 / math.sqrt(M))
        # The optimal bound is safely the maximum of the two
        cns_a_bound = max(baseline_bound, mp_bound)

    lower_bound = cns_a_bound
    upper_bound = 1.0
    for _ in range(steps):
        lb, ub = lower_bound, upper_bound
        lb_ub = lb * ub
        # Calculate Mean Square Error term
        e_sq = (lb**2 + lb_ub + ub**2) / 3.0
        # Calculate components for alpha and bounds update
        K = 2.0 * e_sq**1.5
        L = lb_ub * (lb + ub)
        denom = K + L
        alpha = 6.0 / denom
        c1 = alpha * e_sq
        c3 = -alpha / 3.0
        # Apply the 3rd-order Newton-Schulz update
        A = X @ X.mT
        X = c1 * X + c3 * (A @ X)
        # Update the singular value bounds for the next iteration based on the error
        eps_val = (K - L) / denom
        lower_bound, upper_bound = 1.0 - eps_val, 1.0 + eps_val

    # Transpose back if necessary
    if transposed:
        X = X.mT

    return X
