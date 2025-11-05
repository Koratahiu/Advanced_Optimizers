import torch

@torch.no_grad()
def _newton_schulz_iteration(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cns: bool = False,
    cns_a_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Performs the Newton-Schulz iteration to find the nearest orthogonal matrix.
    This is the core computation of the Muon optimizer.

    Args:
        G (torch.Tensor): The 2D input matrix (momentum-accumulated gradient).
        steps (int): The number of iterations to run.
        eps (float): Small constant for numerical stability during normalization.
        coeffs (tuple[float, float, float]): The (a, b, c) coefficients for the
            quintic polynomial update.
        cns (bool): If True, enables Chebyshev-accelerated Newton-Schulz (CANS)
            using an iterative 3rd-order polynomial with optimal coefficients
            derived at each step.
        cns_a_bound (float): The initial lower bound for singular values when
            using CANS. The upper bound is assumed to be 1.0 after normalization.
    Returns:
        torch.Tensor: The orthogonalized matrix.
    """
    assert G.ndim >= 2

    a, b, c = coeffs

    X = G.to(torch.bfloat16)

    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    if cns:
        # Chebyshev-accelerated Newton-Schulz (CANS) from
        # "Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials"
        # This implements the iterative scheme from Algorithm 1, using the
        # closed-form 3rd-order polynomial from Proposition 2.
        lower_bound = cns_a_bound
        upper_bound = 1.0  # Matrix is normalized, so largest singular value is approx 1.

        for _ in range(steps):
            # Calculate optimal 3rd-order coefficients c1, c3 for p(x) = c1*x + c3*x^3
            # based on the current singular value bounds [lower_bound, upper_bound].
            # Formulas are derived from Proposition 2 and its proof in Appendix B of the paper.
            a_bound, b_bound = lower_bound, upper_bound
            term = a_bound*a_bound + a_bound*b_bound + b_bound*b_bound
            e_sq = term / 3.0

            # Calculate alpha, which scales the polynomial
            common_den_part = 2.0 * (e_sq**1.5)
            ab_part = a_bound*a_bound*b_bound + b_bound*b_bound*a_bound
            alpha_den = common_den_part + ab_part
            alpha = 6.0 / alpha_den

            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            # Apply the 3rd-order Newton-Schulz update
            A = X @ X.mT
            X = c1 * X + c3 * (A @ X)

            # Update the singular value bounds for the next iteration based on the error
            eps_num = common_den_part - ab_part
            eps_val = eps_num / alpha_den
            lower_bound = 1.0 - eps_val
            upper_bound = 1.0 + eps_val
    else:
        # Perform the iterative updates
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    # Transpose back if necessary
    if transposed:
        X = X.mT

    return X.to(G.dtype)
