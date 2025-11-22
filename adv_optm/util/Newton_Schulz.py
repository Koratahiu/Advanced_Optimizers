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
            # X = c1 * X + c3 * (A @ X)
            X = torch.addmm(X, A, X, beta=c1, alpha=c3)

            # Update the singular value bounds for the next iteration based on the error
            eps_num = common_den_part - ab_part
            eps_val = eps_num / alpha_den
            lower_bound = 1.0 - eps_val
            upper_bound = 1.0 + eps_val
    else:
        # Perform the iterative updates
        for _ in range(steps):
            A = X @ X.mT
            # B = b * A + c * (A @ A)
            B = torch.addmm(A, A, A, beta=b, alpha=c)
            # X = a * X + B @ X
            X = torch.addmm(X, B, X, beta=a)

    # Transpose back if necessary
    if transposed:
        X = X.mT

    return X.to(G.dtype)


@torch.no_grad()
def newton_schulz(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cns: bool = False,
    cns_a_bound: float = 1e-4,
    low_rank_ortho: bool = False,
    ortho_rank: int = 128,
) -> torch.Tensor:
    """
    Public entry point for Muon orthogonalization.
    Handles either full Newton-Schulz or Low-Rank Orthogonalization via Gaussian Sketching.

    Args:
        G (torch.Tensor): Input matrix (gradient/update).
        steps (int): NS iterations.
        eps (float): Numerical stability epsilon.
        coeffs (tuple): Polynomial coefficients.
        cns (bool): Use Chebyshev-accelerated Newton-Schulz.
        cns_a_bound (float): CANS lower bound.
        low_rank_ortho (bool): Whether to project to low rank before orthogonalizing.
        ortho_rank (int): Rank for low-rank projection.
    """
    if low_rank_ortho:
        # Low-Rank Orthogonalization based on Gaussian Sketching
        M = G
        r = min(ortho_rank, M.shape[0], M.shape[1])

        if r > 0:
            # 1. Sketch the matrix
            G_sketch = torch.randn(M.shape[1], r, device=M.device, dtype=M.dtype)
            MG = M @ G_sketch

            # 2. QR decomposition to get orthogonal basis Q
            # Handle dtype mismatch for QR if necessary
            if MG.dtype != torch.float32:
                MG_dtype = M.dtype
                Q, _ = torch.linalg.qr(MG.float())
                Q = Q.to(MG_dtype)
            else:
                Q, _ = torch.linalg.qr(MG)

            # 3. Project M onto the basis
            projected_M = Q.T @ M

            # 4. Orthogonalize the smaller projected matrix
            ortho_projected_M = _newton_schulz_iteration(
                projected_M,
                steps=steps,
                eps=eps,
                coeffs=coeffs,
                cns=cns,
                cns_a_bound=cns_a_bound
            )

            # 5. Project back to the original space
            return Q @ ortho_projected_M

    # Fallback (if rank invalid) or Standard Path
    return _newton_schulz_iteration(
        G,
        steps=steps,
        eps=eps,
        coeffs=coeffs,
        cns=cns,
        cns_a_bound=cns_a_bound
    )
