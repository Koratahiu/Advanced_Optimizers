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

    Some optimizations inspired from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/muon.py#L79

    Args:
        G (torch.Tensor): The 2D input matrix (momentum-accumulated gradient).
        steps (int): The number of iterations to run.
        eps (float): Small constant for numerical stability during normalization.
        coeffs (Union[Tuple[float, float, float], List[Tuple[float, float, float]]]):
            The (a, b, c) coefficients for the quintic polynomial update.
        cns (bool): If True, enables Chebyshev-accelerated Newton-Schulz (CANS)
            using an iterative 3rd-order polynomial with optimal coefficients
            derived at each step.
        cns_a_bound (float): The initial lower bound for singular values when
            using CANS. The upper bound is assumed to be 1.0 after normalization.
    Returns:
        torch.Tensor: The orthogonalized matrix.
    """
    assert G.ndim in (2, 3), f"Input must be 2D or 3D, got {G.ndim}D"

    a, b, c = coeffs

    X = G.to(torch.bfloat16)

    # Transpose if needed
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Normalize spectral norm to at most 1
    X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp_min_(eps))

    # Select matrix multiplication function based on dimension (Batched vs Standard)
    mm_fn = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Pre-allocate for performance
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    # Allocating B and C for standard NS to avoid loop allocations
    # We also reuse C for CNS updates to be efficient
    C = torch.empty_like(X)
    if not cns:
        B = torch.empty_like(A)

    if cns:
        # Chebyshev-accelerated Newton-Schulz (CANS) from
        # "Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials"
        # This implements the iterative scheme from Algorithm 1, using the
        # closed-form 3rd-order polynomial from Proposition 2.
        # Note: CANS calculates its own coefficients dynamically, ignoring `coeffs`
        lower_bound = cns_a_bound
        upper_bound = 1.0  # Matrix is normalized, so largest singular value is approx 1.

        for _ in range(steps):
            # Calculate optimal 3rd-order coefficients c1, c3 for p(x) = c1*x + c3*x^3
            # based on the current singular value bounds [lower_bound, upper_bound].
            # Formulas are derived from Proposition 2 and its proof in Appendix B of the paper.
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub

            # Calculate Mean Square Error term
            e_sq = (lb*lb + lb_ub + ub*ub) / 3.0

            # Calculate components for alpha and bounds update
            # K is the error scaling component
            # L is the bound interaction component
            K = 2.0 * e_sq**1.5
            L = lb_ub * (lb + ub)

            denom = K + L

            # Calculate alpha, which scales the polynomial
            alpha = 6.0 / denom

            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            # Apply the 3rd-order Newton-Schulz update
            # A = X @ X.mT
            mm_fn(A, X, X.mT, beta=0.0, alpha=1.0, out=A)
            # X = c1 * X + c3 * (A @ X)
            mm_fn(X, A, X, beta=c1, alpha=c3, out=C)
            X, C = C, X

            # Update the singular value bounds for the next iteration based on the error
            eps_val = (K - L) / denom
            lower_bound, upper_bound = 1.0 - eps_val, 1.0 + eps_val
    else:
        # Standard Quintic Newton-Schulz
        for _ in range(steps):
            # A = X @ X.mT
            mm_fn(A, X, X.mT, beta=0.0, alpha=1.0, out=A)
            # B = b * A + c * (A @ A)
            mm_fn(A, A, A, beta=b, alpha=c, out=B)
            # X = a * X + B @ X
            mm_fn(X, B, X, beta=a, alpha=1.0, out=C)
            X, C = C, X  # swap refs to avoid copy

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
    Source: "Low-rank Orthogonalization for Large-scale Matrix Optimization with Applications
    to Foundation Model Training" (https://arxiv.org/abs/2509.11983)

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
        # Low-Rank Orthogonalization via Gaussian Sketching
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

    # Standard Path
    return _newton_schulz_iteration(
        G,
        steps=steps,
        eps=eps,
        coeffs=coeffs,
        cns=cns,
        cns_a_bound=cns_a_bound
    )

def _is_suitable_for_muon(
        param: torch.Tensor,
        min_dim_size: int = 4,
        max_aspect_ratio: float = 128.,
) -> bool:
    """Check if a parameter is suitable for Muon optimization.
    modified from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/muon.py#L167
    """

    s = param.shape
    # Must have at least 2 non-unit dimensions
    if param.ndim < 2 or sum(1 for dim_size in s if dim_size > 1) < 2:
        return False

    # Unit dimension in first two positions indicates:
    # - Position embeddings (1, seq, dim)
    # - Depthwise convs (out, 1, h, w)
    # - Other degenerate cases possibly not caught by first rule
    if s[0] == 1 or s[1] == 1:
        return False

    if param.ndim >= 3:
        # For 3D+ tensors, check what dimensions will be AFTER flattening
        # since that's what gets passed to Newton-Schulz iteration
        # Flatten mode: (out, in, *spatial) -> (out, in * spatial_prod)
        out_ch = s[0]
        in_ch_with_spatial = 1
        for d in s[1:]:
            in_ch_with_spatial *= d
        check_dims = (out_ch, in_ch_with_spatial)
    else:
        # For 2D tensors, check as-is
        check_dims = s

    # Both dims should be >= minimum size
    min_size = min(check_dims)
    if min_size < min_dim_size:
        return False

    # Aspect ratio shouldn't be too extreme
    max_size = max(check_dims)
    aspect_ratio = max_size / min_size
    if aspect_ratio > max_aspect_ratio:
        return False

    return True

def approx_mars(current_grad: torch.Tensor, last_grad: torch.Tensor, mars_gamma:float, beta1:float, Simplified_AdEMAMix:bool=False):
    """
    The approximated version of MARS-M, proposed in the paper: "MARS-M: When Variance Reduction
    Meets Matrices" (https://arxiv.org/abs/2510.21800). A variance reduction technique that
    incorporates the changes in gradients into the momentum gradient.
    Formula: c_t = g_t + gamma * beta / (1 - beta) * (g_t - g_{t-1}
    """
    if Simplified_AdEMAMix:
        mars_factor = mars_gamma * beta1
    else:
        mars_factor = mars_gamma * beta1 / (1.0 - beta1)
    # Compute corrected gradient c_t
    # c_t = current_grad + mars_factor * (current_grad - last_grad)
    correction = current_grad.sub(last_grad).mul_(mars_factor).add_(current_grad)
    # Update last_grad to current grad for the next step
    last_grad.copy_(current_grad)
    # Use correction as the gradient for subsequent momentum updates
    return correction

def normuon_update(update: torch.Tensor, v_t: torch.Tensor, beta2, eps):
    """
    The scalar state update of NorMuon variant, proposed in the paper: "NorMuon: Making Muon more
    efficient and scalable" (https://arxiv.org/abs/2510.05491). Implement a row-wise normalization via
    2nd moment estimation to balance parameter utilization and retain Muon conditioning.
    """
    # Update 2nd moment estimate
    mean_squared_update = torch.mean(update.square(), dim=1, dtype=v_t.dtype)
    v_t.lerp_(mean_squared_update, 1 - beta2)
    # Normalize update
    del mean_squared_update
    return update.div_(v_t.sqrt().unsqueeze_(1).add_(eps))

def rms_adjustment(update: torch.Tensor, rms_rescaling: bool, lr):
    if rms_rescaling: # RMS-aligned rescaling
        # This is slower due to norm calculations but it worked the best for t2i models.
        rms_target = 0.2 # default (Adam) value for RMS
        update_norm = torch.linalg.vector_norm(update)
        return update.mul_(lr * rms_target * (update.numel()**0.5) / update_norm.clamp_min_(1e-8))
    else:
        # Original Muon scaling
        r, c = update.size(-2), update.size(-1)
        scaling_factor = max(1, r / c) ** 0.5
        return update.mul_(lr * scaling_factor)

def _auto_projection_for_adamuon(raw_update: torch.Tensor, kappa_p: float) -> torch.Tensor:
    """
    Inspired from the paper "Lion Secretly Solves Constrained Optimization,
    As Lyapunov Predicts". (https://arxiv.org/abs/2310.05898)

    The core finding of the Lion-K paper is that the optimal "projection"
    depends on the geometry of the parameters:
    - Linear Layers / Transformers (p=1.0): These weights often benefit from
    coordinate-wise uniformity. The "Sign" update (standard Lion/AdaMuon) works
    best here because it treats every neuron/channel as equally important.
    - Convolutional Layers / UNet (p=2.0): These weights often possess rotational
    invariance. A hard "Sign" update distorts the direction of the gradient vector
    in 4D space (Batch, Channel, H, W). A "Spherical" update (p=2) preserves the
    direction while normalizing the magnitude.

    We take those findings and apply it to AdaMuon raw update.
    """
    EPS = 1e-12
    x = raw_update
    p = kappa_p

    # Standard (p=1) - sign update
    if p == 1.0:
        return x.sign_()

    # Spherical (p=2) - rotation invariant
    if p == 2.0:
        # Normalize (L2=1)
        norm = x.norm(p=2).clamp_min_(EPS)
        x.div_(norm)
        return x

    # General p case - hybrid optimizer
    # Calculate the 'Direction' Numerator: sign(x) * |x|^(p-1)
    num = x.sign() * x.abs().pow_(p - 1)

    # Denominator: ||x||_p^(p-1)
    den = x.norm(p=p).pow_(p - 1).clamp_min_(EPS)
    return num.div_(den)
