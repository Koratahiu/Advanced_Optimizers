import torch

from . import param_update

import math

def scale_update(
    p: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    state: dict | None = None,
) -> torch.Tensor:
    """
    Applies adaptive scaling to the parameter update based on the parameter's
    role (DoRA, OFT, or LoRA/Full Finetuning).

    Args:
        p: The original parameter tensor.
        update: The computed gradient/update tensor to be scaled.
        lr: The learning rate.
        state: The state dict used for spectral normalization.

    Returns:
        The scaled update tensor.
    """
    is_dora_scale = getattr(p, '_is_dora_scale', False)
    is_oft = getattr(p, '_is_oft', False)

    # DoRA Magnitude Scales (1D) or 1D Bias/Norm layers
    if p.ndim < 2 or is_dora_scale:
        return max_abs_normalization(update, dim=None, lr=lr)

    # OFT Block Parameters: shape (k, C(b,2))
    # Normalise by max per-block row norm so that
    #   ‖ΔR_block‖_spec = max_i ‖ΔRᵢ‖_spec ≤ 2 · max_i ‖Δθᵢ‖₂ ≤ target_scale · lr
    if is_oft:
        return max_row_norm_oft_normalization(p, update, lr)

    # LoRA Factors or Full Finetuning weights
    # Scales update to maintain consistent spectral norm across different layer sizes and ranks.
    if p.ndim >= 2:
        d_out = update.shape[0]
        d_in = update.numel() // d_out
        target_scale = 1 if getattr(p, '_is_lora_A', False) else math.sqrt(d_out / d_in) 
        return spectral_normalization(update, state['spectral_u'], state['spectral_v'], lr=lr, target_scale=target_scale)


def scale_eps(eps: float | None, p: torch.Tensor) -> float:
    """
    Scales Adam eps to be scale-invariant.
    """
    if eps is None:
        return (1.0 / math.sqrt(p.numel()))
    else:
        return eps

def adjust_wds(wd: float, cwd: float, p: torch.Tensor) -> tuple[float, float]:
    """
    Adjusts standard weight decay and centered weight decay.
    """
    # DoRA Scale (Magnitude Vector)
    if getattr(p, '_is_dora_scale', False):
        return wd, cwd

    if getattr(p, '_is_oft', False):
        return wd, 0.0

    if p.ndim >= 2:
        is_lora = getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False)
        if is_lora:
            return wd, 0.0

        return wd, cwd

    else:
        # 1D Biases or generic 1D parameters
        # Centered WD safely regularizes the delta without collapsing base feature variance.
        return wd, cwd


def is_spectral(p: torch.Tensor) -> bool:
    """Determines if a parameter should undergo spectral normalization updates."""
    if p.ndim < 2:
        return False
    if getattr(p, '_is_oft', False) or getattr(p, '_is_dora_scale', False)or getattr(p, 'is_vector', False):
        return False
    return getattr(p, 'is_hidden', True)

@torch.no_grad()
def init_spectral_norm(state: dict, p: torch.Tensor):
    """Initializes the singular vectors 'u' and 'v' for the Power Iteration method."""
    gen = param_update.get_generator(p.device)

    d_out = p.shape[0]
    d_in = p.numel() // d_out

    # Initialize v (Right singular vector)
    v = torch.randn(d_in, device=p.device, dtype=p.dtype, generator=gen)
    state['spectral_v'] = v.div_(v.norm().add_(1e-12))

    # Initialize u (Left singular vector)
    u = torch.randn(d_out, device=p.device, dtype=p.dtype, generator=gen)
    state['spectral_u'] = u.div_(u.norm().add_(1e-12))


@torch.no_grad()
def l2_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs L2 normalization on the update tensor."""
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min(1e-12)
    return update.mul_(lr / norm)


@torch.no_grad()
def rms_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs Root Mean Square normalization on the update tensor."""
    n = update.numel() if dim is None else update.shape[dim]
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min(1e-12)
    scale_n = math.sqrt(n)
    return update.mul_(lr * scale_n / norm)

@torch.no_grad()
def max_abs_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """
    Performs L-infinity (Max Absolute) normalization.
    Strictly bounds the maximum update of any single element to 'lr'.
    """
    # ord=float('inf') computes the maximum absolute value
    norm = torch.linalg.vector_norm(update, ord=float('inf'), dim=dim, keepdim=True).clamp_min(1e-12)
    return update.mul_(lr / norm)

@torch.no_grad()
def max_row_norm_oft_normalization(
    p: torch.Tensor,
    update: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """
    Normalizes OFT parameter updates by the maximum per-block (row) L2 norm.

    For OFT params of shape (k, C(b,2)), each row Δθᵢ maps to a skew-symmetric
    matrix ΔQᵢ with ‖ΔQᵢ‖_spec ≤ ‖Δθᵢ‖₂. Via the Cayley linearization
    ΔRᵢ ≈ 2ΔQᵢ, bounding max_i ‖Δθᵢ‖₂ directly controls:

        ‖ΔR_block‖_spec = max_i ‖ΔRᵢ‖_spec ≤ 2 · max_i ‖Δθᵢ‖₂

    Result: Var[Δyⱼ] ≤ (target_scale · lr)² = O(1) for every block configuration.
    """
    scale_factor = getattr(p, '_oft_scale_factor', 1.0)

    # Base target scale: 0.5 for Cayley (R ≈ I + 2Q)
    # Compensate for the forward pass scaling so the effective rotation step is bounded by `lr`
    target_scale = 0.5 * scale_factor 

    # Row norms: shape (k,) - one per block
    row_norms = torch.linalg.vector_norm(update, ord=2, dim=-1)
    # Stability floor: equivalent to a single-element vector norm lower bound
    norm_lb = 1.0 / math.sqrt(update.shape[1])
    max_norm = row_norms.max().clamp_min(norm_lb)

    return update.mul_(lr * target_scale / max_norm)

@torch.no_grad()
def get_oft_structural_invariants(p: torch.Tensor):
    """
    Computes shared structural invariants and the Sparsity Index for OFT matrices.
    Returns tuple:
        n_el: Number of elements.
        b: Size of the orthogonal matrix derived from the elements.
        p_sq: Element-wise squared magnitude.
        p_norm_sq: Squared L2 norm (1D sum reduction).
        S: Sparsity Index / Inverse Participation Ratio.
    """
    n_el = p.shape[-1]
    b = (1.0 + math.sqrt(1.0 + 8.0 * n_el)) / 2.0
    scale_factor = getattr(p, '_oft_scale_factor', 1.0)

    # Element-wise squared magnitude
    p_sq = p.div(scale_factor).square_()

    # Structural invariants (1D sum reductions)
    p_norm_sq = p_sq.sum(dim=-1, keepdim=True)
    p_quad = p_sq.square().sum(dim=-1, keepdim=True)

    # Sparsity Index / Inverse Participation Ratio
    # S = 1.0 -> Single rank-2 rotation (Sparse)
    # S <= 2/b -> Multiple disjoint rotations / Uniform (Dense)
    S = p_quad.div_(p_norm_sq.square().clamp_min_(1e-12))

    return (n_el, b, p_sq, p_norm_sq, S)

@torch.no_grad()
def get_oft_magnitude_correction(p: torch.Tensor, oft_vars: tuple) -> torch.Tensor:
    """
    Approximates the magnitude correction of exact Riemannian preconditioning (M @ G @ M).
    Leverages the 4th moment invariant of skew-symmetric matrices to construct a Sparsity Index (S).
    Dynamically routes the correction between purely isotropic (for dense distributed rotations)
    and purely anisotropic (for sparse singular rotations).
    """
    n_el, b, p_sq, p_norm_sq, S = oft_vars

    # Dynamic Blending Factor (w)
    # Extracts the exact mixture of sparse vs dense topologies
    S_dense = 2.0 / b
    w = S.sub(S_dense).div_(max(1.0 - S_dense, 1e-6)).clamp_(min=0.0, max=1.0)

    # Adaptive Coefficients
    # A solves for the local optimal scale.
    # B mathematically guarantees the metric volume trace is strictly preserved: sum(Correction) == (b - 1) * ||p||^2
    A_max = (b - 1.0) / (b + 1.0)
    A = w.mul_(A_max) 
    B = A.neg().add_(b - 1.0).div_(n_el)

    # Apply dynamic Riemannian Correction
    # Evaluates: A * p_sq + B * ||p||^2 + 1.0
    base_correction = p_sq.mul_(A).add_(p_norm_sq.mul(B)).add_(1.0)

    return base_correction

def get_geodesic_decay_scaler(p: torch.Tensor, oft_vars: tuple) -> torch.Tensor:
    """
    Computes the scalar multiplier for geodesic weight decay.
    Near identity (‖Q‖ ≈ 0), this returns 1.0 (standard L2 decay). 
    Far from identity, it decays towards 0, respecting the bounded 
    geometry of SO(n) so large rotations aren't infinitely penalized.
    Dynamically maps the 4th-moment Sparsity Index to interpolate between
    dense distributed rotations and single sparse rotations.
    """
    n_el, b, _, p_norm_sq, S = oft_vars

    # Map Sparsity to a normalized weight (w in [0, 1])
    S_min = 1.0 / n_el
    w = S.sub_(S_min).div_(max(1.0 - S_min, 1e-6)).clamp_(min=0.0, max=1.0)

    # Estimate Effective Active Rotations (K_eff)
    # If Dense (w=0) -> b/2 rotations (matches standard isotropic bounds)
    # If Sparse (w=1) -> 1 dominant rotation plane
    max_rotations = b / 2.0
    K_eff = w.mul_(1.0 - max_rotations).add_(max_rotations)

    # Isolate the Average Squared Eigenvalue of the active rotations
    x_sq = p_norm_sq.div_(K_eff)
    x = x_sq.sqrt().clamp_min_(1e-8)

    # Calculate Riemannian penalty scaler
    # Geodesic Gradient / Euclidean Gradient ratio: d(theta)/d(lambda) / d(L2)/d(lambda)
    decay_scaler = torch.atan(x) / (x * (1.0 + x_sq))

    return decay_scaler


@torch.no_grad()
def spectral_normalization(
    update: torch.Tensor,
    u_state: torch.Tensor,
    v_state: torch.Tensor,
    lr: float,
    target_scale: float,
) -> torch.Tensor:
    """
    Applies Spectral Normalization via a single step of Power Iteration.
    Implementation follows: "Scalable Optimization in the Modular Norm" (arXiv:2405.14813).
    """
    d_out = update.shape[0]
    d_in = update.numel() // d_out
    update = update.to(u_state.dtype)
    update_flat = update.view(d_out, d_in)

    # Power Iteration step to estimate the largest singular value (sigma)
    # Update v (Right Singular Vector)
    v_raw = torch.mv(update_flat.mT, u_state)
    v_norm = torch.linalg.vector_norm(v_raw)
    candidate_v = v_raw / v_norm.clamp_min(1e-8)
    # Stability: Only update the state if the norm is significant
    next_v = torch.where(v_norm >= 1e-6, candidate_v, v_state)
    v_state.copy_(next_v)

    # Update u (Left Singular Vector)
    u_raw = torch.mv(update_flat, v_state)
    u_norm = torch.linalg.vector_norm(u_raw)
    candidate_u = u_raw / u_norm.clamp_min(1e-8)
    next_u = torch.where(u_norm >= 1e-6, candidate_u, u_state)
    u_state.copy_(next_u)

    # Estimate sigma (The spectral norm)
    sigma = torch.linalg.vecdot(u_state, u_raw)

    spectral_eps = 1.0 / (math.sqrt(d_out) + math.sqrt(d_in))

    # Rescale update
    scale = lr * (target_scale / sigma.clamp_min_(spectral_eps))
    return update.mul_(scale)
