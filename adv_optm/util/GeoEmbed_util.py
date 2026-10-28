import torch
import math

def reshape_2d(p, grad, anchor, radius):
    if p.ndim == 1:
        return p.view(1, -1), grad.view(1, -1), anchor.view(1, -1), radius.view(1, 1)
    return p.view(-1, p.shape[-1]), grad.view(-1, p.shape[-1]), anchor.view(-1, p.shape[-1]), radius.view(-1, 1)

def radial_projection(p_2d: torch.Tensor, g_2d: torch.Tensor, p_norm_sq: torch.Tensor) -> torch.Tensor:
    """Projects Euclidean gradient onto tangent space T_p S^{D-1}."""
    radial_comp = torch.sum(p_2d * g_2d, dim=-1, keepdim=True) / p_norm_sq
    return g_2d - radial_comp * p_2d

def gram_condition(K: int, g_tangent: torch.Tensor, eps: float) -> torch.Tensor:
    """Inter-token polar orthogonalization (Löwdin symmetric orthogonalization)."""
    g_norms = torch.linalg.vector_norm(g_tangent, ord=2, dim=-1, keepdim=True)
    active_mask = (g_norms > eps).float()

    if K > 1:
        try:
            # Gram matrix in tangent space (K, K)
            gram = torch.matmul(g_tangent, g_tangent.mT)
            U, S, Vh = torch.linalg.svd(gram.float())
            inv_sqrt_S = torch.diag_embed(1.0 / torch.sqrt(S.clamp_min(eps)))
            inv_sqrt_gram = (U @ inv_sqrt_S @ Vh).to(g_tangent.dtype)

            # Orthonormal basis
            u_ortho = torch.matmul(inv_sqrt_gram, g_tangent)

            # Re-apply gradient magnitudes so step size scales with loss
            u_tangent = u_ortho * g_norms
        except Exception:
            u_tangent = g_tangent
    else:
        u_tangent = g_tangent

    return u_tangent * active_mask

def geodesic_decay(
    u_tangent: torch.Tensor,
    p_2d: torch.Tensor,
    a_2d: torch.Tensor,
    p_norm_sq: torch.Tensor,
    r_2d: torch.Tensor,
    wd: float,
    eps: float
) -> torch.Tensor:
    """
    Pulls tangent direction toward the Riemannian anchor along the geodesic.
    Fixed sign: Moves along -anchor_tangent in the exponential map.
    """
    anchor_radial = torch.sum(p_2d * a_2d, dim=-1, keepdim=True) / p_norm_sq
    anchor_tangent = a_2d - anchor_radial * p_2d
    anchor_norm = torch.linalg.vector_norm(anchor_tangent, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    anchor_dir = anchor_tangent / anchor_norm

    # Geodesic distance (angle along the sphere arc)
    cos_dist = (torch.sum(p_2d * a_2d, dim=-1, keepdim=True) / (r_2d * r_2d)).clamp(-1.0 + eps, 1.0 - eps)
    geo_angle = torch.acos(cos_dist)

    # SUBTRACT anchor_dir because exponential flow subtracts u_tangent!
    # - (-anchor_dir) = +anchor_dir (moves TOWARD anchor)
    return u_tangent - (wd * geo_angle * anchor_dir)

def parallel_transport(
    p: torch.Tensor,
    p_next: torch.Tensor,
    v: torch.Tensor,
    r_norm_sq: torch.Tensor,
    cos_theta: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Symmetric Levi-Civita Parallel Transport on S^{D-1}(R).
    Transports tangent vector v from T_p S^{D-1} to T_{p_next} S^{D-1} along the geodesic.
    Guarantees: <p_next, v_trans> = 0 and ||v_trans|| = ||v||.
    """
    # <p_next, v>
    dot_qv = torch.sum(p_next * v, dim=-1, keepdim=True)
    
    # Denominator: R^2 * (1 + cos(theta))
    denom = r_norm_sq * (1.0 + cos_theta).clamp_min(eps)
    
    # Transport formula: v - (<p_next, v> / denom) * (p + p_next)
    v_transported = v - (dot_qv / denom) * (p + p_next)
    
    return v_transported