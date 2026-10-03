import torch


def reshape_2d(p, grad, anchor, radius):
    if p.ndim == 1:
        return p.view(1, -1), grad.view(1, -1), anchor.view(1, -1), radius.view(1, 1)
    return p.view(-1, p.shape[-1]), grad.view(-1, p.shape[-1]), anchor.view(-1, p.shape[-1]), radius.view(-1, 1)


def radial_projection(p_2d: torch.Tensor, g_2d: torch.Tensor, p_norm_sq: torch.Tensor) -> torch.Tensor:
    """Projects Euclidean gradient onto tangent space T_p S^{D-1}."""
    radial_comp = torch.sum(p_2d * g_2d, dim=-1, keepdim=True) / p_norm_sq
    return g_2d - radial_comp * p_2d


def gram_condition(K: int, g_tangent: torch.Tensor, p_2d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Löwdin symmetric orthogonalization over ACTIVE tokens only,
    followed by re-projection to guarantee tangent space membership.
    """
    g_norms = torch.linalg.vector_norm(g_tangent, ord=2, dim=-1, keepdim=True)
    active_idx = (g_norms.squeeze(-1) > eps).nonzero(as_tuple=True)[0]
    
    if len(active_idx) == 0:
        return torch.zeros_like(g_tangent)

    u_tangent = torch.zeros_like(g_tangent)

    if len(active_idx) > 1:
        g_act = g_tangent[active_idx]  # (K_act, D)
        # Gram matrix: (K_act, K_act)
        gram = torch.matmul(g_act, g_act.mT)

        # SVD of Gram matrix
        U, S, Vh = torch.linalg.svd(gram)

        # Truncated pseudo-inverse square root
        max_s = S[0].clamp_min(eps)
        rel_thresh = max_s * 1e-4
        valid_mask = S > rel_thresh

        inv_sqrt_S = torch.zeros_like(S)
        inv_sqrt_S[valid_mask] = 1.0 / torch.sqrt(S[valid_mask])
        inv_sqrt_gram = U @ torch.diag_embed(inv_sqrt_S) @ Vh

        u_act = (inv_sqrt_gram @ g_act).to(g_tangent.dtype)
        u_tangent[active_idx] = u_act
    else:
        idx = active_idx[0]
        u_tangent[idx] = g_tangent[idx] / g_norms[idx].clamp_min(eps).to(g_tangent.dtype)

    # Re-project onto each token's tangent space T_{p_i}
    p_norm_sq = torch.sum(p_2d * p_2d, dim=-1, keepdim=True).clamp_min(eps)
    u_tangent = radial_projection(p_2d, u_tangent, p_norm_sq)

    # Normalize active tangent vectors to unit norm
    u_norms = torch.linalg.vector_norm(u_tangent, ord=2, dim=-1, keepdim=True).clamp_min(eps).to(u_tangent.dtype)
    u_tangent[active_idx] = u_tangent[active_idx] / u_norms[active_idx]

    return u_tangent


def geodesic_decay(
    u_tangent: torch.Tensor,
    p_2d: torch.Tensor,
    a_2d: torch.Tensor,
    p_norm_sq: torch.Tensor,
    wd: float,
    active_mask: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Riemannian anchor pull along geodesic: grad(1/2 * dist^2(p, a)) = -log_p(a).
    Bypassed for inactive or zero tokens.
    """
    p_norm = torch.linalg.vector_norm(p_2d, dim=-1, keepdim=True).clamp_min(eps)
    a_norm = torch.linalg.vector_norm(a_2d, dim=-1, keepdim=True).clamp_min(eps)

    # Tangent vector pointing from p towards anchor a
    anchor_radial = (torch.sum(p_2d * a_2d, dim=-1, keepdim=True) / p_norm_sq)
    anchor_tangent = (a_2d - anchor_radial * p_2d)
    anchor_tangent_norm = torch.linalg.vector_norm(anchor_tangent, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    anchor_dir = (anchor_tangent / anchor_tangent_norm)

    # Angular distance
    cos_dist = (torch.sum(p_2d * a_2d, dim=-1, keepdim=True) / (p_norm * a_norm)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    geo_angle = torch.acos(cos_dist)

    # zero out restoring force for any inactive/zero token
    decay_force = (wd * geo_angle * anchor_dir) * active_mask.float()

    return u_tangent - decay_force


def parallel_transport(
    p: torch.Tensor,
    p_next: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Levi-Civita Parallel Transport along great-circle arc:
    Transports v from T_p S^{D-1} to T_{p_next} S^{D-1}.
    """
    p_2d = p
    q_f = p_next
    v_f = v

    dot_qv = torch.sum(q_f * v_f, dim=-1, keepdim=True)
    dot_pp = torch.sum(p_2d * p_2d, dim=-1, keepdim=True)
    dot_pq = torch.sum(p_2d * q_f, dim=-1, keepdim=True)

    denom = (dot_pp + dot_pq).clamp_min(eps)
    v_trans = v_f - (dot_qv / denom) * (p_2d + q_f)

    # Clean up numerical float drift to ensure strict tangency <q, v_trans> = 0
    q_norm_sq = torch.sum(q_f * q_f, dim=-1, keepdim=True).clamp_min(eps)
    radial_leak = torch.sum(q_f * v_trans, dim=-1, keepdim=True) / q_norm_sq
    v_trans = v_trans - radial_leak * q_f

    return v_trans.to(v.dtype)