import torch
from torch.optim import Optimizer
from typing import Optional

from ..util.GeoEmbed_util import (
    reshape_2d,
    radial_projection,
    gram_condition,
    geodesic_decay,
    parallel_transport,
)


class GeoEmbed_adv(Optimizer):
    """
    Hyperspherical Geodesic Optimizer (GeoEmbed) for Token Embeddings with Riemannian Parallel-Transported Momentum.
    
    Operates strictly on the Riemannian manifold S^{D-1}(R):
      1. Riemannian Tangential Projection (Orthogonalization to parameter vector)
      2. Inter-token Polar Orthogonalization (Löwdin symmetric condition)
      3. Manifold-aware Anchor Regularizatio
      4. Riemannian Momentum via Levi-Civita Parallel Transport
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,            # Angular base step size in radians
        momentum: float = 0.95,       # Riemannian momentum factor
        nesterov: bool = True,      # Riemannian Nesterov acceleration
        nesterov_coef: float | None = None,
        weight_decay: float = 0.0,   # Geodesic anchor decay
        eps: float = 1e-12,
    ):
        if not lr > 0.0:
            raise ValueError(f"Invalid angular lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            nesterov_coef=nesterov_coef,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)
        self.init_anchors()

    @torch.no_grad()
    def init_anchors(self):
        for group in self.param_groups:
            for p in group["params"]:
                self._init_anchors(p)

    @torch.no_grad()
    def _init_anchors(self, p):
        state = self.state[p]
        if "anchor" not in state:
            state["anchor"] = p.detach().clone()
            if p.ndim == 1:
                state["radius"] = p.norm(p=2).clamp_min(1e-8)
            else:
                state["radius"] = torch.linalg.vector_norm(p, ord=2, dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            nesterov_coef = group["nesterov_coef"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                self._init_anchors(p)

                grad = p.grad
                state = self.state[p]
                anchor = state["anchor"]
                radius = state["radius"]

                orig_shape = p.shape
                p_2d, g_2d, a_2d, r_2d = reshape_2d(p, grad, anchor, radius)

                # Mask active tokens
                g_norm = torch.linalg.vector_norm(g_2d, ord=2, dim=-1, keepdim=True)
                active_mask = (g_norm > eps)
                if not active_mask.any():
                    continue

                p_norm_sq = torch.sum(p_2d * p_2d, dim=-1, keepdim=True).clamp_min(eps)

                # Project Gradient onto Tangent Space T_p S^{D-1}
                g_tangent = radial_projection(p_2d, g_2d, p_norm_sq)

                # Multi-Token Polar Decoupling
                u_tangent = gram_condition(g_tangent, p_2d, eps=eps)

                # Riemannian Momentum Integration
                if mu > 0.0:
                    # Only update momentum for active tokens; leave inactive tokens frozen
                    act_idx = active_mask.squeeze(-1)
                    if "momentum_buffer" not in state:
                        buf = torch.zeros_like(u_tangent)
                        buf[act_idx] = u_tangent[act_idx]
                        state["momentum_buffer"] = buf
                    else:
                        buf = state["momentum_buffer"]
                        buf[act_idx] = buf[act_idx] * mu + u_tangent[act_idx] * (1.0 - mu)

                    if nesterov:
                        nv = mu if nesterov_coef is None else nesterov_coef
                        update_vec = u_tangent.lerp(buf, nv)
                    else:
                        update_vec = buf
                else:
                    update_vec = u_tangent

                # Decoupled Geodesic Anchor Pull
                if wd > 0.0:
                    update = geodesic_decay(
                        update_vec, p_2d, a_2d, p_norm_sq, wd, 
                        active_mask=active_mask, eps=eps
                    )
                else:
                    update = update_vec

                # Geodesic Exponential Map Flow
                v_norm = torch.linalg.vector_norm(update.float(), ord=2, dim=-1, keepdim=True).to(p.dtype)
                act_v = (v_norm > eps) & active_mask
                cond_act = act_v.any()

                p_next = p_2d.clone()
                if cond_act:
                    # Unit direction along the tangent geodesic
                    u_hat = torch.zeros_like(update)
                    u_hat[act_v.squeeze(-1)] = update[act_v.squeeze(-1)] / v_norm[act_v.squeeze(-1)]

                    # Angular step smoothly scaling with v_norm and saturating at lr
                    theta = lr * torch.tanh(v_norm) * act_v.float()

                    cos_theta = torch.cos(theta)
                    sin_theta = torch.sin(theta)

                    # Geodesic flow preserving sphere radius ||p_next|| == ||p|| == R
                    p_next = p_2d * cos_theta - r_2d * u_hat * sin_theta

                # Parallel Transport Momentum Buffer to T_{p_next} for the Next Step
                if mu > 0.0 and cond_act:
                    state["momentum_buffer"] = parallel_transport(
                        p=p_2d,
                        p_next=p_next,
                        v=state["momentum_buffer"],
                        eps=eps
                    )

                p.copy_(p_next.view(orig_shape))

        return loss