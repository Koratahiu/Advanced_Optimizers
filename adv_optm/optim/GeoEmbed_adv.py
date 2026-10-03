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
        """Stores the initial embedding vectors as Riemannian anchors."""
        for group in self.param_groups:
            for p in group["params"]:
                self._init_anchors(p)

    @torch.no_grad()
    def _init_anchors(self, p):
        state = self.state[p]
        if "anchor" not in state:
            state["anchor"] = p.detach().clone()
            # Pre-calculate target manifold radius per token
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

                # Ensure 2D representation: (K, D) where K is token count, D is dimension
                orig_shape = p.shape
                p_2d, g_2d, a_2d, r_2d = reshape_2d(p, grad, anchor, radius)
                K, D = p_2d.shape

                # Detect active gradient mask to prevent idle token norm shrinkage
                g_norm = torch.linalg.vector_norm(g_2d, ord=2, dim=-1, keepdim=True)
                active_tokens = (g_norm > eps).float()

                if active_tokens.sum() == 0:
                    continue

                p_norm_sq = torch.sum(p_2d * p_2d, dim=-1, keepdim=True).clamp_min(eps)

                # Project Gradient onto Tangent Bundle T_e S^{D-1}
                # Radial projection: alpha = <p, g> / ||p||^2
                g_tangent = radial_projection(p_2d, g_2d, p_norm_sq)

                # Multi-Token Polar Decoupling (Instantaneous Gram-Schmidt)
                u_tangent = gram_condition(K, g_tangent, eps)

                # Geodesic Decay (Manifold Anchor Pull)
                if wd > 0.0:
                    u_tangent = geodesic_decay(u_tangent, p_2d, a_2d, p_norm_sq, r_2d, wd, eps)

                # Riemannian Momentum Integration
                if mu > 0.0:
                    if "momentum_buffer" not in state:
                        # First step: initialize buffer in current tangent space
                        buf = u_tangent.clone()
                    else:
                        # Buffer was already parallel-transported to T_p S^{D-1} at the end of last step!
                        buf = state["momentum_buffer"]
                        buf.mul_(mu).add_(u_tangent, alpha=1.0 - mu)

                    if nesterov:
                        nv_coef = mu if nesterov_coef is None else nesterov_coef
                        update_direction = u_tangent.lerp(buf, nv_coef)
                    else:
                        update_direction = buf
                else:
                    update_direction = u_tangent

                dir_norm = torch.linalg.vector_norm(update_direction, ord=2, dim=-1, keepdim=True)
                u_unit = update_direction / dir_norm.clamp_min(eps)

                # Smoothly saturate large steps, decay to 0 near minimum
                theta = lr * torch.atan(dir_norm) * active_tokens

                # Geodesic Exponential Map Flow
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

                # Exponential flow
                p_next = p_2d * cos_theta - r_2d * u_unit * sin_theta

                # Parallel Transport Momentum Buffer to T_{p_next} for the Next Step
                if mu > 0.0:
                    state["momentum_buffer"] = parallel_transport(
                        p=p_2d,
                        p_next=p_next,
                        v=buf,
                        r_norm_sq=p_norm_sq,
                        cos_theta=cos_theta,
                        eps=eps
                    )

                # Write directly in-place
                p.copy_(p_next.view(orig_shape))

        return loss