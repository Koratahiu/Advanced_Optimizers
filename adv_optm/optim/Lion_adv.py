import torch

from typing import Tuple, Optional

from ..util import param_update
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools

class Lion_adv(torch.optim.Optimizer):
    """
    Implements the SMMF technique for Lion algorithm.

    This optimizer combines the Lion update rule with the memory-saving low-rank
    compression (SMMF) technique from https://arxiv.org/abs/2412.08894.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-4).
        betas (Tuple[float, float], optional): coefficients for computing
            running averages of the update (default: (0.9, 0.99)).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0).
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        vector_reshape (bool, optional): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool, optional): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        cautious_mask (bool): whether to use the cautious masking technique. (default: False).
        clip_threshold (float, optional): whether to clip the gradients norm
            per-parameter as proposed in the paper `Lions and Muons: Optimization via
            Stochastic Frank-Wolfe` (https://arxiv.org/abs/2506.04192) to make Lion more stable
            (default: 0.0).
        nnmf_factor (bool): whether to use the factorization or use the
            uncompressed optimizer. (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        orthogonal_gradient: bool = False,
        cautious_mask: bool = False,
        clip_threshold: float = 0.0,
        nnmf_factor: bool = True,
    ):
        if not lr > 0.0:
            raise ValueError(f"Learning rate must be > 0.0, but got {lr}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"Betas should be in [0.0, 1.0], but got {betas}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Weight decay must be >= 0.0, but got {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            vector_reshape=vector_reshape,
            orthogonal_gradient=orthogonal_gradient,
            clip_threshold=clip_threshold,
        )
        self.stochastic_rounding = stochastic_rounding
        self.cautious_mask = cautious_mask
        self.factored = nnmf_factor
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            # For deterministic stochastic rounding, we need to seed the generator
            # for each device used by the parameters.
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

    @property
    def supports_fused_back_pass(self) -> bool:
        return True

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: Optional[int] = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        grad = p.grad
        if grad.dtype != torch.float32 and self.factored:
            grad = grad.float()
        if group["clip_threshold"] > 0.0:
            grad_norm = torch.norm(grad.detach())
            if grad_norm > group["clip_threshold"]:
                clip_coef = group["clip_threshold"] / grad_norm
                grad.mul_(clip_coef)
        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)
        state = self.state[p]

        # State Initialization
        if 'step' not in state:
            state['step'] = 0

            should_factor = (
                self.factored and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            state['factored'] = should_factor

            dtype = torch.float32 if self.factored else p.dtype

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                state['mu_m_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype) 
                state['mv_m_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                packed_d2 = (d2 + 7) // 8
                state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
            else: # Fallback to standard Lion
                state['exp_avg'] = torch.zeros_like(p, device=p.device, dtype=dtype)

        state['step'] += 1
        beta1, beta2 = group["betas"]
        lr = group["lr"]

        if state['factored']:
            # Factored Path
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum m_{t-1}
            exp_avg = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
            unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
            torch.where(unpacked_sign, exp_avg, -exp_avg, out=exp_avg)
            del unpacked_sign
            if exp_avg.dtype != torch.float32:
                exp_avg = exp_avg.float()

            # Compute update term c_t
            update = torch.lerp(grad_reshaped, exp_avg, beta1).sign_()

            if self.cautious_mask:
                mask = (update * grad_reshaped > 0).to(grad_reshaped.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update.mul_(mask)
                del mask

            # Parameter update
            update = update.view(p.shape).mul_(lr)

            # Standard Lion momentum update
            # m_t = beta2 * m_{t-1} + (1-beta2) * g_t
            exp_avg.lerp_(grad_reshaped, 1 - beta2)
            del grad_reshaped

            # Compress new momentum m_t and store factors
            state['sign'] = _pack_bools(exp_avg > 0)
            _nnmf(exp_avg.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
            del exp_avg

        else:
            # Fallback to standard Lion logic
            exp_avg = state["exp_avg"]

            # Compute update term and sign for the update
            if exp_avg.dtype != torch.float32 and self.factored:
                exp_avg = exp_avg.float()

            update = torch.lerp(grad, exp_avg, beta1).sign_()

            if self.cautious_mask:
                mask = (update * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update.mul_(mask)
                del mask

            update.mul_(lr)

            # Standard Lion momentum update
            exp_avg.lerp_(grad, 1 - beta2)

        param_update.apply_parameter_update(self, p, group, update, lr)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    self.step_parameter(p, group, i)

        return loss