import torch

from typing import Tuple, Optional

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.lion_k import _get_lion_k_update


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
        orthogonal_gradient (bool): whether to orthogonalize the gradient (default: False).
        cautious_mask (bool): whether to use the cautious masking technique. (default: False).
        clip_threshold (float, optional): whether to clip the gradients norm
            per-parameter (default: 0.0).
        kappa_p (float, optional): The p-value for the Lp-norm in Lion-K (domain [1.0, 2.0]).
            - 1.0: Standard Lion (sign update).
            - 2.0: Spherical Lion (normalized L2 update).
            - values between 1.0 and 2.0 interpolate behavior.
            (default: 1.0).
        auto_kappa_p (bool, optional): If True, automatically determines kappa_p based on
            parameter dimensionality. Sets p=2.0 for 4D tensors (Conv2D) (Biases/Norms) to
            use Spherical updates, and p=1.0 for others (Linear/Embeddings) to use Sign
            updates. Overrides explicit kappa_p value. (default: False).
        nnmf_factor (bool): whether to use the factorization or use the
            uncompressed optimizer. (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # Cautious variant
        cautious_mask: bool = False,
        clip_threshold: float = 0.0,
        # Lion-k
        kappa_p: float = 1.0,
        auto_kappa_p: bool = False,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
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
            kappa_p=kappa_p,
            auto_kappa_p=auto_kappa_p,
            nnmf_factor=nnmf_factor,
        )
        self.stochastic_rounding = stochastic_rounding
        self.cautious_mask = cautious_mask
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            # For deterministic stochastic rounding, we need to seed the generator
            # for each device used by the parameters.
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        # Initialize compiled function
        self._compiled_step_parameter = None
        if compiled_optimizer:
            self.compile(fullgraph=True)

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
        state = self.state[p]

        # State Initialization
        if 'step' not in state:
            state['step'] = 0

            state['factored'] = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            dtype = torch.float32 if state['factored'] else p.dtype

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
        lr = group["lr"]

        random_int_tensor = None

        if group.get('compiled_optimizer', False):
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, lr, random_int_tensor)

    def _step_parameter(self, p, grad, state, group, lr, random_int_tensor):
        if grad.dtype != torch.float32 and state['factored']:
            grad = grad.float()
        if group["clip_threshold"] > 0.0:
            grad_norm = torch.norm(grad.detach())
            if grad_norm > group["clip_threshold"]:
                clip_coef = group["clip_threshold"] / grad_norm
                grad.mul_(clip_coef)
        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        # Lion-K Logic
        kappa_p = group.get("kappa_p", 1.0)
        if group.get("auto_kappa_p", False):
            # Apply p=2.0 (Spherical) for 4D (Conv2D)
            # Apply p=1.0 (Sign) for everything else (Linear/Embeddings)
            if p.ndim >= 4:
                kappa_p = 2.0
            else:
                kappa_p = 1.0

        beta1, beta2 = group["betas"]

        if state['factored']:
            # Factored Path
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum m_{t-1}
            exp_avg = _reconstruct_state(state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2)

            # Compute update term c_t
            update = torch.lerp(grad_reshaped, exp_avg, beta1)

            # Standard Lion momentum update
            # m_t = beta2 * m_{t-1} + (1-beta2) * g_t
            exp_avg.lerp_(grad_reshaped, 1 - beta2)
            del grad_reshaped

            # Compress new momentum m_t and store factors
            state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(exp_avg, signed=True)
            del exp_avg

            update = _get_lion_k_update(update, kappa_p)

            if self.cautious_mask:
                mask = (update * grad_reshaped > 0).to(grad_reshaped.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update.mul_(mask)
                del mask

            update = update.view(p.shape).mul_(lr)

        else:
            # Fallback to standard Lion logic
            exp_avg = state["exp_avg"]

            # Compute update term
            update = torch.lerp(grad, exp_avg, beta1)

            update = _get_lion_k_update(update, kappa_p)

            if self.cautious_mask:
                mask = (update * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update.mul_(mask)
                del mask

            update.mul_(lr)

            # Standard Lion momentum update
            exp_avg.lerp_(grad, 1 - beta2)

        param_update.apply_parameter_update(self, p, group, update, lr, random_int_tensor=random_int_tensor)

    def compile(self, *args, **kwargs):
        self._compiled_step_parameter = torch.compile(self._step_parameter, *args, **kwargs)

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
