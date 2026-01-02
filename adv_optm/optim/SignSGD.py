import torch

from typing import Optional

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.lion_k import _get_lion_k_update


class SignSGD(torch.optim.Optimizer):
    """
    Implements an advanced SignSGD algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-4).
        momentum (float, optional): coefficients for computing
            running average of the gradients (default: 0.9).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0).
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        vector_reshape (bool, optional): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool, optional): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        orthogonal_gradient (bool): whether to orthogonalize the gradient (default: False).
        kappa_p (float, optional): The p-value for the Lp-norm in projection-K (domain [1.0, 2.0]).
            - 1.0: Standard (sign update).
            - 2.0: Spherical (normalized L2 update).
            - values between 1.0 and 2.0 interpolate behavior.
            (default: 1.0).
        auto_kappa_p (bool, optional): If True, automatically determines kappa_p based on
            parameter dimensionality. Sets p=2.0 for 4D tensors (Conv2D) (Biases/Norms) to
            use Spherical updates, and p=1.0 for others (Linear/Embeddings) to use Sign
            updates. Overrides explicit kappa_p value. (default: False).
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule.
            This changes the EMA to accumulator and the update numerator to `alpha_grad * grad + mt`, which can be
            more responsive, especially for small batch sizes. (default: False)
        alpha_grad (float): Mixing coefficient for the Simplified AdEMAMix update rule
            (only used when `Simplified_AdEMAMix` is `True`). Controls the weight of the
            current gradient. For small batch sizes, use high values (e.g., 10-100) to be
            more responsive. For large batch sizes, use low values (e.g., 0-1) for
            stability. (default: 100.0)
        nnmf_factor (bool): whether to use the factorization or use the
            uncompressed optimizer. (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.9,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # Projection-k
        kappa_p: float = 1.0,
        auto_kappa_p: bool = True,
        # Simplified_AdEMAMix
        alpha_grad: float = 1.0,
        Simplified_AdEMAMix: bool = False,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
    ):
        if not lr > 0.0:
            raise ValueError(f"Learning rate must be > 0.0, but got {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum should be in [0.0, 1.0], but got {momentum}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Weight decay must be >= 0.0, but got {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            vector_reshape=vector_reshape,
            orthogonal_gradient=orthogonal_gradient,
            kappa_p=kappa_p,
            auto_kappa_p=auto_kappa_p,
            alpha_grad=alpha_grad,
            Simplified_AdEMAMix=Simplified_AdEMAMix,
            nnmf_factor=nnmf_factor,
        )
        self.stochastic_rounding = stochastic_rounding
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
        if group["momentum"] > 0 and len(state) == 0:
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
            else:
                state['exp_avg'] = torch.zeros_like(p, device=p.device, dtype=dtype)

        lr = group["lr"]

        random_int_tensor = None

        if group.get('compiled_optimizer', False):
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
            lr = torch.as_tensor(lr, dtype=torch.float64)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, lr, random_int_tensor)

    def _step_parameter(self, p, grad, state, group, lr, random_int_tensor):
        if grad.dtype != torch.float32 and state['factored']:
            grad = grad.float()

        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        # Projection logic (inspired from Lion-K)
        kappa_p = group.get("kappa_p", 1.0)
        if group.get("auto_kappa_p", False):
            # Apply p=2.0 (Spherical) for >=4D (Conv2D)
            # Apply p=1.0 (Sign) for everything else (Linear/Embeddings)
            if p.ndim >= 4:
                kappa_p = 2.0
            else:
                kappa_p = 1.0

        momentum = group["momentum"]
        Simplified_AdEMAMix = group["Simplified_AdEMAMix"]
        alpha_grad = group["alpha_grad"]

        if state['factored']:
            # Factored Path
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            if momentum > 0:
                # Reconstruct momentum m_{t-1}
                exp_avg = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True)
                exp_avg.mul_(momentum).add_(grad_reshaped)

                if Simplified_AdEMAMix:
                    raw_update = exp_avg + (grad_reshaped * alpha_grad)
                else:
                    raw_update = exp_avg.clone()

                # Compress new momentum m_t and store factors
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(exp_avg, signed=True)
            else:
                raw_update = grad_reshaped.clone()

            update = _get_lion_k_update(raw_update, kappa_p)

            update = update.view(p.shape).mul_(lr)

        else:
            # Fallback to standard SignSGD logic
            if momentum > 0:
                exp_avg = state["exp_avg"]
                exp_avg.mul_(momentum).add_(grad)

                if Simplified_AdEMAMix:
                    raw_update = exp_avg + (grad * alpha_grad)
                else:
                    raw_update = exp_avg.clone()
            else:
                raw_update = grad.clone()

            update = _get_lion_k_update(raw_update, kappa_p)

            update = update.mul_(lr)

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
