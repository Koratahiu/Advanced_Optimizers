import torch

import math

from typing import Optional, Callable

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _grams_update, _cautious_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper

A = 4 / math.pi

class AdamW_adv(torch.optim.Optimizer):
    """
    Implements an advanced AdamW algorithm.
    This is an advanced version of AdamW with optional features like
    low-rank factorization of optimizer states (SMMF), OrthoGrad, etc.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        betas (tuple[float, float]): coefficients used for computing running
            averages of gradient and its square (default: (0.9, 0.999))
        eps (float): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0).
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        use_bias_correction (bool): whether to use bias correction for the first
            and second moment estimates, as in the original Adam paper.
            (default: True)
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        use_atan2 (bool): whether to use the atan2 update rule. (default: False)
        grams_moment (bool): whether to use Grams-style updates. (default: False)
        cautious_mask (bool):  whether to use cautious masking to align the gradient's
            direction with the first moment's.  (default: False)
        orthogonal_gradient (bool): whether to use OrthoGrad.  (default: False)
        use_AdEMAMix (bool): whether to enable the AdEMAMix feature. This adds
            a second, slow-moving average of the momentum (`mt_slow`) which is
            combined with the primary momentum (`mt`) to stabilize updates,
            especially in noisy, small-batch settings. If `False`, the
            optimizer behaves as standard AdamW. (default: False)
        beta3_ema (float): The decay rate for the slow exponential moving average of
            the momentum (only used when `use_AdEMAMix` is `True`). A higher
            value (e.g., 0.9999) gives the EMA a longer memory, making it more
            stable but slower to adapt. A lower value (e.g., 0.999) is often
            better for shorter training runs. (default: 0.9999)
        alpha (float): The mixing coefficient that scales the slow momentum term
            before it is added to the fast momentum term (`update = mt + alpha * mt_slow`).
            A higher value increases the stabilizing influence of the slow
            momentum. (default: 5.0)
        kourkoutas_beta (bool): whether to enable the layer-wise dynamic β₂ logic.
            If `False`, the optimizer behaves as standard AdamW. (default: False)
        beta2_min (float): The minimum value for dynamic β₂, used during periods of
            high gradient variance ("sunspikes"). Must be less than `betas[1]`.
            (default: 0.88)
        ema_alpha (float): The decay rate for the Exponential Moving Average (EMA) of
            the pooled gradient norms. Corresponds to `α` in the paper.
            (default: 0.93)
        tiny_spike (float): A small constant added to the denominator of the
            "sunspike" ratio calculation to prevent division by zero. Corresponds
            to `ε_spike` in the paper. (default: 1e-9)
        k_warmup_steps (int): The number of initial steps during which β₂ is held
            at a fixed beta2 value before the
            dynamic logic activates. (default: 0)
        k_logging (int): if > 0 and kourkoutas_beta=True, enables periodic console
            logging of Kourkoutas-β statistics (min, max, mean of `β₂` across layers)
            every logging steps. Useful for debugging and tuning. Set to 0 to disable
            logging (default: 0).
        layer_key_fn (Optional[Callable]): A function that takes a parameter `p`
            and returns a unique, hashable key representing its "layer" or "bucket".
            If `None`, parameters are bucketed by their memory ID (tensor-wise).
            (default: None)
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        # Adam's Bias Correction
        use_bias_correction: bool = True,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # Adam_atan2 (scale invariant)
        use_atan2: bool = False,
        # Cautious and GRAMS
        cautious_mask: bool = False,
        grams_moment: bool = False,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # AdEMAMix (long-term momentum)
        use_AdEMAMix: bool = False,
        beta3_ema: float = 0.9999,
        alpha: float = 5.0,
        # K-b (adaptive beta2)
        kourkoutas_beta: bool = False,
        beta2_min: float = 0.9,
        ema_alpha: float = 0.95,
        tiny_spike: float = 1e-9,
        k_warmup_steps: int = 0,
        k_logging: int = 0,
        layer_key_fn: Optional[Callable] = None,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Betas should be in [0.0, 1.0). Got {betas}")
        if not (eps >= 0.0):
            raise ValueError(f"Epsilon should be >= 0.0. Got {eps}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if kourkoutas_beta and not (betas[1] > beta2_min):
            raise ValueError(f"For Kourkoutas-β, betas[1] (as beta2_max) must be > beta2_min. Got {betas[1]} and {beta2_min}")

        if cautious_mask and grams_moment:
            print("Warning: cautious is incompatible with grams, Disabling cautious.")
            cautious_mask = False

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "cautious_wd": cautious_wd,
            "vector_reshape": vector_reshape, "use_atan2": use_atan2,
            "orthogonal_gradient": orthogonal_gradient, "use_bias_correction": use_bias_correction,
            "beta3_ema": beta3_ema, "alpha": alpha, "compiled_optimizer": compiled_optimizer,
            "kourkoutas_beta": kourkoutas_beta, "beta2_min": beta2_min, "ema_alpha": ema_alpha,
            "tiny_spike": tiny_spike, "k_warmup_steps": k_warmup_steps, "k_logging": k_logging,
            "nnmf_factor": nnmf_factor
        }
        self.stochastic_rounding = stochastic_rounding
        self.cautious_mask = cautious_mask
        self.grams_moment = grams_moment
        self.use_AdEMAMix = use_AdEMAMix
        self.kourkoutas_beta = kourkoutas_beta
        self.layer_key_fn = layer_key_fn
        super().__init__(params, defaults)

        if self.kourkoutas_beta:
            self.kourkoutas_helper = KourkoutasHelper(self)

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
    def supports_fused_back_pass(self):
        return True

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
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
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
                if group['betas'][0] > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                # AdEMAMix slow moment (m_slow)
                if self.use_AdEMAMix:
                    state['mu_m_slow_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype)
                    state['mv_m_slow_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign_slow'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                # Second moment (v)
                state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:  # Fallback to standard AdamW for non-factored tensors
                # First moment
                if group['betas'][0] > 0:
                    state['exp_avg'] = torch.zeros_like(p, device=device, dtype=dtype)
                # AdEMAMix slow moment
                if self.use_AdEMAMix:
                    state['exp_avg_slow'] = torch.zeros_like(p, device=device, dtype=dtype)
                # Second moment (v)
                state['exp_avg_sq'] = torch.zeros_like(p, device=device, dtype=dtype)

        beta1, beta2 = group['betas']

        current_step = state['step']
        if group.get('kourkoutas_beta', False):
            # Call prepare_step() once at the beginning of the step for all params
            self.kourkoutas_helper.maybe_prepare_step(current_step, p.device)
            # Get the dynamic beta2 calculated in prepare_step()
            beta2 = self.kourkoutas_helper.get_beta2(p, group)

        if group['use_bias_correction']:
            step = current_step + 1
            bias_correction1 = 1.0 - beta1 ** step
            sqrt_bias_correction2 = (1.0 - group['betas'][1] ** step) ** 0.5
        else:
            bias_correction1 = 1
            sqrt_bias_correction2 = 1
        step_size = group['lr'] / bias_correction1

        random_int_tensor = None

        if group.get('compiled_optimizer', False):
            step_size = torch.as_tensor(step_size, dtype=torch.float64)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, step_size, beta1, beta2, sqrt_bias_correction2, random_int_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, step_size, beta1, beta2, sqrt_bias_correction2, random_int_tensor):
        if grad.dtype != torch.float32 and state['factored']:
            grad = grad.float()
        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        if self.use_AdEMAMix:
            beta3_ema = group['beta3_ema']
            alpha = group['alpha']

        if group.get('kourkoutas_beta', False):
            # Accumulate current grad's norm for the *next* step
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors
            if beta1 > 0:
                mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True)

                # Update momentum in full-size
                mt.lerp_(grad_reshaped, 1.0 - beta1)

                # Factorize
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(mt.clone(), signed=True)

                if self.grams_moment:
                    update_mt = _grams_update(mt, grad_reshaped, inplace=True)
                elif self.cautious_mask:
                    update_mt = _cautious_update(mt, grad_reshaped, inplace=True)
                else:
                    update_mt = mt

            vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False)
            vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2)

            if self.use_AdEMAMix:
                mt_slow = _reconstruct_state((state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'], d2), signed=True)

                mt_slow.lerp_(grad_reshaped, 1.0 - beta3_ema)

                if beta1 > 0:
                    update = update_mt.add_(mt_slow, alpha=alpha)
                else:
                    update = grad_reshaped.add(mt_slow, alpha=alpha)
                # Factorize
                state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'] = _factorize_state(mt_slow, signed=True)
                del mt_slow
            else:
                if beta1 > 0:
                    update = update_mt
                else:
                    update = grad_reshaped.clone()

            if group['use_atan2']:
                denom = vt.sqrt()
                denom.div_(sqrt_bias_correction2)
                update.atan2_(denom)
            else:
                denom = vt.sqrt()
                denom.div_(sqrt_bias_correction2).add_(group['eps'])
                update.div_(denom)
            del denom

            # Factorize
            state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt, signed=False)
            del vt

            update_scaling = step_size * A if group['use_atan2'] else step_size
            update = update.view(p.shape).mul_(update_scaling)

        else:  # Standard AdamW logic for non-factored tensors
            if beta1 > 0:
                exp_avg = state['exp_avg']
                exp_avg.lerp_(grad, 1.0 - beta1)

                if self.grams_moment:
                    update_mt = _grams_update(exp_avg, grad)
                elif self.cautious_mask:
                    update_mt = _cautious_update(exp_avg, grad)
                else:
                    update_mt = exp_avg.clone()

            if self.use_AdEMAMix:
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_slow.lerp_(grad, 1.0 - beta3_ema)

                if beta1 > 0:
                    update = update_mt.add_(exp_avg_slow, alpha=alpha)
                else:
                    update = torch.add(grad, exp_avg_slow, alpha=alpha)
            else:
                update = update_mt if beta1 > 0 else grad.clone()

            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if group['use_atan2']:
                denom = exp_avg_sq.sqrt()
                denom.div_(sqrt_bias_correction2)
                update.atan2_(denom)
            else:
                denom = exp_avg_sq.sqrt()
                denom.div_(sqrt_bias_correction2).add_(group['eps'])
                update.div_(denom)
            del denom

            update_scaling = step_size * A if group['use_atan2'] else step_size
            update.mul_(update_scaling)

        param_update.apply_parameter_update(self, p, group, update, step_size, random_int_tensor=random_int_tensor)

    def compile(self, *args, **kwargs):
        self._compiled_step_parameter = torch.compile(self._step_parameter, *args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.step_parameter(p, group, i)

        return loss
