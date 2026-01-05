import torch
import torch.distributed as dist

import math

from typing import Optional, Callable

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _grams_update, _cautious_update

A = 4 / math.pi

class Prodigy_adv(torch.optim.Optimizer):
    """
    Implements an advanced Prodigy algorithm.
    This is an advanced version of Prodigy with optional features like
    low-rank factorization of optimizer states (SMMF), OrthoGrad, etc.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1)
        betas (tuple[float, float]): coefficients used for computing running
            averages of gradient and its square (default: (0.9, 0.999))
        eps (float): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
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
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule.
            This changes the EMA to accumulator and the update numerator to `alpha_grad * grad + mt`, which can be
            more responsive, especially for small batch sizes. Enabling this will
            automatically disable `use_AdEMAMix`, `cautious_mask`, `grams_moment`,
            and `use_atan2`. (default: False)
        alpha_grad (float): Mixing coefficient for the Simplified AdEMAMix update rule
            (only used when `Simplified_AdEMAMix` is `True`). Controls the weight of the
            current gradient. For small batch sizes, use high values (e.g., 10-100) to be
            more responsive. For large batch sizes, use low values (e.g., 0-1) for
            stability. (default: 100.0)
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well.
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
        slice_p (int): Reduce memory usage by calculating LR adaptation statistics on only every
            pth entry of each tensor. For values greater than 1 this an an approximation to standard
            Prodigy. Values ~11 are reasonable (default 11).
        prodigy_steps (int): If greater than zero, disable Prodigy's stepsize adjustments
            after the specified optimiser step and release all state memory required by Prodigy
            (default: 0).
        d_limiter (bool): whether to clamp the new step size estimate (`d_hat`)
            to prevent sudden, volatile increases in the adaptive step size (`d`).
            (default: False)
        kourkoutas_beta (bool): whether to enable the layer-wise dynamic β₂ logic.
            If `False`, the optimizer behaves as standard AdamW/Prodigy. (default: False)
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
    """

    def __init__(
        self,
        params,
        lr: float = 1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
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
        # One-EMA AdEMAMix
        Simplified_AdEMAMix: bool = False,
        alpha_grad: float = 100.0,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
        # prodigy parameters
        beta3: float = None,
        d0: float = 1e-6,
        d_coef: float = 1,
        growth_rate: float = float('inf'),
        safeguard_warmup: bool = False,
        fsdp_in_use: bool = False,
        slice_p: int = 11,
        prodigy_steps: int = 0,
        d_limiter: bool = False,
        # K-b (adaptive beta2)
        kourkoutas_beta: bool = False,
        beta2_min: float = 0.9,
        ema_alpha: float = 0.95,
        tiny_spike: float = 1e-9,
        k_warmup_steps: int = 0,
        k_logging: int = 0,
        layer_key_fn: Optional[Callable] = None,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Betas should be in [0.0, 1.0). Got {betas}")
        if not (eps >= 0.0):
            raise ValueError(f"Epsilon should be >= 0.0. Got {eps}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (prodigy_steps >= 0):
            raise ValueError(f"prodigy_steps should be >= 0. Got {prodigy_steps}")
        if cautious_mask and grams_moment:
            print("Warning: cautious is incompatible with grams, Disabling cautious.")
            cautious_mask = False
        if betas[0] == 0.0 and Simplified_AdEMAMix:
            raise ValueError(f"Beta1 cannot be 0.0 when using Simplified_AdEMAMix. Got {betas[0]}")
        if use_AdEMAMix and Simplified_AdEMAMix:
            print("Warning: use_AdEMAMix is incompatible with Simplified_AdEMAMix, Disabling use_AdEMAMix.")
        if grams_moment and Simplified_AdEMAMix:
            print("Warning: grams is incompatible with Simplified_AdEMAMix, Disabling grams.")
        if cautious_mask and Simplified_AdEMAMix:
            print("Warning: cautious is incompatible with Simplified_AdEMAMix, Disabling cautious.")
        if use_atan2 and Simplified_AdEMAMix:
            print("Warning: use_atan2 is incompatible with Simplified_AdEMAMix. Disabling use_atan2.")
            use_atan2 = False
        if kourkoutas_beta and not (betas[1] > beta2_min):
            raise ValueError(f"For Kourkoutas-β, betas[1] (as beta2_max) must be > beta2_min. Got {betas[1]} and {beta2_min}")
        if Simplified_AdEMAMix and alpha_grad > 0 and not d_limiter:
            # scales d_coef by alpha_grad, this force prodigy to behave well with Simplified_AdEMAMix.
            d_coef = d_coef/alpha_grad

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "cautious_wd": cautious_wd,
            "vector_reshape": vector_reshape, "use_atan2": use_atan2,
            "orthogonal_gradient": orthogonal_gradient,
            "beta3_ema": beta3_ema, "alpha": alpha, "compiled_optimizer": compiled_optimizer,
            "beta3": beta3, "d": d0, "d0": d0, "d_max": d0, "d_numerator": 0.0, "d_coef": d_coef,
            "growth_rate": growth_rate, "safeguard_warmup": safeguard_warmup, "k": 0, "slice_p": slice_p,
            "fsdp_in_use": fsdp_in_use, "prodigy_steps": prodigy_steps, "d_limiter": d_limiter,
            "alpha_grad": alpha_grad,
            "kourkoutas_beta": kourkoutas_beta, "beta2_min": beta2_min, "ema_alpha": ema_alpha,
            "tiny_spike": tiny_spike, "k_warmup_steps": k_warmup_steps, "k_logging": k_logging,
            "nnmf_factor": nnmf_factor,
        }
        self.stochastic_rounding = stochastic_rounding
        self.cautious_mask = cautious_mask and not Simplified_AdEMAMix
        self.grams_moment = grams_moment and not Simplified_AdEMAMix
        self.use_AdEMAMix = use_AdEMAMix and not Simplified_AdEMAMix
        self.Simplified_AdEMAMix = Simplified_AdEMAMix
        self.fsdp_in_use = fsdp_in_use

        self.kourkoutas_beta = kourkoutas_beta
        self.layer_key_fn = layer_key_fn

        super().__init__(params, defaults)

        # Use the device of the first parameter to avoid hardcoding '.cuda()'
        self.device = self.param_groups[0]['params'][0].device

        if self.kourkoutas_beta:
            self.kourkoutas_helper = KourkoutasHelper(self)

        self.init_step()

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

    def init_step(self):
        """Resets accumulators and calculates dlr for the upcoming step."""
        g_group = self.param_groups[0]
        self.beta1, self.beta2_default = g_group['betas']
        self.beta3 = g_group['beta3']
        if self.beta3 is None:
            self.beta3 = math.sqrt(self.beta2_default)

        if hasattr(self, 'd_denom'):
            device = self.d_denom.device
            self.d_denom = torch.tensor(0.0, device=device)
            self.d_numerator = torch.tensor(g_group.get('d_numerator', 0.0) * self.beta3, device=device)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        if hasattr(p, "_fsdp_flattened"):
            self.fsdp_in_use = True

        grad = p.grad
        state = self.state[p]

        # State Initialization
        if 'step' not in state:
            state['step'] = 0

            state['factored'] = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            slice_p = group['slice_p']

            dtype = torch.float32 if state['factored'] else p.dtype
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
                if self.beta1 > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                if self.use_AdEMAMix:
                    state['mu_m_slow_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype)
                    state['mv_m_slow_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign_slow'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                # Second moment (v)
                state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:  # Fallback to standard AdamW for non-factored tensors
                if self.beta1 > 0:
                    state['exp_avg'] = torch.zeros_like(p, device=device, dtype=dtype)
                if self.use_AdEMAMix:
                    state['exp_avg_slow'] = torch.zeros_like(p, dtype=dtype)
                state['exp_avg_sq'] = torch.zeros_like(p, device=device, dtype=dtype)

            # Prodigy states
            state['s'] = torch.zeros_like(p.flatten()[::slice_p]).detach()
            if p.any():
                state['p0'] = p.flatten()[::slice_p].detach().clone()
            else:
                state['p0'] = torch.tensor(0, device=device, dtype=p.dtype)

        if not hasattr(self, 'd_denom'):
            self.d_denom = torch.tensor(0.0, device=p.device)
            self.d_numerator = torch.tensor(group.get('d_numerator', 0.0), device=p.device)

        current_step = state['step']
        if group.get('kourkoutas_beta', False):
            # Call prepare_step() once at the beginning of the step for all params
            self.kourkoutas_helper.maybe_prepare_step(current_step, p.device)
            # Get the dynamic beta2 calculated in prepare_step()
            beta2 = self.kourkoutas_helper.get_beta2(p, group)
        else:
            beta2 = self.beta2_default

        dlr = group['d'] * group['lr']

        random_int_tensor = None

        if group.get('compiled_optimizer', False):
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
            # TODO, workaround until pytorch#169634 is fixed
            d = torch.as_tensor(group['d'], dtype=torch.float64)
            dlr = torch.as_tensor(dlr, dtype=torch.float64)
            step_param_fn = self._compiled_step_parameter
        else:
            d = group['d']
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, beta2, d, dlr, random_int_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, beta2, d, dlr, random_int_tensor):
        if grad.dtype != torch.float32 and state['factored']:
            grad = grad.float()
        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        if self.use_AdEMAMix:
            beta3_ema = group['beta3_ema']
            alpha = group['alpha']
        if self.Simplified_AdEMAMix:
            alpha_grad = group["alpha_grad"]

        if group.get('kourkoutas_beta', False):
            # Accumulate current grad's norm for the *next* step
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors
            if self.beta1 > 0:
                mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True)

                # Update momentum in full-size
                if self.Simplified_AdEMAMix:
                    alpha_mt = d
                else:
                    alpha_mt = d * (1.0 - self.beta1)

                mt.mul_(self.beta1).add_(grad_reshaped, alpha=alpha_mt)

                # Factorize
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(mt.clone(), signed=True)

                if self.grams_moment:
                    update_mt = _grams_update(mt, grad_reshaped, inplace=True)
                elif self.cautious_mask:
                    update_mt = _cautious_update(mt, grad_reshaped, inplace=True)
                else:
                    update_mt = mt

            vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False)
            vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=d * d * (1.0 - beta2))

            if self.use_AdEMAMix:
                mt_slow = _reconstruct_state((state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'], d2), signed=True)

                mt_slow.mul_(beta3_ema).add_(grad_reshaped, alpha=d * (1.0 - beta3_ema))
                if self.beta1 > 0:
                    update = update_mt.add_(mt_slow, alpha=alpha)
                else:
                    update = grad_reshaped.mul(d).add_(mt_slow, alpha=alpha)
                # Factorize
                state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'] = _factorize_state(mt_slow, signed=True)
                del mt_slow
            elif self.Simplified_AdEMAMix:
                update = update_mt.add_(grad_reshaped, alpha=alpha_grad * d)
            else:
                if self.beta1 > 0:
                    update = update_mt
                else:
                    update = grad_reshaped.mul(d)

            # Factorize
            state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt, signed=False)

            if group['use_atan2']:
                denom = vt.sqrt_()
                update.atan2_(denom)
            else:
                denom = vt.sqrt_()
                update.div_(denom.add_(d * group['eps']))
            del vt

            update_scaling = dlr * A if group['use_atan2'] else dlr
            update = update.view(p.shape).mul_(update_scaling)

        else:  # Standard AdamW logic for non-factored tensors
            if self.beta1 > 0:
                exp_avg = state['exp_avg']

                if self.Simplified_AdEMAMix:
                    alpha_mt = d
                else:
                    alpha_mt = d * (1.0 - self.beta1)

                exp_avg.mul_(self.beta1).add_(grad, alpha=alpha_mt)

                if self.grams_moment:
                    update_mt = _grams_update(exp_avg, grad)
                elif self.cautious_mask:
                    update_mt = _cautious_update(exp_avg, grad)
                else:
                    update_mt = exp_avg.clone()

            if self.use_AdEMAMix:
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_slow.mul_(beta3_ema).add_(grad, alpha=d * (1.0 - beta3_ema))
                if self.beta1 > 0:
                    update = update_mt.add_(exp_avg_slow, alpha=alpha)
                else:
                    update = grad.mul(d).add_(exp_avg_slow, alpha=alpha)
            elif self.Simplified_AdEMAMix:
                update = update_mt.add_(grad, alpha=alpha_grad * d)
            else:
                if self.beta1 > 0:
                    update = update_mt
                else:
                    update = grad.mul(d)

            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1.0 - beta2))

            if group['use_atan2']:
                denom = exp_avg_sq.sqrt()
                update.atan2_(denom)
            else:
                denom = exp_avg_sq.sqrt()
                update.div_(denom.add_(d * group['eps']))
            del denom

            update_scaling = dlr * A if group['use_atan2'] else dlr
            update.mul_(update_scaling)

        # --- Accumulate Prodigy stats ---
        prodigy_steps = group['prodigy_steps']
        if prodigy_steps <= 0 or group['k'] < prodigy_steps:
            d0, safeguard_warmup, slice_p = group['d0'], group['safeguard_warmup'], group['slice_p']
            s, p0 = state['s'], state['p0']

            grad_slice = grad.flatten()[::slice_p].float()
            p_slice = p.flatten()[::slice_p].float()
            p0 = p0.float()

            self.d_numerator.add_((d / d0) * dlr * torch.dot(grad_slice, p0 - p_slice))

            alpha = ((d / d0) * d) if safeguard_warmup else ((d / d0) * dlr)
            s.mul_(self.beta3).add_(grad_slice, alpha=alpha)
            self.d_denom.add_(s.abs().sum())

            del s, p0, grad_slice, p_slice, alpha
        else:
            # Free memory if prodigy_steps is reached
            if 's' in state:
                del state['s']
            if 'p0' in state:
                del state['p0']

        param_update.apply_parameter_update(self, p, group, update, dlr, random_int_tensor=random_int_tensor)

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

        self.calculate_d()
        self.init_step()
        return loss

    def calculate_d(self):
        """Calculates the new `d` based on the accumulated stats."""
        g_group = self.param_groups[0]

        # Only perform d-adaptation if prodigy_steps has not been reached
        prodigy_active = not (g_group.get('prodigy_steps', 0) > 0 and g_group['k'] >= g_group['prodigy_steps'])

        if prodigy_active:
            d_max, d_coef, growth_rate = g_group['d_max'], g_group['d_coef'], g_group['growth_rate']

            if self.fsdp_in_use and dist.is_available() and dist.is_initialized():
                dist_tensor = torch.stack([self.d_numerator, self.d_denom])
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0].item()
                global_d_denom = dist_tensor[1].item()
            else:
                global_d_numerator = self.d_numerator.item()
                global_d_denom = self.d_denom.item()

            d_hat = g_group['d']
            if global_d_denom > 0:
                d_hat = d_coef * global_d_numerator / global_d_denom
                if g_group.get('d_limiter', False):
                    d_hat = min(g_group['d'] * (2 ** 0.25), d_hat)
                if g_group['d'] == g_group['d0']:
                    g_group['d'] = max(g_group['d'], d_hat)
                d_max = max(d_max, d_hat)
                g_group['d'] = min(d_max, g_group['d'] * growth_rate)

            for group in self.param_groups:
                group['d_numerator'] = global_d_numerator
                group['d'] = g_group['d']
                group['d_max'] = d_max

        # Increment step counter for all groups, regardless of whether d was updated
        for group in self.param_groups:
            group['k'] += 1
