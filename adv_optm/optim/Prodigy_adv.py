import torch
import torch.distributed as dist

import math

from typing import Optional, Callable

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _init_fisher_wd_scaler, _get_fisher_wd_scaler
from ..util.centered_decay import _init_anchor
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm, scale_eps
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

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
            numerical stability. Set to None for scale invariant eps (vector
            lower bound) (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        fisher_wd (bool): whether to use Fisher Adam (FAdam) weight decay, mapping
            the decay direction through the empirical Fisher information matrix and
            clipping its RMS. (default: False)
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        use_atan2 (bool): whether to use the atan2 update rule. (default: False)
        orthogonal_gradient (str): whether to use OrthoGrad variants. 'disabled': off.
        'flattened': Standard vectorized OrthoGrad. 'iterative': Matrix-wise rank-2 OrthoGrad. (default: disabled)
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
        factored_2nd (bool): whether to keep the first moment uncompressed (dense)
            while only factorizing the second moment. (default: True)
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
        centered_wd (float): Centered Weight Decay coefficient. Instead of decaying weights
            toward zero, they are decayed toward their initial values (anchors). This
            can be used together with standard weight decay. (default: 0.0)
        centered_wd_mode (str): The quantization format used to store the anchor
            weights to save VRAM. Options include:
            'full': Stores anchors in the original parameter's precision.
            'float8': Uses torch.float8_e4m3fn for a balance of precision and memory.
            'int8': Uses 8-bit block-wise quantization (block size 128).
            'int4': Uses 4-bit block-wise quantization (block size 32).
    """

    def __init__(
        self,
        params,
        lr: float = 1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float | None = 1e-8,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        fisher_wd: bool = False,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # Adam_atan2 (scale invariant)
        use_atan2: bool = False,
        # OrthoGrad
        orthogonal_gradient: str = 'disabled', # 'flattened', 'iterative'
        # Nesterov momentum
        nesterov: bool = False,
        nesterov_coef: float | None = None,
        # States precision
        state_precision: str = "auto", # 'fp32', 'factored', 'bf16_sr', 'int8_sr'.
        # Factorized second moment only
        factored_2nd: bool = False,
        # SMMF factorization (legacy)
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
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
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
        if kourkoutas_beta and not (betas[1] > beta2_min):
            raise ValueError(f"For Kourkoutas-β, betas[1] (as beta2_max) must be > beta2_min. Got {betas[1]} and {beta2_min}")

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp16", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        # Legacy backwards compatibility support for `nnmf_factor=True`
        if nnmf_factor:
            state_precision = "factored"

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "fisher_wd": fisher_wd, "cautious_wd": cautious_wd,
            "use_atan2": use_atan2,
            "orthogonal_gradient": orthogonal_gradient,
            "compiled_optimizer": compiled_optimizer,
            "beta3": beta3, "d": d0, "d0": d0, "d_max": d0, "d_numerator": 0.0, "d_coef": d_coef,
            "growth_rate": growth_rate, "safeguard_warmup": safeguard_warmup, "k": 0, "slice_p": slice_p,
            "fsdp_in_use": fsdp_in_use, "prodigy_steps": prodigy_steps, "d_limiter": d_limiter,
            "nesterov": nesterov, "nesterov_coef": nesterov_coef, "state_precision": state_precision,
            "kourkoutas_beta": kourkoutas_beta, "beta2_min": beta2_min, "ema_alpha": ema_alpha,
            "tiny_spike": tiny_spike, "k_warmup_steps": k_warmup_steps, "k_logging": k_logging,
            "centered_wd": centered_wd, "centered_wd_mode": centered_wd_mode,
            "nnmf_factor": nnmf_factor, "vector_reshape": vector_reshape, "factored_2nd": factored_2nd
        }
        self.stochastic_rounding = stochastic_rounding
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

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Overrides default load_state_dict to implement a workaround for PyTorch's
        automatic dtype casting. It ensures factorized states remain float32 for
        stability, preserves integer/float8 quantized anchor states, and forces
        standard states onto the parameter's current dtype/device.
        """
        super().load_state_dict(state_dict)
        param_update.post_process_loaded_state(self)

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

            slice_p = group['slice_p']

            req_precision = group['state_precision']
            is_vector = len(p.shape) == 1 and not group['vector_reshape']

            state['factored'] = req_precision == 'factored' and not is_vector

            state['factored_2nd'] = group.get('factored_2nd', False) and not is_vector

            actual_precision = 'auto' if req_precision == 'factored' else req_precision
            group['actual_state_precision'] = actual_precision

            dtype = torch.float32 if (state['factored'] or req_precision == 'factored') else p.dtype
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
                if group['betas'][0] > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                    state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)

                # Second moment (v)
                state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
                state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
            else:  # Fallback to standard AdamW for non-factored tensors
                # First moment
                if group['betas'][0] > 0:
                    init_state_tensor(state, 'exp_avg', p.shape, actual_precision, p.device, dtype)

                # Second moment (v)
                if state['factored_2nd']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
                    state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
                    state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
                    state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=p.device, dtype=torch.uint8)
                else:
                    init_state_tensor(state, 'exp_avg_sq', p.shape, actual_precision, p.device, dtype, non_neg=True)

            if group.get('spectral_normalization', False) and is_spectral(p):
                init_spectral_norm(state, p)

            _init_anchor(p, state, group)

            _init_fisher_wd_scaler(group, state, p)

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
        random_int_state_tensor = None

        if group.get('compiled_optimizer', False):
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
                random_int_state_tensor = random_int_tensor
            # TODO, workaround until pytorch#169634 is fixed
            d = torch.as_tensor(group['d'])
            dlr = torch.as_tensor(dlr)
            if group['actual_state_precision'] == 'bf16_sr' and random_int_state_tensor is None:
                random_int_state_tensor = param_update._get_random_int_for_sr(p)
            elif group['actual_state_precision'] == 'int8_sr':
                random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
            step_param_fn = self._compiled_step_parameter
        else:
            d = group['d']
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, beta2, d, dlr, random_int_tensor, random_int_state_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, beta2, d, dlr, random_int_tensor, random_int_state_tensor):
        grad = upcast_grad_for_precision(grad, state, group['state_precision'])

        grad = _orthogonalize_gradient(p, grad, group["orthogonal_gradient"])

        nesterov = group.get('nesterov', False)
        nesterov_coef = group.get('nesterov_coef', None)
        use_mt = group['betas'][0] > 0

        if group.get('kourkoutas_beta', False):
            # Accumulate current grad's norm for the *next* step
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)

        adaptive_eps = scale_eps(group['eps'], p)

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors
            if use_mt:
                mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True, shifter=state['shifter'])

                # Update momentum in full-size
                mt.mul_(self.beta1).add_(grad_reshaped, alpha=d * (1.0 - self.beta1))

                # Factorize
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(mt.clone(), signed=True, shifter=state['shifter'])

                update_mt = mt

                if nesterov:
                    nv_coef = self.beta1 if nesterov_coef is None else nesterov_coef
                    update_mt = update_mt.lerp_(grad_reshaped, 1-nv_coef)

            vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])

            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                vt = vt.view_as(p).mul_(beta2).addcmul_(grad, grad * (d * d * (1.0 - beta2))).view_as(grad_reshaped)
            else:
                vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=d * d * (1.0 - beta2))

            if use_mt:
                update = update_mt
            else:
                update = grad_reshaped.mul(d)

            # Factorize
            state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt, signed=False, shifter=state['shifter'])

            if group['use_atan2']:
                denom = vt.sqrt_()
                update.atan2_(denom)
            else:
                denom = vt.sqrt_()
                denom.add_(d * adaptive_eps)
                update.div_(denom)

            wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, group['use_atan2'])

            del vt

            update = update.view(p.shape)

        else:  # Standard AdamW logic for non-factored tensors (or factored_2nd)
            actual_precision = group['actual_state_precision']
            factored_2nd = state.get('factored_2nd', False)

            if use_mt:
                exp_avg = get_state(state, 'exp_avg', actual_precision)
                exp_avg.mul_(self.beta1).add_(grad, alpha=d * (1.0 - self.beta1))

                update_mt = exp_avg.clone()

                if nesterov:
                    nv_coef = self.beta1 if nesterov_coef is None else nesterov_coef
                    update_mt = update_mt.lerp_(grad, 1-nv_coef)

                set_state(state, 'exp_avg', exp_avg, actual_precision, random_int_state_tensor)

            if use_mt:
                update = update_mt
            else:
                update = grad.mul(d)

            if factored_2nd:
                d1, d2 = state['effective_shape']
                exp_avg_sq = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])
                exp_avg_sq = exp_avg_sq.view(p.shape)
            else:
                exp_avg_sq = get_state(state, 'exp_avg_sq', actual_precision)

            grad_vt = grad.float() if factored_2nd else grad

            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                exp_avg_sq.mul_(beta2).addcmul_(grad_vt, grad_vt * (d * d * (1.0 - beta2)))
            else:
                exp_avg_sq.mul_(beta2).addcmul_(grad_vt, grad_vt, value=d * d * (1.0 - beta2))

            if factored_2nd:
                state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(exp_avg_sq.view(d1, d2), signed=False, shifter=state['shifter'])
            else:
                set_state(state, 'exp_avg_sq', exp_avg_sq, actual_precision, random_int_state_tensor, non_neg=True)
            del random_int_state_tensor

            if group['use_atan2']:
                denom = exp_avg_sq.sqrt()
                update.atan2_(denom.to(update.dtype))
            else:
                denom = exp_avg_sq.sqrt()
                denom.add_(d * adaptive_eps)
                update.div_(denom.to(update.dtype))

            wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, group['use_atan2'])

            del denom

        update_scaling = dlr * A if group['use_atan2'] else dlr
        if group.get('spectral_normalization', False):
            update = scale_update(p, update, update_scaling, state=state)
        else:
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

        param_update.apply_parameter_update(self, p, group, update, dlr, random_int_tensor=random_int_tensor, wd_scaler=wd_scaler)

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
