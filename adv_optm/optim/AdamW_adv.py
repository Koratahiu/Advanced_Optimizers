import torch

import math

from typing import Optional, Callable

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _init_fisher_wd_scaler, _get_fisher_wd_scaler
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.foreach_util import _foreach_orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm, scale_eps
from ..util.centered_decay import _init_anchor
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

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
            numerical stability. Set to None for scale invariant eps (vector
            lower bound) (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0).
        fisher_wd (bool): whether to use Fisher Adam (FAdam) weight decay, mapping
            the decay direction through the empirical Fisher information matrix and
            clipping its RMS. (default: False)
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
        nesterov (bool): enables Nesterov momentum (default: False).
        nesterov_coef (float, optional): Nesterov coefficient (default: None).
        orthogonal_gradient (str): whether to use OrthoGrad variants. 'disabled': off.
        'flattened': Standard vectorized OrthoGrad. 'iterative': Matrix-wise rank-2 OrthoGrad. (default: disabled)
        normed_momentum (bool): whether to compute the first moment on the normalized gradient. (default: False)
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
        centered_wd (float): Centered Weight Decay coefficient. Instead of decaying weights
            toward zero, they are decayed toward their initial values (anchors). This
            can be used together with standard weight decay. (default: 0.0)
        centered_wd_mode (str): The quantization format used to store the anchor
            weights to save VRAM. Options include:
            'full': Stores anchors in the original parameter's precision.
            'float8': Uses torch.float8_e4m3fn for a balance of precision and memory.
            'int8': Uses 8-bit block-wise quantization (block size 128).
            'int4': Uses 4-bit block-wise quantization (block size 32).
        spectral_normalization (bool): Enable explicit spectral normalization using power iteration. (default: False)
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
        compiled_optimizer (bool): Compiles the core step function using torch.compile
            for faster execution. (default: False)
        factored_2nd (bool): whether to keep the first moment uncompressed (dense)
            while only factorizing the second moment. (default: False)
        state_precision (str): Precision method for Adopt states. Options: 'auto'
            (parameter precision), 'fp32', 'factored' (SMMF low-rank FP32), 'bf16_sr' (with
            stochastic rounding), 'fp16' , 'int8_sr'. (default: 'auto')
        foreach (bool | None): If True, uses foreach/multi-tensor operations for
            improved throughput on small matrices. Only supports a subset of features.
            (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float | None = 1e-8,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        fisher_wd: bool = False,
        cautious_wd: bool = False,
        # Adam's Bias Correction
        use_bias_correction: bool = True,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # Adam_atan2 (scale invariant)
        use_atan2: bool = False,
        # OrthoGrad
        orthogonal_gradient: str = 'disabled', # 'flattened', 'iterative'
        # Nesterov momentum
        nesterov: bool = False,
        nesterov_coef: float | None = None,
        # Normalization then Momentum
        normed_momentum: bool = False,
        # K-b (adaptive beta2)
        kourkoutas_beta: bool = False,
        beta2_min: float = 0.9,
        ema_alpha: float = 0.95,
        tiny_spike: float = 1e-9,
        k_warmup_steps: int = 0,
        k_logging: int = 0,
        layer_key_fn: Optional[Callable] = None,
        # Spectral Normed Optimizer
        spectral_normalization: bool = False,
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        # States precision
        state_precision: str = "auto", # 'fp32', 'factored', 'bf16_sr', 'int8_sr'.
        # Factorized second moment only
        factored_2nd: bool = False,
        # SMMF factorization (legacy)
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
        # Foreach / multi-tensor
        foreach: bool = False,
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

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp16", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        # Legacy backwards compatibility support for `nnmf_factor=True`
        if nnmf_factor:
            state_precision = "factored"

        # Foreach mode only supports a subset of features (designed for small matrices)
        if foreach:
            _foreach_unsupported = []
            if state_precision not in ("auto", "fp32"):
                _foreach_unsupported.append(f"state_precision='{state_precision}'")
            if factored_2nd:
                _foreach_unsupported.append("factored_2nd")
            if nnmf_factor or state_precision == "factored":
                _foreach_unsupported.append("nnmf_factor / factored state")
            if vector_reshape:
                _foreach_unsupported.append("vector_reshape")
            if compiled_optimizer:
                _foreach_unsupported.append("compiled_optimizer")
            if kourkoutas_beta:
                _foreach_unsupported.append("kourkoutas_beta")
            if centered_wd != 0.0 and centered_wd_mode != 'full':
                _foreach_unsupported.append(f"centered_wd (mode='{centered_wd_mode}')")
            if _foreach_unsupported:
                raise ValueError(
                    f"foreach=True does not support the following features: {', '.join(_foreach_unsupported)}. "
                    "Foreach is intended for small-matrix training and does not implement "
                    "memory-saving or advanced features."
                )

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "fisher_wd": fisher_wd, "cautious_wd": cautious_wd,
            "use_atan2": use_atan2, "nesterov": nesterov, "nesterov_coef": nesterov_coef,
            "normed_momentum": normed_momentum,
            "orthogonal_gradient": orthogonal_gradient, "use_bias_correction": use_bias_correction,
            "compiled_optimizer": compiled_optimizer,
            "kourkoutas_beta": kourkoutas_beta, "beta2_min": beta2_min, "ema_alpha": ema_alpha,
            "tiny_spike": tiny_spike, "k_warmup_steps": k_warmup_steps, "k_logging": k_logging,
            "spectral_normalization": spectral_normalization,
            "centered_wd": centered_wd, "centered_wd_mode": centered_wd_mode,
            "state_precision": state_precision,
            "nnmf_factor": nnmf_factor, "vector_reshape": vector_reshape, "factored_2nd": factored_2nd,
            "foreach": foreach,
        }
        self.stochastic_rounding = stochastic_rounding
        self.kourkoutas_beta = kourkoutas_beta
        self.layer_key_fn = layer_key_fn
        self._init_lr = lr if lr > 0 else 1
        super().__init__(params, defaults)

        self.init_step()

        if self.kourkoutas_beta:
            self.kourkoutas_helper = KourkoutasHelper(self)

        if self.stochastic_rounding:
            # For deterministic stochastic rounding, we need to seed the generator
            # for each device used by the parameters.
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        # Initialize compiled function (by parameter shape)
        self._compiled_step_fns = {}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Overrides default load_state_dict to implement a workaround for PyTorch's
        automatic dtype casting. It ensures factorized states remain float32 for
        stability, preserves integer/float8 quantized anchor states, and forces
        standard states onto the parameter's current dtype/device.
        """
        super().load_state_dict(state_dict)
        param_update.post_process_loaded_state(self)

        # Pre-initialize states for all parameters to stabilize len(self.state).
        # This prevents torch.compile from triggering recompilations due to 
        # len(self.state) changing dynamically during the first step after loading.
        self.init_step()

    @property
    def supports_fused_back_pass(self):
        """Returns whether the optimizer supports fused backward pass."""
        return True

    @property
    def supports_memory_efficient_fp16(self):
        """Returns whether the optimizer supports memory-efficient FP16."""
        return True

    @property
    def supports_flat_params(self):
        """Returns whether the optimizer supports flat parameters."""
        return False

    def init_step(self):
        """Initializes optimizer state for all parameters across all parameter groups."""
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.__init_state(p, group)

    @torch.no_grad()
    def __init_state(self, p, group):
        state = self.state[p]

        # State Initialization
        if 'step' not in state:
            state['step'] = 0

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
                    state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
                else:
                    init_state_tensor(state, 'exp_avg_sq', p.shape, actual_precision, p.device, dtype, non_neg=True)

            if group.get('spectral_normalization', False) and is_spectral(p):
                init_spectral_norm(state, p)

            _init_anchor(p, state, group)

            _init_fisher_wd_scaler(group, state, p)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]
        self.__init_state(p, group)

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
        random_int_state_tensor = None

        if group.get('compiled_optimizer', False):
            step_size = torch.as_tensor(step_size)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
                random_int_state_tensor = random_int_tensor
            if group['actual_state_precision'] == 'bf16_sr' and random_int_state_tensor is None:
                random_int_state_tensor = param_update._get_random_int_for_sr(p)
            elif group['actual_state_precision'] == 'int8_sr':
                random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
            # Cache compiled function per-shape
            cache_key = (p.shape, state.get('factored', False))
            if cache_key not in self._compiled_step_fns:
                self._compiled_step_fns[cache_key] = torch.compile(
                    self._step_parameter,
                    fullgraph=True,
                    dynamic=False
                )
            step_param_fn = self._compiled_step_fns[cache_key]
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, step_size, beta1, beta2, sqrt_bias_correction2, random_int_tensor, random_int_state_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, step_size, beta1, beta2, sqrt_bias_correction2, random_int_tensor, random_int_state_tensor):
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

            vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])

            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                # View vt as p.shape, apply broadcasting with beta2 and grad, then view back to (d1, d2)
                vt = vt.view_as(p).mul_(beta2).addcmul_(grad, grad * (1.0 - beta2)).view_as(grad_reshaped)
            else:
                vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2)

            # Factorize
            for key, val in zip(('mu_v_nmf', 'mv_v_nmf'), _factorize_state(vt, signed=False, shifter=state['shifter'])):
                state[key].copy_(val)

            if group['use_atan2']:
                denom = vt.sqrt_()
                denom.div_(sqrt_bias_correction2)
                if group.get('normed_momentum', False):
                    grad_reshaped.atan2_(denom)
            else:
                denom = vt.sqrt_()
                denom.div_(sqrt_bias_correction2).add_(adaptive_eps)
                if group.get('normed_momentum', False):
                    grad_reshaped.div_(denom)

            # Reconstruct momentum from previous step's factors
            if use_mt:
                mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True, shifter=state['shifter'])

                # Update momentum in full-size
                mt.lerp_(grad_reshaped, 1.0 - beta1)

                # Factorize
                for key, val in zip(('mu_m_nmf', 'mv_m_nmf', 'sign'), _factorize_state(mt.clone(), signed=True, shifter=state['shifter'])):
                    state[key].copy_(val)

                update_mt = mt

                if nesterov:
                    nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                    update_mt = update_mt.lerp_(grad_reshaped, 1-nv_coef)

            if use_mt:
                update = update_mt
            else:
                update = grad_reshaped.clone()

            if not group.get('normed_momentum', False):
                if group['use_atan2']:
                    update.atan2_(denom)
                else:
                    update.div_(denom)

            wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, group['use_atan2'])

            del vt

            update = update.view(p.shape)

        else:  # Standard AdamW logic for non-factored tensors (or factored_2nd)
            actual_precision = group['actual_state_precision']
            factored_2nd = state.get('factored_2nd', False)

            if factored_2nd:
                d1, d2 = state['effective_shape']
                exp_avg_sq = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])
                exp_avg_sq = exp_avg_sq.view(p.shape)
            else:
                exp_avg_sq = get_state(state, 'exp_avg_sq', actual_precision)

            grad_vt = grad.float() if factored_2nd else grad

            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                exp_avg_sq.mul_(beta2).addcmul_(grad_vt, grad_vt * (1.0 - beta2))
            else:
                exp_avg_sq.mul_(beta2).addcmul_(grad_vt, grad_vt, value=1.0 - beta2)

            if factored_2nd:
                for key, val in zip(('mu_v_nmf', 'mv_v_nmf'), _factorize_state(exp_avg_sq.view(d1, d2), signed=False, shifter=state['shifter'])):
                    state[key].copy_(val)
            else:
                set_state(state, 'exp_avg_sq', exp_avg_sq, actual_precision, random_int_state_tensor, non_neg=True)

            if group['use_atan2']:
                denom = exp_avg_sq.sqrt()
                denom.div_(sqrt_bias_correction2)
                if group.get('normed_momentum', False):
                    grad.atan2_(denom.to(grad.dtype))
            else:
                denom = exp_avg_sq.sqrt()
                denom.div_(sqrt_bias_correction2).add_(adaptive_eps)
                if group.get('normed_momentum', False):
                    grad.div_(denom.to(grad.dtype))

            if use_mt:
                exp_avg = get_state(state, 'exp_avg', actual_precision)
                exp_avg.lerp_(grad, 1.0 - beta1)

                update_mt = exp_avg.clone()

                if nesterov:
                    nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                    update_mt = update_mt.lerp_(grad, 1-nv_coef)

                set_state(state, 'exp_avg', exp_avg, actual_precision, random_int_state_tensor)

            update = update_mt if use_mt else grad.clone()

            if not group.get('normed_momentum', False):
                if group['use_atan2']:
                    update.atan2_(denom.to(update.dtype))
                else:
                    update.div_(denom.to(update.dtype))

            wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, group['use_atan2'])

            del denom, random_int_state_tensor

        update_scaling = step_size * A if group['use_atan2'] else step_size
        if group.get('spectral_normalization', False):
            update = scale_update(p, update, update_scaling, state=state)
        else:
            update.mul_(update_scaling)

        param_update.apply_parameter_update(self, p, group, update, group['lr'], random_int_tensor=random_int_tensor, wd_scaler=wd_scaler)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get('foreach', False):
                self._foreach_step(group)
            else:
                for i, p in enumerate(group['params']):
                    self.step_parameter(p, group, i)

        return loss

    @torch.no_grad()
    def _foreach_step(self, group: dict) -> None:
        """
        Foreach/multi-tensor optimization step.
        Uses torch._foreach_* ops where available; falls back to iterative
        list-comprehension style for ops without foreach support (e.g. atan2).
        """
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return

        grads = [p.grad for p in params]

        # Orthogonalize gradients if needed
        ortho_mode = group.get('orthogonal_gradient', 'disabled')
        grads = _foreach_orthogonalize_gradient(params, grads, ortho_mode)

        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        anchors = []
        wd_scalers_init = []

        for p in params:
            state = self.state[p]
            self.__init_state(p, group)
            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            state_steps.append(torch.tensor(state['step'], dtype=torch.float32, device=p.device))

            # Collect anchor for centered_wd (full mode only in foreach)
            cwd = group.get('centered_wd', 0.0)
            if cwd != 0.0 and 'anchor_data' in state:
                anchors.append(state['anchor_data'])
            else:
                anchors.append(None)

            # Collect initial wd_scaler for fisher_wd
            wd_scalers_init.append(state.get("wd_scaler", torch.tensor(1.0, device=p.device)))

        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        use_atan2 = group['use_atan2']
        nesterov = group.get('nesterov', False)
        nesterov_coef = group.get('nesterov_coef', None)
        use_mt = beta1 > 0
        use_bias_correction = group.get('use_bias_correction', True)
        cautious = group.get('cautious_wd', False)
        fisher_wd = group.get('fisher_wd', False)
        spectral_norm = group.get('spectral_normalization', False)
        cwd = group.get('centered_wd', 0.0)
        decay_factor = lr / self._init_lr
        adaptive_eps = eps

        # Group tensors by (device, dtype) for foreach ops
        grouped = self._group_by_device_dtype(params, grads, exp_avgs, exp_avg_sqs, state_steps)

        for (g_params, g_grads, g_exp_avgs, g_exp_avg_sqs, g_steps) in grouped.values():
            if len(g_params) == 0:
                continue

            # Update steps
            torch._foreach_add_(g_steps, 1.0)

            # Bias correction
            if use_bias_correction:
                step_tensor = g_steps[0]
                bias_correction1 = 1.0 - beta1 ** step_tensor
                sqrt_bias_correction2 = (1.0 - beta2 ** step_tensor) ** 0.5
                bc1_scalar = float(bias_correction1)
                sbc2_scalar = float(sqrt_bias_correction2)
            else:
                bc1_scalar = 1.0
                sbc2_scalar = 1.0

            step_size = lr / bc1_scalar if use_bias_correction else lr

            # Second moment update
            torch._foreach_mul_(g_exp_avg_sqs, beta2)
            torch._foreach_addcmul_(g_exp_avg_sqs, g_grads, g_grads, value=1.0 - beta2)

            # First moment update
            if use_mt:
                torch._foreach_lerp_(g_exp_avgs, g_grads, 1.0 - beta1)

            # Compute denom: sqrt(exp_avg_sq) / sbc2 + eps
            denom = torch._foreach_sqrt(g_exp_avg_sqs)
            torch._foreach_div_(denom, sbc2_scalar)
            torch._foreach_add_(denom, adaptive_eps)

            # Compute updates (unscaled)
            if use_atan2:
                # atan2 is not available as foreach op - fallback to iterative
                updates = []
                for i, p in enumerate(g_params):
                    if use_mt:
                        update = g_exp_avgs[i].clone()
                        if nesterov:
                            nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                            update = update.lerp_(g_grads[i], 1.0 - nv_coef)
                        update.atan2_(denom[i])
                    else:
                        update = g_grads[i].clone()
                        update.atan2_(denom[i])
                    updates.append(update)
            else:
                if use_mt:
                    updates = torch._foreach_clone(g_exp_avgs)
                    if nesterov:
                        nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                        torch._foreach_lerp_(updates, g_grads, 1.0 - nv_coef)
                    torch._foreach_div_(updates, denom)
                else:
                    updates = torch._foreach_clone(g_grads)
                    torch._foreach_div_(updates, denom)

            # Spectral normalization (per-parameter, applied to update)
            if spectral_norm:
                for i, p in enumerate(g_params):
                    updates[i] = scale_update(p, updates[i], step_size, state=self.state[p])
            else:
                torch._foreach_mul_(updates, -step_size)

            # Apply main update to params
            torch._foreach_add_(g_params, updates)

            # Weight decay
            wd_scalar = weight_decay * decay_factor if weight_decay != 0 else None
            cwd_scalar = cwd * decay_factor if cwd != 0 else None

            if wd_scalar is not None or cwd_scalar is not None:
                # Compute global indices for this group
                group_start = 0
                for gp in grouped.values():
                    if gp is grouped[(g_params[0].device, g_params[0].dtype)]:
                        break
                    group_start += len(gp[0])
                # Compute fisher_wd scalers if needed
                if fisher_wd and wd_scalar is not None:
                    for i, p in enumerate(g_params):
                        idx = group_start + i
                        wd_scaler = _get_fisher_wd_scaler(group, wd_scalers_init[idx], p, denom[i], use_atan2)
                        wd_scalers_init[idx].copy_(wd_scaler)

                # Standard weight decay
                if wd_scalar is not None:
                    if fisher_wd:
                        for i, p in enumerate(g_params):
                            idx = group_start + i
                            scaled = wd_scalar * wd_scalers_init[idx]
                            if cautious:
                                mask = (g_params[i] * updates[i] >= 0).to(p.dtype)
                                g_params[i].addcmul_(g_params[i], mask * scaled, value=-1.0)
                                del mask
                            else:
                                g_params[i].mul_(1.0 - scaled)
                    elif cautious:
                        for i, p in enumerate(g_params):
                            mask = (g_params[i] * updates[i] >= 0).to(p.dtype)
                            g_params[i].addcmul_(g_params[i], mask * wd_scalar, value=-1.0)
                            del mask
                    else:
                        torch._foreach_mul_(g_params, 1.0 - wd_scalar)

                # Centered weight decay
                if cwd_scalar is not None:
                    for i, p in enumerate(g_params):
                        idx = group_start + i
                        anchor = anchors[idx]
                        if anchor is not None:
                            decay_target = g_params[i].sub(anchor)
                            if cautious:
                                mask = (g_params[i] * updates[i] >= 0).to(p.dtype)
                                g_params[i].addcmul_(decay_target, mask * cwd_scalar, value=-1.0)
                                del mask
                            else:
                                g_params[i].add_(decay_target, alpha=-cwd_scalar)
                            del decay_target

            # Update state steps in self.state
            for p, s in zip(g_params, g_steps):
                self.state[p]['step'] = int(s)

    @staticmethod
    def _group_by_device_dtype(params, grads, exp_avgs, exp_avg_sqs, state_steps):
        """Groups tensors by (device, dtype) for foreach operations."""
        groups: dict = {}
        for i in range(len(params)):
            key = (params[i].device, params[i].dtype)
            if key not in groups:
                groups[key] = [[], [], [], [], []]
            groups[key][0].append(params[i])
            groups[key][1].append(grads[i])
            groups[key][2].append(exp_avgs[i])
            groups[key][3].append(exp_avg_sqs[i])
            groups[key][4].append(state_steps[i])
        return groups
