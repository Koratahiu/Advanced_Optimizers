import torch
from typing import Callable, Optional

import math

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state, _nnmf
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util.update_util import _init_fisher_wd_scaler, _get_fisher_wd_scaler
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm, scale_eps
from ..util.centered_decay import _init_anchor
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

A = 4 / math.pi

class Adopt_adv(torch.optim.Optimizer):
    """
    Implements an advanced ADOPT algorithm.

    The ADOPT update rule modifies Adam by:
    1.  **Initialization:** The second moment `vt` is initialized as `v₀ = g₀²`.
    2.  **Decorrelation:** The current gradient is normalized using the second-moment estimate
        from the *previous* step (`v_{t-1}`).
    3.  **Order of Operations:** This normalization occurs *before* updating the
        first-moment (momentum) estimate.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-4)
        betas (tuple[float, float]): coefficients used for computing running
            averages of momentum and variance (default: (0.9, 0.9999))
        eps (float): term added to the denominator to improve
            numerical stability. Set to None for scale invariant eps (vector
            lower bound) (default: 1e-6)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        fisher_wd (bool): whether to use Fisher Adam (FAdam) weight decay, mapping
            the decay direction through the empirical Fisher information matrix and
            clipping its RMS. (default: False)
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        clip_lambda (Callable, optional): A function that takes the current step
            and returns a value to clip the normalized gradient. Only used when
            `use_atan2` is False. (default: `lambda step: step**0.25`)
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices for low-rank compression (default: True).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        use_atan2 (bool): whether to use an atan2-based normalization, which can
            improve stability by removing the need for `eps`. (default: False)
        orthogonal_gradient (bool): whether to use OrthoGrad. (default: False)
        kourkoutas_beta (bool): whether to enable the layer-wise dynamic β₂ logic.
            If `False`, the optimizer behaves as standard Adopt. (default: False)
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
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
        factored_2nd (bool): whether to keep the first moment uncompressed (dense)
            while only factorizing the second moment. (default: False)
        state_precision (str): Precision method for Adopt states. Options: 'auto'
            (parameter precision), 'fp32', 'factored' (SMMF low-rank FP32), 'bf16_sr' (with
            stochastic rounding), 'fp16' , 'int8_sr'. (default: 'auto')
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: float | None = 1e-6,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        fisher_wd: bool = False,
        cautious_wd: bool = False,
        # ADOPT clipping
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        # Adam_atan2 (scale invariant)
        use_atan2: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: str = 'disabled', # 'flattened', 'iterative'
        # Nesterov momentum
        nesterov: bool = False,
        nesterov_coef: float | None = None,
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

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "fisher_wd": fisher_wd, "cautious_wd": cautious_wd, "orthogonal_gradient": orthogonal_gradient,
            "nesterov": nesterov, "nesterov_coef": nesterov_coef,
            "kourkoutas_beta": kourkoutas_beta, "beta2_min": beta2_min, "ema_alpha": ema_alpha,
            "tiny_spike": tiny_spike, "k_warmup_steps": k_warmup_steps, "k_logging": k_logging,
            "spectral_normalization": spectral_normalization,
            "centered_wd": centered_wd,
            "centered_wd_mode": centered_wd_mode,
            "state_precision": state_precision,
            "nnmf_factor": nnmf_factor, "vector_reshape": vector_reshape, "factored_2nd": factored_2nd,
            "compiled_optimizer": compiled_optimizer,
        }
        self.clip_lambda = clip_lambda
        self.stochastic_rounding = stochastic_rounding
        self.use_atan2 = use_atan2
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
    def supports_fused_back_pass(self): return True
    @property
    def supports_memory_efficient_fp16(self): return True
    @property
    def supports_flat_params(self): return False

    def init_step(self):
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.__init_state(p, group)

    @torch.no_grad()
    def __init_state(self, p, group):
        state = self.state[p]

        # State Initialization
        if 'step' not in state:
            grad = p.grad
            state['step'] = 0

            req_precision = group['state_precision']
            is_vector = len(p.shape) == 1 and not group['vector_reshape']

            state['factored'] = req_precision == 'factored' and not is_vector
            state['factored_2nd'] = group.get('factored_2nd', False) and not is_vector

            actual_precision = 'auto' if req_precision == 'factored' else req_precision
            group['actual_state_precision'] = actual_precision

            dtype = torch.float32 if (state['factored'] or req_precision == 'factored') else p.dtype

            vt_dtype = torch.float32 if (state['factored'] or state['factored_2nd'] or req_precision in ['factored', 'bf16_sr', 'int8_sr']) else dtype
            vt_init = grad.pow(2).to(vt_dtype)

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
                if group['betas'][0] > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                    state['mv_m_nmf'] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                    state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=p.device, dtype=torch.uint8)

                # Second moment (v)
                state['mu_v_nmf'], state['mv_v_nmf'] = _nnmf(vt_init.view(d1, d2))
                del vt_init
            else: # Fallback for non-factored tensors (or factored_2nd)
                if group['betas'][0] > 0:
                    init_state_tensor(state, 'exp_avg', p.shape, actual_precision, p.device, dtype)

                if state['factored_2nd']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
                    state['mu_v_nmf'], state['mv_v_nmf'] = _nnmf(vt_init.view(d1, d2))
                    state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=p.device, dtype=torch.uint8)
                else:
                    init_state_tensor(state, 'exp_avg_sq', p.shape, actual_precision, p.device, dtype)
                    set_state(state, 'exp_avg_sq', vt_init, actual_precision, None, non_neg=True)
                del vt_init

            if group.get('spectral_normalization', False) and is_spectral(p):
                init_spectral_norm(state, p)

            _init_anchor(p, state, group)

            _init_fisher_wd_scaler(group, state, p)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]
        self.__init_state(p, group)

        beta1, beta2 = group['betas']

        if group.get('kourkoutas_beta', False):
            if 'step' not in state:
                current_step = 0
            else:
                current_step = state['step']
            # Call prepare_step() once at the beginning of the step for all params
            self.kourkoutas_helper.maybe_prepare_step(current_step, p.device)
            # Get the dynamic beta2 calculated in prepare_step()
            beta2 = self.kourkoutas_helper.get_beta2(p, group)

        current_step = state['step']

        # The first step is for initialization only (skip when use_atan2 as it's scale invariant).
        if state['step'] == 0 and not (self.use_atan2 or group.get('spectral_normalization', False)):
            state['step'] += 1
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)
            return

        random_int_tensor = None
        random_int_state_tensor = None

        if group.get('compiled_optimizer', False):
            lr = torch.as_tensor(group['lr'])
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
                random_int_state_tensor = random_int_tensor
            if group['actual_state_precision'] == 'bf16_sr' and random_int_state_tensor is None:
                random_int_state_tensor = param_update._get_random_int_for_sr(p)
            elif group['actual_state_precision'] == 'int8_sr':
                random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
            step_param_fn = self._compiled_step_parameter
        else:
            lr = group['lr']
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, lr, beta1, beta2, random_int_tensor, random_int_state_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, lr, beta1, beta2, random_int_tensor, random_int_state_tensor):
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

            # Reconstruct v_{t-1}
            vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])

            # ADOPT Step A: Decorrelate g_t using v_{t-1}
            denom = vt.sqrt()
            wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, self.use_atan2, adaptive_eps)

            # Update second moment v_t for the *next* step using raw g_t
            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                # View vt as p.shape, apply broadcasting with beta2 and grad, then view back to (d1, d2)
                vt = vt.view_as(p).mul_(beta2).addcmul_(grad, grad * (1.0 - beta2)).view_as(grad_reshaped)
            else:
                vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2)
            # Factorize
            state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt, signed=False, shifter=state['shifter'])
            del vt

            if self.use_atan2:
                normalized_grad = torch.atan2(grad_reshaped, denom, out=denom)
            else:
                normalized_grad = torch.div(grad_reshaped, denom.add_(adaptive_eps), out=denom)
                if self.clip_lambda is not None:
                    clip_val = self.clip_lambda(state['step'])
                    normalized_grad.clamp_(-clip_val, clip_val)

            # ADOPT Step B: Update momentum m_t using normalized gradient
            if use_mt:
                mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True, shifter=state['shifter'])

                mt.lerp_(normalized_grad, 1.0 - beta1)

                # Factorize
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(mt.clone(), signed=True, shifter=state['shifter'])

                update_mt = mt

                if nesterov:
                    nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                    update_mt = update_mt.lerp_(grad_reshaped, 1-nv_coef)

            if use_mt:
                update = update_mt
                del normalized_grad
            else:
                update = normalized_grad

            update = update.view(p.shape)

        else: # Standard ADOPT logic for non-factored tensors (or factored_2nd)
            actual_precision = group['actual_state_precision']
            factored_2nd = state.get('factored_2nd', False)

            if factored_2nd:
                d1, d2 = state['effective_shape']
                vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])
                vt = vt.view(p.shape).to(grad.dtype)
            else:
                vt = get_state(state, 'exp_avg_sq', actual_precision) # v_{t-1}

            # ADOPT Step A: Decorrelate g_t using v_{t-1}
            denom = vt.sqrt()
            wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, self.use_atan2, adaptive_eps)

            if self.use_atan2:
                normalized_grad = torch.atan2(grad, denom, out=denom)
            else:
                normalized_grad = torch.div(grad, denom.add_(adaptive_eps), out=denom)
                if self.clip_lambda is not None:
                    clip_val = self.clip_lambda(state['step'])
                    normalized_grad.clamp_(-clip_val, clip_val)

            # ADOPT Step B: Update momentum m_t
            if use_mt:
                mt = get_state(state, 'exp_avg', actual_precision) # m_{t-1}
                mt.lerp_(normalized_grad, 1.0 - beta1)

                update_mt = mt.clone()

                if nesterov:
                    nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                    update_mt = update_mt.lerp_(grad, 1-nv_coef)

                set_state(state, 'exp_avg', mt, actual_precision, random_int_state_tensor)

            if use_mt:
                update = update_mt
                del normalized_grad
            else:
                update = normalized_grad

            grad_vt, vt = (grad.float(), vt.float()) if factored_2nd else (grad, vt)

            # Update second moment v_t for the next step using raw g_t
            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                vt.mul_(beta2).addcmul_(grad_vt, grad_vt * (1.0 - beta2))
            else:
                vt.mul_(beta2).addcmul_(grad_vt, grad_vt, value=1 - beta2)

            if factored_2nd:
                state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt.view(d1, d2), signed=False, shifter=state['shifter'])
            else:
                set_state(state, 'exp_avg_sq', vt, actual_precision, random_int_state_tensor, non_neg=True)
            del random_int_state_tensor

        update_scaling = lr * A if self.use_atan2 else lr

        # Parameter Update
        param_update.apply_parameter_update(self, p, group, update, lr, update_scaling, random_int_tensor=random_int_tensor, wd_scaler=wd_scaler)

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
