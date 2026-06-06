import torch
import math

from typing import Optional

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm
from ..util.centered_decay import _init_anchor, dequantize_anchor
from ..util.signed_util import apply_stochastic_sign_, get_signsgd_wd_target
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

A = 4 / math.pi

class SignSGD_adv(torch.optim.Optimizer):
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
        stochastic_sign (bool): whether to use the Stochastic Sign operator. (default: False)
        centered_wd (float): Centered Weight Decay coefficient. Instead of decaying weights
            toward zero, they are decayed toward their initial values (anchors). This
            can be used together with standard weight decay. (default: 0.0)
        centered_wd_mode (str): The quantization format used to store the anchor
            weights to save VRAM. Options include:
            'full': Stores anchors in the original parameter's precision.
            'float8': Uses torch.float8_e4m3fn for a balance of precision and memory.
            'int8': Uses 8-bit block-wise quantization (block size 128).
            'int4': Uses 4-bit block-wise quantization (block size 32).
        state_precision (str): Precision method for Adopt states. Options: 'auto'
            (parameter precision), 'fp32', 'factored' (SMMF low-rank FP32), 'bf16_sr' (with
            stochastic rounding), 'fp16' , 'int8_sr'. (default: 'auto')
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
        # weight decay features
        geometric_wd: bool = False,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: str = 'disabled', # 'flattened', 'iterative'
        # Stochastic Sign Operator
        stochastic_sign: bool = False,
        # Nesterov momentum
        nesterov: bool = False,
        nesterov_coef: float | None = None,
        # Normalization then Momentum
        normed_momentum: bool = False,
        # SNR Precondition (requires normed_momentum)
        snr_cond: bool = False,
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        # States precision
        state_precision: str = "auto", # 'fp32', 'factored', 'bf16_sr', 'int8_sr'.
        # Spectral Normed Optimizer
        spectral_normalization: bool = False,
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
        if snr_cond and not normed_momentum and not momentum > 0:
            raise NotImplementedError(f"snr_cond is intended to be used with normed_momentum")

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp16", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        # Legacy backwards compatibility support for `nnmf_factor=True`
        if nnmf_factor:
            state_precision = "factored"

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            geometric_wd=geometric_wd,
            vector_reshape=vector_reshape,
            orthogonal_gradient=orthogonal_gradient,
            stochastic_sign=stochastic_sign,
            nesterov=nesterov,
            nesterov_coef=nesterov_coef,
            normed_momentum=normed_momentum,
            snr_cond=snr_cond,
            spectral_normalization=spectral_normalization,
            centered_wd= centered_wd,
            centered_wd_mode= centered_wd_mode,
            state_precision=state_precision,
            nnmf_factor=nnmf_factor,
        )
        self.stochastic_rounding = stochastic_rounding
        self._init_lr = lr if lr > 0 else 1
        super().__init__(params, defaults)

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
    def supports_fused_back_pass(self) -> bool:
        return True

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    def init_step(self):
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.__init_state(p, group)

    @torch.no_grad()
    def __init_state(self, p, group):
        state = self.state[p]
        # State Initialization
        if group["momentum"] > 0 and len(state) == 0:
            req_precision = group['state_precision']
            is_vector = len(p.shape) == 1 and not group['vector_reshape']

            state['factored'] = req_precision == 'factored' and not is_vector

            actual_precision = 'auto' if req_precision == 'factored' else req_precision
            group['actual_state_precision'] = actual_precision

            dtype = torch.float32 if (state['factored'] or req_precision == 'factored') else p.dtype

            if group["momentum"] > 0:
                if state['factored']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
                    state['mu_m_nmf'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                    state['mv_m_nmf'] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                    state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=p.device, dtype=torch.uint8)
                else:
                    init_state_tensor(state, 'exp_avg', p.shape, actual_precision, p.device, dtype)

            if group.get('spectral_normalization', False) and is_spectral(p):
                init_spectral_norm(state, p)

            state["step"] = 0

            _init_anchor(p, state, group)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: Optional[int] = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]
        self.__init_state(p, group)

        lr = group["lr"]

        random_int_tensor = None
        random_noise_tensor = None
        random_int_state_tensor = None

        is_vector = p.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)

        if group.get('compiled_optimizer', False):
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
                random_int_state_tensor = random_int_tensor

            if group.get('momentum', 0) > 0 and not state.get('factored', False):
                if group['actual_state_precision'] == 'bf16_sr' and random_int_state_tensor is None:
                    random_int_state_tensor = param_update._get_random_int_for_sr(p)
                elif group['actual_state_precision'] == 'int8_sr':
                    random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)

            if group.get('stochastic_sign', False) and not is_vector:
                random_noise_tensor = param_update._get_random_noise_for_sso(p)

            lr = torch.as_tensor(lr)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, lr, random_int_tensor, random_noise_tensor, random_int_state_tensor)

        state["step"] += 1

    def _step_parameter(self, p, grad, state, group, lr, random_int_tensor, random_noise_tensor, random_int_state_tensor=None):
        grad = upcast_grad_for_precision(grad, state, group['state_precision'])

        is_vector = grad.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)

        momentum = group["momentum"]
        nesterov = group.get('nesterov', False)
        nesterov_coef = group.get('nesterov_coef', None)
        sso = group.get('stochastic_sign', False)
        normed_mt = group.get('normed_momentum', False)
        snr_cond = group.get('snr_cond', False) and normed_mt and momentum > 0

        denom = None
        wd_target = None
        cwd_target = None

        grad = _orthogonalize_gradient(p, grad, group["orthogonal_gradient"])

        if normed_mt:
            if sso:
                grad = apply_stochastic_sign_(grad, noise=random_noise_tensor, is_vector=is_vector)
            else:
                grad = grad.sign_()

        if state.get('factored'):
            # Factored Path
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            if momentum > 0:
                # Reconstruct momentum m_{t-1}
                exp_avg = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True, shifter=state['shifter'])

                if snr_cond:
                    denom = (1.0 - exp_avg.square()).clamp_min_(1e-30).sqrt_().view_as(p)

                if nesterov and normed_mt:
                    # Scale the normalized gradient using empirical buffer magnitude (SNR recovery)
                    normed_grad = exp_avg.abs().mul_(grad_reshaped)

                exp_avg.lerp_(grad_reshaped, 1 - momentum)

                if nesterov:
                    nv_coef = momentum if nesterov_coef is None else nesterov_coef
                    if normed_mt:
                        raw_update = normed_grad.lerp_(exp_avg, nv_coef)
                    else:
                        raw_update = grad_reshaped.lerp(exp_avg, nv_coef)
                else:
                    raw_update = exp_avg.clone()

                # Compress new momentum m_t and store factors
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(exp_avg, signed=True, shifter=state['shifter'])
            else:
                raw_update = grad_reshaped.clone()

            raw_update = raw_update.view(p.shape)

        else:
            # Fallback to standard SignSGD logic
            if momentum > 0:
                actual_precision = group['actual_state_precision']
                exp_avg = get_state(state, 'exp_avg', actual_precision)

                if snr_cond:
                    denom = (1.0 - exp_avg.square()).clamp_min_(1e-30).sqrt_()

                if nesterov and normed_mt:
                    # Scale the normalized gradient using empirical buffer magnitude (SNR recovery)
                    normed_grad = exp_avg.abs().mul_(grad)

                exp_avg.lerp_(grad, 1 - momentum)

                if nesterov:
                    nv_coef = momentum if nesterov_coef is None else nesterov_coef
                    if normed_mt:
                        raw_update = normed_grad.lerp_(exp_avg, nv_coef)
                    else:
                        raw_update = grad.lerp(exp_avg, nv_coef)
                else:
                    raw_update = exp_avg.clone()

                set_state(state, 'exp_avg', exp_avg, actual_precision, random_int_state_tensor)
            else:
                raw_update = grad.clone()

        if not group.get('normed_momentum', False):
            if sso:
                update = apply_stochastic_sign_(raw_update, noise=random_noise_tensor, is_vector=is_vector)
            else:
                update = raw_update.sign_()
        else:
            update = raw_update

        if snr_cond:
            update.atan2_(denom)

        if group.get('geometric_wd', False) and group["weight_decay"] > 0 :
            wd_target = get_signsgd_wd_target(p, denom=denom, stochastic_sign=sso, noise=random_noise_tensor, is_vector=is_vector)

            if group.get('centered_wd', 0.0) > 0 and 'anchor_data' in state:
                anchor = dequantize_anchor(p, state, group, p.dtype)
                cwd_target = get_signsgd_wd_target(p.sub(anchor), denom=denom, stochastic_sign=sso, noise=random_noise_tensor, is_vector=is_vector)
                del anchor

        if group.get('spectral_normalization', False):
            update = scale_update(p, update, lr, state=state)
        else:
            update_scaling = lr * A if snr_cond else lr
            update.mul_(update_scaling)

        param_update.apply_parameter_update(self, p, group, update, lr, random_int_tensor=random_int_tensor, wd_target=wd_target, cwd_target=cwd_target)

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