import torch

import math

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _grams_update, _cautious_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm
from ..util.centered_decay import _init_anchor
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision
from ..util.sinkhorn import apply_sr_sinkhorn, _sinkhorn_sq_grad, get_sinkhorn_wd_scaler
from ..util.signed_util import apply_stochastic_sign_

class SinkSGD_adv(torch.optim.Optimizer):
    """
    Implements an advanced Stochastic Gradient Descent (SGD) with Sinkhorn Iterative Normalization (SinkSGD) algorithm.
    This is an advanced version of SinkSGD with optional features like
    low-rank factorization of optimizer states (SMMF), OrthoGrad, etc.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0)
        weight_decay (float): weight decay (L2 penalty or decoupled) (default: 0).
        nesterov (bool): enables Nesterov momentum. Only applicable when momentum
            is non-zero. (default: False)
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        orthogonal_gradient (bool): whether to use OrthoGrad. (default: False)
        centered_wd (float): Centered Weight Decay coefficient. Instead of decaying weights
            toward zero, they are decayed toward their initial values (anchors). This
            can be used together with standard weight decay. (default: 0.0)
        centered_wd_mode (str): The quantization format used to store the anchor
            weights to save VRAM. Options include:
            'full', 'float8', 'int8', 'int4'. (default: 'float8')
        nnmf_factor (bool): whether to use factorization or disable it. (default: False)
        state_precision (str): Precision method for states. Options: 'auto'
            (parameter precision), 'fp32', 'factored' (SMMF low-rank FP32), 'bf16_sr',
            'fp8_sr', 'int8_sr'. (default: 'auto')
        compiled_optimizer (bool): Compiles the core step function using torch.compile
            for faster execution. (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        # Sinkhorn Iterative Normalization
        sinkhorn_iterations: int = 5,
        orthogonal_sinkhorn: bool = False,
        # Normalization then Momentum
        normed_momentum: bool = False,
        # Centered Variance Precondition
        centered_vt: bool = False,
        # Nesterov Momentum
        nesterov: bool = False,
        nesterov_coef: float | None = None,
        # weight decay features
        geometric_wd: bool = False,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # Spectral Normed Optimizer
        spectral_normalization: bool = False,
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        # States precision
        state_precision: str = "auto",
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (momentum >= 0.0):
            raise ValueError(f"Momentum should be >= 0.0. Got {momentum}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp8_sr", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        if nnmf_factor:
            state_precision = "factored"

        defaults = {
            "lr": lr, "momentum": momentum,
            "weight_decay": weight_decay, "nesterov": nesterov, "nesterov_coef": nesterov_coef, "normed_momentum": normed_momentum, "centered_vt": centered_vt,
            "geometric_wd": geometric_wd, "cautious_wd": cautious_wd,
            "orthogonal_gradient": orthogonal_gradient, 
            "compiled_optimizer": compiled_optimizer,
            "sinkhorn_iterations": sinkhorn_iterations,
            "orthogonal_sinkhorn": orthogonal_sinkhorn,
            "spectral_normalization": spectral_normalization,
            "centered_wd": centered_wd, "centered_wd_mode": centered_wd_mode,
            "state_precision": state_precision,
            "nnmf_factor": nnmf_factor, "vector_reshape": vector_reshape
        }
        self.stochastic_rounding = stochastic_rounding
        self._init_lr = lr
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        self.init_step()

        self._compiled_step_parameter = None
        if compiled_optimizer:
            self.compile(fullgraph=True)

    def load_state_dict(self, state_dict: dict) -> None:
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

            actual_precision = 'auto' if req_precision == 'factored' else req_precision
            if actual_precision != 'auto' and (p.numel() < 10000 or p.ndim == 1):
                actual_precision = 'fp32'
            group['actual_state_precision'] = actual_precision

            dtype = torch.float32 if (state['factored'] or req_precision == 'factored') else p.dtype
            device = p.device

            if group['momentum'] != 0:
                if state['factored']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']

                    if group['momentum'] != 0:
                        state['mu_b_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
                        state['mv_b_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
                        packed_d2 = (d2 + 7) // 8
                        state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                else: 
                    if group['momentum'] != 0:
                        init_state_tensor(state, 'momentum_buffer', p.shape, actual_precision, p.device, dtype)

                if group.get('centered_vt', False):
                    p_shape = p.shape
                    state['vt_row'] = torch.zeros(p_shape[:-1], device=device, dtype=torch.float32)
                    state['vt_col'] = torch.zeros(p_shape[:-2] + p_shape[-1:], device=device, dtype=torch.float32)

            if group.get('spectral_normalization', False) and is_spectral(p):
                init_spectral_norm(state, p)

            _init_anchor(p, state, group)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]
        self.__init_state(p, group)

        step_size = group['lr']

        random_int_tensor = None
        random_int_state_tensor = None
        sign_noise = None

        if group.get('compiled_optimizer', False):
            step_size = torch.as_tensor(step_size)
            is_vector = grad.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
            if is_vector:
                sign_noise = param_update._get_random_noise_for_sso(p)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                random_int_tensor = param_update._get_random_int_for_sr(p)
                random_int_state_tensor = random_int_tensor
            if group['actual_state_precision'] == 'bf16_sr' and random_int_state_tensor is None:
                random_int_state_tensor = param_update._get_random_int_for_sr(p)
            elif group['actual_state_precision'] == 'int8_sr':
                random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
            elif group['actual_state_precision'] == 'fp8_sr':
                random_int_state_tensor = param_update._get_random_int_for_fp8_sr(p)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, step_size, random_int_tensor, random_int_state_tensor, sign_noise)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, step_size, random_int_tensor, random_int_state_tensor, sign_noise):
        grad = upcast_grad_for_precision(grad, state, group['state_precision'])
        is_vector = grad.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
        sinkhorn_iterations = group['sinkhorn_iterations']
        orthogonal_sinkhorn = group['orthogonal_sinkhorn']

        momentum = group['momentum']
        nesterov = group['nesterov']
        nesterov_coef = group.get('nesterov_coef', None)

        if group.get('normed_momentum', False):
            if not is_vector:
                # Sinkhorn iterative normalization
                grad = apply_sr_sinkhorn(grad, iters=sinkhorn_iterations, p=p, ortho_project=orthogonal_sinkhorn)
            else:
                # For vectors, apply adaptive stochastic sign
                grad = apply_stochastic_sign_(grad, sign_noise, is_vector=is_vector)

        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            if momentum != 0:
                buf = _reconstruct_state((state['mu_b_nmf'], state['mv_b_nmf'], state['sign'], d2), signed=True)
                buf.lerp_(grad_reshaped, 1 - momentum)

                # Factorize updated buffer
                state['mu_b_nmf'], state['mv_b_nmf'], state['sign'] = _factorize_state(buf.clone(), signed=True)

                if nesterov:
                    nv_coef = momentum if nesterov_coef is None else nesterov_coef
                    update = grad_reshaped.lerp(buf, nv_coef)
                else:
                    update = buf.clone()
            else:
                update = grad_reshaped.clone()

            update = update.view(p.shape)

        else:  # Standard logic for non-factored tensors
            actual_precision = group['actual_state_precision']

            if momentum != 0:
                buf = get_state(state, 'momentum_buffer', actual_precision)

                if group.get('centered_vt', False):
                    vt_row, vt_col = state['vt_row'], state['vt_col']
                    grad_vt = grad - buf
                    grad_vt_sq = grad_vt * grad_vt
                    mean_row_grad = grad_vt_sq.mean(dim=-1)
                    mean_col_grad = grad_vt_sq.mean(dim=-2)
                    vt_row.mul_(momentum).add_(mean_row_grad, alpha=1.0 - momentum)
                    vt_col.mul_(momentum).add_(mean_col_grad, alpha=1.0 - momentum)
                    if nesterov:
                        nv_coef = momentum if nesterov_coef is None else nesterov_coef
                        vt_row = vt_row.lerp(mean_row_grad, 1.0 - nv_coef)
                        vt_col = vt_col.lerp(mean_col_grad, 1.0 - nv_coef)
                    vt = _sinkhorn_sq_grad(vt_row, vt_col)
                else:
                    vt_row = None
                    vt_col = None

                buf.lerp_(grad, 1 - momentum)

                set_state(state, 'momentum_buffer', buf, actual_precision, random_int_state_tensor)

                if nesterov:
                    nv_coef = momentum if nesterov_coef is None else nesterov_coef
                    update = grad.lerp(buf, nv_coef)
                else:
                    update = buf.clone()
            else:
                update = grad.clone()

            del random_int_state_tensor

        if group.get('centered_vt', False):
            denom = vt
            update.atan2_(denom)
        else:
            denom = None

        if not group.get('normed_momentum', False):
            if not is_vector:
                # Sinkhorn iterative normalization
                update = apply_sr_sinkhorn(update, iters=sinkhorn_iterations, p=p, ortho_project=orthogonal_sinkhorn)
            else:
                # For vectors, apply adaptive stochastic sign
                update = apply_stochastic_sign_(update, sign_noise, is_vector=is_vector)

        if group.get('geometric_wd', False):
            wd_scaler = get_sinkhorn_wd_scaler(p, row_denom=vt_row, col_denom=vt_col)
        else:
            wd_scaler = None

        update_scaling = step_size
        if group.get('spectral_normalization', False):
            update = scale_update(p, update, update_scaling, state=state)
        else:
            if group.get('centered_vt', False):
                update_scaling = update_scaling * (4/math.pi)
            update.mul_(update_scaling)

        param_update.apply_parameter_update(self, p, group, update, step_size, random_int_tensor=random_int_tensor, wd_scaler=wd_scaler)

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