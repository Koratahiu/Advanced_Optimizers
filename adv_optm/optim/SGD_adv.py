import torch

from typing import Optional, Callable

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _grams_update, _cautious_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm
from ..util.centered_decay import _init_anchor
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

class SGD_adv(torch.optim.Optimizer):
    """
    Implements an advanced Stochastic Gradient Descent (SGD) algorithm.
    This is an advanced version of SGD with optional features like
    low-rank factorization of optimizer states (SMMF), OrthoGrad, 
    Cautious updating, and AdEMAMix extensions.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum factor (default: 0)
        dampening (float): dampening for momentum (default: 0)
        weight_decay (float): weight decay (L2 penalty or decoupled) (default: 0).
        nesterov (bool): enables Nesterov momentum. Only applicable when momentum
            is non-zero. (default: False)
        decoupled_wd (bool): whether to apply decoupled weight decay (like AdamW) 
            instead of standard L2 penalty. (default: False)
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        grams_moment (bool): whether to use Grams-style updates. (default: False)
        cautious_mask (bool): whether to use cautious masking to align the gradient's
            direction with the momentum. (default: False)
        orthogonal_gradient (bool): whether to use OrthoGrad. (default: False)
        scaled_optm (bool): Enable Scaled Optimizer parameter scaling. (default: False)
        centered_wd (float): Centered Weight Decay coefficient. Instead of decaying weights
            toward zero, they are decayed toward their initial values (anchors). This
            can be used together with standard weight decay. (default: 0.0)
        centered_wd_mode (str): The quantization format used to store the anchor
            weights to save VRAM. Options include:
            'full', 'float8', 'int8', 'int4'. (default: 'float8')
        nnmf_factor (bool): whether to use factorization or disable it. (default: False)
        state_precision (str): Precision method for states. Options: 'auto'
            (parameter precision), 'fp32', 'factored' (SMMF low-rank FP32), 'bf16_sr',
            'fp8_sr', 'uint8_sr'. (default: 'auto')
        compiled_optimizer (bool): Compiles the core step function using torch.compile
            for faster execution. (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        # Decoupled/cautious weight decay
        decoupled_wd: bool = False,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # Cautious and GRAMS
        cautious_mask: bool = False,
        grams_moment: bool = False,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # Scaled Optimizer
        scaled_optm: bool = False,
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
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        if cautious_mask and grams_moment:
            print("Warning: cautious is incompatible with grams, Disabling cautious.")
            cautious_mask = False

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp8_sr", "uint8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        if nnmf_factor:
            state_precision = "factored"

        defaults = {
            "lr": lr, "momentum": momentum, "dampening": dampening, 
            "weight_decay": weight_decay, "nesterov": nesterov,
            "decoupled_wd": decoupled_wd, "cautious_wd": cautious_wd,
            "orthogonal_gradient": orthogonal_gradient, 
            "compiled_optimizer": compiled_optimizer,
            "scaled_optm": scaled_optm,
            "centered_wd": centered_wd, "centered_wd_mode": centered_wd_mode,
            "state_precision": state_precision,
            "nnmf_factor": nnmf_factor, "vector_reshape": vector_reshape
        }
        self.stochastic_rounding = stochastic_rounding
        self.cautious_mask = cautious_mask
        self.grams_moment = grams_moment
        self._init_lr = lr
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

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

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        grad = p.grad
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
            state['actual_state_precision'] = actual_precision

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

            if group.get('scaled_optm', False) and is_spectral(p):
                init_spectral_norm(group, state, p)

            _init_anchor(p, state, group)

        step_size = group['lr']

        random_int_tensor = None
        random_int_state_tensor = None

        if group.get('compiled_optimizer', False):
            step_size = torch.as_tensor(step_size)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                random_int_tensor = param_update._get_random_int_for_sr(p)
                random_int_state_tensor = random_int_tensor
            if state['actual_state_precision'] == 'bf16_sr' and random_int_state_tensor is None:
                random_int_state_tensor = param_update._get_random_int_for_sr(p)
            elif state['actual_state_precision'] == 'uint8_sr':
                random_int_state_tensor = param_update._get_random_int_for_uint8_sr(p)
            elif state['actual_state_precision'] == 'fp8_sr':
                random_int_state_tensor = param_update._get_random_int_for_fp8_sr(p)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, step_size, random_int_tensor, random_int_state_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, step_size, random_int_tensor, random_int_state_tensor):
        grad = upcast_grad_for_precision(grad, state, group['state_precision'])

        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            if momentum != 0:
                buf = _reconstruct_state((state['mu_b_nmf'], state['mv_b_nmf'], state['sign'], d2), signed=True)
                buf.mul_(momentum).add_(grad_reshaped, alpha=1 - dampening)

                # Factorize updated buffer
                state['mu_b_nmf'], state['mv_b_nmf'], state['sign'] = _factorize_state(buf.clone(), signed=True)

                if self.grams_moment:
                    update_mt = _grams_update(buf, grad_reshaped, inplace=True)
                elif self.cautious_mask:
                    update_mt = _cautious_update(buf, grad_reshaped, inplace=True)
                else:
                    update_mt = buf

                if nesterov:
                    update = grad_reshaped.add(update_mt, alpha=momentum)
                else:
                    update = update_mt
            else:
                update = grad_reshaped.clone()

            update = update.view(p.shape)

        else:  # Standard logic for non-factored tensors
            actual_precision = state.get('actual_state_precision', 'auto')

            if momentum != 0:
                buf = get_state(state, 'momentum_buffer', actual_precision)

                if state['step'] == 0:
                    buf.copy_(grad)
                else:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if self.grams_moment:
                    update_mt = _grams_update(buf, grad)
                elif self.cautious_mask:
                    update_mt = _cautious_update(buf, grad)
                else:
                    update_mt = buf.clone()

                set_state(state, 'momentum_buffer', buf, actual_precision, random_int_state_tensor)

                if nesterov:
                    update = grad.add(update_mt, alpha=momentum)
                else:
                    update = update_mt
            else:
                update = grad.clone()

            del random_int_state_tensor

        update_scaling = step_size
        if group.get('scaled_optm', False):
            update = scale_update(p, update, update_scaling, vector_state=state.get('spectral_v'))
        else:
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