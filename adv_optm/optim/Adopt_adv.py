import torch
from typing import Callable, Optional

import math

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state, _nnmf
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util.update_util import _grams_update, _cautious_update, _scale_sim_AdEMAMix_update, _init_fisher_wd_scaler, _get_fisher_wd_scaler
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
            numerical stability (default: 1e-6)
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
        cautious_mask (bool):  whether to use cautious masking to align the gradient's
            direction with the first moment's.  (default: False)
        grams_moment (bool): whether to combine the gradient's direction with the
            first moment's magnitude (default: False).
        orthogonal_gradient (bool): whether to use OrthoGrad. (default: False)
        use_AdEMAMix (bool): whether to enable the AdEMAMix feature. This adds
            a second, slow-moving average of the momentum (`mt_slow`) which is
            combined with the primary momentum (`mt`) to stabilize updates,
            especially in noisy, small-batch settings. If `False`, the
            optimizer behaves as standard ADOPT. (default: False)
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
            stochastic rounding), 'fp8_sr', 'uint8_sr'. (default: 'auto')
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
        state_precision: str = "auto", # 'fp32', 'factored', 'bf16_sr', 'fp8_sr', 'uint8_sr'.
        # SMMF factorization (legacy)
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        factored_2nd: bool = False,
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
        if cautious_mask and grams_moment:
            print("Warning: cautious is incompatible with grams, Disabling cautious.")
            cautious_mask = False
        if betas[0] == 0.0 and Simplified_AdEMAMix:
            raise ValueError(f"Beta1 cannot be 0.0 when using Simplified_AdEMAMix. Got {betas[0]}")
        if kourkoutas_beta and not (betas[1] > beta2_min):
            raise ValueError(f"For Kourkoutas-β, betas[1] (as beta2_max) must be > beta2_min. Got {betas[1]} and {beta2_min}")
        if use_AdEMAMix and Simplified_AdEMAMix:
            print("Warning: use_AdEMAMix is incompatible with Simplified_AdEMAMix, Disabling use_AdEMAMix.")
        if grams_moment and Simplified_AdEMAMix:
            print("Warning: grams is incompatible with Simplified_AdEMAMix, Disabling grams.")
        if cautious_mask and Simplified_AdEMAMix:
            print("Warning: cautious is incompatible with Simplified_AdEMAMix, Disabling cautious.")


        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp8_sr", "uint8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        # Legacy backwards compatibility support for `nnmf_factor=True`
        if nnmf_factor:
            state_precision = "factored"

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "fisher_wd": fisher_wd, "cautious_wd": cautious_wd,
            "beta3_ema": beta3_ema, "alpha": alpha,
            "alpha_grad": alpha_grad,
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
        self.use_atan2 = use_atan2 and not Simplified_AdEMAMix
        self.cautious_mask = cautious_mask and not Simplified_AdEMAMix
        self.grams_moment = grams_moment and not Simplified_AdEMAMix
        self.orthogonal_gradient = orthogonal_gradient
        self.use_AdEMAMix = use_AdEMAMix and not Simplified_AdEMAMix
        self.Simplified_AdEMAMix = Simplified_AdEMAMix
        self.kourkoutas_beta = kourkoutas_beta
        self.layer_key_fn = layer_key_fn
        self._init_lr = lr
        super().__init__(params, defaults)

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

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]

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

        # State Initialization
        if 'step' not in state:
            state['step'] = 0

            req_precision = group['state_precision']
            is_vector = len(p.shape) == 1 and not group['vector_reshape']

            state['factored'] = req_precision == 'factored' and not is_vector
            state['factored_2nd'] = group.get('factored_2nd', False) and not is_vector

            actual_precision = 'auto' if req_precision == 'factored' else req_precision
            if actual_precision != 'auto' and (p.numel() < 10000 or p.ndim == 1):
                actual_precision = 'fp32'
            state['actual_state_precision'] = actual_precision

            dtype = torch.float32 if (state['factored'] or req_precision == 'factored') else p.dtype

            vt_dtype = torch.float32 if (state['factored'] or state['factored_2nd'] or req_precision in ['factored', 'bf16_sr', 'fp8_sr', 'uint8_sr']) else dtype
            vt_init = grad.pow(2).to(vt_dtype)
            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                vt_init.mul_(beta2).addcmul_(grad.to(vt_dtype), grad.to(vt_dtype) * (1.0 - beta2))
            else:
                vt_init.mul_(beta2).addcmul_(grad.to(vt_dtype), grad.to(vt_dtype), value=1.0 - beta2)

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
                if group['betas'][0] > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                    state['mv_m_nmf'] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                # AdEMAMix slow moment (m_slow)
                if self.use_AdEMAMix:
                    state['mu_m_slow_nmf'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                    state['mv_m_slow_nmf'] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                    packed_d2 = (d2 + 7) // 8
                    state['sign_slow'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)

                # Second moment (v)
                state['mu_v_nmf'], state['mv_v_nmf'] = _nnmf(vt_init.view(d1, d2))
                del vt_init
            else: # Fallback for non-factored tensors (or factored_2nd)
                if group['betas'][0] > 0:
                    init_state_tensor(state, 'exp_avg', p.shape, actual_precision, p.device, dtype)
                if self.use_AdEMAMix:
                    init_state_tensor(state, 'exp_avg_slow', p.shape, actual_precision, p.device, dtype)

                if state['factored_2nd']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
                    state['mu_v_nmf'], state['mv_v_nmf'] = _nnmf(vt_init.view(d1, d2))
                else:
                    init_state_tensor(state, 'exp_avg_sq', p.shape, actual_precision, p.device, dtype)
                    set_state(state, 'exp_avg_sq', vt_init, actual_precision, None)
                del vt_init

            if group.get('spectral_normalization', False) and is_spectral(p):
                init_spectral_norm(group, state, p)

            _init_anchor(p, state, group)

            _init_fisher_wd_scaler(group, state, p)

        current_step = state['step']

        # The first step is for initialization only (skip when use_atan2 as it's scale invariant).
        if state['step'] == 0 and not (self.use_atan2 or group.get('spectral_normalization', False)):
            state['step'] += 1
            return

        random_int_tensor = None
        random_int_state_tensor = None

        if group.get('compiled_optimizer', False):
            lr = torch.as_tensor(group['lr'])
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
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
            lr = group['lr']
            step_param_fn = self._step_parameter

        if self.Simplified_AdEMAMix:
            lr = _scale_sim_AdEMAMix_update(beta1, state['step'] + 1, group["alpha_grad"], lr, group.get('spectral_normalization', False))

        step_param_fn(p, grad, state, group, lr, beta1, beta2, random_int_tensor, random_int_state_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, lr, beta1, beta2, random_int_tensor, random_int_state_tensor):
        grad = upcast_grad_for_precision(grad, state, group['state_precision'])

        if self.orthogonal_gradient:
            grad = _orthogonalize_gradient(p, grad)

        if self.use_AdEMAMix:
            beta3_ema = group['beta3_ema']
            alpha = group['alpha']
        if self.Simplified_AdEMAMix:
            alpha_grad = group["alpha_grad"]

        if group.get('kourkoutas_beta', False):
            # Accumulate current grad's norm for the *next* step
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)

        adaptive_eps = scale_eps(group['eps'], p)

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct v_{t-1}
            vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False)

            # ADOPT Step A: Decorrelate g_t using v_{t-1}
            denom = vt.sqrt()
            wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, self.use_atan2, adaptive_eps)

            # Update second moment v_t for the *next* step using raw g_t
            if isinstance(beta2, torch.Tensor) and beta2.dim() > 0:
                vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped * (1.0 - beta2))
            else:
                vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2)
            # Factorize
            state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt, signed=False)
            del vt

            if self.use_atan2:
                normalized_grad = torch.atan2(grad_reshaped, denom, out=denom)
            else:
                normalized_grad = torch.div(grad_reshaped, denom.add_(adaptive_eps), out=denom)
                if self.clip_lambda is not None:
                    clip_val = self.clip_lambda(state['step'])
                    normalized_grad.clamp_(-clip_val, clip_val)

            # ADOPT Step B: Update momentum m_t using normalized gradient
            if beta1 > 0:
                mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True)

                if self.Simplified_AdEMAMix:
                    mt.mul_(beta1).add_(normalized_grad, alpha=1.0)
                else:
                    mt.lerp_(normalized_grad, 1.0 - beta1)

                # Factorize
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(mt.clone(), signed=True)

                if self.grams_moment:
                    update_mt = _grams_update(mt, grad_reshaped, inplace=True)
                elif self.cautious_mask:
                    update_mt = _cautious_update(mt, grad_reshaped, inplace=True)
                else:
                    update_mt = mt

            if self.use_AdEMAMix:
                # Reconstruct AdEMAMix EMA
                mt_slow = _reconstruct_state((state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'], d2), signed=True)

                mt_slow.lerp_(normalized_grad, 1.0 - beta3_ema)

                if beta1 > 0:
                    update = update_mt.add_(mt_slow, alpha=alpha)
                    del normalized_grad
                else:
                    update = normalized_grad.add_(mt_slow, alpha=alpha)
                # Factorize
                state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'] = _factorize_state(mt_slow, signed=True)
                del mt_slow

            elif self.Simplified_AdEMAMix:
                update = update_mt.add_(normalized_grad, alpha=alpha_grad)
                del normalized_grad
            else:
                if beta1 > 0:
                    update = update_mt
                    del normalized_grad
                else:
                    update = normalized_grad

            update = update.view(p.shape)

        else: # Standard ADOPT logic for non-factored tensors (or factored_2nd)
            actual_precision = state.get('actual_state_precision', 'auto')
            factored_2nd = state.get('factored_2nd', False)

            if factored_2nd:
                d1, d2 = state['effective_shape']
                vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False)
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
            if beta1 > 0:
                mt = get_state(state, 'exp_avg', actual_precision) # m_{t-1}
                if self.Simplified_AdEMAMix:
                    mt.mul_(beta1).add_(normalized_grad, alpha=1.0)
                else:
                    mt.lerp_(normalized_grad, 1.0 - beta1)

                if self.grams_moment:
                    update_mt = _grams_update(mt, grad)
                elif self.cautious_mask:
                    update_mt = _cautious_update(mt, grad)
                else:
                    update_mt = mt.clone()

                set_state(state, 'exp_avg', mt, actual_precision, random_int_state_tensor)

            if self.use_AdEMAMix:
                m_slow = get_state(state, 'exp_avg_slow', actual_precision)
                m_slow.lerp_(normalized_grad, 1.0 - beta3_ema)
                if beta1 > 0:
                    update = update_mt.add_(m_slow, alpha=alpha)
                    del normalized_grad
                else:
                    update = normalized_grad.add_(m_slow, alpha=alpha)
                set_state(state, 'exp_avg_slow', m_slow, actual_precision, random_int_state_tensor)
            elif self.Simplified_AdEMAMix:
                update = update_mt.add_(normalized_grad, alpha=alpha_grad)
            else:
                if beta1 > 0:
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
                state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt.view(d1, d2), signed=False)
            else:
                set_state(state, 'exp_avg_sq', vt, actual_precision, random_int_state_tensor)
            del random_int_state_tensor

        update_scaling = lr * A if self.use_atan2 else lr

        if group.get('spectral_normalization', False):
            update = scale_update(p, update, update_scaling, vector_state=state.get('spectral_v'))
        else:
            update.mul_(update_scaling)

        # Parameter Update
        param_update.apply_parameter_update(self, p, group, update, lr, random_int_tensor=random_int_tensor, wd_scaler=wd_scaler)

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
