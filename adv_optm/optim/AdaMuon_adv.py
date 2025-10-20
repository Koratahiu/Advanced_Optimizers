import torch
from typing import Optional, Callable

from .AdamW_adv import AdamW_adv
from ..util.MuonAdam_helper import MuonAdamHelper
from ..util.Kourkoutas import KourkoutasHelper

from ..util.BF16_Stochastic_Rounding import add_stochastic_
from ..util.Newton_Schulz import _newton_schulz_iteration
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools

class AdaMuon_adv(torch.optim.Optimizer):
    """
    Implements the AdaMuon optimizer algorithm.

    AdaMuon combines the geometry-aware updates of Muon with the element-wise
    adaptivity of Adam. It is designed for 2D parameters (e.g., linear layers)
    and can handle higher-dimensional parameters by flattening.

    The algorithm incorporates three key mechanisms:
    1.  A sign-stabilized orthogonal update, where the sign of the momentum is
        orthogonalized instead of the momentum itself.
    2.  An element-wise second momentum estimator applied to the orthogonalized
        update directions.
    3.  An RMS-aligned rescaling strategy to match the update magnitude of Adam,
        allowing for reuse of learning rate schedules.

    Can also operate in a hybrid mode, using an auxiliary AdamW
    optimizer for specific parameters (e.g., biases, norms, embeddings) as
    defined by a `layer_key_fn`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): learning rate (default: 1e-3).
        betas (tuple[float, float]): coefficients used for both first and second moment
            estimation (default: (0.95, 0.95))
        weight_decay (float): weight decay (L2 penalty) (default: 0.1).
        eps (float): term added to the denominator for adaptive scaling to improve
            numerical stability (default: 1e-8).
        rms_target (float): The target Root-Mean-Square value for the final update
            vector, used for RMS-aligned rescaling. Allows for the reuse of existing Adam
            learning rate schedules. (default: 0.2).
        ns_steps (int): number of Newton-Schulz iterations to perform (default: 5).
        ns_eps (float): epsilon for Newton-Schulz normalization stability (default: 1e-7).
        ns_coeffs (tuple[float, float, float]): The (a, b, c) coefficients for the
            quintic polynomial in the Newton-Schulz iteration.
            (default: (3.4445, -4.7750, 2.0315)).
        stochastic_rounding (bool): whether to use stochastic rounding for
            BF16 parameter updates (default: True).
        nesterov (bool): enables Nesterov momentum (default: False).
        use_atan2 (bool): whether to use the atan2 update rule. (default: False)
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule.
            This changes the update  to `alpha_grad * grad + mt`, which can be
            more responsive, especially for small batch sizes. (default: False)
        alpha_grad (float): Mixing coefficient for the Simplified AdEMAMix update rule
            (only used when `Simplified_AdEMAMix` is `True`). Controls the weight of the
            current gradient. For small batch sizes, use high values (e.g., 10-100) to be
            more responsive. For large batch sizes, use low values (e.g., 0-1) for
            stability. (default: 100.0)
        vector_reshape_muon (bool): whether to reshape 1D vectors into 2D
            matrices for muon NewtonSchulz (default: False).
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
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
        MuonWithAuxAdam (bool): If True, enables the hybrid optimizer mode.
            Parameters designated by `layer_key_fn` will be optimized with
            AdamW_adv instead of Muon. (default: False)
        layer_key_fn (Optional[Callable]): A function that takes a parameter `p`
            and returns a key. If the key is 'adam', the parameter is handled by
            the auxiliary AdamW optimizer. All other keys are handled by Muon.
            Only used when `MuonWithAuxAdam` is True. (default: None)
        adam_kwargs (Optional[dict]): A dictionary of keyword arguments to pass
            to the auxiliary AdamW_adv optimizer. Only used when
            `MuonWithAuxAdam` is True. (default: None)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        weight_decay: float = 0.1,
        eps: float = 1e-8,
        rms_target: float = 0.2,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        stochastic_rounding: bool = True,
        use_atan2: bool = False,
        nesterov: bool = False,
        Simplified_AdEMAMix: bool = False,
        alpha_grad: float = 100.0,
        vector_reshape_muon: bool = False,
        vector_reshape: bool = False,
        nnmf_factor: bool = False,
        # K-b parameters
        kourkoutas_beta: bool = False,
        beta2_min: float = 0.9,
        ema_alpha: float = 0.95,
        tiny_spike: float = 1e-9,
        k_warmup_steps: int = 0,
        k_logging: int = 0,
        layer_key_kb_fn: Optional[Callable] = None,
        # hybrid optimizer mode
        MuonWithAuxAdam: bool = False,
        layer_key_fn: Optional[Callable] = None,
        muon_adam_lr: float = 1e-4,
        adam_kwargs: Optional[dict] = None,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (ns_steps > 0):
            raise ValueError(f"Newton-Schulz steps should be > 0. Got {ns_steps}")
        if Simplified_AdEMAMix and nesterov:
            print("Warning: nesterov is incompatible with Simplified_AdEMAMix, Disabling cautious.")
            nesterov = False

        muon_defaults = {
            "lr": lr, "betas": betas, "weight_decay": weight_decay,
            "eps": eps, "rms_target": rms_target, "ns_steps": ns_steps,
            "ns_eps": ns_eps, "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
            "vector_reshape_muon": vector_reshape_muon,
            "nesterov":nesterov, "use_atan2":use_atan2,
            "Simplified_AdEMAMix": Simplified_AdEMAMix, "alpha_grad": alpha_grad, 
            "_kourkoutas_beta": kourkoutas_beta, "beta2_min": beta2_min, "ema_alpha": ema_alpha,
            "tiny_spike": tiny_spike, "k_warmup_steps": k_warmup_steps, "k_logging": k_logging,
        }
        self.stochastic_rounding = stochastic_rounding
        self._kourkoutas_beta = kourkoutas_beta
        self._kourkoutas_helper = None
        self.layer_key_kb_fn = layer_key_kb_fn
        self.MuonWithAuxAdam = MuonWithAuxAdam
        self.helper = None
        self.aux_adam = None
 
        if not self.MuonWithAuxAdam:
            super().__init__(params, muon_defaults)
            return

        # HYBRID OPTIMIZER LOGIC
        adam_kwargs = adam_kwargs or {}
        self.aux_adam = AdamW_adv(
            [],
            lr=muon_adam_lr,
            **adam_kwargs,
            _is_delegate=True
        )
        adam_defaults = self.aux_adam.defaults

        final_param_groups = []
        _layer_key_fn = layer_key_fn if layer_key_fn is not None else lambda p: 'muon'

        for group in params:
            # All params in a group are of the same type
            first_param = group['params'][0]
            key = _layer_key_fn(first_param)
            optim_type = 'adam' if key == 'adam' else 'muon'

            new_group = group.copy()
            defaults_to_use = adam_defaults if optim_type == 'adam' else muon_defaults

            for key, value in defaults_to_use.items():
                new_group.setdefault(key, value)

            final_param_groups.append(new_group)

        super().__init__(final_param_groups, {})
        
        # Now that self is initialized, create the helper
        self.helper = MuonAdamHelper(self, layer_key_fn)
        

    @property
    def supports_fused_back_pass(self):
        return True

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    @property
    def kourkoutas_helper(self):
        """
        Exposes the kourkoutas_helper from the auxiliary AdamW optimizer,
        if it exists. This allows external access for logging K-b.
        """
        if self.aux_adam and hasattr(self.aux_adam, 'kourkoutas_helper'):
            return self.aux_adam.kourkoutas_helper
        return None

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if group['_kourkoutas_beta'] and self._kourkoutas_helper is None:
            self._kourkoutas_helper = KourkoutasHelper(self)

        if self.MuonWithAuxAdam:
            optim_type = self.helper.get_optimizer_type(p)
            if optim_type == 'adam':
                # Delegate to the AdamW_adv optimizer's logic.
                # We need to temporarily "lend" our state and param_groups
                self.aux_adam.state = self.state
                self.aux_adam.param_groups = self.param_groups
                
                # Ensure the aux optimizer uses the same Kourkoutas helper instance.
                if self._kourkoutas_helper is not None:
                    self.aux_adam.kourkoutas_helper = self._kourkoutas_helper

                self.aux_adam.step_parameter(p, group, i)
                return

        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]


        # State Initialization
        if 'step' not in state:
            state['step'] = 0

            should_factor = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            state['factored'] = should_factor

            state['reshaped_1d_muon'] = len(p.shape) == 1 and group['vector_reshape_muon']

            dtype = torch.float32 if group['nnmf_factor'] else p.dtype
            device = p.device
            if state['factored'] or state['reshaped_1d_muon']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
            if state['factored']:
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype) 
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                    state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype) 
                    state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:
                if len(p.shape) >= 2:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['second_momentum_buffer'] = torch.zeros_like(p)
                if state['reshaped_1d_muon']:
                    state['momentum_buffer'] = torch.zeros((d1, d2), device=device, dtype=dtype)
                    state['second_momentum_buffer'] = torch.zeros((d1, d2), device=device, dtype=dtype)
                elif len(p.shape) == 1:
                    state['momentum_buffer'] = torch.zeros_like(p)

        # Retrieve hyperparameters
        beta1, beta2 = group['betas']
        current_step = state['step']
        nesterov = group['nesterov']
        Simplified_AdEMAMix = group['Simplified_AdEMAMix']
        alpha_grad = group['alpha_grad']

        if state['factored']: # Factored AdaMuon

            # Reconstruct momentum from previous step's factors & sign
            d1, d2 = state['effective_shape']
            mt_buf = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
            unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
            torch.where(unpacked_sign, mt_buf, -mt_buf, out=mt_buf)
            del unpacked_sign

            # Update momentum in full-size
            grad_reshaped = grad.view(d1, d2)
            mt_buf.mul_(beta1).add_(grad_reshaped)

            if nesterov:
                signed_m_buf = torch.sign(grad_reshaped.add(mt_buf, alpha=beta1))
            elif Simplified_AdEMAMix:
                signed_m_buf = torch.sign(mt_buf.add(grad_reshaped, alpha=alpha_grad))
            else:
                signed_m_buf = torch.sign(mt_buf)
            del grad_reshaped

            update = _newton_schulz_iteration(
                signed_m_buf,
                steps=group['ns_steps'],
                eps=group['ns_eps'],
                coeffs=group['ns_coeffs'],
            )

            if group['_kourkoutas_beta']:
                # Call prepare_step() once at the beginning of the step for all params
                self._kourkoutas_helper.maybe_prepare_step(current_step)
                # Accumulate current sign-stabilized orthogonal update's norm for the *next* step
                self._kourkoutas_helper.accumulate_gradient_sq_norm(p, update.view(p.shape))
                # Get the dynamic beta2 calculated in prepare_step()
                beta2 = self._kourkoutas_helper.get_beta2(p, group, current_step)

            # Reconstruct second momentum from previous step's factors
            vt_buf = _unnmf((state['mu_v_nmf'], state['mv_v_nmf']))

            # Update second momentum in full-size
            vt_buf.mul_(beta2).addcmul_(update, update, value=1 - beta2)

            # Apply second momentum update (adaptive scaling)
            if group['use_atan2']:
                a = 1.2732395
                denom = vt_buf.sqrt()
                update.atan2_(denom).mul_(a)
            else:
                denom = vt_buf.sqrt().add_(group['eps'])
                update.div_(denom)
            del denom

            # RMS-aligned rescaling
            rms_target = group['rms_target']
            num_elements = update.numel()
            scaling_factor = rms_target * (num_elements ** 0.5) / (update.norm())

            update.mul_(scaling_factor)
            update = update.view(p.shape).mul_(group['lr'])
            del num_elements, scaling_factor

            # Compress updated moments and store new factors
            state['sign'] = _pack_bools(mt_buf > 0)
            _nnmf(mt_buf.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
            del mt_buf

            _nnmf(vt_buf.abs(), out=(state['mu_v_nmf'], state['mv_v_nmf']))
            del vt_buf

        else: # Standard AdaMuon logic for non-factored tensors

            if len(p.shape) >= 2 or state['reshaped_1d_muon']:

                # Momentum update
                mt_buf = state['momentum_buffer']
                if state['reshaped_1d_muon']:
                    d1, d2 = state['effective_shape']
                    grad_reshaped = grad.view(d1, d2)
                    mt_buf.mul_(beta1).add_(grad_reshaped)
                    if nesterov:
                        signed_m_buf = torch.sign(grad_reshaped.add(mt_buf, alpha=beta1))
                    elif Simplified_AdEMAMix:
                        signed_m_buf = torch.sign(mt_buf.add(grad_reshaped, alpha=alpha_grad))
                    else:
                        signed_m_buf = torch.sign(mt_buf)
                    del grad_reshaped
                else:
                    mt_buf.mul_(beta1).add_(grad)
                    if nesterov:
                        signed_m_buf = torch.sign(grad.add(mt_buf, alpha=beta1))
                    elif Simplified_AdEMAMix:
                        signed_m_buf = torch.sign(mt_buf.add(grad, alpha=alpha_grad))
                    else:
                        signed_m_buf = torch.sign(mt_buf)

                # Flatten if necessary (e.g., for Conv layers)
                if len(p.shape) > 2:
                    signed_m_buf = signed_m_buf.view(p.shape[0], -1)

                # NewtonSchulz
                update = _newton_schulz_iteration(
                    signed_m_buf,
                    steps=group['ns_steps'],
                    eps=group['ns_eps'],
                    coeffs=group['ns_coeffs'],
                )

                if len(p.shape) > 2 or state['reshaped_1d_muon']:
                    update = update.view(p.shape)

            if group['_kourkoutas_beta']:
                # Call prepare_step() once at the beginning of the step for all params
                self._kourkoutas_helper.maybe_prepare_step(current_step)
                # Accumulate current sign-stabilized orthogonal update's norm for the *next* step
                self._kourkoutas_helper.accumulate_gradient_sq_norm(p, update)
                # Get the dynamic beta2 calculated in prepare_step()
                beta2 = self._kourkoutas_helper.get_beta2(p, group, current_step)

                vt_buf = state['second_momentum_buffer']
                vt_buf.mul_(beta2).addcmul_(update, update, value=1 - beta2)

                # Apply second momentum update (adaptive scaling)
                if group['use_atan2']:
                    a = 1.2732395
                    denom = vt_buf.sqrt()
                    update.atan2_(denom).mul_(a)
                else:
                    denom = vt_buf.sqrt().add_(group['eps'])
                    update.div_(denom)
                del denom

                # RMS-aligned rescaling
                rms_target = group['rms_target']
                num_elements = update.numel()
                scaling_factor = rms_target * (num_elements ** 0.5) / (update.norm())

                update.mul_(scaling_factor)
                del num_elements, scaling_factor

                update.mul_(group['lr'])

            else: # Fallback to standard SGD with momentum for 1D params (biases, etc.) when not reshaped
                # Momentum update
                mt_buf = state['momentum_buffer']
                mt_buf.mul_(beta1).add_(grad)
                if nesterov:
                    update = grad.add(mt_buf, alpha=beta1)
                elif Simplified_AdEMAMix:
                    signed_m_buf = torch.sign(mt_buf.add(grad, alpha=alpha_grad))
                else:
                    update = mt_buf.clone()
                update.mul_(group['lr'])

        # Decoupled weight decay
        if group["weight_decay"] != 0:
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-group["weight_decay"] * group["lr"])
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p.data, -update)
        else:
            p.data.add_(-update)
        del update

        state['step'] += 1

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
