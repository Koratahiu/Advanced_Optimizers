import torch

import math

from ..util import param_update
from ..util.Muon_util import newton_schulz, _is_suitable_for_muon, rms_adjustment, normuon_update, approx_mars, _auto_projection_for_adamuon
from ..util.scaled_optm import spectral_normalization, init_spectral_norm, scale_eps
from ..util.factorization_util import _get_effective_shape, _factorize_state, _reconstruct_state
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util import Muon_AuxAdam
from ..util.centered_decay import _init_anchor
from typing import Optional
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

A = 4 / math.pi

class AdaMuon_adv(torch.optim.Optimizer):
    """
    Implements an advanced AdaMuon optimizer algorithm.

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

    When `MuonWithAuxAdam` is enabled, this single optimizer class handles both
    'muon' and 'adam' parameter groups, dispatching to the appropriate logic internally.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): learning rate (default: 1e-3).
        betas (tuple[float, float]): coefficients used for both first and second moment
            estimation (default: (0.95, 0.95))
        weight_decay (float): weight decay (L2 penalty) (default: 0.1).
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        eps (float): term added to the denominator for adaptive scaling to improve
            numerical stability (default: 1e-8).
        rms_rescaling (bool): Use Root-Mean-Square for the final update
            vector, used for RMS-aligned rescaling. Allows for the reuse of existing Adam
            learning rate schedules. (default: True).
        ns_steps (int): number of Newton-Schulz iterations to perform (default: 5).
        ns_eps (float): epsilon for Newton-Schulz normalization stability. When None
            it's derived from scale invariant rule (default: 1e-7).
        ns_coeffs (tuple[float, float, float]): The (a, b, c) coefficients for the
            quintic polynomial in the Newton-Schulz iteration.
            (default: (3.4445, -4.7750, 2.0315)).
        stochastic_rounding (bool): whether to use stochastic rounding for
            BF16 parameter updates (default: True).
        orthogonal_gradient (bool): whether to use OrthoGrad.  (default: False)
        nesterov (bool): enables Nesterov momentum (default: False).
        use_atan2 (bool): whether to use the atan2 update rule. (default: False)
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        kappa_p (float, optional): The p-value for the update geometry  (domain [1.0, 2.0]).
            - 1.0: Standard (sign update).
            - 2.0: Spherical (normalized L2 update).
            - values between 1.0 and 2.0 interpolate behavior.
            (default: 1.0).
        auto_projection (bool): Automatically determines kappa_p based on parameter
            dimensionality. Sets p=2.0 for 4D tensors (Conv2D) to use Spherical updates,
            and p=1.0 for others (Linear/Embeddings) to use Sign updates. Overrides explicit
            kappa_p value. (default: True).
        low_rank_ortho (bool): If True, enables low-rank orthogonalization, which
            projects the update to a lower rank before orthogonalization.
            (default: False)
        ortho_rank (int): The rank for low-rank orthogonalization.
            (default: 128)
        accelerated_ns (bool): If True, enables Chebyshev-accelerated Newton-Schulz, which
            dynamically calculates optimal 3rd-order polynomial coefficients. (default: False)
        cns_a_bound (float): Initial lower bound for singular values for CANS. When None
            it's derived from scale invariant rule (default: None).
        approx_mars (bool): If True, enables Approximated MARS-M variance reduction.
        fom the paper "MARS-M: When Variance Reduction Meets Matrices"
            (default: False)
        mars_gamma (float): The scaling coefficient for MARS gradient correction.
            (default: 0.025)
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
        use_muon (bool | None): whether to use Muon or AuxAdamW. MUST be provided
            either here or via `optim_type` in parameter groups. (default: None)
        state_precision (str): Precision for Muon optimizer states. Options: 'auto' (parameter dtype), 'fp32',
            'bf16_sr' (BF16 with stochastic rounding), 'fp8_sr', 'int8_sr'.
            (default: 'auto')
        factored_2nd (bool): Factorize only the second moment (v_t) using SMMF
            low-rank compression while keeping the first moment (momentum_buffer)
            dense. Ignored when `nnmf_factor=True` (full SMMF) or `normuon_variant=True`.
            Combines well with `state_precision` on the first moment. (default: False)
        n_layers (int): The depth of the network (L). Required for optimal epsilon scaling. (default: 1)
        spectral_normalization (bool): Enable explicit spectral normalization using power iteration. (default: False)
        --- Auxiliary AdamW_adv Parameters (used for 'adam' groups) ---
        adam_betas (tuple[float, float]): Betas for the AdamW optimizer part.
        adam_eps (float): Epsilon for the AdamW optimizer part.
        adam_weight_decay (float): Weight decay for the AdamW optimizer part.
        adam_fisher_wd (bool): Fisher Adam (FAdam) weight decay for the AdamW part. (default: False)
        adam_use_bias_correction (bool): Bias correction for AdamW.
        adam_use_atan2 (bool): Atan2 update rule for AdamW.
        adam_orthogonal_gradient (bool): OrthoGrad for AdamW.
        adam_nesterov (bool): Nesterov momentum for AdamW. (default: False)
        adam_nesterov_coef (float, optional): Nesterov coefficient for AdamW. (default: None)
        adam_kourkoutas_beta (bool): Kourkoutas-β for AdamW.
        adam_beta2_min (float): Minimum beta2 for Kourkoutas-β. (default: 0.9)
        adam_ema_alpha (float): EMA alpha for Kourkoutas-β. (default: 0.95)
        adam_tiny_spike (float): Tiny spike for Kourkoutas-β. (default: 1e-9)
        adam_k_warmup_steps (int): Warmup steps for Kourkoutas-β. (default: 0)
        adam_spectral_normalization (bool): Enable explicit spectral normalization for AdamW. (default: False)
        adam_state_precision (str): Precision for AuxAdam states. Options: 'auto', 'fp32', 'bf16_sr', 'fp16', 'fp8_sr', 'int8_sr', 'factored'. (default: 'auto')
        adam_nnmf_factor (bool): 1-bit factored for AdamW.
        adam_factored_2nd (bool): Factorize only the second moment (v_t) for AuxAdam. (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        # Decoupled/cautious weight decay
        weight_decay: float = 0,
        cautious_wd: bool = False,
        # Nesterov momentum
        nesterov: bool = True,
        nesterov_coef: float | None = None,
        # RMS Rescaling
        rms_rescaling: bool = True,
        # Newton Schulz
        ns_steps: int = 5,
        ns_eps: float | None = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # Adam_atan2 (scale invariant)
        use_atan2: bool = False,
        # NorMuon
        normuon_variant: bool = False,
        # Boolean to spilt param
        use_muon: bool | None = None,
        # States precision (Muon path)
        state_precision: str = "auto",  # 'fp32', 'bf16_sr', 'fp8_sr', 'int8_sr'
        # Factorized second moment only
        factored_2nd: bool = False,
        # Update geometry parameters
        kappa_p: float = 1.0,
        auto_projection: bool = True,
        # Low-rank Muon
        low_rank_ortho: bool = False,
        ortho_rank: int = 128,
        # SMMF factorization
        vector_reshape: bool = False,
        nnmf_factor: bool = False,
        # CANS
        accelerated_ns: bool = False,
        cns_a_bound: float | None = None,
        # MARS-M
        approx_mars: bool = False,
        mars_gamma: float = 0.025,
        # Spectral Normalization
        n_layers: int = 1,
        spectral_normalization: bool = False,
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        # torch.compile
        compiled_optimizer: bool = False,
        # --- AdamW_adv specific parameters ---
        adam_betas: tuple[float, float] = (0.9, 0.99),
        adam_eps: float | None = 1e-8,
        adam_weight_decay: float = 0.0,
        adam_fisher_wd: bool = False,
        adam_use_bias_correction: bool = True,
        adam_use_atan2: bool = False,
        adam_orthogonal_gradient: bool = False,
        adam_nesterov: bool = False,
        adam_nesterov_coef: float | None = None,
        adam_kourkoutas_beta: bool = False,
        adam_beta2_min: float = 0.9,
        adam_ema_alpha: float = 0.95,
        adam_tiny_spike: float = 1e-9,
        adam_k_warmup_steps: int = 0,
        adam_spectral_normalization: bool = False,
        adam_state_precision: str = "auto",
        adam_nnmf_factor: bool = False,
        adam_factored_2nd: bool = False,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (ns_steps > 0):
            raise ValueError(f"Newton-Schulz steps should be > 0. Got {ns_steps}")
        if spectral_normalization and rms_rescaling:
            print("Warning: spectral_normalization is incompatible with rms_rescaling, Disabling rms_rescaling.")
            rms_rescaling = False
        if spectral_normalization and accelerated_ns:
            ValueError("spectral_normalization violates accelerated Newton-Schulz assumptions. Pick one of them.")

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp16", "fp8_sr", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        adam_state_precision = adam_state_precision.lower()
        if adam_state_precision not in valid_precisions:
            raise ValueError(f"adam_state_precision must be one of {valid_precisions}. Got {adam_state_precision}")

        defaults = {
            "lr": lr, "betas": betas, "weight_decay": weight_decay, "cautious_wd": cautious_wd,
            "eps": eps, "rms_rescaling": rms_rescaling, "ns_steps": ns_steps,
            "ns_eps": ns_eps, "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
            "nesterov":nesterov, "nesterov_coef": nesterov_coef, "use_atan2":use_atan2,
            "normuon_variant": normuon_variant, "orthogonal_gradient": orthogonal_gradient,
            "compiled_optimizer":compiled_optimizer,
            "use_muon": use_muon,
            # States precision (Muon path)
            "state_precision": state_precision,
            # Factorized second moment only (Muon path)
            "factored_2nd": factored_2nd,
            # Lion-K
            "kappa_p": kappa_p, "auto_projection": auto_projection,
            # Low-rank Ortho
            "low_rank_ortho": low_rank_ortho, "ortho_rank": ortho_rank,
            # CANS
            "accelerated_ns": accelerated_ns, "cns_a_bound": cns_a_bound,
            # MARS-M
            "approx_mars": approx_mars, "mars_gamma": mars_gamma,
            # Spectral Normalization
            "n_layers": n_layers, "spectral_normalization": spectral_normalization,
            # Centered WD
            "centered_wd": centered_wd,
            "centered_wd_mode": centered_wd_mode,
            # AdamW_adv defaults
            "adam_betas": adam_betas, "adam_eps": adam_eps, "adam_weight_decay": adam_weight_decay,
            "adam_fisher_wd": adam_fisher_wd,
            "adam_use_bias_correction": adam_use_bias_correction, "adam_use_atan2": adam_use_atan2,
            "adam_orthogonal_gradient": adam_orthogonal_gradient,
            "adam_nesterov": adam_nesterov, "adam_nesterov_coef": adam_nesterov_coef,
            "adam_kourkoutas_beta": adam_kourkoutas_beta, "adam_beta2_min": adam_beta2_min,
            "adam_ema_alpha": adam_ema_alpha, "adam_tiny_spike": adam_tiny_spike,
            "adam_k_warmup_steps": adam_k_warmup_steps,
            "adam_spectral_normalization": adam_spectral_normalization,
            "adam_state_precision": adam_state_precision,
            "adam_nnmf_factor": adam_nnmf_factor, "adam_factored_2nd": adam_factored_2nd,
        }
        self.stochastic_rounding = stochastic_rounding
        self._init_lr = lr

        super().__init__(params, defaults)

        self.init_step()

        self.kourkoutas_helper = None
        if any(group.get('adam_kourkoutas_beta', False) for group in self.param_groups):
            self.kourkoutas_helper = KourkoutasHelper(self)

        if self.stochastic_rounding:
            # For deterministic stochastic rounding, we need to seed the generator
            # for each device used by the parameters.
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        # Initialize compiled function
        self._compiled_muon_step_parameter = None
        self._compiled_adam_step_parameter = None
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
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.__init_state(p, group)

    @torch.no_grad()
    def __init_state(self, p, group):
        state = self.state[p]

        if 'is_muon' in state:
            return

        if group.get('use_muon') is not None:
            state['is_muon'] = group['use_muon']
        elif group.get('optim_type') is not None:
            state['is_muon'] = group['optim_type'] == 'muon'
        else: # Auto-detect per parameter
            state['is_muon'] = _is_suitable_for_muon(p)

        if state['is_muon']:

            state['factored'] = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )
            dtype = torch.float32 if state['factored'] else p.dtype
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                state['mu_mbuf_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                state['mv_mbuf_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                packed_d2 = (d2 + 7) // 8
                state['sign_buf'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
                if not group['normuon_variant']:
                    state['mu_vbuf_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_vbuf_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:
                # Determine effective state precision (small tensors always use fp32)
                req_precision = group.get('state_precision', 'auto')
                actual_precision = 'auto' if req_precision == 'factored' else req_precision
                group['actual_state_precision'] = actual_precision

                # factored_2nd: factorize v_t only; ignored for NorMuon (no v_t) and tiny params
                use_factored_2nd = (
                    group.get('factored_2nd', False)
                    and not group['normuon_variant']
                    and not (len(p.shape) == 1 and not group['vector_reshape'])
                )
                state['factored_2nd'] = use_factored_2nd

                default_dtype = p.dtype
                init_state_tensor(state, 'momentum_buffer', p.shape, actual_precision, p.device, default_dtype)

                if use_factored_2nd:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
                    state['mu_vbuf_nmf'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                    state['mv_vbuf_nmf'] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                elif not group['normuon_variant']:
                    init_state_tensor(state, 'second_momentum_buffer', p.shape, actual_precision, p.device, default_dtype, non_neg=True)

            # NorMuon state initialization
            if group['normuon_variant']:
                if state['factored']:
                    d1, _ = state['effective_shape']
                    state['normuon_v'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                elif len(p.shape) >= 2:
                    state['normuon_v'] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)

            # Spectral Normalization
            if group.get('spectral_normalization', False):
                init_spectral_norm(state, p)

            # MARS-M state initialization
            if group.get('approx_mars', False):
                # Note: This requires full-rank memory even if factored
                state['last_grad'] = torch.zeros_like(p, device=device, dtype=p.dtype)

            _init_anchor(p, state, group)

            group['adam_kourkoutas_beta'] = False
            state['is_muon'] = True # Workaround as group was acting weirdly; passing muon params in adam path

        else: # AdamW
            Muon_AuxAdam._init_auxadam_state(self, p, group)
            state['is_muon'] = False

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        grad = p.grad
        if grad is None:
            return

        state = self.state[p]

        self.__init_state(p, group)

        is_compiled = group.get('compiled_optimizer', False)

        random_int_tensor = None
        if p.dtype == torch.bfloat16 and self.stochastic_rounding and is_compiled:
            # Pre-generate random tensor for stochastic rounding if needed.
            random_int_tensor = param_update._get_random_int_for_sr(p)

        if not state['is_muon']: # AdamW path
            step = state['step']

            beta1_adam, beta2_adam = group['adam_betas']

            if self.kourkoutas_helper:
                # Prepare Kourkoutas-β once per optimizer step.
                self.kourkoutas_helper.maybe_prepare_step(step, p.device)
                # Get the dynamic beta2_adam calculated in prepare_step()
                beta2_adam = self.kourkoutas_helper.get_beta2(p, group)

            if group['adam_use_bias_correction']:
                current_step = step + 1
                beta1_adam, beta2_adam = group['adam_betas']
                bias_correction1 = 1.0 - beta1_adam ** current_step
                sqrt_bias_correction2 = (1.0 - beta2_adam ** current_step)**0.5
            else:
                bias_correction1 = 1.0
                sqrt_bias_correction2 = 1.0

            step_size = group['lr'] / bias_correction1

            random_int_state_tensor = None
            if is_compiled:
                step_size = torch.as_tensor(step_size)
                adam_step_param = self._compiled_adam_step_parameter
                
                # Generate state SR random tensor when compiled
                actual_precision = group.get('adam_actual_state_precision', 'auto')
                random_int_state_tensor = random_int_tensor
                if actual_precision == 'bf16_sr' and random_int_state_tensor is None:
                    random_int_state_tensor = param_update._get_random_int_for_sr(p)
                elif actual_precision == 'int8_sr':
                    random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
                elif actual_precision == 'fp8_sr':
                    random_int_state_tensor = param_update._get_random_int_for_fp8_sr(p)
            else:
                adam_step_param = Muon_AuxAdam._adam_step_parameter

            adam_step_param(self, p, grad, state, group, beta1_adam, beta2_adam, sqrt_bias_correction2, step_size, random_int_tensor, random_int_state_tensor)

            state['step'] += 1

        else: # Muon path
            if is_compiled:
                lr = torch.as_tensor(group['lr'])
                muon_step_param = self._compiled_muon_step_parameter

                # Generate state SR random tensor when compiled
                actual_precision = group['actual_state_precision']
                random_int_state_tensor = random_int_tensor
                if actual_precision == 'bf16_sr' and random_int_state_tensor is None:
                    random_int_state_tensor = param_update._get_random_int_for_sr(p)
                elif actual_precision == 'int8_sr':
                    random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
                elif actual_precision == 'fp8_sr':
                    random_int_state_tensor = param_update._get_random_int_for_fp8_sr(p)
                if group['low_rank_ortho']:
                    random_G_sketch = param_update._get_random_noise_for_low_rank_ortho(p, group['ortho_rank'])
            else:
                lr = group['lr']
                random_int_state_tensor = None
                random_G_sketch = None
                muon_step_param = self._muon_step_parameter

            muon_step_param(p, grad, state, group, lr, random_int_tensor, random_int_state_tensor, random_G_sketch)

    def compile(self, *args, **kwargs):
        self._compiled_muon_step_parameter = torch.compile(self._muon_step_parameter, *args, **kwargs)
        self._compiled_adam_step_parameter = torch.compile(Muon_AuxAdam._adam_step_parameter, *args, **kwargs)

    @torch.no_grad()
    def _muon_step_parameter(self, p, grad, state, group, lr, random_int_tensor, random_int_state_tensor, random_G_sketch):
        # Upcast grad for low-precision state modes (non-factored path)
        grad = upcast_grad_for_precision(grad, state, group.get('state_precision', 'auto'))
        beta1, beta2 = group['betas']
        nesterov = group['nesterov']
        nesterov_coef = group.get('nesterov_coef', None)

        # Update geometry
        kappa_p = group.get("kappa_p", 1.0)
        if group.get("auto_projection", False):
            # Auto projection depending on dimension
            # Apply p=2.0 (Spherical) for 4D (Conv2D)
            # Apply p=1.0 (Sign) for everything else (Linear/Embeddings)
            if p.ndim >= 4:
                kappa_p = 2.0
            else:
                kappa_p = 1.0

        ns_eps = group['ns_eps']
        adaptive_eps = scale_eps(group['eps'], p)

        # MARS-M Approximated (Variance Reduction)
        if group.get('approx_mars', False):
            grad = approx_mars(grad, state['last_grad'], group['mars_gamma'], beta1)


        if group.get("orthogonal_gradient"):
            grad = _orthogonalize_gradient(p, grad)

        if state['factored']: # Factored Muon
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors & sign
            mt_buf = _reconstruct_state((state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'], d2), signed=True, shifter=state['shifter'])

            # Update momentum in full-size
            mt_buf.lerp_(grad_reshaped, 1 - beta1)

            if nesterov:
                nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                update = grad_reshaped.lerp(mt_buf, nv_coef)
            else:
                update = mt_buf.clone()

            # Factorize
            state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'] = _factorize_state(mt_buf, signed=True, shifter=state['shifter'])
            del mt_buf

            # Apply update projection
            update = _auto_projection_for_adamuon(update, kappa_p)

            # Orthogonalization step
            update = newton_schulz(
                update,
                steps=group['ns_steps'],
                eps=ns_eps,
                coeffs=group['ns_coeffs'],
                cns=group['accelerated_ns'],
                cns_a_bound=group['cns_a_bound'],
                low_rank_ortho=group['low_rank_ortho'],
                ortho_rank=group['ortho_rank'],
                G_sketch=random_G_sketch,
                compiled=group.get('compiled_optimizer', False)
            )

            if group['normuon_variant']:
                normuon_update(update, state['normuon_v'], beta2, group['eps'])
            else:
                # Reconstruct second momentum from previous step's factors
                vt_buf = _reconstruct_state((state['mu_vbuf_nmf'], state['mv_vbuf_nmf']), signed=False, shifter=state['shifter'])
                # Update second momentum in full-size
                vt_buf.mul_(beta2).addcmul_(update, update, value=1 - beta2)
                state['mu_vbuf_nmf'], state['mv_vbuf_nmf'] = _factorize_state(vt_buf, signed=False, shifter=state['shifter'])
                # Apply second momentum update (adaptive scaling)
                if group['use_atan2']:
                    denom = vt_buf.sqrt_()
                    update.atan2_(denom)
                else:
                    denom = vt_buf.sqrt_().add_(adaptive_eps)
                    update.div_(denom)
                del denom, vt_buf

            update = update.reshape(p.shape)

        else: # Standard AdaMuon logic for non-factored tensors
            original_shape = p.shape
            actual_precision = group['actual_state_precision']
            factored_2nd = state.get('factored_2nd', False)

            # Momentum update
            mt_buf = get_state(state, 'momentum_buffer', actual_precision)
            mt_buf.lerp_(grad, 1 - beta1)

            if nesterov:
                nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                update = grad.lerp(mt_buf, nv_coef)
            else:
                update = mt_buf.clone()

            set_state(state, 'momentum_buffer', mt_buf, actual_precision, random_int_state_tensor)

            # Apply update projection
            update = _auto_projection_for_adamuon(update, kappa_p)

            # Flatten if necessary (e.g., for Conv layers)
            update = update.flatten(1)

            # Orthogonalization step
            update = newton_schulz(
                update,
                steps=group['ns_steps'],
                eps=ns_eps,
                coeffs=group['ns_coeffs'],
                cns=group['accelerated_ns'],
                cns_a_bound=group['cns_a_bound'],
                low_rank_ortho=group['low_rank_ortho'],
                ortho_rank=group['ortho_rank'],
                G_sketch=random_G_sketch,
                compiled=group.get('compiled_optimizer', False)
            )

            # NorMuon Logic
            if group['normuon_variant']:
                normuon_update(update, state['normuon_v'], beta2, group['eps'])
            elif factored_2nd:
                # Factorized second moment: reconstruct → update → re-factorize
                d1, d2 = state['effective_shape']
                update = update.view(original_shape)
                update_f32 = update.float()
                vt_buf = _reconstruct_state((state['mu_vbuf_nmf'], state['mv_vbuf_nmf']), signed=False, shifter=state['shifter'])
                vt_buf.mul_(beta2).addcmul_(update_f32.view(d1, d2), update_f32.view(d1, d2), value=1 - beta2)
                state['mu_vbuf_nmf'], state['mv_vbuf_nmf'] = _factorize_state(vt_buf, signed=False, shifter=state['shifter'])
                # Apply second moment scaling
                if group['use_atan2']:
                    denom = vt_buf.sqrt_().view(original_shape)
                    update.atan2_(denom.to(update.dtype))
                else:
                    denom = vt_buf.sqrt_().add_(adaptive_eps).view(original_shape)
                    update.div_(denom.to(update.dtype))
                del denom, vt_buf, update_f32
            else:
                # Original AdaMuon Logic
                update = update.view(original_shape)
                vt_buf = get_state(state, 'second_momentum_buffer', actual_precision)
                vt_buf.mul_(beta2).addcmul_(update, update, value=1 - beta2)
                # Apply second momentum update (adaptive scaling)
                if group['use_atan2']:
                    denom = vt_buf.sqrt()
                    update.atan2_(denom)
                else:
                    denom = vt_buf.sqrt().add_(adaptive_eps)
                    update.div_(denom)
                set_state(state, 'second_momentum_buffer', vt_buf, actual_precision, random_int_state_tensor, non_neg=True)
                del denom

        step_scale = lr * A if group['use_atan2'] and not group['normuon_variant'] else lr

        if group.get('spectral_normalization', False):
            # Spectral Normalization
            spectral_normalization(update, state['spectral_u'], state['spectral_v'], step_scale)
        else:
            # RMS-aligned rescaling
            rms_adjustment(update, group['rms_rescaling'], step_scale)

        update = update.reshape(original_shape)

        param_update.apply_parameter_update(self, p, group, update, lr, random_int_tensor=random_int_tensor)

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
