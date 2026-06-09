import torch

from ..util import param_update
from ..util.Muon_util import newton_schulz, _is_suitable_for_muon, rms_adjustment, normuon_update, approx_mars
from ..util.scaled_optm import spectral_normalization, init_spectral_norm
from ..util.factorization_util import _get_effective_shape, _factorize_state, _reconstruct_state
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util import Muon_AuxAdam
from ..util.centered_decay import _init_anchor
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

class Muon_adv(torch.optim.Optimizer):
    """
    Implements an advanced Muon algorithm, with an integrated auxiliary AdamW optimizer.

    Muon (MomentUm Orthogonalized by Newton-Schulz) is an optimizer designed for
    the hidden layers of neural networks. It applies SGD with momentum and then
    orthogonalizes the resulting update matrix using a Newton-Schulz iteration.

    When `MuonWithAuxAdam` is enabled, this single optimizer class handles both
    'muon' and 'adam' parameter groups, dispatching to the appropriate logic internally.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float): momentum factor (default: 0.9).
        weight_decay (float): weight decay (L2 penalty) (default: 0).
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        nesterov (bool): enables Nesterov momentum (default: True).
        ns_steps (int): number of Newton-Schulz iterations to perform (default: 5).
        ns_eps (float): epsilon for Newton-Schulz normalization stability. When None
            it's derived from scale invariant rule (default: 1e-7).
        ns_coeffs (tuple[float, float, float]): The (a, b, c) coefficients for the
            quintic polynomial in the Newton-Schulz iteration.
            (default: (3.4445, -4.7750, 2.0315)).
        stochastic_rounding (bool): whether to use stochastic rounding for
            BF16 parameter updates (default: True).
        orthogonal_gradient (str): whether to use OrthoGrad variants. 'disabled': off.
        'flattened': Standard vectorized OrthoGrad. 'iterative': Matrix-wise rank-2 OrthoGrad. (default: disabled)
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
        use_muon (bool | None): whether to use Muon or AuxAdamW. MUST be provided
            either here or via `optim_type` in parameter groups. (default: None)
        state_precision (str): Precision for Muon optimizer states. Options: 'auto' (parameter dtype), 'fp32',
            'bf16_sr' (BF16 with stochastic rounding), 'int8_sr'.
            (default: 'auto')
        low_rank_ortho (bool): If True, enables low-rank orthogonalization, which
            projects the update to a lower rank before orthogonalization.
            (default: False)
        ortho_rank (int): The rank for low-rank orthogonalization.
            (default: 128)
        normuon_variant (bool): If True, enables the NorMuon update rule, which adds
            neuron-wise normalization. (default: False)
        beta2_normuon (float): The exponential decay rate for the second moment estimates
            used in NorMuon. (default: 0.95)
        normuon_eps (float): Epsilon for NorMuon normalization stability. (default: 1e-8)
        rms_rescaling (bool): Use Root-Mean-Square for the final update
            vector, used for RMS-aligned rescaling. Allows for the reuse of existing Adam
            learning rate schedules. (default: True).
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
        n_layers (int): The depth of the network (L). Required for optimal epsilon scaling. (default: 1)
        spectral_normalization (bool): Enable explicit spectral normalization using power iteration. (default: False)
        --- Auxiliary AdamW_adv Parameters (used for 'adam' groups) ---
        adam_betas (tuple[float, float]): Betas for the AdamW optimizer part.
        adam_eps (float): Epsilon for the AdamW optimizer part.
        adam_weight_decay (float): Weight decay for the AdamW optimizer part.
        adam_fisher_wd (bool): Fisher Adam (FAdam) weight decay for the AdamW part. (default: False)
        adam_use_bias_correction (bool): Bias correction for AdamW.
        adam_use_atan2 (bool): Atan2 update rule for AdamW.
        adam_orthogonal_gradient (str): OrthoGrad for AdamW.
        adam_nesterov (bool): Nesterov momentum for AdamW. (default: False)
        adam_nesterov_coef (float, optional): Nesterov coefficient for AdamW. (default: None)
        adam_kourkoutas_beta (bool): Kourkoutas-β for AdamW.
        adam_beta2_min (float): Minimum beta2 for Kourkoutas-β. (default: 0.9)
        adam_ema_alpha (float): EMA alpha for Kourkoutas-β. (default: 0.95)
        adam_tiny_spike (float): Tiny spike for Kourkoutas-β. (default: 1e-9)
        adam_k_warmup_steps (int): Warmup steps for Kourkoutas-β. (default: 0)
        adam_spectral_normalization (bool): Enable explicit spectral normalization for AdamW. (default: False)
        adam_state_precision (str): Precision for AuxAdam states. Options: 'auto', 'fp32', 'bf16_sr', 'fp16', 'int8_sr', 'factored'. (default: 'auto')
        adam_nnmf_factor (bool): 1-bit factored for AdamW.
        adam_factored_2nd (bool): Factorize only the second moment (v_t) for AuxAdam. (default: False)
        """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.95,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        # Nesterov momentum
        nesterov: bool = True,
        nesterov_coef: float | None = None,
        # Newton Schulz
        ns_steps: int = 5,
        ns_eps: float | None = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: str = 'disabled', # 'flattened', 'iterative'
        # RMS Rescaling
        rms_rescaling: bool = True,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # Boolean to spilt param
        use_muon: bool | None = None,
        # States precision (Muon path)
        state_precision: str = "auto",  # 'fp32', 'bf16_sr', 'int8_sr'
        # Low-rank Muon
        low_rank_ortho: bool = False,
        ortho_rank: int = 128,
        # NorMuon
        normuon_variant: bool = False,
        beta2_normuon: float = 0.95,
        normuon_eps: float = 1e-8,
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
        adam_orthogonal_gradient: str = 'disabled', # 'flattened', 'iterative'
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
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 should be in [0.0, 1.0). Got {beta1}")
        if normuon_variant and not (0.0 <= beta2_normuon < 1.0):
            raise ValueError(f"beta2_normuon should be in [0.0, 1.0) for NorMuon. Got {beta2_normuon}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (ns_steps > 0):
            raise ValueError(f"Newton-Schulz steps should be > 0. Got {ns_steps}")
        if spectral_normalization and rms_rescaling:
            print("Warning: spectral_normalization is incompatible with rms_rescaling, Disabling rms_rescaling.")
            rms_rescaling = False
        if spectral_normalization and accelerated_ns:
            raise ValueError("spectral_normalization violates accelerated Newton-Schulz assumptions. Pick one of them.")

        # Legacy backwards compatibility support for `nnmf_factor=True`
        if nnmf_factor:
            state_precision = "factored"

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp16", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        adam_state_precision = adam_state_precision.lower()
        if adam_state_precision not in valid_precisions:
            raise ValueError(f"adam_state_precision must be one of {valid_precisions}. Got {adam_state_precision}")

        defaults = {
            "lr": lr, "beta1": beta1, "weight_decay": weight_decay, "cautious_wd": cautious_wd,
            "nesterov": nesterov, "nesterov_coef": nesterov_coef, "ns_steps": ns_steps, "ns_eps": ns_eps,
            "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,  "rms_rescaling": rms_rescaling,
            "orthogonal_gradient": orthogonal_gradient,
            'compiled_optimizer': compiled_optimizer,
            "use_muon": use_muon,
            # States precision (Muon path)
            "state_precision": state_precision,
            # Low-rank Ortho
            "low_rank_ortho": low_rank_ortho, "ortho_rank": ortho_rank,
            # NorMuon
            "normuon_variant": normuon_variant, "beta2_normuon": beta2_normuon,
            "normuon_eps": normuon_eps,
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
        self.compiled_optimizer = compiled_optimizer
        self._init_lr = lr if lr > 0 else 1

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

            req_precision = group['state_precision']
            is_vector = len(p.shape) == 1 and not group['vector_reshape']

            state['factored'] = req_precision == 'factored' and not is_vector
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
            else:
                # Determine effective state precision (small tensors always use fp32)
                req_precision = group.get('state_precision', 'auto')
                actual_precision = req_precision
                group['actual_state_precision'] = actual_precision

                default_dtype = p.dtype
                init_state_tensor(state, 'momentum_buffer', p.shape, actual_precision, p.device, default_dtype)

            # Spectral Normalization
            if group.get('spectral_normalization', False):
                init_spectral_norm(state, p)

            # MARS-M state initialization
            if group.get('approx_mars', False):
                # Note: This requires full-rank memory even if factored
                state['last_grad'] = torch.zeros_like(p, device=device, dtype=p.dtype)

            # NorMuon state initialization
            if group['normuon_variant']:
                if state['factored']:
                    d1, _ = state['effective_shape']
                    state['normuon_v'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                elif len(p.shape) >= 2:
                    state['normuon_v'] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)

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
                
                actual_precision = group.get('adam_actual_state_precision', 'auto')
                random_int_state_tensor = random_int_tensor
                if actual_precision == 'bf16_sr' and random_int_state_tensor is None:
                    random_int_state_tensor = param_update._get_random_int_for_sr(p)
                elif actual_precision == 'int8_sr':
                    random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
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

        beta1 = group['beta1']
        nesterov = group['nesterov']
        nesterov_coef = group.get('nesterov_coef', None)

        ns_eps = group['ns_eps']

        # MARS-M Approximated (Variance Reduction)
        if group.get('approx_mars', False):
            grad = approx_mars(grad, state['last_grad'], group['mars_gamma'], beta1)

        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()

        grad = _orthogonalize_gradient(p, grad, group.get("orthogonal_gradient"))

        if state['factored']: # Factored Muon
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors & sign
            mt_buf = _reconstruct_state((state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'], d2), signed=True, shifter=state['shifter'])

            # Update momentum in full-size
            mt_buf.lerp_(grad_reshaped, 1 - beta1)

            if nesterov:
                # Nesterov momentum
                nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                update = grad_reshaped.lerp(mt_buf, nv_coef)
            else:
                # Standard momentum
                update = mt_buf.clone()

            # Factorize
            state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'] = _factorize_state(mt_buf, signed=True, shifter=state['shifter'])
            del mt_buf

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
                normuon_update(update, state['normuon_v'], group['beta2_normuon'], group['normuon_eps'])

            update = update.reshape(p.shape)

        else: # Standard Muon logic for non-factored tensors

            if len(p.shape) >= 2:

                original_shape = p.shape
                actual_precision = group['actual_state_precision']

                # Momentum update
                mt_buf = get_state(state, 'momentum_buffer', actual_precision)
                mt_buf.lerp_(grad, 1 - beta1)

                if nesterov:
                    # Nesterov momentum
                    nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                    update = grad.lerp(mt_buf, nv_coef)
                else:
                    # Standard momentum
                    update = mt_buf.clone()

                set_state(state, 'momentum_buffer', mt_buf, actual_precision, random_int_state_tensor)

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
                    normuon_update(update, state['normuon_v'], group['beta2_normuon'], group['normuon_eps'])

        step_scale = lr * rms_adjustment(update, group['rms_rescaling']) if not group.get('spectral_normalization', False) else lr

        update = update.reshape(original_shape)

        param_update.apply_parameter_update(self, p, group, update, lr, step_size=step_scale, random_int_tensor=random_int_tensor)

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
