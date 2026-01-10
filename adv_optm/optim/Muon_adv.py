import torch

from ..util import param_update
from ..util.Muon_util import newton_schulz, _is_suitable_for_muon, rms_adjustment, normuon_update, approx_mars, spectral_norm_update, get_spectral_scaling
from ..util.factorization_util import _get_effective_shape, _factorize_state, _reconstruct_state
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util import Muon_AuxAdam

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
        ns_eps (float): epsilon for Newton-Schulz normalization stability (default: 1e-7).
        ns_coeffs (tuple[float, float, float]): The (a, b, c) coefficients for the
            quintic polynomial in the Newton-Schulz iteration.
            (default: (3.4445, -4.7750, 2.0315)).
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule.
            This changes the update  to `alpha_grad * grad + mt`, which can be
            more responsive, especially for small batch sizes. (default: False)
        alpha_grad (float): Mixing coefficient for the Simplified AdEMAMix update rule
            (only used when `Simplified_AdEMAMix` is `True`). Controls the weight of the
            current gradient. For small batch sizes, use high values (e.g., 10-100) to be
            more responsive. For large batch sizes, use low values (e.g., 0-1) for
            stability. (default: 100.0)
        stochastic_rounding (bool): whether to use stochastic rounding for
            BF16 parameter updates (default: True).
        orthogonal_gradient (bool): whether to use OrthoGrad.  (default: False)
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
        use_muon (bool | None): whether to use Muon or AuxAdamW. MUST be provided
            either here or via `optim_type` in parameter groups. (default: None)
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
        cns_a_bound (float): Initial lower bound for singular values for CANS. (default: 1e-4)
        approx_mars (bool): If True, enables Approximated MARS-M variance reduction.
        fom the paper "MARS-M: When Variance Reduction Meets Matrices"
            (default: False)
        mars_gamma (float): The scaling coefficient for MARS gradient correction.
            (default: 0.025)
        n_layers (int): The depth of the network (L). Required for optimal epsilon scaling. (default: 1)
        spectral_normalization (bool): Enable explicit spectral normalization using power iteration. (default: True)
        --- Auxiliary AdamW_adv Parameters (used for 'adam' groups) ---
        adam_betas (tuple[float, float]): Betas for the AdamW optimizer part.
        adam_eps (float): Epsilon for the AdamW optimizer part.
        adam_weight_decay (float): Weight decay for the AdamW optimizer part.
        adam_use_bias_correction (bool): Bias correction for AdamW.
        adam_use_atan2 (bool): Atan2 update rule for AdamW.
        adam_cautious_mask (bool): Cautious masking for AdamW.
        adam_grams_moment (bool): Grams-style updates for AdamW.
        adam_orthogonal_gradient (bool): OrthoGrad for AdamW.
        adam_use_AdEMAMix (bool): AdEMAMix for AdamW.
        adam_beta3_ema (float): Beta3 for AdEMAMix.
        adam_alpha (float): Alpha for AdEMAMix.
        adam_kourkoutas_beta (bool): Kourkoutas-β for AdamW.
        adam_nnmf_factor (bool): 1-bit factored for AdamW.
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
        # Newton Schulz
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # RMS Rescaling
        rms_rescaling: bool = True,
        # One-EMA AdEMAMix
        Simplified_AdEMAMix: bool = False,
        alpha_grad: float = 100.0,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # Boolean to spilt param
        use_muon: bool | None = None,
        # Low-rank Muon
        low_rank_ortho: bool = False,
        ortho_rank: int = 128,
        # NorMuon
        normuon_variant: bool = False,
        beta2_normuon: float = 0.95,
        normuon_eps: float = 1e-8,
        # CANS
        accelerated_ns: bool = False,
        cns_a_bound: float = 1e-4,
        # MARS-M
        approx_mars: bool = False,
        mars_gamma: float = 0.025,
        # Spectral Normalization
        n_layers: int = 1,
        spectral_normalization: bool = True,
        # torch.compile
        compiled_optimizer: bool = False,
        # --- AdamW_adv specific parameters ---
        adam_betas: tuple[float, float] = (0.9, 0.99),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.0,
        adam_use_bias_correction: bool = True,
        adam_use_atan2: bool = False,
        adam_cautious_mask: bool = False,
        adam_grams_moment: bool = False,
        adam_orthogonal_gradient: bool = False,
        adam_use_AdEMAMix: bool = False,
        adam_beta3_ema: float = 0.9999,
        adam_alpha: float = 5.0,
        adam_kourkoutas_beta: bool = False,
        adam_beta2_min: float = 0.9,
        adam_ema_alpha: float = 0.95,
        adam_tiny_spike: float = 1e-9,
        adam_k_warmup_steps: int = 0,
        adam_nnmf_factor: bool = False,
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
        if Simplified_AdEMAMix and nesterov:
            print("Warning: nesterov is incompatible with Simplified_AdEMAMix, Disabling nesterov.")
            nesterov = False

        defaults = {
            "lr": lr, "beta1": beta1, "weight_decay": weight_decay, "cautious_wd": cautious_wd,
            "nesterov": nesterov, "ns_steps": ns_steps, "ns_eps": ns_eps,
            "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,  "rms_rescaling": rms_rescaling,
            "Simplified_AdEMAMix": Simplified_AdEMAMix, "alpha_grad": alpha_grad,
            "orthogonal_gradient": orthogonal_gradient,
            'compiled_optimizer': compiled_optimizer,
            "use_muon": use_muon,
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
            # AdamW_adv defaults
            "adam_betas": adam_betas, "adam_eps": adam_eps, "adam_weight_decay": adam_weight_decay,
            "adam_use_bias_correction": adam_use_bias_correction, "adam_use_atan2": adam_use_atan2,
            "adam_cautious_mask": adam_cautious_mask, "adam_grams_moment": adam_grams_moment,
            "adam_orthogonal_gradient": adam_orthogonal_gradient,
            "adam_use_AdEMAMix": adam_use_AdEMAMix, "adam_beta3_ema": adam_beta3_ema, "adam_alpha": adam_alpha,
            "adam_kourkoutas_beta": adam_kourkoutas_beta, "adam_beta2_min": adam_beta2_min,
            "adam_ema_alpha": adam_ema_alpha, "adam_tiny_spike": adam_tiny_spike,
            "adam_k_warmup_steps": adam_k_warmup_steps,
            "adam_nnmf_factor":adam_nnmf_factor,
        }
        self.stochastic_rounding = stochastic_rounding
        self.compiled_optimizer = compiled_optimizer

        super().__init__(params, defaults)

        # Validate that every group has a determined optimizer type
        for i, group in enumerate(self.param_groups):
            if group.get('use_muon') is None and group.get('optim_type') is None:
                # Automatic shape-based detection if not explicit
                has_muon_shape = False
                for p in group['params']:
                    has_muon_shape = _is_suitable_for_muon(p)
                    if has_muon_shape:
                        group['use_muon'] = True
                    else:
                        group['use_muon'] = False

            if group.get('use_muon') is None: # Fallback
                 group['use_muon'] = group.get('optim_type') == 'muon'

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

        if group['use_muon']:

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
            else:
                state['momentum_buffer'] = torch.zeros_like(p)

            # Spectral Normalization
            if group.get('spectral_normalization', False):
                # Case A: Factored Muon
                if state['factored']:
                    d1, d2 = state['effective_shape']
                    # We need a vector matching the 'inner' dimension d2
                    state['spectral_v'] = torch.randn(d2, device=device, dtype=torch.float32)

                # Case B: Standard Muon (Linear, Conv2d, etc.)
                elif len(p.shape) >= 2:
                    # Since Muon performs `update.flatten(1)`, the matrix becomes
                    # (p.shape[0], product_of_rest).
                    d_in_flat = p.numel() // p.shape[0]
                    
                    state['spectral_v'] = torch.randn(d_in_flat, device=device, dtype=torch.float32)

                # Normalize initial vector for stability
                if 'spectral_v' in state:
                     state['spectral_v'].div_(state['spectral_v'].norm())

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

            if is_compiled:
                step_size = torch.as_tensor(step_size, dtype=torch.float64)
                adam_step_param = self._compiled_adam_step_parameter
            else:
                adam_step_param = Muon_AuxAdam._adam_step_parameter

            adam_step_param(self, p, grad, state, group, beta1_adam, beta2_adam, sqrt_bias_correction2, step_size, random_int_tensor)

            state['step'] += 1

        else: # Muon path
            if is_compiled:
                lr = torch.as_tensor(group['lr'], dtype=torch.float64)
                muon_step_param = self._compiled_muon_step_parameter
            else:
                lr = group['lr']
                muon_step_param = self._muon_step_parameter
            
            muon_step_param(p, grad, state, group, lr, random_int_tensor)

    def compile(self, *args, **kwargs):
        self._compiled_muon_step_parameter = torch.compile(self._muon_step_parameter, *args, **kwargs)
        self._compiled_adam_step_parameter = torch.compile(Muon_AuxAdam._adam_step_parameter, *args, **kwargs)

    @torch.no_grad()
    def _muon_step_parameter(self, p, grad, state, group, lr, random_int_tensor):


        beta1 = group['beta1']
        nesterov = group['nesterov']
        Simplified_AdEMAMix = group['Simplified_AdEMAMix']
        alpha_grad = group['alpha_grad']

        if group.get('spectral_normalization', False):
            # Compute Scaling Factors
            if state['factored']:
                shape_for_scaling = torch.Size(state['effective_shape'])
            else:
                shape_for_scaling = p.shape

            scaled_eps, normuon_eps, spectral_target, wd_scale = get_spectral_scaling(shape_for_scaling, group['n_layers'])

            weight_decay = group['weight_decay'] * wd_scale
            ns_eps = scaled_eps
        else:
            weight_decay = group['weight_decay']
            ns_eps = group['ns_eps']
            normuon_eps = group['normuon_eps']

        # MARS-M Approximated (Variance Reduction)
        if group.get('approx_mars', False):
            grad = approx_mars(grad, state['last_grad'], group['mars_gamma'], beta1, Simplified_AdEMAMix=Simplified_AdEMAMix)

        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()

        if group.get("orthogonal_gradient"):
            grad = _orthogonalize_gradient(p, grad)

        if state['factored']: # Factored Muon
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors & sign
            mt_buf = _reconstruct_state((state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'], d2), signed=True)

            # Update momentum in full-size
            if not Simplified_AdEMAMix:
                mt_buf.lerp_(grad_reshaped, 1 - beta1)
            else:
                mt_buf.mul_(beta1).add_(grad_reshaped)

            if nesterov:
                # Nesterov momentum
                update = grad_reshaped.lerp(mt_buf, beta1)
            elif Simplified_AdEMAMix:
                update = torch.add(mt_buf, grad_reshaped, alpha=alpha_grad)
            else:
                # Standard momentum
                update = mt_buf.clone()

            # Factorize
            state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'] = _factorize_state(mt_buf, signed=True)
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
                spectral_normalization=group.get('spectral_normalization', False)
            )

            if group['normuon_variant']:
                normuon_update(update, state['normuon_v'], group['beta2_normuon'], normuon_eps)

            if group.get('spectral_normalization', False):
                # Spectral Normalization
                spectral_norm_update(update, state['spectral_v'], spectral_target, lr)
            else:
                # Factored RMS-aligned scaling
                rms_adjustment(update, group['rms_rescaling'], lr)

            update = update.reshape(p.shape)

        else: # Standard Muon logic for non-factored tensors

            if len(p.shape) >= 2:

                original_shape = p.shape

                # Momentum update
                mt_buf = state['momentum_buffer']
                if not Simplified_AdEMAMix:
                    mt_buf.lerp_(grad, 1 - beta1)
                else:
                    mt_buf.mul_(beta1).add_(grad)

                if nesterov:
                    # Nesterov momentum
                    update = grad.lerp(mt_buf, beta1)
                elif Simplified_AdEMAMix:
                    update = torch.add(mt_buf, grad, alpha=alpha_grad)
                else:
                    # Standard momentum
                    update = mt_buf.clone()

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
                    spectral_normalization=group.get('spectral_normalization', False)
                )

                # NorMuon Logic
                if group['normuon_variant']:
                    normuon_update(update, state['normuon_v'], group['beta2_normuon'], normuon_eps)

                if group.get('spectral_normalization', False):
                    # Spectral Normalization
                    spectral_norm_update(update, state['spectral_v'], spectral_target, lr)
                else:
                    # RMS-aligned rescaling
                    rms_adjustment(update, group['rms_rescaling'], lr)

                update = update.reshape(original_shape)

        param_update.apply_parameter_update(self, p, group, update, lr, wd=weight_decay, random_int_tensor=random_int_tensor)

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
