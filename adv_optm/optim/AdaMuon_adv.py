import torch

import math

from ..util import param_update
from ..util.Muon_util import newton_schulz, _is_suitable_for_muon, rms_adjustment, normuon_update, approx_mars, _auto_projection_for_adamuon
from ..util.factorization_util import _get_effective_shape, _factorize_state, _reconstruct_state
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper
from ..util import Muon_AuxAdam

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
        ns_eps (float): epsilon for Newton-Schulz normalization stability (default: 1e-7).
        ns_coeffs (tuple[float, float, float]): The (a, b, c) coefficients for the
            quintic polynomial in the Newton-Schulz iteration.
            (default: (3.4445, -4.7750, 2.0315)).
        stochastic_rounding (bool): whether to use stochastic rounding for
            BF16 parameter updates (default: True).
        orthogonal_gradient (bool): whether to use OrthoGrad.  (default: False)
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
        cns_a_bound (float): Initial lower bound for singular values for CANS. (default: 1e-4)
        approx_mars (bool): If True, enables Approximated MARS-M variance reduction.
        fom the paper "MARS-M: When Variance Reduction Meets Matrices"
            (default: False)
        mars_gamma (float): The scaling coefficient for MARS gradient correction.
            (default: 0.025)
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
        use_muon (bool | None): whether to use Muon or AuxAdamW. MUST be provided
            either here or via `optim_type` in parameter groups. (default: None)
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
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        # Decoupled/cautious weight decay
        weight_decay: float = 0,
        cautious_wd: bool = False,
        # Nesterov momentum
        nesterov: bool = False,
        # RMS Rescaling
        rms_rescaling: bool = True,
        # Newton Schulz
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # Adam_atan2 (scale invariant)
        use_atan2: bool = False,
        # One-EMA AdEMAMix
        Simplified_AdEMAMix: bool = False,
        alpha_grad: float = 100.0,
        # NorMuon
        normuon_variant: bool = False,
        # Boolean to spilt param
        use_muon: bool | None = None,
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
        cns_a_bound: float = 1e-4,
        # MARS-M
        approx_mars: bool = False,
        mars_gamma: float = 0.025,
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
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (ns_steps > 0):
            raise ValueError(f"Newton-Schulz steps should be > 0. Got {ns_steps}")
        if Simplified_AdEMAMix and nesterov:
            print("Warning: nesterov is incompatible with Simplified_AdEMAMix, Disabling nesterov.")
            nesterov = False

        defaults = {
            "lr": lr, "betas": betas, "weight_decay": weight_decay, "cautious_wd": cautious_wd,
            "eps": eps, "rms_rescaling": rms_rescaling, "ns_steps": ns_steps,
            "ns_eps": ns_eps, "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
            "nesterov":nesterov, "use_atan2":use_atan2,
            "Simplified_AdEMAMix": Simplified_AdEMAMix, "alpha_grad": alpha_grad,
            "normuon_variant": normuon_variant, "orthogonal_gradient": orthogonal_gradient,
            "compiled_optimizer":compiled_optimizer,
            "use_muon": use_muon,
            # Lion-K
            "kappa_p": kappa_p, "auto_projection": auto_projection,
            # Low-rank Ortho
            "low_rank_ortho": low_rank_ortho, "ortho_rank": ortho_rank,
            # CANS
            "accelerated_ns": accelerated_ns, "cns_a_bound": cns_a_bound,
            # MARS-M
            "approx_mars": approx_mars, "mars_gamma": mars_gamma,
            # AdamW_adv defaults
            "adam_betas": adam_betas, "adam_eps": adam_eps, "adam_weight_decay": adam_weight_decay,
            "adam_use_bias_correction": adam_use_bias_correction, "adam_use_atan2": adam_use_atan2,
            "adam_cautious_mask": adam_cautious_mask, "adam_grams_moment": adam_grams_moment,
            "adam_orthogonal_gradient": adam_orthogonal_gradient,
            "adam_use_AdEMAMix": adam_use_AdEMAMix, "adam_beta3_ema": adam_beta3_ema, "adam_alpha": adam_alpha,
            "adam_kourkoutas_beta": adam_kourkoutas_beta, "adam_beta2_min": adam_beta2_min,
            "adam_ema_alpha": adam_ema_alpha, "adam_tiny_spike": adam_tiny_spike,
            "adam_k_warmup_steps": adam_k_warmup_steps, "adam_nnmf_factor": adam_nnmf_factor,
        }
        self.stochastic_rounding = stochastic_rounding

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
                    if not group['normuon_variant']:
                        state['mu_vbuf_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                        state['mv_vbuf_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:
                if not group['normuon_variant']:
                    state['second_momentum_buffer'] = torch.zeros_like(p)
                state['momentum_buffer'] = torch.zeros_like(p)

            # NorMuon state initialization
            if group['normuon_variant']:
                if state['factored']:
                    state['normuon_v'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                elif len(p.shape) >= 2:
                    state['normuon_v'] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)

            # MARS-M state initialization
            if group.get('approx_mars', False):
                # Note: This requires full-rank memory even if factored
                state['last_grad'] = torch.zeros_like(p, device=device, dtype=dtype)

            group['adam_kourkoutas_beta'] = False
            state['is_muon'] = True # Workaround as group was acting weirdly; passing muon params in adam path

        else: # AdamW
            Muon_AuxAdam._init_auxadam_state(self, p, group)
            state['is_muon'] = False

    @torch.no_grad()
    def _muon_step_parameter(self, p, grad, state, group, is_compiled, random_int_tensor):


        @torch.compile(fullgraph=True, disable= not is_compiled)
        def compiled_muon_step_parameter(state, grad, group, lr, random_int_tensor):
            beta1, beta2 = group['betas']
            nesterov = group['nesterov']
            Simplified_AdEMAMix = group['Simplified_AdEMAMix']
            alpha_grad = group['alpha_grad']

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

            if grad.dtype != torch.float32 and state.get('factored', False):
                grad = grad.float()

            # MARS-M Approximated (Variance Reduction)
            if group.get('approx_mars', False):
                grad = approx_mars(grad, state['last_grad'], group['mars_gamma'], beta1)

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
                    update = grad_reshaped.lerp(mt_buf, beta1)
                elif Simplified_AdEMAMix:
                    update = torch.add(mt_buf, grad_reshaped, alpha=alpha_grad)
                else:
                    update = mt_buf.clone()

                # Factorize
                state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'] = _factorize_state(mt_buf, signed=True)
                del mt_buf

                # Apply update projection
                update = _auto_projection_for_adamuon(update, kappa_p)

                # Orthogonalization step
                update = newton_schulz(
                    update,
                    steps=group['ns_steps'],
                    eps=group['ns_eps'],
                    coeffs=group['ns_coeffs'],
                    cns=group['accelerated_ns'],
                    cns_a_bound=group['cns_a_bound'],
                    low_rank_ortho=group['low_rank_ortho'],
                    ortho_rank=group['ortho_rank']
                )

                if group['normuon_variant']:
                    normuon_update(update, state['normuon_v'], beta2, group['eps'])
                else:
                    # Reconstruct second momentum from previous step's factors
                    vt_buf = _reconstruct_state((state['mu_vbuf_nmf'], state['mv_vbuf_nmf']), signed=False)
                    # Update second momentum in full-size
                    vt_buf.mul_(beta2).addcmul_(update, update, value=1 - beta2)
                    # Apply second momentum update (adaptive scaling)
                    if group['use_atan2']:
                        denom = vt_buf.sqrt()
                        update.atan2_(denom)
                    else:
                        denom = vt_buf.sqrt().add_(group['eps'])
                        update.div_(denom)
                    state['mu_vbuf_nmf'], state['mv_vbuf_nmf'] = _factorize_state(vt_buf, signed=False)
                    del denom, vt_buf

                # RMS-aligned scaling
                step_scale = lr * A if group['use_atan2'] and not group['normuon_variant'] else lr
                rms_adjustment(update, group['rms_rescaling'], step_scale)

                update = update.reshape(p.shape)

            else: # Standard AdaMuon logic for non-factored tensors
                original_shape = p.shape

                # Momentum update
                mt_buf = state['momentum_buffer']
                if not Simplified_AdEMAMix:
                    mt_buf.lerp_(grad, 1 - beta1)
                else:
                    mt_buf.mul_(beta1).add_(grad)

                if nesterov:
                    update = grad.lerp(mt_buf, beta1)
                elif Simplified_AdEMAMix:
                    update = mt_buf.add(grad, alpha=alpha_grad)
                else:
                    update = mt_buf.clone()

                # Apply update projection
                update = _auto_projection_for_adamuon(update, kappa_p)

                # Flatten if necessary (e.g., for Conv layers)
                update = update.flatten(1)

                # Orthogonalization step
                update = newton_schulz(
                    update,
                    steps=group['ns_steps'],
                    eps=group['ns_eps'],
                    coeffs=group['ns_coeffs'],
                    cns=group['accelerated_ns'],
                    cns_a_bound=group['cns_a_bound'],
                    low_rank_ortho=group['low_rank_ortho'],
                    ortho_rank=group['ortho_rank']
                )

                # NorMuon Logic
                if group['normuon_variant']:
                    normuon_update(update, state['normuon_v'], beta2, group['eps'])
                else:
                    # Original AdaMuon Logic
                    update = update.view(original_shape)
                    vt_buf = state['second_momentum_buffer']
                    vt_buf.mul_(beta2).addcmul_(update, update, value=1 - beta2)
                    # Apply second momentum update (adaptive scaling)
                    if group['use_atan2']:
                        denom = vt_buf.sqrt()
                        update.atan2_(denom)
                    else:
                        denom = vt_buf.sqrt().add_(group['eps'])
                        update.div_(denom)
                    del denom

                # RMS-aligned rescaling
                step_scale = lr * A if group['use_atan2'] and not group['normuon_variant'] else lr
                rms_adjustment(update, group['rms_rescaling'], step_scale)

                update = update.reshape(original_shape)

            param_update.apply_parameter_update(self, p, group, update, lr, random_int_tensor=random_int_tensor)

        if group.get('compiled_optimizer', False):
            lr = torch.as_tensor(group['lr'])
        else:
            lr = group['lr']
        compiled_muon_step_parameter(state, grad, group, lr, random_int_tensor)

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
            Muon_AuxAdam._adam_step_parameter(self, p, grad, state, group, is_compiled, random_int_tensor)
        else: # Muon path
            self._muon_step_parameter(p, grad, state, group, is_compiled, random_int_tensor)

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
