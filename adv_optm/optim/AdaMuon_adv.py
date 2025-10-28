import torch

from ..util.BF16_Stochastic_Rounding import add_stochastic_
from ..util.Newton_Schulz import _newton_schulz_iteration
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper

class AdaMuon_adv(torch.optim.Optimizer):
    """
    IImplements an advanced AdaMuon optimizer algorithm.

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
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        low_rank_ortho (bool): If True, enables low-rank orthogonalization, which
            projects the update to a lower rank before orthogonalization.
            (default: False)
        ortho_rank (int): The rank for low-rank orthogonalization.
            (default: 128)
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
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
        stochastic_rounding: bool = False,
        use_atan2: bool = False,
        nesterov: bool = False,
        Simplified_AdEMAMix: bool = False,
        alpha_grad: float = 100.0,
        vector_reshape: bool = False,
        # Low-rank Muon
        low_rank_ortho: bool = False,
        ortho_rank: int = 128,
        nnmf_factor: bool = False,
        # Compiled
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
            "lr": lr, "betas": betas, "weight_decay": weight_decay,
            "eps": eps, "rms_target": rms_target, "ns_steps": ns_steps,
            "ns_eps": ns_eps, "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
            "nesterov":nesterov, "use_atan2":use_atan2,
            "Simplified_AdEMAMix": Simplified_AdEMAMix, "alpha_grad": alpha_grad,
            # Low-rank Ortho
            "low_rank_ortho": low_rank_ortho, "ortho_rank": ortho_rank,
            "compiled_optimizer":compiled_optimizer,
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

        self.global_step = 0 # For Adam bias correction and Kourkoutas
        self.kourkoutas_helper = None
        if any(group.get('adam_kourkoutas_beta', False) for group in self.param_groups):
            self.kourkoutas_helper = KourkoutasHelper(self)

        self.init_step()

        # Initialize compiled functions to None
        self._compiled_muon_step = None
        self._compiled_adam_step = None

        if compiled_optimizer:
            print("Compiling AdaMuon_adv optimizer paths...")
            torch._dynamo.config.cache_size_limit = 8192
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

        if len(state) > 0:
            return

        optim_type = group.get('optim_type', 'muon')

        if optim_type == 'muon':

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
                    state['mu_vbuf_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_vbuf_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:
                if len(p.shape) >= 2:
                    state['second_momentum_buffer'] = torch.zeros_like(p)
                state['momentum_buffer'] = torch.zeros_like(p)


        elif optim_type == 'adam':

            state['factored'] = (
                group['adam_nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )
            dtype = torch.float32 if state['factored'] else p.dtype
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                # First moment (m)
                if group['adam_betas'][0] > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    if not group.get('adam_grams_moment'):
                        packed_d2 = (d2 + 7) // 8
                        state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                if group.get('adam_use_AdEMAMix'):
                    state['mu_m_slow_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype)
                    state['mv_m_slow_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign_slow'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                # Second moment (v)
                state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:  # Fallback to standard AdamW for non-factored tensors
                if group['adam_betas'][0] > 0:
                    state['exp_avg'] = torch.zeros_like(p, device=device, dtype=dtype)
                if group.get('adam_use_AdEMAMix'):
                    state['exp_avg_slow'] = torch.zeros_like(p, device=device, dtype=dtype)
                state['exp_avg_sq'] = torch.zeros_like(p, device=device, dtype=dtype)

    @torch.no_grad()
    def _muon_step_parameter(self, p, grad, state, group, lr):
        # Retrieve hyperparameters
        beta1, beta2 = group['betas']
        nesterov = group['nesterov']
        Simplified_AdEMAMix = group['Simplified_AdEMAMix']
        alpha_grad = group['alpha_grad']

        if state['factored']: # Factored AdaMuon

            # Reconstruct momentum from previous step's factors & sign
            d1, d2 = state['effective_shape']
            mt_buf = _unnmf((state['mu_mbuf_nmf'], state['mv_mbuf_nmf']))
            unpacked_sign = _unpack_bools(state['sign_buf'], original_m=d2)
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

            # Orthogonalization step
            if group['low_rank_ortho']:
                # Low-Rank Orthogonalization on the reconstructed matrix
                M = signed_m_buf
                r = min(group['ortho_rank'], M.shape[0], M.shape[1])
                if r > 0:
                    G_sketch = torch.randn(M.shape[1], r, device=M.device, dtype=M.dtype)
                    MG = M @ G_sketch
                    if MG.dtype != torch.float32:
                        MG_dtype = M.dtype
                        Q, _ = torch.linalg.qr(MG.float())
                        Q = Q.to(MG_dtype)
                    else:
                        Q, _ = torch.linalg.qr(MG)
                    projected_M = Q.T @ M
                    ortho_projected_M = _newton_schulz_iteration(
                        projected_M, steps=group['ns_steps'], eps=group['ns_eps'], coeffs=group['ns_coeffs']
                    )
                    update = Q @ ortho_projected_M
                else: # Fallback for invalid rank
                    update = _newton_schulz_iteration(
                        signed_m_buf, steps=group['ns_steps'], eps=group['ns_eps'], coeffs=group['ns_coeffs']
                    )
            else:
                # Original full Newton-Schulz
                update = _newton_schulz_iteration(
                    signed_m_buf,
                    steps=group['ns_steps'],
                    eps=group['ns_eps'],
                    coeffs=group['ns_coeffs'],
                )
            del signed_m_buf

            # Reconstruct second momentum from previous step's factors
            vt_buf = _unnmf((state['mu_vbuf_nmf'], state['mv_vbuf_nmf']))

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
            # Add eps to prevent division by zero
            update.mul_(rms_target * (num_elements ** 0.5) / (update.norm() + group['eps']))

            update = update.view(p.shape).mul_(lr)
            del num_elements

            # Compress updated moments and store new factors
            state['sign_buf'] = _pack_bools(mt_buf > 0)
            _nnmf(mt_buf.abs(), out=(state['mu_mbuf_nmf'], state['mv_mbuf_nmf']))
            del mt_buf

            _nnmf(vt_buf.abs(), out=(state['mu_vbuf_nmf'], state['mv_vbuf_nmf']))
            del vt_buf

        else: # Standard AdaMuon logic for non-factored tensors

            if len(p.shape) >= 2:

                original_shape = p.shape

                # Momentum update
                mt_buf = state['momentum_buffer']
                mt_buf.mul_(beta1).add_(grad)

                if nesterov:
                    signed_m_buf = torch.sign(grad.add(mt_buf, alpha=beta1))
                elif Simplified_AdEMAMix:
                    signed_m_buf = torch.sign(mt_buf.add(grad, alpha=alpha_grad))
                else:
                    signed_m_buf = torch.sign(mt_buf)

                # Flatten if necessary (e.g., for Conv layers)
                signed_m_buf = signed_m_buf.view(original_shape[0], -1)

                # Orthogonalization step
                if group['low_rank_ortho']:
                    # Low-Rank Orthogonalization on the reconstructed matrix
                    M = signed_m_buf
                    r = min(group['ortho_rank'], M.shape[0], M.shape[1])
                    if r > 0:
                        G_sketch = torch.randn(M.shape[1], r, device=M.device, dtype=M.dtype)
                        MG = M @ G_sketch
                        if MG.dtype != torch.float32:
                            MG_dtype = M.dtype
                            Q, _ = torch.linalg.qr(MG.float())
                            Q = Q.to(MG_dtype)
                        else:
                            Q, _ = torch.linalg.qr(MG)
                        projected_M = Q.T @ M
                        ortho_projected_M = _newton_schulz_iteration(
                            projected_M, steps=group['ns_steps'], eps=group['ns_eps'], coeffs=group['ns_coeffs']
                        )
                        update = Q @ ortho_projected_M
                    else: # Fallback for invalid rank
                        update = _newton_schulz_iteration(
                            signed_m_buf, steps=group['ns_steps'], eps=group['ns_eps'], coeffs=group['ns_coeffs']
                        )
                else:
                    # Original full Newton-Schulz
                    update = _newton_schulz_iteration(
                        signed_m_buf,
                        steps=group['ns_steps'],
                        eps=group['ns_eps'],
                        coeffs=group['ns_coeffs'],
                    )
                del signed_m_buf

                update = update.view(original_shape)

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
                # Add eps to prevent division by zero
                update.mul_(rms_target * (num_elements ** 0.5) / (update.norm() + group['eps']))

                del num_elements

                update.mul_(lr)

            else: # Fallback to standard SGD with momentum for 1D params (biases, etc.)
                # Momentum update
                mt_buf = state['momentum_buffer']
                mt_buf.mul_(beta1).add_(grad)
                if nesterov:
                    # Nesterov momentum
                    update = grad.add(mt_buf, alpha=beta1)
#                 elif Simplified_AdEMAMix: # TODO, it will break SGD since it requires x100 lower LR
#                     update = mt_buf.add(grad, alpha=alpha_grad)
                else:
                    update = mt_buf.clone()
                update.mul_(lr)

        # Decoupled weight decay
        if group["weight_decay"] != 0:
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-group["weight_decay"] * lr)
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * lr)

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p.data, -update)
        else:
            p.data.add_(-update)
        del update

    @torch.no_grad()
    def _adam_step_parameter(self, p, grad, state, group, lr, bias_correction1, bias_correction2):
        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()
        if group.get("adam_orthogonal_gradient"):
            grad = _orthogonalize_gradient(p, grad)

        beta1_adam, beta2_adam = group['adam_betas']

        if group.get('adam_kourkoutas_beta', False):
            # Accumulate current grad's norm for the *next* step
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)
            # Get the dynamic beta2_adam calculated in prepare_step()
            beta2_adam = self.kourkoutas_helper.get_beta2(p, group)

        step_size = lr / bias_correction1

        if group.get('adam_use_AdEMAMix'):
            beta3_ema = group['adam_beta3_ema']
            alpha = group['adam_alpha']

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors
            if beta1_adam > 0:
                mt = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
                if not group.get('adam_grams_moment'):
                    unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
                    torch.where(unpacked_sign, mt, -mt, out=mt)
                    del unpacked_sign
                # Update momentum in full-size
                mt.mul_(beta1_adam).add_(grad_reshaped, alpha=1.0 - beta1_adam)
                if group.get('adam_grams_moment'):
                    mt = (grad_reshaped.sign().mul_(mt.abs()))
                elif group.get('adam_cautious_mask'):
                    mask = (mt * grad_reshaped > 0).to(grad_reshaped.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    mt.mul_(mask)
                    del mask

            vt = _unnmf((state['mu_v_nmf'], state['mv_v_nmf']))
            vt.mul_(beta2_adam).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2_adam)

            if group.get('adam_use_AdEMAMix'):
                mt_slow = _unnmf((state['mu_m_slow_nmf'], state['mv_m_slow_nmf']))
                if state['sign_slow'].dtype != torch.uint8:
                    state['sign_slow'] = state['sign_slow'].to(torch.uint8)
                unpacked_sign_slow = _unpack_bools(state['sign_slow'], original_m=d2)
                torch.where(unpacked_sign_slow, mt_slow, -mt_slow, out=mt_slow)
                del unpacked_sign_slow

                mt_slow.mul_(beta3_ema).add_(grad_reshaped, alpha=1.0 - beta3_ema)
                if beta1_adam > 0:
                    update = torch.add(mt, mt_slow, alpha=alpha)
                else:
                    update = torch.add(grad_reshaped, mt_slow, alpha=alpha)
            else:
                update = mt.clone() if beta1_adam > 0 else grad_reshaped.clone()
            del grad_reshaped

            if group['adam_use_atan2']:
                a = 1.2732395
                denom = (vt.sqrt() / (bias_correction2**0.5))
                update.atan2_(denom).mul_(a)
            else:
                denom = (vt.sqrt() / (bias_correction2**0.5)).add_(group['adam_eps'])
                update.div_(denom)
            del denom

            update = update.view(p.shape).mul_(step_size)

            # Compress updated moments and store new factors
            if beta1_adam > 0:
                if not group.get('adam_grams_moment'):
                    state['sign'] = _pack_bools(mt > 0)
                _nnmf(mt.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
                del mt
            if group.get('adam_use_AdEMAMix'):
                state['sign_slow'] = _pack_bools(mt_slow > 0)
                _nnmf(mt_slow.abs(), out=(state['mu_m_slow_nmf'], state['mv_m_slow_nmf']))
                del mt_slow
            _nnmf(vt, out=(state['mu_v_nmf'], state['mv_v_nmf']))
            del vt

        else:  # Standard AdamW logic for non-factored tensors
            exp_avg_sq = state['exp_avg_sq']

            if beta1_adam > 0:
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1_adam).add_(grad, alpha=1 - beta1_adam)
                if group.get('adam_grams_moment'):
                    exp_avg = grad.sign().mul_(exp_avg.abs())
                elif group.get('adam_cautious_mask'):
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    exp_avg.mul_(mask)
                    del mask

            if group.get('adam_use_AdEMAMix'):
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_slow.mul_(beta3_ema).add_(grad, alpha=1 - beta3_ema)
                if beta1_adam > 0:
                    update = torch.add(exp_avg, exp_avg_slow, alpha=alpha)
                else:
                    update = torch.add(grad, exp_avg_slow, alpha=alpha)
            else:
                update = exp_avg.clone() if beta1_adam > 0 else grad.clone()

            exp_avg_sq.mul_(beta2_adam).addcmul_(grad, grad.conj(), value=1 - beta2_adam)

            if group.get('adam_use_atan2'):
                a = 1.2732395
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5))
                update.atan2_(denom).mul_(a)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group['adam_eps'])
                update.div_(denom)
            del denom

            update.mul_(step_size)

        # Decoupled weight decay
        if group["adam_weight_decay"] != 0:
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-group["adam_weight_decay"] * lr)
            else:
                p.data.add_(p.data, alpha=-group["adam_weight_decay"] * lr)

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p.data, -update)
        else:
            p.data.add_(-update)
        del update

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        grad = p.grad
        if grad is None:
            return
        state = self.state[p]



        # Determine if using Adam or Muon based on state keys
        # We can use optm_type but I see this as a safer way.
        if 'momentum_buffer' in state or 'mu_mbuf_nmf' in state:
            use_adam = False
        else:
            use_adam = True

        lr = group['lr']
        is_compiled = group.get('compiled_optimizer', False)

        if use_adam:
            if self.kourkoutas_helper:
                # Prepare Kourkoutas-β once per optimizer step.
                self.kourkoutas_helper.maybe_prepare_step(self.global_step)
            # Adam-specific setup (bias correction)
            if group['adam_use_bias_correction']:
                current_step = self.global_step + 1
                beta1_adam, beta2_adam = group['adam_betas']
                bias_correction1 = 1.0 - beta1_adam ** current_step
                bias_correction2 = 1.0 - beta2_adam ** current_step
            else:
                bias_correction1 = 1.0
                bias_correction2 = 1.0
            # Dispatch to compiled or uncompiled Adam step
            if is_compiled and self._compiled_adam_step is not None:
                # Tensors must be used for compiled functions
                lr_tensor = torch.tensor(lr, device=p.device)
                bc1_tensor = torch.tensor(bias_correction1, device=p.device)
                bc2_tensor = torch.tensor(bias_correction2, device=p.device)
                self._compiled_adam_step(p, grad, state, group, lr_tensor, bc1_tensor, bc2_tensor)
            else:
                self._adam_step_parameter(p, grad, state, group, lr, bias_correction1, bias_correction2)
        else: # Muon path
            # Dispatch to compiled or uncompiled Muon step
            if is_compiled and self._compiled_muon_step is not None:
                lr_tensor = torch.tensor(lr, device=p.device)
                self._compiled_muon_step(p, grad, state, group, lr_tensor)
            else:
                self._muon_step_parameter(p, grad, state, group, lr)


    def compile(self, *args, **kwargs):
        print("Compiling AdaMuon step path...")
        self._compiled_muon_step = torch.compile(self._muon_step_parameter, *args, **kwargs)
        print("Compiling AuxAdam step path...")
        self._compiled_adam_step = torch.compile(self._adam_step_parameter, *args, **kwargs)

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

        self.global_step += 1

        return loss
