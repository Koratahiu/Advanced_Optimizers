import torch

from ..util.BF16_Stochastic_Rounding import add_stochastic_
from ..util.Newton_Schulz import _newton_schulz_iteration
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.Kourkoutas import KourkoutasHelper

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
        beta1 (float): momentum factor for Muon groups (default: 0.9).
        weight_decay (float): weight decay for Muon groups (L2 penalty) (default: 0).
        nesterov (bool): enables Nesterov momentum for Muon groups (default: True).
        ns_steps (int): number of Newton-Schulz iterations for Muon groups (default: 5).
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule for Muon groups.
        alpha_grad (float): Mixing coefficient for Simplified AdEMAMix.
        stochastic_rounding (bool): whether to use stochastic rounding for
            BF16 parameter updates (default: True).
        nnmf_factor (bool): whether to use factorization for all groups.
        low_rank_ortho (bool): If True, enables low-rank orthogonalization for Muon groups.
        ortho_rank (int): The rank for low-rank orthogonalization.
        normuon_variant (bool): If True, enables the NorMuon update rule for Muon groups.
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
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        Simplified_AdEMAMix: bool = False,
        alpha_grad: float = 100.0,
        stochastic_rounding: bool = True,
        vector_reshape_muon: bool = False,
        vector_reshape: bool = False,
        nnmf_factor: bool = False,
        # Low-rank Muon
        low_rank_ortho: bool = False,
        ortho_rank: int = 128,
        # NorMuon additions
        normuon_variant: bool = False,
        beta2_normuon: float = 0.95,
        normuon_eps: float = 1e-8,
        normuon_lr_scale: float = 0.2,
        normuon_atan2: bool = False,
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
            "lr": lr, "beta1": beta1, "weight_decay": weight_decay,
            "nesterov": nesterov, "ns_steps": ns_steps, "ns_eps": ns_eps,
            "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
            "vector_reshape_muon": vector_reshape_muon,
            "Simplified_AdEMAMix": Simplified_AdEMAMix, "alpha_grad": alpha_grad,
            # Low-rank Ortho
            "low_rank_ortho": low_rank_ortho, "ortho_rank": ortho_rank,
            # NorMuon
            "normuon_variant": normuon_variant, "beta2_normuon": beta2_normuon,
            "normuon_eps": normuon_eps, "normuon_lr_scale": normuon_lr_scale,
            "normuon_atan2": normuon_atan2,
            # AdamW_adv defaults
            "adam_betas": adam_betas, "adam_eps": adam_eps, "adam_weight_decay": adam_weight_decay,
            "adam_use_bias_correction": adam_use_bias_correction, "adam_use_atan2": adam_use_atan2,
            "adam_cautious_mask": adam_cautious_mask, "adam_grams_moment": adam_grams_moment,
            "adam_orthogonal_gradient": adam_orthogonal_gradient,
            "adam_use_AdEMAMix": adam_use_AdEMAMix, "adam_beta3_ema": adam_beta3_ema, "adam_alpha": adam_alpha,
            "adam_kourkoutas_beta": adam_kourkoutas_beta, "adam_beta2_min": adam_beta2_min,
            "adam_ema_alpha": adam_ema_alpha, "adam_tiny_spike": adam_tiny_spike,
            "adam_k_warmup_steps": adam_k_warmup_steps,
        }
        self.stochastic_rounding = stochastic_rounding
        self.compiled_optimizer = compiled_optimizer

        super().__init__(params, defaults)

        self.global_step = 0 # For Adam bias correction and Kourkoutas
        self.kourkoutas_helper = None
        if any(group.get('adam_kourkoutas_beta', False) for group in self.param_groups):
            self.kourkoutas_helper = KourkoutasHelper(self)

        self.init_step()

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

        if len(state) > 0:
            return

        optim_type = group.get('optim_type', 'muon')

        state['factored'] = (
            group['nnmf_factor'] and
            not (len(p.shape) == 1 and not group['vector_reshape'])
        )
        dtype = torch.float32 if state['factored'] else p.dtype
        device = p.device

        if optim_type == 'muon':
            state['reshaped_1d_muon'] = len(p.shape) == 1 and group['vector_reshape_muon']

            if state['factored'] or state['reshaped_1d_muon']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

            if state['factored']: # Muon factored state
                state['mu_mbuf_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                state['mv_mbuf_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                packed_d2 = (d2 + 7) // 8
                state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
            else: # Muon non-factored state
                if len(p.shape) >= 2 or state['reshaped_1d_muon']:
                    shape = (d1, d2) if state['reshaped_1d_muon'] else p.shape
                    state['momentum_buffer'] = torch.zeros(shape, device=device, dtype=dtype)
                elif len(p.shape) == 1:
                    state['momentum_buffer'] = torch.zeros_like(p, device=device, dtype=dtype)

            if group.get('normuon_variant'): # NorMuon state
                if state['factored']:
                    state['normuon_v'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                elif len(p.shape) >= 2 or state['reshaped_1d_muon']:
                    num_rows = p.shape[0] if len(p.shape) >= 2 else state['effective_shape'][0]
                    state['normuon_v'] = torch.zeros(num_rows, device=p.device, dtype=torch.float32)

        elif optim_type == 'adam':
            if state['factored']: # Adam factored state
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                if group['adam_betas'][0] > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                if group.get('adam_use_AdEMAMix'):
                    state['mu_m_slow_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_m_slow_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign_slow'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:  # Adam non-factored state
                if group['adam_betas'][0] > 0:
                    state['exp_avg'] = torch.zeros_like(p, device=device, dtype=dtype)
                if group.get('adam_use_AdEMAMix'):
                    state['exp_avg_slow'] = torch.zeros_like(p, device=device, dtype=dtype)
                state['exp_avg_sq'] = torch.zeros_like(p, device=device, dtype=dtype)

    @torch.no_grad()
    def _muon_step_parameter(self, p, grad, state, group, lr):
        beta1 = group['beta1']
        nesterov = group['nesterov']
        Simplified_AdEMAMix = group['Simplified_AdEMAMix']
        alpha_grad = group['alpha_grad']

        if state['factored']: # Factored Muon

            # Reconstruct momentum from previous step's factors & sign
            d1, d2 = state['effective_shape']
            mt_buf = _unnmf((state['mu_mbuf_nmf'], state['mv_mbuf_nmf']))
            unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
            torch.where(unpacked_sign, mt_buf, -mt_buf, out=mt_buf)
            del unpacked_sign

            # Update momentum in full-size
            grad_reshaped = grad.view(d1, d2)
            mt_buf.mul_(beta1).add_(grad_reshaped)

            if nesterov:
                # Nesterov momentum
                update = grad_reshaped.add(mt_buf, alpha=beta1)
            elif Simplified_AdEMAMix:
                update = torch.add(mt_buf, grad_reshaped, alpha=alpha_grad)
            else:
                # Standard momentum
                update = mt_buf.clone()
            del grad_reshaped

            # Orthogonalization step
            if group['low_rank_ortho']:
                # Low-Rank Orthogonalization on the reconstructed matrix
                M = update
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
                        update, steps=group['ns_steps'], eps=group['ns_eps'], coeffs=group['ns_coeffs']
                    )
            else:
                # Original full Newton-Schulz
                update = _newton_schulz_iteration(
                    update,
                    steps=group['ns_steps'],
                    eps=group['ns_eps'],
                    coeffs=group['ns_coeffs'],
                )


            if group['normuon_variant'] and 'normuon_v' in state:
                v_t = state['normuon_v']
                beta2_normuon = group['beta2_normuon']
                # Update 2nd moment estimate
                mean_squared_update = torch.mean(update.square(), dim=1)
                v_t.mul_(beta2_normuon).add_(mean_squared_update, alpha=1 - beta2_normuon)
                # Normalize update
                if group['normuon_atan2']:
                    a = 1.2732395
                    update.atan2_(v_t.sqrt().unsqueeze(1)).mul_(a)
                else:
                    update.div_(v_t.sqrt().unsqueeze(1).add_(group['normuon_eps']))
                # Scale learning rate
                update_norm = torch.linalg.vector_norm(update)

                scaled_lr = group['normuon_lr_scale'] * lr * (p.numel()**0.5) / update_norm.add_(group['normuon_eps'])

                update = update.view(p.shape).mul_(scaled_lr)
                del update_norm, scaled_lr
            else: # Original Muon learning rate application
                update = update.view(p.shape).mul_(lr)

            state['sign'] = _pack_bools(mt_buf > 0)
            _nnmf(mt_buf.abs(), out=(state['mu_mbuf_nmf'], state['mv_mbuf_nmf']))
            del mt_buf

        else: # Standard Muon logic for non-factored tensors

            if len(p.shape) >= 2 or state['reshaped_1d_muon']:

                original_shape = p.shape

                # Momentum update
                mt_buf = state['momentum_buffer']
                if state['reshaped_1d_muon']:
                    d1, d2 = state['effective_shape']
                    grad_reshaped = grad.view(d1, d2)
                    mt_buf.mul_(beta1).add_(grad_reshaped)
                else:
                    mt_buf.mul_(beta1).add_(grad)

                if nesterov:
                    # Nesterov momentum
                    if state['reshaped_1d_muon']:
                        update = grad_reshaped.add(mt_buf, alpha=beta1)
                        del grad_reshaped
                    else:
                        update = grad.add(mt_buf, alpha=beta1)
                elif Simplified_AdEMAMix:
                    if state['reshaped_1d_muon']:
                        update = torch.add(mt_buf, grad_reshaped, alpha=alpha_grad)
                        del grad_reshaped
                    else:
                        update = torch.add(mt_buf, grad, alpha=alpha_grad)
                else:
                    # Standard momentum
                    update = mt_buf.clone()

                # flatten to 2D for orthogonalization.
                # This is a no-op for 2D tensors and correctly flattens 4D+ tensors.
                # This removes the dynamic control flow that breaks torch.compile.
                update = update.view(original_shape[0], -1)

                # Orthogonalization step
                if group['low_rank_ortho']:
                    # Low-Rank Orthogonalization based on Gaussian Sketching
                    M = update
                    r = min(group['ortho_rank'], M.shape[0], M.shape[1])

                    if r > 0:
                        # 1. Sketch the matrix
                        G_sketch = torch.randn(M.shape[1], r, device=M.device, dtype=M.dtype)
                        MG = M @ G_sketch

                        # 2. QR decomposition to get orthogonal basis Q
                        if MG.dtype != torch.float32:
                            MG_dtype = M.dtype
                            Q, _ = torch.linalg.qr(MG.float())
                            Q = Q.to(MG_dtype)
                        else:
                            Q, _ = torch.linalg.qr(MG)

                        # 3. Project M onto the basis
                        projected_M = Q.T @ M

                        # 4. Orthogonalize the smaller projected matrix
                        ortho_projected_M = _newton_schulz_iteration(
                            projected_M,
                            steps=group['ns_steps'],
                            eps=group['ns_eps'],
                            coeffs=group['ns_coeffs'],
                        )

                        # 5. Project back to the original space
                        update = Q @ ortho_projected_M
                    else: # Fallback for invalid rank
                        update = _newton_schulz_iteration(
                            update,
                            steps=group['ns_steps'],
                            eps=group['ns_eps'],
                            coeffs=group['ns_coeffs'],
                        )
                else:
                    # Original NewtonSchulz
                    update = _newton_schulz_iteration(
                        update,
                        steps=group['ns_steps'],
                        eps=group['ns_eps'],
                        coeffs=group['ns_coeffs'],
                    )

                # NorMuon Logic
                if group['normuon_variant'] and 'normuon_v' in state:
                    v_t = state['normuon_v']
                    beta2_normuon = group['beta2_normuon']
                    # Update 2nd moment estimate
                    mean_squared_update = torch.mean(update.square(), dim=1)
                    v_t.mul_(beta2_normuon).add_(mean_squared_update, alpha=1 - beta2_normuon)
                    # Normalize update
                    if group['normuon_atan2']:
                        a = 1.2732395
                        update.atan2_(v_t.sqrt().unsqueeze(1)).mul_(a)
                    else:
                        update.div_(v_t.sqrt().unsqueeze(1).add_(group['normuon_eps']))
                    # Scale learning rate
                    update_norm = torch.linalg.vector_norm(update)
                    scaled_lr = group['normuon_lr_scale'] * lr * (p.numel()**0.5) / update_norm.add_(group['normuon_eps'])
                    update.mul_(scaled_lr)
                else: # Original Muon learning rate application
                    update.mul_(lr)

                # reshape back to the original shape.
                update = update.view(original_shape)


            else: # Fallback to standard SGD with momentum for 1D params (biases, etc.)
                # Momentum update
                mt_buf = state['momentum_buffer']
                mt_buf.mul_(beta1).add_(grad)
                if nesterov:
                    # Nesterov momentum
                    update = grad.add(mt_buf, alpha=beta1)
                elif Simplified_AdEMAMix:
                    update = torch.add(mt_buf, grad, alpha=alpha_grad)
                else:
                    # Standard momentum
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
    def _adam_step_parameter(self, p, grad, state, group, lr):
        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()
        if group.get("adam_orthogonal_gradient"):
            grad = _orthogonalize_gradient(p, grad)

        beta1, beta2 = group['adam_betas']

        if group.get('adam_kourkoutas_beta', False) and self.kourkoutas_helper:
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)
            beta2 = self.kourkoutas_helper.get_beta2(p, group)

        if group.get('adam_use_bias_correction'):
            current_step = self.global_step + 1
            bias_correction1 = 1.0 - beta1 ** current_step
            bias_correction2 = 1.0 - beta2 ** current_step
        else:
            bias_correction1 = 1.0
            bias_correction2 = 1.0

        step_size = lr / bias_correction1

        if state.get('factored', False):
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            if beta1 > 0:
                mt = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
                if not group.get('adam_grams_moment'):
                    unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
                    torch.where(unpacked_sign, mt, -mt, out=mt)
                    del unpacked_sign
                mt.mul_(beta1).add_(grad_reshaped, alpha=1.0 - beta1)
                if group.get('adam_grams_moment'):
                    mt = (grad_reshaped.sign().mul_(mt.abs()))
                elif group.get('adam_cautious_mask'):
                    mask = (mt * grad_reshaped > 0).to(grad_reshaped.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    mt.mul_(mask)
                    del mask

            vt = _unnmf((state['mu_v_nmf'], state['mv_v_nmf']))
            vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2)

            if group.get('adam_use_AdEMAMix'):
                mt_slow = _unnmf((state['mu_m_slow_nmf'], state['mv_m_slow_nmf']))
                unpacked_sign_slow = _unpack_bools(state['sign_slow'], original_m=d2)
                torch.where(unpacked_sign_slow, mt_slow, -mt_slow, out=mt_slow)
                del unpacked_sign_slow
                mt_slow.mul_(group['adam_beta3_ema']).add_(grad_reshaped, alpha=1.0 - group['adam_beta3_ema'])
                update = torch.add(mt, mt_slow, alpha=group['adam_alpha']) if beta1 > 0 else torch.add(grad_reshaped, mt_slow, alpha=group['adam_alpha'])
            else:
                update = mt.clone() if beta1 > 0 else grad_reshaped.clone()
            del grad_reshaped

            if group.get('adam_use_atan2'):
                denom = (vt.sqrt() / (bias_correction2**0.5))
                update.atan2_(denom).mul_(1.2732395)
            else:
                denom = (vt.sqrt() / (bias_correction2**0.5)).add_(group['adam_eps'])
                update.div_(denom)
            del denom

            update = update.view(p.shape).mul_(step_size)

            if beta1 > 0:
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

            if beta1 > 0:
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                if group.get('adam_grams_moment'):
                    exp_avg = grad.sign().mul_(exp_avg.abs())
                elif group.get('adam_cautious_mask'):
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    exp_avg.mul_(mask)
                    del mask

            if group.get('adam_use_AdEMAMix'):
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_slow.mul_(group['adam_beta3_ema']).add_(grad, alpha=1 - group['adam_beta3_ema'])
                update = torch.add(exp_avg, exp_avg_slow, alpha=group['adam_alpha']) if beta1 > 0 else torch.add(grad, exp_avg_slow, alpha=group['adam_alpha'])
            else:
                update = exp_avg.clone() if beta1 > 0 else grad.clone()

            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if group.get('adam_use_atan2'):
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5))
                update.atan2_(denom).mul_(1.2732395)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group['adam_eps'])
                update.div_(denom)
            del denom

            update.mul_(step_size)

        # Decoupled weight decay
        wd = group.get("adam_weight_decay", 0.0)
        if wd != 0:
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-wd * lr)
            else:
                p.data.add_(p.data, alpha=-wd * lr)

        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
            add_stochastic_(p.data, -update)
        else:
            p.data.add_(-update)
        del update


    @torch.no_grad()
    def __step_parameter(self, p: torch.Tensor, group: dict, lr: float | torch.Tensor):
        """
        Internal step logic, intended to be compiled.
        Handles a single parameter update.
        """
        grad = p.grad
        if grad is None:
            return

        state = self.state[p]
        optim_type = group.get('optim_type', 'muon')

        if optim_type == 'muon':
            self._muon_step_parameter(p, grad, state, group, lr)
        elif optim_type == 'adam':
            self._adam_step_parameter(p, grad, state, group, lr)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if self.kourkoutas_helper:
            # Prepare Kourkoutas-β once per step using the global step counter.
            self.kourkoutas_helper.maybe_prepare_step(self.global_step)

        if p.grad is None:
            return

        if not group.get('compiled_optimizer', False):
            self.__step_parameter(p, group, group['lr'])
        else:
            # Note: Tensors must be used for compiled functions
            lr_tensor = torch.tensor(group['lr'], device=p.device)
            self._compiled_step_parameter(p, group, lr_tensor)


    def compile(self, *args, **kwargs):
        self._compiled_step_parameter = torch.compile(self.__step_parameter, *args, **kwargs)

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