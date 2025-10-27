import torch
torch._dynamo.config.force_parameter_static_shapes = False

from ..util.BF16_Stochastic_Rounding import add_stochastic_
from ..util.Newton_Schulz import _newton_schulz_iteration
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools

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
        vector_reshape: bool = False,
        # Low-rank Muon
        low_rank_ortho: bool = False,
        ortho_rank: int = 128,
        nnmf_factor: bool = False,
        # Compiled
        compiled_optimizer: bool = True,
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

        defaults = {
            "lr": lr, "betas": betas, "weight_decay": weight_decay,
            "eps": eps, "rms_target": rms_target, "ns_steps": ns_steps,
            "ns_eps": ns_eps, "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
            "nesterov":nesterov, "use_atan2":use_atan2,
            "Simplified_AdEMAMix": Simplified_AdEMAMix, "alpha_grad": alpha_grad,
            # Low-rank Ortho
            "low_rank_ortho": low_rank_ortho, "ortho_rank": ortho_rank,
            "compiled_optimizer":compiled_optimizer
        }
        self.stochastic_rounding = stochastic_rounding

        super().__init__(params, defaults)

        self.init_step()

        if compiled_optimizer:
            self.compile(fullgraph=False, dynamic=True)

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

        if len(state) == 0:

            should_factor = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            state['factored'] = should_factor

            dtype = torch.float32 if group['nnmf_factor'] else p.dtype
            device = p.device

            if state['factored']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                    state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                    state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:
                if len(p.shape) >= 2:
                    state['second_momentum_buffer'] = torch.zeros_like(p)
                state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def __step_parameter(self, p: torch.Tensor, group: dict):
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]


        # Retrieve hyperparameters
        beta1, beta2 = group['betas']
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
            # Add eps to prevent division by zero
            scaling_factor = rms_target * (num_elements ** 0.5) / (update.norm() + group['eps'])

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
                scaling_factor = rms_target * (num_elements ** 0.5) / (update.norm() + group['eps'])

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


    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if not group.get('compiled_optimizer', False):
            self.__step_parameter(p, group)
        else:
            self._compiled_step_parameter(p, group)

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

        return loss
