import torch


from ..util.BF16_Stochastic_Rounding import add_stochastic_
from ..util.Newton_Schulz import _newton_schulz_iteration
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools

class Muon_adv(torch.optim.Optimizer):
    """
    Implements an advanced Muon algorithm.

    Muon (MomentUm Orthogonalized by Newton-Schulz) is an optimizer designed for
    the hidden layers of neural networks. It applies SGD with momentum and then
    orthogonalizes the resulting update matrix using a Newton-Schulz iteration.

    NorMuon (Neuron-wise Normalized Muon) extends this by adding neuron-level
    adaptive learning rates, combining the benefits of orthogonalization with
    second-order momentum statistics.

    This implementation is designed for 2D parameters (e.g., linear layers) and
    can handle other-dimensional parameters (e.g., 1D bias, 4D convolutional layers) by
    flattening/reshaping them.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float): momentum factor (default: 0.9).
        weight_decay (float): weight decay (L2 penalty) (default: 0).
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
        vector_reshape_muon (bool): whether to reshape 1D vectors into 2D
            matrices for muon NewtonSchulz (default: False).
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
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
        normuon_lr_scale (float): Scaling factor for the NorMuon learning rate.
            (default: 0.2)
        normuon_atan2 (bool): whether to use the atan2 for NorMuon. (default: False)
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
        }
        self.stochastic_rounding = stochastic_rounding
        self.compiled_optimizer = compiled_optimizer

        super().__init__(params, defaults)

        self.init_step()

        if compiled_optimizer:
            self.compile(fullgraph=True) #FIXME

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

            state['factored'] = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

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
            else:
                if len(p.shape) >= 2:
                    state['momentum_buffer'] = torch.zeros_like(p)
                if state['reshaped_1d_muon']:
                    state['momentum_buffer'] = torch.zeros((d1, d2), device=device, dtype=dtype)
                elif len(p.shape) == 1:
                    state['momentum_buffer'] = torch.zeros_like(p)

            # NorMuon state initialization
            if group['normuon_variant']:
                if len(p.shape) >= 2 or state['reshaped_1d_muon']:
                    num_rows = p.shape[0] if len(p.shape) >= 2 else state['effective_shape'][0]
                    state['normuon_v'] = torch.zeros(num_rows, device=p.device, dtype=torch.float32)

    @torch.no_grad()
    def __step_parameter(self, p: torch.Tensor, group: dict):
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]


        beta1 = group['beta1']
        nesterov = group['nesterov']
        Simplified_AdEMAMix = group['Simplified_AdEMAMix']
        alpha_grad = group['alpha_grad']

        if state['factored']: # Factored Muon

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

                scaled_lr = group['normuon_lr_scale'] * group['lr'] * (p.numel()**0.5) / update_norm.add_(group['normuon_eps'])

                update = update.view(p.shape).mul_(scaled_lr)
                del update_norm, scaled_lr
            else: # Original Muon learning rate application
                update = update.view(p.shape).mul_(group['lr'])

            state['sign'] = _pack_bools(mt_buf > 0)
            _nnmf(mt_buf.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
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
                    scaled_lr = group['normuon_lr_scale'] * group['lr'] * (p.numel()**0.5) / update_norm.add_(group['normuon_eps'])
                    update.mul_(scaled_lr)
                else: # Original Muon learning rate application
                    update.mul_(group['lr'])

                # reshape back to the original shape.
                update = update.view(original_shape)


            else: # Fallback to standard SGD with momentum for 1D params (biases, etc.) when not reshaped
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
        if not self.compiled_optimizer:
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