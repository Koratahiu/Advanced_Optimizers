import torch
from typing import Optional, Callable

from .AdamW_adv import AdamW_adv
from ..util.MuonAdam_helper import MuonAdamHelper

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

    This implementation is designed for 2D parameters (e.g., linear layers) and
    can handle other-dimensional parameters (e.g., 1D bias, 4D convolutional layers) by
    flattening/reshaping them.
    
    Can also operate in a hybrid mode, using an auxiliary AdamW
    optimizer for specific parameters (e.g., biases, norms, embeddings) as
    defined by a `layer_key_fn`.

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
        stochastic_rounding (bool): whether to use stochastic rounding for
            BF16 parameter updates (default: True).
        vector_reshape_muon (bool): whether to reshape 1D vectors into 2D
            matrices for muon NewtonSchulz (default: False).
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
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
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        stochastic_rounding: bool = True,
        vector_reshape_muon: bool = False,
        vector_reshape: bool = True,
        nnmf_factor: bool = False,
        # hybrid optimizer mode
        MuonWithAuxAdam: bool = False,
        layer_key_fn: Optional[Callable] = None,
        muon_adam_lr: float = 1e-4,
        adam_kwargs: Optional[dict] = None,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 should be in [0.0, 1.0). Got {beta1}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not (ns_steps > 0):
            raise ValueError(f"Newton-Schulz steps should be > 0. Got {ns_steps}")

        defaults = {
            "lr": lr, "beta1": beta1, "weight_decay": weight_decay,
            "nesterov": nesterov, "ns_steps": ns_steps, "ns_eps": ns_eps,
            "ns_coeffs": ns_coeffs, "nnmf_factor": nnmf_factor,
            "vector_reshape": vector_reshape,
            "vector_reshape_muon": vector_reshape_muon,
        }
        self.stochastic_rounding = stochastic_rounding
        
        self.MuonWithAuxAdam = MuonWithAuxAdam
        self.helper = None
        self.aux_adam = None
 
        if self.MuonWithAuxAdam:
            adam_kwargs = adam_kwargs or {}
            # Create a delegate AdamW optimizer to get its default hyperparameters.
            self.aux_adam = AdamW_adv(
                [],
                lr=muon_adam_lr,
                **adam_kwargs,
                _is_delegate=True
            )
            # Update the defaults dictionary
            defaults.update(self.aux_adam.defaults)
        
        super().__init__(params, defaults)

        if self.MuonWithAuxAdam:
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

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if self.MuonWithAuxAdam:
            optim_type = self.helper.get_optimizer_type(p)
            if optim_type == 'adam':
                # Delegate to the AdamW_adv optimizer's logic.
                # We need to temporarily "lend" our state and param_groups
                # to the delegate so it has the full context to work with,
                # especially for features like Kourkoutas-beta.
                self.aux_adam.state = self.state
                self.aux_adam.param_groups = self.param_groups
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
            if group['vector_reshape'] or state['reshaped_1d_muon']:
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

        beta1 = group['beta1']
        nesterov = group['nesterov']

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
            else:
                # Standard momentum
                update = mt_buf.clone()
            del grad_reshaped

            update = _newton_schulz_iteration(
                update,
                steps=group['ns_steps'],
                eps=group['ns_eps'],
                coeffs=group['ns_coeffs'],
            )

            update = update.view(p.shape).mul_(group['lr'])

            state['sign'] = _pack_bools(mt_buf > 0)
            _nnmf(mt_buf.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
            del mt_buf

        else: # Standard Muon logic for non-factored tensors

            if len(p.shape) >= 2 or state['reshaped_1d_muon']:

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
                else:
                    # Standard momentum
                    update = mt_buf.clone()

                # For Conv layers (4D) or other high-dim tensors, flatten to 2D
                if len(p.shape) > 2:
                    update = update.view(p.shape[0], -1)

                # NewtonSchulz
                update = _newton_schulz_iteration(
                    update,
                    steps=group['ns_steps'],
                    eps=group['ns_eps'],
                    coeffs=group['ns_coeffs'],
                )

                # Reshape back to original if we flattened or reshaped
                if len(p.shape) > 2 or state['reshaped_1d_muon']:
                    update = update.view(p.shape)

                update.mul_(group['lr'])
            
            else: # Fallback to standard SGD with momentum for 1D params (biases, etc.) when not reshaped
                # Momentum update
                mt_buf = state['momentum_buffer']
                mt_buf.mul_(beta1).add_(grad)
                if nesterov:
                    # Nesterov momentum
                    update = grad.add(mt_buf, alpha=beta1)
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
