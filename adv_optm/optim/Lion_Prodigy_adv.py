import torch
import torch.distributed as dist

import math

from typing import Tuple, Optional

from ..util import param_update
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools

class Lion_Prodigy_adv(torch.optim.Optimizer):
    """
    Implements the SMMF technique and Prodigy D-Adaptation method for Lion algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-4).
        betas (Tuple[float, float], optional): coefficients for computing
            running averages of the update (default: (0.9, 0.99)).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0).
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        vector_reshape (bool, optional): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool, optional): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        cautious_mask (bool): whether to use the cautious masking technique. (default: False).
        clip_threshold (float, optional): whether to clip the gradients norm
            per-parameter as proposed in the paper `Lions and Muons: Optimization via
            Stochastic Frank-Wolfe` (https://arxiv.org/abs/2506.04192) to make Lion more stable
            (default: 0.0).
        nnmf_factor (bool): whether to use the factorization or use the
            uncompressed optimizer. (default: True)
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
        slice_p (int): Reduce memory usage by calculating LR adaptation statistics on only every 
            pth entry of each tensor. For values greater than 1 this an an approximation to standard 
            Prodigy. Values ~11 are reasonable (default 11).
        prodigy_steps (int): If greater than zero, disable Prodigy's stepsize adjustments
            after the specified optimiser step and release all state memory required by Prodigy
            (default: 0).
        d_limiter (bool): whether to clamp the new step size estimate (`d_hat`)
            to prevent sudden, volatile increases in the adaptive step size (`d`).
            (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        orthogonal_gradient: bool = False,
        cautious_mask: bool = False,
        clip_threshold: float = 0.0,
        nnmf_factor: bool = False,
        # prodigy parameters
        beta3: float = None,
        d0: float = 1e-6,
        d_coef: float = 1,
        growth_rate: float = float('inf'),
        safeguard_warmup: bool = False,
        fsdp_in_use: bool = False,
        slice_p: int = 11,
        prodigy_steps: int = 0,
        d_limiter: bool = True,
    ):
        if not lr > 0.0:
            raise ValueError(f"Learning rate must be > 0.0, but got {lr}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"Betas should be in [0.0, 1.0], but got {betas}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Weight decay must be >= 0.0, but got {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            vector_reshape=vector_reshape,
            orthogonal_gradient=orthogonal_gradient,
            clip_threshold=clip_threshold,
            beta3=beta3, d=d0, d0=d0, d_max=d0, d_numerator=0.0, d_coef=d_coef,
            growth_rate=growth_rate, safeguard_warmup=safeguard_warmup, k=0, slice_p=slice_p,
            fsdp_in_use=fsdp_in_use,
            prodigy_steps=prodigy_steps,
            d_limiter=d_limiter,
        )
        self.stochastic_rounding = stochastic_rounding
        self.cautious_mask = cautious_mask
        self.factored = nnmf_factor
        self.fsdp_in_use = fsdp_in_use
        super().__init__(params, defaults)
        # Global state for accumulating metrics across parameter updates within a single step.
        self.init_step()

        if self.stochastic_rounding:
            # For deterministic stochastic rounding, we need to seed the generator
            # for each device used by the parameters.
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

    @property
    def supports_fused_back_pass(self) -> bool:
        return True

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    def init_step(self):
        """Resets accumulators and calculates dlr for the upcoming step."""
        self.d_denom = torch.tensor(0.0, device=self.device)

        g_group = self.param_groups[0]
        self.beta1, self.beta2 = g_group['betas']
        self.beta3 = g_group['beta3']
        if self.beta3 is None:
            self.beta3 = math.sqrt(self.beta2)

        k = g_group['k']
        self.d = g_group['d']
        lr = g_group['lr']

        self.dlr = self.d * lr

        self.d_numerator = torch.tensor(g_group.get('d_numerator', 0.0) * self.beta3, device=self.device)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: Optional[int] = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        if hasattr(p, "_fsdp_flattened"):
            self.fsdp_in_use = True

        grad = p.grad
        if grad.dtype != torch.float32 and self.factored:
            grad = grad.float()
        if group["clip_threshold"] > 0.0:
            grad_norm = torch.norm(grad.detach())
            if grad_norm > group["clip_threshold"]:
                clip_coef = group["clip_threshold"] / grad_norm
                grad.mul_(clip_coef)
        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)
        state = self.state[p]

        # State Initialization
        if 'step' not in state:
            state['step'] = 0

            should_factor = (
                self.factored and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            state['factored'] = should_factor

            dtype = torch.float32 if self.factored else p.dtype

            slice_p = group['slice_p']

            # D-Adaptation states
            state['s'] = torch.zeros_like(p.flatten()[::slice_p]).detach()
            if p.any():
                state['p0'] = p.flatten()[::slice_p].detach().clone()
            else:
                state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                state['mu_m_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype) 
                state['mv_m_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                packed_d2 = (d2 + 7) // 8
                state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
            else: # Fallback to standard Lion
                state['exp_avg'] = torch.zeros_like(p, device=p.device, dtype=dtype)

        if state['factored']:
            # Factored Path
            d1, d2 = state['effective_shape']

            # Calculate scaled reshaped gradient for factored Prodigy step (g_t * d)
            grad_scaled_reshaped = grad.view(d1, d2) * self.d

            # Reconstruct momentum m_{t-1}
            exp_avg = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
            unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
            torch.where(unpacked_sign, exp_avg, -exp_avg, out=exp_avg)
            del unpacked_sign
            if exp_avg.dtype != torch.float32:
                exp_avg = exp_avg.float()

            # Compute update term c_t = β1*m_{t-1} + (1-β1)*g_t*d
            update = torch.lerp(grad_scaled_reshaped, exp_avg, self.beta1).sign_()

            if self.cautious_mask:
                mask = (update * grad_scaled_reshaped > 0).to(grad_scaled_reshaped.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update.mul_(mask)
                del mask

            update = update.view(p.shape).mul_(self.dlr)

            # Update momentum m_t = β2*m_{t-1} + (1-β2)*d*g_t
            exp_avg.lerp_(grad_scaled_reshaped, 1 - self.beta2)
            del grad_scaled_reshaped

            # Compress new momentum m_t and store factors
            state['sign'] = _pack_bools(exp_avg > 0)
            _nnmf(exp_avg.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
            del exp_avg

        else:
            # Fallback to standard Lion logic
            exp_avg = state["exp_avg"]

            # Calculate scaled gradient for Prodigy step (g_t * d)
            grad_scaled = grad * self.d

            # Compute update term and sign for the update
            if exp_avg.dtype != torch.float32 and self.factored:
                exp_avg = exp_avg.float()

            # Compute update term c_t
            update = torch.lerp(grad_scaled, exp_avg, self.beta1).sign_()

            if self.cautious_mask:
                mask = (update * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update.mul_(mask)
                del mask

            update.mul_(self.dlr)

            # Update momentum using fused lerp
            exp_avg.lerp_(grad_scaled, 1 - self.beta2)
            del grad_scaled

        prodigy_steps = group['prodigy_steps']
        if prodigy_steps <= 0 or group['k'] < prodigy_steps:
            # --- Accumulate Prodigy stats ---
            d0, safeguard_warmup, slice_p = group['d0'], group['safeguard_warmup'], group['slice_p']
            s, p0 = state['s'], state['p0']

            grad_slice = grad.flatten()[::slice_p].float()
            p_slice = p.flatten()[::slice_p].float()
            p0 = p0.float()

            self.d_numerator.add_((self.d / d0) * self.dlr * torch.dot(grad_slice, p0.data - p_slice))

            alpha = ((self.d / d0) * self.d) if safeguard_warmup else ((self.d / d0) * self.dlr)
            s.mul_(self.beta3).add_(grad_slice, alpha=alpha)
            self.d_denom.add_(s.abs().sum())

            del s, p0, grad_slice, p_slice, alpha
        else:
            # Free memory if prodigy_steps is reached
            if 's' in state:
                del state['s']
            if 'p0' in state:
                del state['p0']

        param_update.apply_parameter_update(self, p, group, update, self.dlr)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    self.step_parameter(p, group, i)


        self.calculate_d()
        self.init_step()
        return loss

    def calculate_d(self):
        """Calculates the new `d` based on the accumulated stats."""
        g_group = self.param_groups[0]

        # Only perform d-adaptation if prodigy_steps has not been reached
        prodigy_active = not (g_group.get('prodigy_steps', 0) > 0 and g_group['k'] >= g_group['prodigy_steps'])

        if prodigy_active:
            d_max, d_coef, growth_rate = g_group['d_max'], g_group['d_coef'], g_group['growth_rate']

            if self.fsdp_in_use and dist.is_available() and dist.is_initialized():
                dist_tensor = torch.stack([self.d_numerator, self.d_denom])
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0].item()
                global_d_denom = dist_tensor[1].item()
            else:
                global_d_numerator = self.d_numerator.item()
                global_d_denom = self.d_denom.item()

            d_hat = self.d
            if global_d_denom > 0:
                d_hat = d_coef * global_d_numerator / global_d_denom
                if g_group.get('d_limiter', False):
                    d_hat = min(self.d * (2 ** 0.25), d_hat)
                if self.d == g_group['d0']:
                    self.d = max(self.d, d_hat)
                d_max = max(d_max, d_hat)
                self.d = min(d_max, self.d * growth_rate)

            for group in self.param_groups:
                group['d_numerator'] = global_d_numerator
                group['d'] = self.d
                group['d_max'] = d_max

        # Increment step counter for all groups, regardless of whether d was updated
        for group in self.param_groups:
            group['k'] += 1
