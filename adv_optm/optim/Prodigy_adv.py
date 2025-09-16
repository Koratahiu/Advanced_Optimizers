import torch
import torch.distributed as dist

import math

from ..util.BF16_Stochastic_Rounding import add_stochastic_
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools

class Prodigy_adv(torch.optim.Optimizer):
    """
    Implements a factored Prodigy/AdamW algorithm.
    This is an advanced version of Prodigy with optional features like
    low-rank factorization of optimizer states (SMMF), OrthoGrad, etc.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1)
        betas (tuple[float, float]): coefficients used for computing running
            averages of gradient and its square (default: (0.9, 0.999))
        eps (float): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        use_atan2 (bool): whether to use the atan2 update rule. (default: False)
        use_grams (bool): whether to use Grams-style updates. (default: False)
        use_cautious (bool):  whether to use cautious masking to align the gradient's
            direction with the first moment's.  (default: False)
        use_orthograd (bool): whether to use OrthoGrad.  (default: False)
        use_AdEMAMix (bool): whether to enable the AdEMAMix feature. This adds
            a second, slow-moving average of the momentum (`mt_slow`) which is
            combined with the primary momentum (`mt`) to stabilize updates,
            especially in noisy, small-batch settings. If `False`, the
            optimizer behaves as standard AdamW. (default: False)
        beta3_ema (float): The decay rate for the slow exponential moving average of
            the momentum (only used when `use_AdEMAMix` is `True`). A higher
            value (e.g., 0.9999) gives the EMA a longer memory, making it more
            stable but slower to adapt. A lower value (e.g., 0.999) is often
            better for shorter training runs. (default: 0.9999)
        alpha (float): The mixing coefficient that scales the slow momentum term
            before it is added to the fast momentum term (`update = mt + alpha * mt_slow`).
            A higher value increases the stabilizing influence of the slow
            momentum. (default: 5.0)
        t_alpha (Optional[int]): The number of steps for a linear warmup of the
            `alpha` parameter (only used when `use_AdEMAMix` is `True`). This is
            highly recommended to prevent instability at the beginning of training,
            as it gradually introduces the stabilizing slow momentum term. During
            the warmup, `alpha` ramps from 0 to its target value. If `None`,
            the scheduler is disabled and th
        factored (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: True)
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
    """

    def __init__(
        self,
        params,
        lr: float = 1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        use_atan2: bool = False,
        use_cautious: bool = False,
        use_grams: bool = False,
        use_orthograd: bool = False,
        use_AdEMAMix: bool = False,
        beta3_ema: float = 0.9999,
        alpha: float = 5.0,
        t_alpha: int | None = None,
        factored: bool = True,
        # prodigy parameters
        beta3: float = None,
        d0: float = 1e-6,
        d_coef: float = 1,
        growth_rate: float = float('inf'),
        safeguard_warmup: bool = False,
        fsdp_in_use: bool = False,
        slice_p: int = 11,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Betas should be in [0.0, 1.0). Got {betas}")
        if not (eps >= 0.0):
            raise ValueError(f"Epsilon should be >= 0.0. Got {eps}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "vector_reshape": vector_reshape, "use_atan2": use_atan2,
            "use_orthograd": use_orthograd,
            "beta3_ema": beta3_ema, "alpha": alpha, "t_alpha": t_alpha,
            "beta3": beta3, "d": d0, "d0": d0, "d_max": d0, "d_numerator": 0.0, "d_coef": d_coef,
            "growth_rate": growth_rate, "safeguard_warmup": safeguard_warmup, "k": 0, "slice_p": slice_p,
            "fsdp_in_use": fsdp_in_use,
        }
        self.stochastic_rounding = stochastic_rounding
        self.use_cautious = use_cautious
        self.use_grams = use_grams
        self.use_AdEMAMix = use_AdEMAMix
        self.factored = factored
        self.fsdp_in_use = fsdp_in_use
        super().__init__(params, defaults)
        self.init_step()

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
        """Resets accumulators and calculates dlr for the upcoming step."""
        self.d_denom = 0.0
        
        g_group = self.param_groups[0]
        self.beta1, self.beta2 = g_group['betas']
        self.beta3 = g_group['beta3']
        if self.beta3 is None:
            self.beta3 = math.sqrt(self.beta2)
        
        k = g_group['k']
        self.d = g_group['d']
        lr = g_group['lr']

        self.dlr = self.d * lr

        self.d_numerator = g_group.get('d_numerator', 0.0) * self.beta3

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        if hasattr(p, "_fsdp_flattened"):
            self.fsdp_in_use = True

        grad = p.grad
        if grad.dtype != torch.float32 and self.factored:
            grad = grad.float()
        if group["use_orthograd"]:
            grad = _orthogonalize_gradient(p, grad)
        state = self.state[p]

        # State Initialization
        if len(state) == 0:
            state['step'] = 0

            should_factor = (
                self.factored and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            state['factored'] = should_factor

            slice_p = group['slice_p']

            dtype = torch.float32 if self.factored else p.dtype
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
                if self.beta1 > 0:
                    state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype) 
                    state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                    if not self.use_grams:
                        packed_d2 = (d2 + 7) // 8
                        state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                if self.use_AdEMAMix:
                    state['mu_m_slow_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype) 
                    state['mv_m_slow_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                    packed_d2 = (d2 + 7) // 8
                    state['sign_slow'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                # Second moment (v)
                state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype) 
                state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:  # Fallback to standard AdamW for non-factored tensors
                if self.beta1 > 0:
                    state['exp_avg'] = torch.zeros_like(p, device=device, dtype=dtype)
                if self.use_AdEMAMix:
                    state['exp_avg_slow'] = torch.zeros_like(p, dtype=dtype)
                state['exp_avg_sq'] = torch.zeros_like(p, device=device, dtype=dtype)

            state['s'] = torch.zeros_like(p.flatten()[::slice_p]).detach()
            if p.any():
                state['p0'] = p.flatten()[::slice_p].detach().clone()
            else:
                state['p0'] = torch.tensor(0, device=device, dtype=p.dtype)

        if self.use_AdEMAMix:
            beta3_ema = group['beta3_ema']
            alpha = group['alpha']
            t_alpha = group['t_alpha']
            current_step = state['step'] + 1
            alpha_t = alpha
            if t_alpha is not None and t_alpha > 0 and current_step < t_alpha:
                alpha_t = min(current_step * alpha / t_alpha, alpha)

        if state['factored']:
            d1, d2 = state['effective_shape']

            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors
            if self.beta1 > 0:
                mt = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
                if not self.use_grams:
                    unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
                    torch.where(unpacked_sign, mt, -mt, out=mt)
                    del unpacked_sign
                # Update momentum in full-size
                mt.mul_(self.beta1).add_(grad_reshaped, alpha=self.d * (1.0 - self.beta1))
                if self.use_grams:
                    mt.copy_(grad_reshaped.sign() * mt.abs())
                elif self.use_cautious:
                    mask = (mt * grad_reshaped > 0).to(grad_reshaped.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    mt.mul_(mask)
                    del mask

            vt = _unnmf((state['mu_v_nmf'], state['mv_v_nmf']))
            vt.mul_(self.beta2).addcmul_(grad_reshaped, grad_reshaped, value=self.d * self.d * (1.0 - self.beta2))

            if self.use_AdEMAMix:
                mt_slow = _unnmf((state['mu_m_slow_nmf'], state['mv_m_slow_nmf']))
                if state['sign_slow'].dtype != torch.uint8:
                    state['sign_slow'] = state['sign_slow'].to(torch.uint8)
                unpacked_sign_slow = _unpack_bools(state['sign_slow'], original_m=d2)
                torch.where(unpacked_sign_slow, mt_slow, -mt_slow, out=mt_slow)
                del unpacked_sign_slow
                mt_slow.mul_(beta3_ema).add_(grad_reshaped, alpha=self.d * (1.0 - beta3_ema))
                update = mt + (alpha_t * mt_slow) if self.beta1 > 0 else grad_reshaped + (alpha_t * mt_slow)
            else:
                update = mt.clone() if self.beta1 > 0 else grad_reshaped.clone()
            del grad_reshaped

            if group['use_atan2']:
                a = 1.2732395
                denom = vt.sqrt()
                update.atan2_(denom).mul_(a)
            else:
                denom = vt.sqrt()
                update.div_(denom.add_(self.d * group['eps']))
            del denom

            update.view(p.shape).mul_(self.dlr)

            # Compress updated moments and store new factors
            if self.beta1 > 0:
                if not self.use_grams:
                    state['sign'] = _pack_bools(mt > 0)
                _nnmf(mt.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
                del mt
            if self.use_AdEMAMix:
                state['sign_slow'] = _pack_bools(mt_slow > 0)
                _nnmf(mt_slow.abs(), out=(state['mu_m_slow_nmf'], state['mv_m_slow_nmf']))
                del mt_slow
            _nnmf(vt, out=(state['mu_v_nmf'], state['mv_v_nmf']))
            del vt

        else:  # Standard AdamW logic for non-factored tensors
            exp_avg_sq = state['exp_avg_sq']

            if self.beta1 > 0:
                exp_avg = state['exp_avg']
                exp_avg.mul_(self.beta1).add_(grad, alpha=self.d * (1.0 - self.beta1))
                if self.use_grams:
                    exp_avg = grad.sign() * exp_avg.abs()
                elif self.use_cautious:
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    exp_avg.mul_(mask)
                    del mask

            if self.use_AdEMAMix:
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_slow.mul_(beta3_ema).add_(grad, alpha=self.d * (1.0 - beta3_ema))
                update = exp_avg + (alpha_t * exp_avg_slow) if self.beta1 > 0 else grad + (alpha_t * exp_avg_slow)
            else:
                update = exp_avg.clone() if self.beta1 > 0 else grad.clone()

            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad.conj(), value=self.d * self.d * (1.0 - self.beta2))

            if group['use_atan2']:
                a = 1.2732395
                denom = exp_avg_sq.sqrt()
                update.atan2_(denom).mul_(a)
            else:
                denom = exp_avg_sq.sqrt()
                update.div_(denom.add_(self.d * group['eps']))
            del denom

            update.mul_(self.dlr)

        # --- Accumulate Prodigy stats ---
        d0, safeguard_warmup, slice_p = group['d0'], group['safeguard_warmup'], group['slice_p']
        s, p0 = state['s'], state['p0']
        grad_flat = grad.flatten().float()
        p_flat = p.data.flatten().float()
        p0 = p0.float()

        self.d_numerator += (self.d / d0) * self.dlr * torch.dot(grad_flat[::slice_p], p0.data - p_flat[::slice_p]).item()

        alpha = ((self.d / d0) * self.d) if safeguard_warmup else ((self.d / d0) * self.dlr)
        s.mul_(self.beta3).add_(grad_flat[::slice_p], alpha=alpha)
        self.d_denom += s.abs().sum().item()

        del s, p0, grad_flat, p_flat, alpha

        # Decoupled weight decay
        if group["weight_decay"] != 0:
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                add_stochastic_(p.data, p.data, alpha=-group["weight_decay"] * self.dlr)
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * self.dlr)

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


        self.calculate_d()
        self.init_step()
        return loss

    def calculate_d(self):
        """Calculates the new `d` based on the accumulated stats."""
        g_group = self.param_groups[0]
        d_max, d_coef, growth_rate = g_group['d_max'], g_group['d_coef'], g_group['growth_rate']
        
        if self.fsdp_in_use and dist.is_available() and dist.is_initialized():
            # Use the device of the first parameter to avoid hardcoding '.cuda()'
            device = self.param_groups[0]['params'][0].device
            dist_tensor = torch.tensor([self.d_numerator, self.d_denom], device=device)
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
            global_d_numerator = dist_tensor[0].item()
            global_d_denom = dist_tensor[1].item()
        else:
            global_d_numerator = self.d_numerator
            global_d_denom = self.d_denom

        d_hat = self.d
        if global_d_denom > 0:
            d_hat = d_coef * global_d_numerator / global_d_denom
            if self.d == g_group['d0']:
                self.d = max(self.d, d_hat)
            d_max = max(d_max, d_hat)
            self.d = min(d_max, self.d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d'] = self.d
            group['d_max'] = d_max
            group['k'] += 1
