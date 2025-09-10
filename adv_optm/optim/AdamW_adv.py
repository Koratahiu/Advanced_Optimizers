import torch
from typing import Optional

from ..util.BF16_Stochastic_Rounding import add_stochastic_
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools

class AdamW_adv(torch.optim.Optimizer):
    """
    Implements a factored AdamW algorithm.
    This is an advanced version of AdamW with optional features like
    low-rank factorization of optimizer states (SMMF), OrthoGrad, etc.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        betas (tuple[float, float]): coefficients used for computing running
            averages of gradient and its square (default: (0.9, 0.999))
        eps (float): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0)
        use_bias_correction (boolean): Turn on Adam's bias correction. (default: False)
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
        beta3 (float): The decay rate for the slow exponential moving average of
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
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_bias_correction: bool = False,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        use_atan2: bool = False,
        use_cautious: bool = False,
        use_grams: bool = False,
        use_orthograd: bool = False,
        use_AdEMAMix: bool = False,
        beta3: float = 0.9999,
        alpha: float = 5.0,
        t_alpha: int | None = None,
        factored: bool = True,
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
            "use_orthograd": use_orthograd, "use_bias_correction": use_bias_correction,
            "beta3": beta3, "alpha": alpha, "t_alpha": t_alpha,
        }
        self.stochastic_rounding = stochastic_rounding
        self.use_cautious = use_cautious
        self.use_grams = use_grams
        self.use_AdEMAMix = use_AdEMAMix
        self.factored = factored
        super().__init__(params, defaults)

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
        if p.grad is None:
            return

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

            dtype = torch.float32 if self.factored else p.dtype
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
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
                state['exp_avg'] = torch.zeros_like(p, device=device, dtype=dtype)
                if self.use_AdEMAMix:
                    state['exp_avg_slow'] = torch.zeros_like(p, dtype=dtype)
                state['exp_avg_sq'] = torch.zeros_like(p, device=device, dtype=dtype)

        beta1, beta2 = group['betas']
        if self.use_AdEMAMix:
            beta3 = group['beta3']
            alpha = group['alpha']
            t_alpha = group['t_alpha']
            current_step = state['step'] + 1
            alpha_t = alpha
            if t_alpha is not None and t_alpha > 0 and current_step < t_alpha:
                alpha_t = min(current_step * alpha / t_alpha, alpha)

        if state['factored']:
            d1, d2 = state['effective_shape']

            # Reconstruct momentum from previous step's factors
            mt = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
            if not self.use_grams:
                unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
                torch.where(unpacked_sign, mt, -mt, out=mt)
                del unpacked_sign
            # Update momentum in full-size
            grad_reshaped = grad.view(d1, d2)
            mt.mul_(beta1).add_(grad_reshaped, alpha=1.0 - beta1)
            if self.use_grams:
                mt.copy_(grad_reshaped.sign() * mt.abs())
            elif self.use_cautious:
                mask = (mt * grad_reshaped > 0).to(grad_reshaped.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                mt.mul_(mask)
                del mask

            vt = _unnmf((state['mu_v_nmf'], state['mv_v_nmf']))
            vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2)

            if self.use_AdEMAMix:
                mt_slow = _unnmf((state['mu_m_slow_nmf'], state['mv_m_slow_nmf']))
                if state['sign_slow'].dtype != torch.uint8:
                    state['sign_slow'] = state['sign_slow'].to(torch.uint8)
                unpacked_sign_slow = _unpack_bools(state['sign_slow'], original_m=d2)
                torch.where(unpacked_sign_slow, mt_slow, -mt_slow, out=mt_slow)
                del unpacked_sign_slow

                mt_slow.mul_(beta3).add_(grad_reshaped, alpha=1.0 - beta3)
                update_m = mt + (alpha_t * mt_slow)
            else:
                update_m = mt
            del grad_reshaped

            if group['use_atan2']:
                a = 1.2732395
                denom = vt.sqrt()
                update = torch.atan2(update_m, denom).mul_(a)
            else:
                denom = vt.sqrt().add_(group['eps'])
                update = update_m / denom
            del update_m, denom

            update = update.view(p.shape)
            update.mul_(group['lr'])

            # Compress updated moments and store new factors
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
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            if self.use_grams:
                exp_avg = grad.sign() * exp_avg.abs()
            elif self.use_cautious:
                mask = (exp_avg * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                exp_avg.mul_(mask)
                del mask

            if self.use_AdEMAMix:
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                update_m = exp_avg + (alpha_t * exp_avg_slow)
            else:
                update_m = exp_avg

            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if group['use_atan2']:
                a = 1.2732395
                denom = exp_avg_sq.sqrt()
                update = torch.atan2(update_m, denom).mul_(a)
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                update = update_m / denom
            del update_m, denom

            update = update.mul_(group['lr'])

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