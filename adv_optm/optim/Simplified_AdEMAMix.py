import torch
from typing import Optional, Callable

import math

from ..util.BF16_Stochastic_Rounding import add_stochastic_, set_seed as set_stochastic_rounding_seed
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools
from ..util.Kourkoutas import KourkoutasHelper

# A little helper from the original simplified_AdEMAMix
def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

    def f(beta, eps=1e-8):
        return math.log(0.5)/math.log(beta+eps)-1

    def f_inv(t):
        return math.pow(0.5, 1/(t+1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0-a) * f(beta_start) + a * f(beta_end))
    return beta_end

class Simplified_AdEMAMix(torch.optim.Optimizer):
    """
    Implements the Simplified AdEMAMix algorithm.
    Refactored from:
    https://github.com/DepenM/Simplified-AdEMAMix/blob/main/simplified_AdEMAMix.py

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-5)
        betas (tuple[float, float]): coefficients used for computing running
            averages of gradient and its square (default: (0.99, 0.999))
        eps (float): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0).
        alpha_grad (float): Coeficient for mixing the current gradient and EMA. for small batch
            sizes set it to high values, up to 100. And for large batch sized set it to small
            value, down to 0. (default: 100)
        beta1_warmup (int, optional): number of warmup steps used to increase beta1 (default: None)
        min_beta1 (float, optional): minimum value of beta1 to start from (default 0.9)
        vector_reshape (bool): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        orthogonal_gradient (bool): whether to use OrthoGrad. (default: False)
        kourkoutas_beta (bool): whether to enable the layer-wise dynamic β₂ logic.
            If `False`, the optimizer behaves as standard Simplified_AdEMAMix. (default: False)
        beta2_min (float): The minimum value for dynamic β₂, used during periods of
            high gradient variance ("sunspikes"). Must be less than `betas[1]`.
            (default: 0.88)
        ema_alpha (float): The decay rate for the Exponential Moving Average (EMA) of
            the pooled gradient norms. Corresponds to `α` in the paper.
            (default: 0.93)
        tiny_spike (float): A small constant added to the denominator of the
            "sunspike" ratio calculation to prevent division by zero. Corresponds
            to `ε_spike` in the paper. (default: 1e-9)
        k_warmup_steps (int): The number of initial steps during which β₂ is held
            at a fixed beta2 value before the
            dynamic logic activates. (default: 0)
        k_logging (int): if > 0 and kourkoutas_beta=True, enables periodic console
            logging of Kourkoutas-β statistics (min, max, mean of `β₂` across layers)
            every logging steps. Useful for debugging and tuning. Set to 0 to disable
            logging (default: 0). 
        layer_key_fn (Optional[Callable]): A function that takes a parameter `p`
            and returns a unique, hashable key representing its "layer" or "bucket".
            If `None`, parameters are bucketed by their memory ID (tensor-wise).
            (default: None)
        nnmf_factor (bool): whether to use the factorization or disable it to use
            the uncompressed optimizer. (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-5,
        betas: tuple[float, float] = (0.99, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        alpha_grad: float = 100.0,
        beta1_warmup: int | None = None,
        min_beta1: float | None = 0.9,
        use_bias_correction: bool = True,
        vector_reshape: bool = True,
        stochastic_rounding: bool = True,
        orthogonal_gradient: bool = False,
        kourkoutas_beta: bool = False,
        beta2_min: float = 0.9,
        ema_alpha: float = 0.95,
        tiny_spike: float = 1e-9,
        k_warmup_steps: int = 0,
        k_logging: int = 0,
        layer_key_fn: Optional[Callable] = None,
        nnmf_factor: bool = False,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Betas should be in [0.0, 1.0). Got {betas}")
        if not (eps >= 0.0):
            raise ValueError(f"Epsilon should be >= 0.0. Got {eps}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if not 0.0 <= alpha_grad:
            raise ValueError("Invalid alpha value: {}".format(alpha_grad))
        if kourkoutas_beta and not (betas[1] > beta2_min): raise ValueError(f"For Kourkoutas-β, betas[1] (as beta2_max) must be > beta2_min. Got {betas[1]} and {beta2_min}")

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "alpha_grad": alpha_grad, "beta1_warmup": beta1_warmup, "min_beta1": min_beta1,
            "vector_reshape": vector_reshape,
            "orthogonal_gradient": orthogonal_gradient, "use_bias_correction": use_bias_correction,
            "kourkoutas_beta": kourkoutas_beta, "beta2_min": beta2_min, "ema_alpha": ema_alpha,
            "tiny_spike": tiny_spike, "k_warmup_steps": k_warmup_steps, "k_logging": k_logging,
        }
        self.stochastic_rounding = stochastic_rounding
        self.factored = nnmf_factor
        self.kourkoutas_beta = kourkoutas_beta
        self.layer_key_fn = layer_key_fn
        super().__init__(params, defaults)

        if self.kourkoutas_beta:
            self.kourkoutas_helper = KourkoutasHelper(self)

        if self.stochastic_rounding:
            # For deterministic stochastic rounding, we need to seed the generator
            # for each device used by the parameters.
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                set_stochastic_rounding_seed(device)

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
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']

                # First moment (m)
                state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=dtype) 
                state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                packed_d2 = (d2 + 7) // 8
                state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                # Second moment (v)
                state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=dtype) 
                state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
            else:  # Fallback to standard optimizer for non-factored tensors
                state['exp_avg'] = torch.zeros_like(p, device=device, dtype=dtype)
                state['exp_avg_sq'] = torch.zeros_like(p, device=device, dtype=dtype)
            
            if group['use_bias_correction']:
                state['num_sum'] = 0.0
                state['den_sum'] = 0.0
            else:
                state['num_sum'] = 1.0
                state['den_sum'] = 1.0

        beta1_final, beta2 = group["betas"]

        current_step = state['step']
        if group.get('kourkoutas_beta', False):
            # Call prepare_step() once at the beginning of the step for all params
            self.kourkoutas_helper.maybe_prepare_step(current_step)
            # Accumulate current grad's norm for the *next* step
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)
            # Get the dynamic beta2 calculated in prepare_step()
            beta2 = self.kourkoutas_helper.get_beta2(p, group)

        beta1_warmup = group["beta1_warmup"]
        alpha_grad = group["alpha_grad"]

        if beta1_warmup is not None:
            step = state['step'] + 1
            beta1 = linear_hl_warmup_scheduler(step, beta_end=beta1_final, beta_start=group['min_beta1'], warmup=beta1_warmup)
        else:
            beta1 = beta1_final

        if group['use_bias_correction']:
            state['num_sum'] = beta1 * state['num_sum'] + 1.0
            if group.get('kourkoutas_beta', False):
                state['den_sum'] = group['betas'][1] * state['den_sum'] + (1.0 - group['betas'][1])
            else:
                state['den_sum'] = beta2 * state['den_sum'] + (1.0 - beta2)

        if state['factored']:
            d1, d2 = state['effective_shape']

            # Reconstruct momentum from previous step's factors
            mt = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
            unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
            torch.where(unpacked_sign, mt, -mt, out=mt)
            del unpacked_sign
            # Update momentum in full-size
            grad_reshaped = grad.view(d1, d2)
            mt.mul_(beta1).add_(grad_reshaped, alpha=1.0)

            vt = _unnmf((state['mu_v_nmf'], state['mv_v_nmf']))
            vt.mul_(beta2).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2)

            update = torch.add(mt, grad_reshaped, alpha=alpha_grad)
            del grad_reshaped

            denom = vt.sqrt().add_(group['eps'] * math.sqrt(state['den_sum']))
            update.div_(denom)
            del denom

            if group['use_bias_correction']:
                update = (update / state['num_sum']) * math.sqrt(state['den_sum'])

            update = update.view(p.shape).mul_(group['lr'])

            # Compress updated moments and store new factors
            state['sign'] = _pack_bools(mt > 0)
            _nnmf(mt.abs(), out=(state['mu_m_nmf'], state['mv_m_nmf']))
            del mt
            _nnmf(vt, out=(state['mu_v_nmf'], state['mv_v_nmf']))
            del vt

        else:  # Standard optimizer logic for non-factored tensors
            exp_avg_sq = state['exp_avg_sq']

            exp_avg = state['exp_avg']
            exp_avg.mul_(beta1).add_(grad, alpha=1.0)

            update = torch.add(exp_avg, grad, alpha=alpha_grad)

            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(group['eps'] * math.sqrt(state['den_sum']))
            update.div_(denom)
            del denom

            if group['use_bias_correction']:
                update = (update / state['num_sum']) * math.sqrt(state['den_sum'])

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