import torch

import math

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _init_fisher_wd_scaler, _get_fisher_wd_scaler
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm, scale_eps
from ..util.centered_decay import _init_anchor
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision

A = 4 / math.pi

@torch.no_grad()
def _init_auxadam_state(self, p, group):
    state = self.state[p]

    state['step'] = 0

    req_precision = group.get('adam_state_precision', 'auto')
    is_vector = len(p.shape) == 1 and not group.get('vector_reshape', False)

    state['factored'] = (
        (group.get('adam_nnmf_factor', False) or req_precision == 'factored') and
        not is_vector
    )

    state['factored_2nd'] = group.get('adam_factored_2nd', False) and not is_vector

    actual_precision = 'auto' if req_precision == 'factored' else req_precision
    group['adam_actual_state_precision'] = actual_precision

    dtype = torch.float32 if (state['factored'] or req_precision == 'factored') else p.dtype
    device = p.device

    if state['factored']:
        state['effective_shape'] = _get_effective_shape(p.numel())
        d1, d2 = state['effective_shape']
        # First moment (m)
        if group['adam_betas'][0] > 0:
            state['mu_m_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
            state['mv_m_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
            packed_d2 = (d2 + 7) // 8
            state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
            state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
        # Second moment (v)
        state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
        state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
    else:  # Fallback to standard AdamW for non-factored tensors
        if group['adam_betas'][0] > 0:
            init_state_tensor(state, 'exp_avg', p.shape, actual_precision, p.device, dtype)

        if state.get('factored_2nd', False):
            state['effective_shape'] = _get_effective_shape(p.numel())
            d1, d2 = state['effective_shape']
            state['mu_v_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
            state['mv_v_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
            state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
        else:
            init_state_tensor(state, 'exp_avg_sq', p.shape, actual_precision, p.device, dtype, non_neg=True)

    if group.get('adam_spectral_normalization', False) and is_spectral(p):
        init_spectral_norm(state, p)

    _init_anchor(p, state, group)
    _init_fisher_wd_scaler(group, state, p)


@torch.no_grad()
def _adam_step_parameter(self, p, grad, state, group, beta1_adam, beta2_adam, sqrt_bias_correction2, step_size, random_int_tensor, random_int_state_tensor=None):
    grad = upcast_grad_for_precision(grad, state, group.get('adam_state_precision', 'auto'))

    grad = _orthogonalize_gradient(p, grad, group.get("adam_orthogonal_gradient"))

    if hasattr(self, 'kourkoutas_helper') and self.kourkoutas_helper:
        # Accumulate current grad's norm for the *next* step
        self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)

    nesterov = group.get('adam_nesterov', False)
    nesterov_coef = group.get('adam_nesterov_coef', None)
    use_mt = group['adam_betas'][0] > 0

    adaptive_eps = scale_eps(group['adam_eps'], p)

    if state['factored']:
        d1, d2 = state['effective_shape']
        grad_reshaped = grad.view(d1, d2)

        # Reconstruct momentum from previous step's factors
        if use_mt:
            mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True, shifter=state['shifter'])

            # Update momentum in full-size
            mt.lerp_(grad_reshaped, 1.0 - beta1_adam)

            # Factorize
            state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(mt.clone(), signed=True, shifter=state['shifter'])

            update_mt = mt

            if nesterov:
                nv_coef = beta1_adam if nesterov_coef is None else nesterov_coef
                update_mt = update_mt.lerp_(grad_reshaped, 1 - nv_coef)

        vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])
        if isinstance(beta2_adam, torch.Tensor) and beta2_adam.dim() > 0:
            vt = vt.view_as(p).mul_(beta2_adam).addcmul_(grad, grad * (1.0 - beta2_adam)).view_as(grad_reshaped)
        else:
            vt.mul_(beta2_adam).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2_adam)

        if use_mt:
            update = update_mt
        else:
            update = grad_reshaped.clone()

        # Factorize
        state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt, signed=False, shifter=state['shifter'])

        if group.get('adam_use_atan2'):
            denom = vt.sqrt_()
            denom.div_(sqrt_bias_correction2)
            update.atan2_(denom)
        else:
            denom = vt.sqrt_()
            denom.div_(sqrt_bias_correction2).add_(adaptive_eps)
            update.div_(denom)

        wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, group.get('adam_use_atan2'))

        del vt

        update = update.view(p.shape)

    else:  # Standard AdamW logic for non-factored tensors
        actual_precision = group.get('adam_actual_state_precision', 'auto')
        factored_2nd = state.get('factored_2nd', False)

        if use_mt:
            exp_avg = get_state(state, 'exp_avg', actual_precision)
            exp_avg.lerp_(grad, 1.0 - beta1_adam)

            update_mt = exp_avg.clone()

            if nesterov:
                nv_coef = beta1_adam if nesterov_coef is None else nesterov_coef
                update_mt = update_mt.lerp_(grad, 1 - nv_coef)

            set_state(state, 'exp_avg', exp_avg, actual_precision, random_int_state_tensor)

        update = update_mt if use_mt else grad.clone()

        if factored_2nd:
            d1, d2 = state['effective_shape']
            exp_avg_sq = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False, shifter=state['shifter'])
            exp_avg_sq = exp_avg_sq.view(p.shape)
        else:
            exp_avg_sq = get_state(state, 'exp_avg_sq', actual_precision)

        grad_vt = grad.float() if factored_2nd else grad

        if isinstance(beta2_adam, torch.Tensor) and beta2_adam.dim() > 0:
            exp_avg_sq.mul_(beta2_adam).addcmul_(grad_vt, grad_vt * (1.0 - beta2_adam))
        else:
            exp_avg_sq.mul_(beta2_adam).addcmul_(grad_vt, grad_vt, value=1.0 - beta2_adam)

        if factored_2nd:
            state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(exp_avg_sq.view(d1, d2), signed=False, shifter=state['shifter'])
        else:
            set_state(state, 'exp_avg_sq', exp_avg_sq, actual_precision, random_int_state_tensor, non_neg=True)
        del random_int_state_tensor

        if group.get('adam_use_atan2'):
            denom = exp_avg_sq.sqrt()
            denom.div_(sqrt_bias_correction2)
            update.atan2_(denom.to(update.dtype))
        else:
            denom = exp_avg_sq.sqrt()
            denom.div_(sqrt_bias_correction2).add_(adaptive_eps)
            update.div_(denom.to(update.dtype))

        wd_scaler = _get_fisher_wd_scaler(group, state.get("wd_scaler"), p, denom, group.get('adam_use_atan2'))
        del denom

    update_scaling = step_size * A if group.get('adam_use_atan2') else step_size

    if group.get('adam_spectral_normalization', False):
        update = scale_update(p, update, update_scaling, state=state)
    else:
        update.mul_(update_scaling)

    param_update.apply_parameter_update(self, p, group, update, step_size, group["adam_weight_decay"], random_int_tensor=random_int_tensor, wd_scaler=wd_scaler)