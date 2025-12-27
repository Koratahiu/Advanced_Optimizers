import torch

import math

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.update_util import _grams_update, _cautious_update

A = torch.as_tensor(4 / math.pi)

@torch.no_grad()
def _init_auxadam_state(self, p, group):
    state = self.state[p]

    state['step'] = 0

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
def _adam_step_parameter(self, p, grad, state, group, is_compiled, random_int_tensor):

    step = state['step']

    beta1_adam, beta2_adam = group['adam_betas']

    if self.kourkoutas_helper:
        # Prepare Kourkoutas-β once per optimizer step.
        self.kourkoutas_helper.maybe_prepare_step(step, p.device)
        # Get the dynamic beta2_adam calculated in prepare_step()
        beta2_adam = self.kourkoutas_helper.get_beta2(p, group)

    if group['adam_use_bias_correction']:
        current_step = step + 1
        beta1_adam, beta2_adam = group['adam_betas']
        bias_correction1 = 1.0 - beta1_adam ** current_step
        sqrt_bias_correction2 = (1.0 - beta2_adam ** current_step)**0.5
    else:
        bias_correction1 = 1.0
        sqrt_bias_correction2 = 1.0

    state['step'] += 1

    step_size = group['lr'] / bias_correction1

    @torch.compile(fullgraph=True, disable= not is_compiled)
    def compiled_muon_step_parameter(state, grad, group, step_size, sqrt_bias_correction2, random_int_tensor):
        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()
        if group.get("adam_orthogonal_gradient"):
            grad = _orthogonalize_gradient(p, grad)

        if self.kourkoutas_helper:
            # Accumulate current grad's norm for the *next* step
            self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)

        if group.get('adam_use_AdEMAMix'):
            beta3_ema = group['adam_beta3_ema']
            alpha = group['adam_alpha']

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum from previous step's factors
            if beta1_adam > 0:
                mt = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True)

                # Update momentum in full-size
                mt.lerp_(grad_reshaped, 1.0 - beta1_adam)

                # Factorize
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(mt.clone(), signed=True)

                if group.get('adam_grams_moment'):
                    update_mt = _grams_update(mt, grad_reshaped, inplace=True)
                elif group.get('adam_cautious_mask'):
                    update_mt = _cautious_update(mt, grad_reshaped, inplace=True)
                else:
                    update_mt = mt

            vt = _reconstruct_state((state['mu_v_nmf'], state['mv_v_nmf']), signed=False)
            vt.mul_(beta2_adam).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2_adam)

            if group.get('adam_use_AdEMAMix'):
                mt_slow = _reconstruct_state((state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'], d2), signed=True)

                mt_slow.lerp_(grad_reshaped, 1.0 - beta3_ema)

                if beta1_adam > 0:
                    update = update_mt.add_(mt_slow, alpha=alpha)
                else:
                    update = grad_reshaped.add(mt_slow, alpha=alpha)
                # Factorize
                state['mu_m_slow_nmf'], state['mv_m_slow_nmf'], state['sign_slow'] = _factorize_state(mt_slow, signed=True)
                del mt_slow
            else:
                if beta1_adam > 0:
                    update = update_mt
                else:
                    update = grad_reshaped.clone()

            if group['adam_use_atan2']:
                denom = vt.sqrt()
                denom.div_(sqrt_bias_correction2)
                update.atan2_(denom)
            else:
                denom = vt.sqrt()
                denom.div_(sqrt_bias_correction2).add_(group['adam_eps'])
                update.div_(denom)
            del denom

            # Factorize
            state['mu_v_nmf'], state['mv_v_nmf'] = _factorize_state(vt, signed=False)
            del vt

            update_scaling = step_size * A if group['use_atan2'] else step_size
            update = update.view(p.shape).mul_(update_scaling)

        else:  # Standard AdamW logic for non-factored tensors
            if beta1_adam > 0:
                exp_avg = state['exp_avg']
                exp_avg.lerp_(grad, 1.0 - beta1_adam)

                if group.get('adam_grams_moment'):
                    update_mt = _grams_update(exp_avg, grad)
                elif group.get('adam_cautious_mask'):
                    update_mt = _cautious_update(exp_avg, grad)
                else:
                    update_mt = exp_avg.clone()

            if group.get('adam_use_AdEMAMix'):
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_slow.lerp_(grad, 1.0 - beta3_ema)

                if beta1_adam > 0:
                    update = update_mt.add_(exp_avg_slow, alpha=alpha)
                else:
                    update = torch.add(grad, exp_avg_slow, alpha=alpha)
            else:
                update = update_mt if beta1_adam > 0 else grad.clone()

            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.mul_(beta2_adam).addcmul_(grad, grad, value=1 - beta2_adam)

            if group.get('adam_use_atan2'):
                denom = exp_avg_sq.sqrt()
                denom.div_(sqrt_bias_correction2)
                update.atan2_(denom)
            else:
                denom = exp_avg_sq.sqrt()
                denom.div_(sqrt_bias_correction2).add_(group['adam_eps'])
                update.div_(denom)
            del denom

            update_scaling = step_size * A if group['adam_use_atan2'] else step_size
            update.mul_(update_scaling)

        param_update.apply_parameter_update(self, p, group, update, step_size, group["adam_weight_decay"], random_int_tensor=random_int_tensor)

    compiled_muon_step_parameter(state, grad, group, step_size, sqrt_bias_correction2, random_int_tensor)
