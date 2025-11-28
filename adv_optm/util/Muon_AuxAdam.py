import torch
from ..util import param_update
from ..util.Effective_Shape import _get_effective_shape
from ..util.NNMF import _nnmf,_unnmf
from ..util.One_Bit_Boolean import _pack_bools, _unpack_bools
from ..util.OrthoGrad import _orthogonalize_gradient

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
            if not group.get('adam_grams_moment'):
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
def _adam_step_parameter(self, p, grad, state, group, lr, bias_correction1, bias_correction2):
    if grad.dtype != torch.float32 and state.get('factored', False):
        grad = grad.float()
    if group.get("adam_orthogonal_gradient"):
        grad = _orthogonalize_gradient(p, grad)

    beta1_adam, beta2_adam = group['adam_betas']

    if group.get('adam_kourkoutas_beta', False):
        # Accumulate current grad's norm for the *next* step
        self.kourkoutas_helper.accumulate_gradient_sq_norm(p, grad)
        # Get the dynamic beta2_adam calculated in prepare_step()
        beta2_adam = self.kourkoutas_helper.get_beta2(p, group)

    step_size = lr / bias_correction1

    if group.get('adam_use_AdEMAMix'):
        beta3_ema = group['adam_beta3_ema']
        alpha = group['adam_alpha']

    if state['factored']:
        d1, d2 = state['effective_shape']
        grad_reshaped = grad.view(d1, d2)

        # Reconstruct momentum from previous step's factors
        if beta1_adam > 0:
            mt = _unnmf((state['mu_m_nmf'], state['mv_m_nmf']))
            if not group.get('adam_grams_moment'):
                unpacked_sign = _unpack_bools(state['sign'], original_m=d2)
                torch.where(unpacked_sign, mt, -mt, out=mt)
                del unpacked_sign

            # Update momentum in full-size
            mt.lerp_(grad_reshaped, 1.0 - beta1_adam)

            if group.get('adam_grams_moment'):
                update_mt = (grad_reshaped.sign().mul_(mt.abs()))
            elif group.get('adam_cautious_mask'):
                mask = (mt * grad_reshaped > 0).to(grad_reshaped.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update_mt = mt.mul(mask)
                del mask
            else:
                update_mt = mt.clone()

        vt = _unnmf((state['mu_v_nmf'], state['mv_v_nmf']))
        vt.mul_(beta2_adam).addcmul_(grad_reshaped, grad_reshaped, value=1.0 - beta2_adam)

        if group.get('adam_use_AdEMAMix'):
            mt_slow = _unnmf((state['mu_m_slow_nmf'], state['mv_m_slow_nmf']))
            if state['sign_slow'].dtype != torch.uint8:
                state['sign_slow'] = state['sign_slow'].to(torch.uint8)
            unpacked_sign_slow = _unpack_bools(state['sign_slow'], original_m=d2)
            torch.where(unpacked_sign_slow, mt_slow, -mt_slow, out=mt_slow)
            del unpacked_sign_slow

            mt_slow.lerp_(grad_reshaped, 1.0 - beta3_ema)

            if beta1_adam > 0:
                update = update_mt.add_(mt_slow, alpha=alpha)
            else:
                update = torch.add(grad_reshaped, mt_slow, alpha=alpha)
        else:
            update = update_mt if beta1_adam > 0 else grad_reshaped.clone()
        del grad_reshaped

        if group['adam_use_atan2']:
            a = 1.2732395
            denom = (vt.sqrt() / (bias_correction2**0.5))
            update.atan2_(denom).mul_(a)
        else:
            denom = (vt.sqrt() / (bias_correction2**0.5)).add_(group['adam_eps'])
            update.div_(denom)
        del denom

        update = update.view(p.shape).mul_(step_size)

        # Compress updated moments and store new factors
        if beta1_adam > 0:
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

        if beta1_adam > 0:
            exp_avg = state['exp_avg']
            exp_avg.lerp_(grad, 1.0 - beta1_adam)

            if group.get('adam_grams_moment'):
                update_mt = grad.sign().mul_(exp_avg.abs())
            elif group.get('adam_cautious_mask'):
                mask = (exp_avg * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                update_mt = exp_avg.mul(mask)
                del mask
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

        exp_avg_sq.mul_(beta2_adam).addcmul_(grad, grad.conj(), value=1 - beta2_adam)

        if group.get('adam_use_atan2'):
            a = 1.2732395
            denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5))
            update.atan2_(denom).mul_(a)
        else:
            denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group['adam_eps'])
            update.div_(denom)
        del denom

        update.mul_(step_size)

    param_update.apply_parameter_update(self, p, group, update, lr, group["adam_weight_decay"])