import torch

import math

from ...util.scaled_optm import scale_eps
from ...util.foreach.fh_scaled_optm import foreach_scale_update, _foreach_collect_spectral_vars

from ...util.foreach.fh_orthograd import _foreach_orthogonalize_gradient
from ...util.foreach.fh_param_update import foreach_apply_parameter_update
from ...util.foreach.fh_update_util import _foreach_get_fisher_wd_scaler

A = 4 / math.pi

@torch.no_grad()
def foreach_step(self, group: dict) -> None:
    """
    Foreach/multi-tensor optimization step.
    Uses torch._foreach_* ops where available; falls back to iterative
    list-comprehension style for ops without foreach support (e.g. atan2).
    """
    params = [p for p in group['params'] if p.grad is not None]
    if not params:
        return

    grads = [p.grad for p in params]

    exp_avgs = []
    exp_avg_sqs = []
    state_steps = []
    anchors = []
    adaptive_eps = []

    for p in params:
        state = self.state[p]
        self._init_state(p, group)
        exp_avgs.append(state['exp_avg'])
        exp_avg_sqs.append(state['exp_avg_sq'])
        state['step'] += 1
        state_steps.append(torch.tensor(state['step']))

        # Collect anchor for centered_wd (full mode only in foreach)
        cwd = group.get('centered_wd', 0.0)
        if cwd != 0.0 and 'anchor_data' in state:
            anchors.append(state['anchor_data'])

        eps = group['eps']
        adaptive_eps.append(scale_eps(eps, p))

    # Group anchors and adaptive_eps by (device, dtype) to match the parameter grouping
    anchor_groups: dict = {}
    eps_groups: dict = {}
    for i, p in enumerate(params):
        key = (p.device, p.dtype)
        if cwd != 0.0 and i < len(anchors):
            if key not in anchor_groups:
                anchor_groups[key] = []
            anchor_groups[key].append(anchors[i])
        if key not in eps_groups:
            eps_groups[key] = []
        eps_groups[key].append(adaptive_eps[i])

    lr = torch.as_tensor(group['lr']) if group.get('compiled_optimizer', False) else group['lr']
    if group.get('compiled_optimizer', False):
        self._compiled_foreach_step = torch.compile(
            _foreach_step,
            fullgraph=True,
            dynamic=False
        )
        foreach_step_fn = self._compiled_foreach_step
    else:
        foreach_step_fn = _foreach_step

    foreach_step_fn(self, group, params, grads, exp_avgs, exp_avg_sqs, state_steps, anchors, anchor_groups, eps_groups, lr, adaptive_eps)


@torch.no_grad()
def _foreach_step(self, group: dict, params, grads, exp_avgs, exp_avg_sqs, state_steps, anchors, anchor_groups, eps_groups, lr, adaptive_eps) -> None:
    beta1, beta2 = group['betas']
    use_atan2 = group['use_atan2']
    nesterov = group.get('nesterov', False)
    nesterov_coef = group.get('nesterov_coef', None)
    use_mt = beta1 > 0
    use_bias_correction = group.get('use_bias_correction', True)
    fisher_wd = group.get('fisher_wd', False)
    cwd = group.get('centered_wd', 0.0)

    # Orthogonalize gradients if needed
    ortho_mode = group.get('orthogonal_gradient', 'disabled')
    grads = _foreach_orthogonalize_gradient(params, grads, ortho_mode)

    # Group tensors by (device, dtype) for foreach ops
    grouped = _group_by_device_dtype(params, grads, exp_avgs, exp_avg_sqs, state_steps)

    for (g_params, g_grads, g_exp_avgs, g_exp_avg_sqs, g_steps) in grouped.values():
        if len(g_params) == 0:
            continue

        # Bias correction
        if use_bias_correction:
            step_tensor = g_steps[0]
            bc1_scalar = 1.0 - beta1 ** step_tensor
            sbc2_scalar = (1.0 - beta2 ** step_tensor) ** 0.5
        else:
            bc1_scalar = 1.0
            sbc2_scalar = 1.0

        step_size = lr / bc1_scalar if use_bias_correction else lr

        # Second moment update
        torch._foreach_mul_(g_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(g_exp_avg_sqs, g_grads, g_grads, value=1.0 - beta2)

        # First moment update
        if use_mt:
            torch._foreach_lerp_(g_exp_avgs, g_grads, 1.0 - beta1)

        # Compute denom: sqrt(exp_avg_sq) / sbc2 + eps
        denom = torch._foreach_sqrt(g_exp_avg_sqs)
        torch._foreach_div_(denom, sbc2_scalar)
        if not use_atan2:
            group_adaptive_eps = eps_groups.get((g_params[0].device, g_params[0].dtype), None)
            torch._foreach_add_(denom, group_adaptive_eps)

        # Compute updates (unscaled)
        if use_mt:
            updates = torch._foreach_clone(g_exp_avgs)
            if nesterov:
                nv_coef = beta1 if nesterov_coef is None else nesterov_coef
                torch._foreach_lerp_(updates, g_grads, 1.0 - nv_coef)
        else:
            updates = torch._foreach_clone(g_grads)

        if use_atan2:
            for i, p in enumerate(g_params):
                updates[i].atan2_(denom[i])
        else:
            torch._foreach_div_(updates, denom)

        # Spectral normalization via foreach_scale_update
        if group.get('spectral_normalization', False):
            u_states, v_states, spectral_targets = _foreach_collect_spectral_vars(self, g_params, step_size)
            updates = foreach_scale_update(
                g_params, updates, step_size,
                u_state=u_states, v_state=v_states,
                target_scale=spectral_targets,
            )
        else:
            if use_atan2:
                step_size = step_size * A
            torch._foreach_mul_(updates, step_size)

        # Compute fisher_wd scalers if needed
        wd_scalers = None
        if fisher_wd:
            wd_scalers = _foreach_get_fisher_wd_scaler(group, g_params, denom, group['eps'])

        # Apply update and weight decay via foreach helper
        group_anchors = anchor_groups.get((g_params[0].device, g_params[0].dtype), None) if cwd != 0.0 else None
        foreach_apply_parameter_update(
            self, g_params, group, updates, step_size,
            wd_scaler=wd_scalers,
            anchors=group_anchors if cwd != 0.0 else None,
        )

def _group_by_device_dtype(params, grads, exp_avgs, exp_avg_sqs, state_steps):
    """Groups tensors by (device, dtype) for foreach operations."""
    groups: dict = {}
    for i in range(len(params)):
        key = (params[i].device, params[i].dtype)
        if key not in groups:
            groups[key] = [[], [], [], [], []]
        groups[key][0].append(params[i])
        groups[key][1].append(grads[i])
        groups[key][2].append(exp_avgs[i])
        groups[key][3].append(exp_avg_sqs[i])
        groups[key][4].append(state_steps[i])
    return groups
