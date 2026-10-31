import torch
from torch import Tensor

import math

from ...util.foreach.fh_scaled_optm import foreach_scale_update, _foreach_collect_spectral_vars, foreach_max_abs_normalization
from ...util.foreach.fh_orthograd import _foreach_orthogonalize_gradient
from ...util.foreach.fh_param_update import foreach_apply_parameter_update
from ...util.foreach.fh_signed_util import (
    apply_stochastic_sign_ as _foreach_apply_stochastic_sign_,
    foreach_get_signsgd_wd_target,
)

A = 4 / math.pi


@torch.no_grad()
def foreach_step(self, group: dict) -> None:
    """
    Foreach/multi-tensor optimization step for SignSGD_adv.
    Uses torch._foreach_* ops where available; falls back to iterative
    list-comprehension style for ops without foreach support (e.g. atan2).
    """
    params = [p for p in group['params'] if p.grad is not None]
    if not params:
        return

    grads = [p.grad for p in params]

    momentum_buffers = []
    state_steps = []
    anchors = []

    for p in params:
        state = self.state[p]
        self._SignSGD_adv__init_state(p, group)
        momentum = group['momentum']
        if momentum > 0:
            momentum_buffers.append(state['exp_avg'])
        state['step'] += 1
        state_steps.append(torch.tensor(state['step']))

        # Collect anchor for centered_wd (full mode only in foreach)
        cwd = group.get('centered_wd', 0.0)
        if cwd != 0.0 and 'anchor_data' in state:
            anchors.append(state['anchor_data'])

    # Group anchors by (device, dtype, is_vector) to match the matrix/vector
    # parameter grouping used by the step functions.
    anchor_groups: dict = {}
    cwd = group.get('centered_wd', 0.0)
    for i, p in enumerate(params):
        is_vector = p.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
        key = (p.device, p.dtype, is_vector)
        if cwd != 0.0 and i < len(anchors):
            if key not in anchor_groups:
                anchor_groups[key] = []
            anchor_groups[key].append(anchors[i])

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

    foreach_step_fn(self, group, params, grads, momentum_buffers, state_steps, anchor_groups, lr)


@torch.no_grad()
def _foreach_step(
    self,
    group: dict,
    params,
    grads,
    momentum_buffers,
    state_steps,
    anchor_groups,
    lr,
) -> None:
    momentum = group['momentum']
    sso = group.get('stochastic_sign', False)
    normed_mt = group.get('normed_momentum', False)
    nesterov = group.get('nesterov', False)
    nesterov_coef = group.get('nesterov_coef', None)
    snr_cond = group.get('snr_cond', False) and normed_mt and momentum > 0
    geometric_wd = group.get('geometric_wd', False)
    cwd = group.get('centered_wd', 0.0)

    # Orthogonalize gradients if needed
    ortho_mode = group.get('orthogonal_gradient', 'disabled')
    grads = _foreach_orthogonalize_gradient(params, grads, ortho_mode)

    # Group tensors by (device, dtype) for foreach ops
    grouped = _group_by_device_dtype(params, grads, momentum_buffers, state_steps)

    for (g_params, g_grads, g_momentum_buffers, g_steps) in grouped.values():
        if len(g_params) == 0:
            continue

        # Separate vector and matrix params
        vec_params, vec_grads, vec_mbufs = [], [], []
        mat_params, mat_grads, mat_mbufs = [], [], []
        for p, g, mbuf in zip(g_params, g_grads, g_momentum_buffers):
            is_vector = p.ndim < 2 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
            if is_vector:
                vec_params.append(p)
                vec_grads.append(g)
                vec_mbufs.append(mbuf)
            else:
                mat_params.append(p)
                mat_grads.append(g)
                mat_mbufs.append(mbuf)

        # Matrix path
        if mat_params:
            mat_updates, mat_wd_targets = _foreach_step_matrix(
                self, mat_params, mat_grads, mat_mbufs, g_steps, group,
                momentum, sso, normed_mt, nesterov, nesterov_coef,
                snr_cond, geometric_wd, lr,
            )
            group_anchors = anchor_groups.get(
                (mat_params[0].device, mat_params[0].dtype, False), None) if cwd != 0.0 else None
            foreach_apply_parameter_update(
                self, mat_params, group, mat_updates, lr,
                wd_target=mat_wd_targets,
                anchors=group_anchors if cwd != 0.0 else None,
            )

        # Vector path
        if vec_params:
            vec_updates, vec_wd_targets = _foreach_step_vector(
                self, vec_params, vec_grads, vec_mbufs, g_steps, group,
                momentum, sso, normed_mt, nesterov, nesterov_coef,
                snr_cond, geometric_wd, lr,
            )
            group_anchors = anchor_groups.get(
                (vec_params[0].device, vec_params[0].dtype, True), None) if cwd != 0.0 else None
            foreach_apply_parameter_update(
                self, vec_params, group, vec_updates, lr,
                wd_target=vec_wd_targets,
                anchors=group_anchors if cwd != 0.0 else None,
            )


@torch.no_grad()
def _foreach_step_matrix(
    self,
    params,
    grads,
    momentum_buffers,
    state_steps,
    group,
    momentum,
    sso,
    normed_mt,
    nesterov,
    nesterov_coef,
    snr_cond,
    geometric_wd,
    lr,
):
    """Foreach optimization step for 2D+ matrix parameters."""
    # Apply sign or stochastic sign to gradient if normed_momentum
    if normed_mt:
        if sso:
            # SSO needs noise; generate per-tensor noise for matrix params
            noise = [_get_random_noise_for_sso(g) for g in grads]
            _foreach_apply_stochastic_sign_(grads, noise=noise, is_vector=False)
        else:
            torch._foreach_sign_(grads)

    # Compute SNR conditioning denominators if needed
    denom = None
    if snr_cond and momentum > 0:
        # denom = sqrt(1 - buf^2)
        denom = torch._foreach_mul(momentum_buffers, momentum_buffers)
        torch._foreach_sub_(denom, 1.0)
        torch._foreach_neg_(denom)
        torch._foreach_clamp_min_(denom, 1e-30)
        torch._foreach_sqrt_(denom)

    # Compute update (momentum or plain gradient)
    if momentum > 0:
        updates = torch._foreach_clone(momentum_buffers)
        if nesterov and normed_mt:
            # When normed_momentum is True, scale the normalized gradient
            # using empirical buffer magnitude (SNR recovery)
            nv_coef = momentum if nesterov_coef is None else nesterov_coef
            normed_grads = torch._foreach_abs(momentum_buffers)
            torch._foreach_mul_(normed_grads, grads)
            torch._foreach_lerp_(updates, normed_grads, 1.0 - nv_coef)
            del normed_grads
        elif nesterov:
            nv_coef = momentum if nesterov_coef is None else nesterov_coef
            torch._foreach_lerp_(updates, grads, 1.0 - nv_coef)
    else:
        updates = torch._foreach_clone(grads)

    # SNR conditioning: atan2 with denom
    if snr_cond and momentum > 0:
        for i, _ in enumerate(updates):
            updates[i].atan2_(denom[i])

    if not normed_mt:
        if sso:
            noise = [_get_random_noise_for_sso(u) for u in updates]
            _foreach_apply_stochastic_sign_(updates, noise=noise, is_vector=False)
        else:
            torch._foreach_sign_(updates)

    # Compute geometric_wd target if needed
    wd_targets = None
    if geometric_wd and group["weight_decay"] > 0:
        wd_targets = foreach_get_signsgd_wd_target(params, denom=denom)

    # Scale updates
    if group.get('spectral_normalization', False):
        u_states, v_states, spectral_targets = _foreach_collect_spectral_vars(self, params, lr)
        updates = foreach_scale_update(params, updates, lr, u_state=u_states, v_state=v_states, target_scale=spectral_targets)
    else:
        scaling = lr
        if snr_cond:
            scaling = scaling * A
        torch._foreach_mul_(updates, scaling)

    return updates, wd_targets


@torch.no_grad()
def _foreach_step_vector(
    self,
    params,
    grads,
    momentum_buffers,
    state_steps,
    group,
    momentum,
    sso,
    normed_mt,
    nesterov,
    nesterov_coef,
    snr_cond,
    geometric_wd,
    lr,
):
    """Foreach optimization step for 1D vector parameters."""
    # Apply sign or stochastic sign to gradient if normed_momentum
    if normed_mt:
        if sso:
            noise = [_get_random_noise_for_sso(g) for g in grads]
            _foreach_apply_stochastic_sign_(grads, noise=noise, is_vector=True)
        else:
            torch._foreach_sign_(grads)

    # Compute SNR conditioning denominator if needed
    denom = None
    if snr_cond and momentum > 0:
        denom = torch._foreach_mul(momentum_buffers, momentum_buffers)
        torch._foreach_sub_(denom, 1.0)
        torch._foreach_neg_(denom)
        torch._foreach_clamp_min_(denom, 1e-30)
        torch._foreach_sqrt_(denom)

    # Compute update
    if momentum > 0:
        updates = torch._foreach_clone(momentum_buffers)
        if nesterov and normed_mt:
            # When normed_momentum is True, scale the normalized gradient
            # using empirical buffer magnitude (SNR recovery)
            nv_coef = momentum if nesterov_coef is None else nesterov_coef
            normed_grads = torch._foreach_abs(momentum_buffers)
            torch._foreach_mul_(normed_grads, grads)
            torch._foreach_lerp_(updates, normed_grads, 1.0 - nv_coef)
            del normed_grads
        elif nesterov:
            nv_coef = momentum if nesterov_coef is None else nesterov_coef
            torch._foreach_lerp_(updates, grads, 1.0 - nv_coef)
    else:
        updates = torch._foreach_clone(grads)

    # SNR conditioning: atan2 with denom
    if snr_cond and momentum > 0:
        for i, _ in enumerate(updates):
            updates[i].atan2_(denom[i])

    if not normed_mt:
        if sso:
            noise = [_get_random_noise_for_sso(u) for u in updates]
            _foreach_apply_stochastic_sign_(updates, noise=noise, is_vector=True)
        else:
            torch._foreach_sign_(updates)

    # Compute geometric_wd target if needed
    wd_targets = None
    if geometric_wd and group["weight_decay"] > 0:
        wd_targets = foreach_get_signsgd_wd_target(params, denom=denom)

    # Scale updates
    if group.get('spectral_normalization', False):
        updates = foreach_max_abs_normalization(updates, lr)
    else:
        scaling = lr
        if snr_cond:
            scaling = scaling * A
        torch._foreach_mul_(updates, scaling)

    return updates, wd_targets


def _group_by_device_dtype(params, grads, momentum_buffers, state_steps):
    """Groups tensors by (device, dtype) for foreach operations."""
    groups: dict = {}
    for i in range(len(params)):
        key = (params[i].device, params[i].dtype)
        if key not in groups:
            groups[key] = [[], [], [], []]
        groups[key][0].append(params[i])
        groups[key][1].append(grads[i])
        # momentum_buffers may be empty when momentum=0
        groups[key][2].append(momentum_buffers[i] if momentum_buffers else None)
        groups[key][3].append(state_steps[i])
    return groups


def _get_random_noise_for_sso(source: torch.Tensor) -> torch.Tensor:
    """
    Generates a random noise tensor for Stochastic Sign operator.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    from ...util import param_update
    return param_update._get_random_noise_for_sso(source)

