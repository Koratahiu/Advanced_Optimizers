import torch

import math

from ...util.foreach.fh_scaled_optm import foreach_scale_update, _foreach_collect_spectral_vars, foreach_max_abs_normalization

from ...util.foreach.fh_orthograd import _foreach_orthogonalize_gradient
from ...util.foreach.fh_param_update import foreach_apply_parameter_update
from ...util.foreach.fh_sinkhorn import (
    foreach_apply_sr_sinkhorn,
    foreach_get_sinkhorn_wd_scaler,
)
from ...util.foreach.fh_signed_util import foreach_get_signsgd_wd_target


A = 4 / math.pi


@torch.no_grad()
def foreach_step(self, group: dict) -> None:
    """
    Foreach/multi-tensor optimization step for SinkSGD.
    Uses torch._foreach_* ops where available; falls back to iterative
    list-comprehension style for ops without foreach support.
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
        self._SinkSGD_adv__init_state(p, group)
        momentum_buffers.append(state.get('momentum_buffer', None))
        state['step'] += 1
        state_steps.append(torch.tensor(state['step']))

        # Collect anchor for centered_wd (full mode only in foreach)
        cwd = group.get('centered_wd', 0.0)
        if cwd != 0.0 and 'anchor_data' in state:
            anchors.append(state['anchor_data'])

    # Group anchors by (device, dtype) to match the parameter grouping
    anchor_groups: dict = {}
    cwd = group.get('centered_wd', 0.0)
    for i, p in enumerate(params):
        key = (p.device, p.dtype)
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
    normed_mt = group.get('normed_momentum', False)
    nesterov = group.get('nesterov', False)
    nesterov_coef = group.get('nesterov_coef', None)
    snr_cond = group.get('snr_cond', False)
    sinkhorn_iterations = group['sinkhorn_iterations']
    orthogonal_sinkhorn = group['orthogonal_sinkhorn']
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
            mat_updates, mat_wd_scalers = _foreach_step_matrix(
                self, mat_params, mat_grads, mat_mbufs, g_steps, group,
                sinkhorn_iterations, orthogonal_sinkhorn,
                momentum, normed_mt, nesterov, nesterov_coef,
                snr_cond, geometric_wd, lr,
            )
            group_anchors = anchor_groups.get((mat_params[0].device, mat_params[0].dtype), None) if cwd != 0.0 else None
            foreach_apply_parameter_update(
                self, mat_params, group, mat_updates, lr,
                wd_scaler=mat_wd_scalers,
                anchors=group_anchors if cwd != 0.0 else None,
            )

        # Vector path
        if vec_params:
            vec_updates, vec_wd_targets = _foreach_step_vector(
                self, vec_params, vec_grads, vec_mbufs, g_steps, group,
                momentum, normed_mt, nesterov, nesterov_coef,
                snr_cond, geometric_wd, lr,
            )
            group_anchors = anchor_groups.get((vec_params[0].device, vec_params[0].dtype), None) if cwd != 0.0 else None
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
    sinkhorn_iterations,
    orthogonal_sinkhorn,
    momentum,
    normed_mt,
    nesterov,
    nesterov_coef,
    snr_cond,
    geometric_wd,
    lr,
):
    """Foreach optimization step for 2D+ matrix parameters."""
    # Apply Sinkhorn to gradient if normed_momentum
    if normed_mt:
        grads = foreach_apply_sr_sinkhorn(grads, iters=sinkhorn_iterations, params=params, ortho_project=orthogonal_sinkhorn)

    # Update momentum buffer
    if momentum != 0:
        torch._foreach_lerp_(momentum_buffers, grads, 1.0 - momentum)

    # Compute SNR conditioning denominators if needed
    vt_row = []
    vt_col = []
    if snr_cond and momentum != 0:
        # Compute buf^2, then mean along each dim
        buf_sq = torch._foreach_mul(momentum_buffers, momentum_buffers)
        for i, b_sq in enumerate(buf_sq):
            b_sq = b_sq[i].view(b_sq[i].shape[0], -1)
            vt_row.append(torch.mean(b_sq, dim=1))
            vt_col.append(torch.mean(b_sq, dim=0))
        del buf_sq

        for v_list in (vt_row, vt_col):
            torch._foreach_sub_(v_list, 1.0)
            torch._foreach_neg_(v_list)
            torch._foreach_clamp_min_(v_list, 1e-30)
            torch._foreach_rsqrt_(v_list)
        vt_row = [v.unsqueeze_(1) for v in vt_row]
        vt_col = [v.unsqueeze_(0) for v in vt_col]

    # Compute update (keep as list for in-place mutation)
    if momentum != 0:
        updates = torch._foreach_clone(momentum_buffers)
    else:
        updates = torch._foreach_clone(grads)

    # SNR conditioning: row/col precondition + atan
    if snr_cond:
        update_2d = [updates[i].view(updates[i].shape[0], -1) for i, _ in enumerate(params)]
        torch._foreach_mul_(update_2d, vt_row)
        torch._foreach_mul_(update_2d, vt_col)
        torch._foreach_atan_(update_2d)

    # Apply Sinkhorn to update if not normed_momentum
    if not normed_mt:
        update = foreach_apply_sr_sinkhorn(update, iters=sinkhorn_iterations, params=params, ortho_project=orthogonal_sinkhorn)

    # Compute geometric_wd scaler if needed
    wd_scalers = None
    if geometric_wd and group["weight_decay"] > 0:
        wd_scalers = foreach_get_sinkhorn_wd_scaler(params, row_denom=vt_row, col_denom=vt_col)

    # Scale updates
    if group.get('spectral_normalization', False):
        u_states, v_states, spectral_targets = _foreach_collect_spectral_vars(self, params, lr)
        updates = foreach_scale_update(params, updates, lr, u_state=u_states, v_state=v_states, target_scale=spectral_targets)
    else:
        scaling = lr
        if snr_cond:
            scaling = scaling * A
        torch._foreach_mul_(updates, scaling)

    return updates, wd_scalers


@torch.no_grad()
def _foreach_step_vector(
    self,
    params,
    grads,
    momentum_buffers,
    state_steps,
    group,
    momentum,
    normed_mt,
    nesterov,
    nesterov_coef,
    snr_cond,
    geometric_wd,
    lr,
):
    """Foreach optimization step for 1D vector parameters."""
    # Apply sign to gradient if normed_momentum
    if normed_mt:
        torch._foreach_sign_(grads)

    # Compute SNR conditioning denominator if needed
    denom = None
    if snr_cond and momentum != 0:
        denom = torch._foreach_mul(momentum_buffers, momentum_buffers)
        torch._foreach_sub_(denom, 1.0)
        torch._foreach_neg_(denom)
        torch._foreach_clamp_min_(denom, 1e-30)
        torch._foreach_sqrt_(denom)

    # Compute update
    if momentum != 0:
        updates = torch._foreach_clone(momentum_buffers)
    else:
        updates = torch._foreach_clone(grads)

    # SNR conditioning: atan2 with denom
    if snr_cond:
        for i, _ in enumerate(updates):
            updates[i].atan2_(denom[i])

    if not normed_mt:
        torch._foreach_sign_(updates)

    # Compute geometric_wd target if needed
    wd_targets = None
    if geometric_wd and group["weight_decay"] > 0:
        wd_targets = foreach_get_signsgd_wd_target(params, denom=denom)

    if group.get('spectral_normalization', False):
        update = foreach_max_abs_normalization(update, lr)
    else:
        # Scale updates
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
        groups[key][2].append(momentum_buffers[i])
        groups[key][3].append(state_steps[i])
    return groups
