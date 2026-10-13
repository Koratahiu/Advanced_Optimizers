import torch
from torch import Tensor

from typing import Dict, Any

from ..scaled_optm import adjust_wds, scale_wd

def _apply_weight_decay(
    p_calc: list[torch.Tensor] | tuple[Tensor, ...],
    update_calc: list[torch.Tensor] | tuple[Tensor, ...],
    group: Dict[str, Any],
    scaled_wd: float | Tensor | list[Tensor] | tuple[Tensor, ...] | None,
    scaled_cwd: float | Tensor | list[Tensor] | tuple[Tensor, ...] | None,
    wd_target: Tensor | None = None,
    cwd_target: Tensor | None = None,
    anchors: list[torch.Tensor] | tuple[Tensor, ...] | None = None,
) -> None:
    """
    Apply decoupled weight decay (Standard and/or Centered) independently.
    """
    cautious = group.get('cautious_wd', False)

    # Normalize targets to lists (single Tensor passed in is wrapped)
    if wd_target is not None and not isinstance(wd_target, (list, tuple)):
        wd_target = [wd_target]
    if cwd_target is not None and not isinstance(cwd_target, (list, tuple)):
        cwd_target = [cwd_target]

    # Standard Weight Decay (pulls toward zero).
    # Per-tensor wd lists may contain None entries (e.g. vectors excluded by
    # scale_wd); subset to the active entries on local copies.
    if isinstance(scaled_wd, (list, tuple)):
        if not any(v is not None for v in scaled_wd):
            scaled_wd = None
    if scaled_wd is not None:
        wd_p = p_calc
        wd_update = update_calc
        wd_vals = scaled_wd
        wd_target_local = wd_target if wd_target is not None else p_calc
        if isinstance(scaled_wd, (list, tuple)):
            keep = [i for i, v in enumerate(scaled_wd) if v is not None]
            wd_p = [p_calc[i] for i in keep]
            wd_update = [update_calc[i] for i in keep]
            wd_vals = [scaled_wd[i] for i in keep]
            wd_target_local = [wd_target_local[i] for i in keep]
        # Cautious Weight Decay: only decay if the update pushes in the same direction as the decay
        if cautious:
            # Cautious decoupled weight decay
            masks = torch._foreach_mul(wd_update, wd_p)
            torch._foreach_sign_(masks)
            torch._foreach_add_(masks, 1.0)
            torch._foreach_clamp_max_(masks, 1.0)  # 1.0 for dot>=0, 0.0 for dot<0
            if isinstance(wd_vals, Tensor):
                if wd_vals.dim() == 0:
                    torch._foreach_mul_(masks, wd_vals)
                else:
                    torch._foreach_mul_(masks, [wd_vals])
                torch._foreach_addcmul_(wd_p, wd_target_local, masks, value=-1.0)
            elif isinstance(wd_vals, (list, tuple)):
                torch._foreach_mul_(masks, wd_vals)
                torch._foreach_addcmul_(wd_p, wd_target_local, masks, value=-1.0)
            else:
                torch._foreach_addcmul_(wd_p, wd_target_local, masks, value=-wd_vals)
        else:
            if isinstance(wd_vals, Tensor):
                scaled_targets = torch._foreach_mul(wd_target_local, -wd_vals)
                torch._foreach_add_(wd_p, scaled_targets)
            elif isinstance(wd_vals, (list, tuple)):
                scaled_targets = torch._foreach_mul(wd_target_local, wd_vals)
                torch._foreach_add_(wd_p, scaled_targets, alpha=-1.0)
            else:
                torch._foreach_add_(wd_p, wd_target_local, alpha=-wd_vals)


    # Centered Weight Decay (pulls toward anchor)
    if isinstance(scaled_cwd, (list, tuple)):
        if not any(v is not None for v in scaled_cwd):
            scaled_cwd = None
    if scaled_cwd is not None and (anchors or cwd_target is not None):
        cwd_p = p_calc
        cwd_update = update_calc
        cwd_vals = scaled_cwd
        if isinstance(scaled_cwd, (list, tuple)):
            keep = [i for i, v in enumerate(scaled_cwd) if v is not None]
            cwd_p = [p_calc[i] for i in keep]
            cwd_update = [update_calc[i] for i in keep]
            cwd_vals = [scaled_cwd[i] for i in keep]
            if cwd_target is not None:
                cwd_target = [cwd_target[i] for i in keep]
            if anchors is not None:
                anchors = [anchors[i] for i in keep]
        if cwd_target is not None:
            decay_target = cwd_target
        else:
            anchor = anchors
            decay_target = torch._foreach_sub(cwd_p, anchor)

        del anchors

        if cautious:
            masks = torch._foreach_mul(cwd_update, decay_target)
            torch._foreach_sign_(masks)
            torch._foreach_add_(masks, 1.0)
            torch._foreach_clamp_max_(masks, 1.0) # Produces 1.0 for >= 0, 0.0 for < 0

            if isinstance(cwd_vals, Tensor):
                if cwd_vals.dim() == 0:
                    torch._foreach_mul_(masks, cwd_vals)
                else:
                    torch._foreach_mul_(masks, [cwd_vals])
                torch._foreach_addcmul_(cwd_p, decay_target, masks, value=-1.0)
            elif isinstance(cwd_vals, (list, tuple)):
                torch._foreach_mul_(masks, cwd_vals)
                torch._foreach_addcmul_(cwd_p, decay_target, masks, value=-1.0)
            else:
                torch._foreach_addcmul_(cwd_p, decay_target, masks, value=-cwd_vals)
        else:
            if isinstance(cwd_vals, torch.Tensor):
                scaled_targets = torch._foreach_mul(decay_target, -cwd_vals)
                torch._foreach_add_(cwd_p, scaled_targets)
            elif isinstance(cwd_vals, (list, tuple)):
                scaled_targets = torch._foreach_mul(decay_target, cwd_vals)
                torch._foreach_add_(cwd_p, scaled_targets, alpha=-1.0)
            else:
                torch._foreach_add_(cwd_p, decay_target, alpha=-cwd_vals)

        if cwd_target is None:
            del decay_target


def foreach_apply_parameter_update(
    self,
    params: list[torch.Tensor] | tuple[Tensor, ...],
    group: Dict[str, Any],
    update: list[torch.Tensor] | tuple[Tensor, ...],
    lr: float | Tensor,
    wd: float | None = None,
    decoupled: bool = False,
    wd_scaler: float | Tensor | None = None,
    wd_target: Tensor | None = None,
    cwd_target: Tensor | None = None,
    anchors: list[torch.Tensor] | tuple[Tensor, ...] | None = None,
) -> None:
    """
    Applies decoupled weight decay (standard, cautious, centered) and the final
    parameter update to p in-place.
    Using torch._foreach ops.

    Args:
        p: The parameter tensor whose data (p) will be updated.
        group: The parameter group dictionary (must contain "weight_decay").
        update: The pre-calculated update tensor (e.g., scaled gradient or momentum term).
        lr: The current learning rate.
        wd: Optional float value for weight decay, if another value other than group["weight_decay"] is needed.
        random_int_tensor: Optional pre-generated random tensor for stochastic
            rounding. Required for the `torch.compile` path.
        decoupled: Whenever to use the true decoupled weight decay.
        wd_scaler: A multiplier/tensor to scale the calculated wd/cwd magnitude (e.g. for Fisher Adam WD).
    """
    wd = group["weight_decay"] if wd is None else wd
    cwd = group.get("centered_wd", 0.0)
    scaled_wd_group = group.get("scaled_wd", False)
    decoupled = scaled_wd_group or decoupled

    # Calculate global decay factor for decoupled vs standard
    decay_factor = (lr / self._init_lr) if decoupled else lr

    scaled_wd = (wd * decay_factor) if wd != 0 else None
    scaled_cwd = (cwd * decay_factor) if cwd != 0 else None

    # Per-parameter wd/cwd adjustment and dimension scaling, mirroring
    # apply_parameter_update (adjust_wds + scale_wd). Without this the
    # foreach path decays OFT/vector/matrix params with the raw wd.
    if scaled_wd is not None or scaled_cwd is not None:
        scaled_wds: list[float | None] = []
        scaled_cwds: list[float | None] = []
        for p in params:
            p_wd, p_cwd = adjust_wds(wd, cwd, p)
            w = p_wd * decay_factor if p_wd != 0 else None
            c = p_cwd * decay_factor if p_cwd != 0 else None
            if w is not None:
                w = scale_wd(w, p, skip_vectors=True)
            if c is not None:
                c = scale_wd(c, p, skip_vectors=False)
            scaled_wds.append(w)
            scaled_cwds.append(c)
        scaled_wd = scaled_wds
        scaled_cwd = scaled_cwds

    if wd_scaler is not None:
        if isinstance(wd_scaler, (list, tuple)):
            if scaled_wd is not None:
                if isinstance(scaled_wd, (list, tuple)):
                    scaled_wd = [s * sc if s is not None else None for s, sc in zip(scaled_wd, wd_scaler)]
                else:
                    scaled_wd = torch._foreach_mul_(wd_scaler, scaled_wd)
            if scaled_cwd is not None:
                if isinstance(scaled_cwd, (list, tuple)):
                    scaled_cwd = [s * sc if s is not None else None for s, sc in zip(scaled_cwd, wd_scaler)]
                else:
                    scaled_cwd = torch._foreach_mul_(wd_scaler, scaled_cwd)
        else:
            if scaled_wd is not None:
                if isinstance(scaled_wd, (list, tuple)):
                    scaled_wd = [s * wd_scaler if s is not None else None for s in scaled_wd]
                else:
                    scaled_wd = scaled_wd * wd_scaler
            if scaled_cwd is not None:
                if isinstance(scaled_cwd, (list, tuple)):
                    scaled_cwd = [s * wd_scaler if s is not None else None for s in scaled_cwd]
                else:
                    scaled_cwd = scaled_cwd * wd_scaler

    if scaled_wd is not None or scaled_cwd is not None:
        _apply_weight_decay(params, update, group, scaled_wd, scaled_cwd, wd_target, cwd_target, anchors)

    # Apply main update
    torch._foreach_add_(params, update, alpha=-1.0)

    del update
