import torch
from torch import Tensor

from typing import Dict, Any

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

    # Normalize wd_target to a list (single Tensor passed in is wrapped)
    if wd_target is not None and not isinstance(wd_target, list):
        wd_target = [wd_target]

    # Standard Weight Decay (pulls toward zero)
    if scaled_wd is not None:
        if wd_target is None:
            wd_target = p_calc
        # Cautious Weight Decay: only decay if the update pushes in the same direction as the decay
        if cautious:
            # Cautious decoupled weight decay
            masks = torch._foreach_mul(update_calc, p_calc)
            torch._foreach_sign_(masks)
            torch._foreach_add_(masks, 1.0)
            torch._foreach_clamp_max_(masks, 1.0)  # 1.0 for dot>=0, 0.0 for dot<0
            if isinstance(scaled_wd, Tensor):
                if scaled_wd.dim() == 0:
                    torch._foreach_mul_(masks, scaled_wd)
                else:
                    torch._foreach_mul_(masks, [scaled_wd])
                torch._foreach_addcmul_(p_calc, wd_target, masks, value=-1.0)
            elif isinstance(scaled_wd, (list, tuple)):
                torch._foreach_mul_(masks, scaled_wd)
                torch._foreach_addcmul_(p_calc, wd_target, masks, value=-1.0)
            else:
                torch._foreach_addcmul_(p_calc, wd_target, masks, value=-scaled_wd)
        else:
            if isinstance(scaled_wd, Tensor):
                scaled_targets = torch._foreach_mul(wd_target, -scaled_wd)
                torch._foreach_add_(p_calc, scaled_targets)
            elif isinstance(scaled_wd, (list, tuple)):
                scaled_targets = torch._foreach_mul(wd_target, scaled_wd)
                torch._foreach_add_(p_calc, scaled_targets, alpha=-1.0)
            else:
                torch._foreach_add_(p_calc, wd_target, alpha=-scaled_wd)


    # Normalize cwd_target to a list (single Tensor passed in is wrapped)
    if cwd_target is not None and not isinstance(cwd_target, list):
        cwd_target = [cwd_target]

    # Centered Weight Decay (pulls toward anchor)
    if scaled_cwd is not None and (anchors or cwd_target is not None):
        if cwd_target is not None:
            decay_target = cwd_target
        else:
            anchor = anchors
            decay_target = torch._foreach_sub(p_calc, anchor)

        del anchors

        if cautious:
            masks = torch._foreach_mul(update_calc, decay_target)
            torch._foreach_sign_(masks)
            torch._foreach_add_(masks, 1.0)
            torch._foreach_clamp_max_(masks, 1.0) # Produces 1.0 for >= 0, 0.0 for < 0

            if isinstance(scaled_cwd, Tensor):
                if scaled_cwd.dim() == 0:
                    torch._foreach_mul_(masks, scaled_cwd)
                else:
                    torch._foreach_mul_(masks, [scaled_cwd])
                torch._foreach_addcmul_(p_calc, decay_target, masks, value=-1.0)
            elif isinstance(scaled_cwd, (list, tuple)):
                torch._foreach_mul_(masks, scaled_cwd)
                torch._foreach_addcmul_(p_calc, decay_target, masks, value=-1.0)
            else:
                torch._foreach_addcmul_(p_calc, decay_target, masks, value=-scaled_cwd)
        else:
            if isinstance(scaled_cwd, torch.Tensor):
                scaled_targets = torch._foreach_mul(decay_target, -scaled_cwd)
                torch._foreach_add_(p_calc, scaled_targets)
            elif isinstance(scaled_cwd, (list, tuple)):
                scaled_targets = torch._foreach_mul(decay_target, scaled_cwd)
                torch._foreach_add_(p_calc, scaled_targets, alpha=-1.0)
            else:
                torch._foreach_add_(p_calc, decay_target, alpha=-scaled_cwd)

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

    # Calculate global decay factor for decoupled vs standard
    decay_factor = (lr / self._init_lr) if decoupled else lr

    scaled_wd = (wd * decay_factor) if wd != 0 else None
    scaled_cwd = (cwd * decay_factor) if cwd != 0 else None

    if wd_scaler is not None:
        if isinstance(wd_scaler, (list, tuple)):
            if scaled_wd is not None:
                scaled_wd = torch._foreach_mul_(wd_scaler, scaled_wd)
            if scaled_cwd is not None:
                scaled_cwd = torch._foreach_mul_(wd_scaler, scaled_cwd)
        else:
            if scaled_wd is not None:
                scaled_wd = scaled_wd * wd_scaler
            if scaled_cwd is not None:
                scaled_cwd = scaled_cwd * wd_scaler

    if scaled_wd is not None or scaled_cwd is not None:
        _apply_weight_decay(params, update, group, scaled_wd, scaled_cwd, wd_target, cwd_target, anchors)

    # Apply main update
    torch._foreach_add_(params, update, alpha=-1.0)

    del update
