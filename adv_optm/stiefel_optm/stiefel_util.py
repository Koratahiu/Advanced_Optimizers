import torch

from torch import Tensor

from typing import Dict, Any

def trangent_proj(p, update, lr):
    """
    [Stiefel-LoRA] Step 1: Tangent Space Projection
    Formula: update = update - p @ sym(p.T @ update)
    """
    pt_u = torch.matmul(p.t(), update)
    sym_pt_u = 0.5 * (pt_u + pt_u.t())
    # Project the update onto the tangent space
    update.sub_(torch.matmul(p, sym_pt_u))
    del pt_u, sym_pt_u
    rms_rescaling(update, lr)
    return update

def qr_retraction(p):
    """[Stiefel-LoRA] Step 2: Manifold Retraction (QR Decomposition)"""
    Q, R = torch.linalg.qr(p)
    d = R.diagonal().sign_()
    Q *= d
    return p.copy_(Q)

def rms_rescaling(update, lr):
    """Rescales update to have Frobenius norm exactly equal to lr."""
    norm = torch.linalg.vector_norm(update).clamp_min_(1e-12)
    return update.mul_(lr/norm)

def set_flags_AB(p):
    """
    Identify if parameter is LoRA A or B factor based on dimensions.
    B: (dim_out, rank) - tall matrix, Stiefel manifold
    A: (rank, dim_in) - wide matrix, Euclidean space
    """
    B = p.ndim == 2 and p.shape[0] > p.shape[1]
    A = p.ndim == 2 and p.shape[0] < p.shape[1]
    return B, A

def apply_stiefel_update(
    self,
    p: Tensor,
    group: Dict[str, Any],
    update: Tensor,
    lr: float | Tensor,
    wd: float | None = None,
    random_int_tensor: Tensor | None = None,
    is_B: bool | None = None,
    is_A: bool | None = None,
) -> None:
    from ..util.param_update import _copy_stochastic_core_, copy_stochastic_
    wd = group["weight_decay"] if wd is None else wd
    cautious = group.get('cautious_wd', False)

    if is_B:
        # Disable weight decay for the ortho matrix B
        wd = 0

    scaled_wd = wd * (lr / self._init_lr)

    # Compute full update in float32 if using bfloat16 with stochastic rounding
    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        p_fp32 = p.float()
        update_fp32 = update.float()

        if is_B:
            update_fp32 = trangent_proj(p_fp32, update_fp32, lr)
            p_fp32.add_(-update_fp32)
            p_fp32 = qr_retraction(p_fp32)
            if random_int_tensor is not None:
                _copy_stochastic_core_(p, p_fp32, random_int_tensor)
                del random_int_tensor
            else:
                copy_stochastic_(p, p_fp32)
            del p_fp32, update_fp32
            return

        # Apply weight decay if needed
        if wd != 0:
            if cautious:
                # Cautious Weight Decay
                mask = (update_fp32 * p_fp32 >= 0).float()
                p_fp32.addcmul_(p_fp32, mask, value=-scaled_wd)
                del mask
            else:
                # Standard decoupled weight decay
                p_fp32.add_(p_fp32, alpha=-scaled_wd)

        # Apply main update
        p_fp32.add_(-update_fp32)

        # Single stochastic rounding at the end
        if random_int_tensor is not None:
            # Compiled path: use the pre-computed random tensor
            _copy_stochastic_core_(p, p_fp32, random_int_tensor)
            del random_int_tensor
        else:
            # Uncompiled path: generate randoms inside
            copy_stochastic_(p, p_fp32)
        del p_fp32, update_fp32

    else:
        # Standard path for non-bfloat16 or without stochastic rounding

        if is_B:
            update = trangent_proj(p, update, lr)
            p.add_(-update)
            p = qr_retraction(p)
            return

        if wd != 0:
            if cautious:
                # Cautious Weight Decay
                mask = (update * p >= 0).to(p.dtype)
                p.addcmul_(p, mask, value=-scaled_wd)
                del mask
            else:
                # Standard decoupled weight decay
                p.add_(p, alpha=-scaled_wd)

        # Apply main update
        p.add_(-update)

    del update