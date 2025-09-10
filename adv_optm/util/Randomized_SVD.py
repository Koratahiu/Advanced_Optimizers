import torch
from torch import Tensor

from typing import Tuple

def _rsvd(A: torch.Tensor, rank: int, oversampling: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs Randomized SVD."""
    orig_dtype, device, (m, n) = A.dtype, A.device, A.shape
    A_float = A.float()
    l, true_rank = rank + oversampling, min(m, n, rank)
    
    if true_rank == 0:
        return (
            torch.zeros(m, rank, dtype=orig_dtype, device=device),
            torch.zeros(rank, dtype=orig_dtype, device=device),
            torch.zeros(rank, n, dtype=orig_dtype, device=device),
        )
    
    if l >= min(m, n):  # Fallback to full SVD
        U_full, S_full, Vh_full = torch.linalg.svd(A_float, full_matrices=False)
        U, S, Vh = U_full[:, :true_rank], S_full[:true_rank], Vh_full[:true_rank, :]
    else:  # Standard RSVD path
        Omega = torch.randn(n, l, dtype=A_float.dtype, device=device)
        Y = A_float @ Omega
        Q, _ = torch.linalg.qr(Y.float())
        B = Q.T @ A_float
        U_tilde, S, Vh = torch.linalg.svd(B.float(), full_matrices=False)
        U, S, Vh = (Q @ U_tilde)[:, :true_rank], S[:true_rank], Vh[:true_rank, :]
        
    if true_rank < rank: # Pad factors with zeros
        U_padded = torch.zeros(m, rank, dtype=A_float.dtype, device=device)
        S_padded = torch.zeros(rank, dtype=A_float.dtype, device=device)
        Vh_padded = torch.zeros(rank, n, dtype=A_float.dtype, device=device)
        U_padded[:, :true_rank], S_padded[:true_rank], Vh_padded[:true_rank, :] = U, S, Vh
        U, S, Vh = U_padded, S_padded, Vh_padded
        
    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)