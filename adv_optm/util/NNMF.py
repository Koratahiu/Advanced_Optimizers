import torch

def _unnmf(row_col: tuple) -> torch.Tensor:
    """Reconstructs a matrix from its rank-1 factors (outer product)."""
    row, col = row_col
    # Ensure both tensors are float32
    return torch.outer(row.float(), col.float())

def _nnmf(matrix: torch.Tensor, out: tuple):
    """Performs a rank-1 non-negative matrix factorization."""
    shape = matrix.shape

    torch.sum(matrix, dim=1, out=out[0], dtype=torch.float32)
    torch.sum(matrix, dim=0, out=out[1], dtype=torch.float32)

    # Normalize one of the factors for stability
    EPSILON = 1e-12
    if shape[0] < shape[1]:
        scale = out[0].sum()
        out[0].div_(scale.clamp_(min=EPSILON))
    else:
        scale = out[1].sum()
        out[1].div_(scale.clamp_(min=EPSILON))
