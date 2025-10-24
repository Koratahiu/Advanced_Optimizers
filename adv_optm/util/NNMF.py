import torch

def _unnmf(row_col: tuple) -> torch.Tensor:
    """Reconstructs a matrix from its rank-1 factors (outer product)."""
    return torch.outer(row_col[0], row_col[1])

def _nnmf(matrix: torch.Tensor, out: tuple):
    """Performs a rank-1 non-negative matrix factorization."""
    shape = matrix.shape
    torch.sum(matrix, dim=1, out=out[0])
    torch.sum(matrix, dim=0, out=out[1])
    
    # Add a small epsilon for numerical stability and to remove 
    # data-dependent branching, making it compatible with torch.dynamo.
    epsilon = 1e-12 

    # Normalize one of the factors for stability
    if shape[0] < shape[1]:
        scale = out[0].sum()
        out[0].div_(scale + epsilon)
    else:
        scale = out[1].sum()
        out[1].div_(scale + epsilon)