import torch

@torch.no_grad()
def _newton_schulz_iteration(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
) -> torch.Tensor:
    """
    Performs the Newton-Schulz iteration to find the nearest orthogonal matrix.
    This is the core computation of the Muon optimizer.

    Args:
        G (torch.Tensor): The 2D input matrix (momentum-accumulated gradient).
        steps (int): The number of iterations to run.
        eps (float): Small constant for numerical stability during normalization.
        coeffs (tuple[float, float, float]): The (a, b, c) coefficients for the
            quintic polynomial update.

    Returns:
        torch.Tensor: The orthogonalized matrix.
    """
    assert G.ndim == 2, "Newton-Schulz iteration only supports 2D matrices."

    a, b, c = coeffs

    X = G.to(torch.bfloat16)

    # Normalize the matrix
    X.div_(X.norm() + eps)

    # Handle non-square matrices by transposing the taller one
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    # Perform the iterative updates
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    # Transpose back if necessary
    if transposed:
        X = X.T

    return X.to(G.dtype)