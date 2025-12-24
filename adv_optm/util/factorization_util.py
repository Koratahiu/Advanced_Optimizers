import torch


# ------------------------------------------
# Factorization and reconstruction cycle
# ------------------------------------------

@torch.no_grad()
def _reconstruct_state(factors: tuple, signed: bool):
    """
    Reconstruct full state from its factors and optionally sign
    """
    if signed:
        mu_factor, mv_factor, sign, d2 = factors
        full_state = _unnmf((mu_factor, mv_factor))
        unpacked_sign = _unpack_bools(sign, original_m=d2)
        torch.where(unpacked_sign, full_state, -full_state, out=full_state)
        del unpacked_sign
        # We cast here to FP32 to prevent issues where the states are saved in 16-bit
        return full_state.float()
    else:
        # mu_factor, mv_factor = factors
        full_state = _unnmf(factors)
        return full_state.float()

@torch.no_grad()
def _factorize_state(full_state: torch.Tensor, signed: bool):
    """
    Compress a full state to its two rank-1 factors and optionally 1-bit sign
    """
    if signed:
        sign = _pack_bools(full_state > 0)
        mu_factor, mv_factor = _nnmf(full_state.abs_())
        return mu_factor, mv_factor, sign

    mu_factor, mv_factor = _nnmf(full_state.abs_())
    return mu_factor, mv_factor

# ------------------------------------------
# SMMF rank-1 NNMF factorization, logic modified from:
# https://github.com/eai-lab/SMMF/blob/ee2b953c79f4205c2506e1e2ede06b3b960f9b5d/SMMF/SMMF.py#L201
# ------------------------------------------

def _get_effective_shape(numel: int) -> tuple[int, int]:
    """Finds two factors of numel that are closest to its square root."""
    if numel <= 0:
        return (0, 0)
    for i in reversed(range(1, int(numel ** 0.5) + 1)):
        if numel % i == 0:
            return (numel // i, i)
    return (numel, 1)

@torch.no_grad()
def _unnmf(row_col: tuple) -> torch.Tensor:
    """Reconstructs a matrix from its rank-1 factors (outer product)."""
    return torch.outer(row_col[0], row_col[1])

@torch.no_grad()
def _nnmf(matrix: torch.Tensor):
    """Performs a rank-1 non-negative matrix factorization."""
    shape = matrix.shape
    M, N = shape

    # Calculate the initial factors (sums)
    # mu_factor is the sum along dim=1 (rows), shape (M)
    # mv_factor is the sum along dim=0 (columns), shape (N)
    mu_factor = torch.sum(matrix, dim=1, dtype=torch.float32)
    mv_factor = torch.sum(matrix, dim=0, dtype=torch.float32)

    # Normalize one of the factors for stability
    EPSILON = 1e-12
    if M < N:
        scale = mu_factor.sum()
        mu_factor.div_(scale.clamp_min(EPSILON))
    else:
        scale = mv_factor.sum()
        mv_factor.div_(scale.clamp_min(EPSILON))

    return mu_factor, mv_factor

# ------------------------------------------
# Losslessly converts 8-bit boolean to 1-bit sign and vice versa
# ------------------------------------------

@torch.no_grad()
def _pack_bools(tensor: torch.Tensor) -> torch.Tensor:
    """Packs a boolean tensor into a uint8 tensor to achieve 1-bit storage."""
    n, m = tensor.shape
    packed_m = (m + 7) // 8
    padded_tensor = torch.nn.functional.pad(tensor, (0, packed_m * 8 - m), 'constant', 0)
    reshaped = padded_tensor.view(n, packed_m, 8)
    shifter = torch.arange(8, device=tensor.device, dtype=torch.uint8)
    packed = (reshaped.to(torch.uint8) * (2**shifter)).sum(dim=2).to(torch.uint8)
    return packed

@torch.no_grad()
def _unpack_bools(packed_tensor: torch.Tensor, original_m: int) -> torch.Tensor:
    """Unpacks a uint8 tensor back into a boolean tensor."""
    if packed_tensor.dtype != torch.uint8:
        packed_tensor = packed_tensor.to(torch.uint8)
    shifter = (2**torch.arange(8, device=packed_tensor.device, dtype=torch.uint8)).view(1, 1, 8)
    unpacked_padded = (packed_tensor.unsqueeze(2) & shifter) != 0
    unpacked = unpacked_padded.view(packed_tensor.shape[0], -1)[:, :original_m]
    return unpacked
