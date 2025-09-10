import torch

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