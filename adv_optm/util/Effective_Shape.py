def _get_effective_shape(numel: int) -> tuple[int, int]:
    """Finds two factors of numel that are closest to its square root."""
    if numel <= 0:
        return (0, 0)
    for i in reversed(range(1, int(numel ** 0.5) + 1)):
        if numel % i == 0:
            return (numel // i, i)
    return (numel, 1)