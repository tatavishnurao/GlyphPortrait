from __future__ import annotations


def normalize_seed(seed: int | None, default_seed: int = 42) -> int:
    if seed is None:
        return default_seed
    if seed < 0:
        return abs(seed)
    return seed
