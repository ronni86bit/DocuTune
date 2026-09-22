"""Deterministic seeding helpers."""

from __future__ import annotations

import random


def set_seeds(seed: int) -> None:
    """Seed all available RNG sources for reproducibility.

    torch/numpy seeding is applied only when those libraries are installed,
    so this stays safe in GPU-free environments.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def derive_seed(base_seed: int, *parts: object) -> int:
    """Deterministically derive a sub-seed from a base seed and label parts.

    Pure integer arithmetic (not Python's salted hash) so results are stable
    across processes and platforms.
    """
    value = base_seed * 1_000_003
    for part in parts:
        text = str(part)
        for ch in text:
            value = (value * 131 + ord(ch)) % (2**31 - 1)
    return value
