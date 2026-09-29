"""Reproducibility and small shared helpers.

Contents:
  - set_global_seed() : seed Python, NumPy and PyTorch in one call
  - make_rng()        : a NumPy Generator owned by an agent
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness the project touches.

    Exact bit-for-bit reproducibility on GPU would also need
    `torch.use_deterministic_algorithms(True)`; on CPU this is enough.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- legacy global state used by some libs
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(seed)


def make_rng(seed: int | None) -> np.random.Generator:
    """Agents own their RNG instead of relying on global state."""
    return np.random.default_rng(seed)
