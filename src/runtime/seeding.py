"""Deterministic seeding for reproducible audit runs."""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (if available) ``torch`` from one value.

    Shared by the endpoint-additive audits. ``torch`` is imported lazily so
    non-torch callers (e.g. the EBM audit) do not pay the import cost; the CUDA
    branch is a no-op when no GPU is present.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
