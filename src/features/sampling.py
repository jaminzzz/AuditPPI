"""Class-stratified index sampling for training-matrix preparation.

Row-level sampling utilities (as opposed to the column/feature selection in
:mod:`src.features.feature_selection`): pick a class-balanced subset of row
indices, used to cap training size and to carve the inner validation split that
drives early stopping in the fingerprint baseline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def stratified_subsample(y: np.ndarray, max_rows: Optional[int], seed: int) -> Optional[np.ndarray]:
    """Indices of a class-stratified subsample (None ⇒ keep all)."""
    if max_rows is None or max_rows >= len(y):
        return None
    rng = np.random.default_rng(seed)
    chosen = []
    remaining = max_rows
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        idx = np.flatnonzero(y == cls)
        n_take = remaining if i == len(classes) - 1 else min(
            int(round(max_rows * len(idx) / len(y))), len(idx), remaining)
        chosen.append(rng.choice(idx, size=n_take, replace=False))
        remaining -= n_take
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


__all__ = ["stratified_subsample"]
