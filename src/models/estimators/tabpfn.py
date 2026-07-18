"""TabPFN integration with support for the repository-vendored source tree."""

from __future__ import annotations

import sys

import numpy as np

from conf.model import DEFAULT_SEED
from conf.paths import TABPFN_SRC


def fit_tabpfn(
    Xtr,
    ytr,
    *,
    n_estimators: int = 4,
    subsample_samples: int = 50000,
    ignore_limits: bool = True,
    seed: int = DEFAULT_SEED,
    device: str = "cuda",
):
    try:
        from tabpfn import TabPFNClassifier
    except (ImportError, ModuleNotFoundError):
        if TABPFN_SRC.exists() and str(TABPFN_SRC) not in sys.path:
            sys.path.insert(0, str(TABPFN_SRC))
        from tabpfn import TabPFNClassifier

    inference_config = (
        {"SUBSAMPLE_SAMPLES": subsample_samples} if subsample_samples > 0 else None
    )
    clf = TabPFNClassifier(
        device=device,
        n_estimators=n_estimators,
        ignore_pretraining_limits=ignore_limits,
        inference_config=inference_config,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    return clf


def predict_proba_chunked(clf, X: np.ndarray, batch_size: int = 5000) -> np.ndarray:
    """Return positive-class probabilities without materializing one huge query."""
    if batch_size <= 0 or X.shape[0] <= batch_size:
        return clf.predict_proba(X)[:, 1]
    parts = [
        clf.predict_proba(X[start : start + batch_size])[:, 1]
        for start in range(0, X.shape[0], batch_size)
    ]
    return np.concatenate(parts)


__all__ = ["fit_tabpfn", "predict_proba_chunked"]
