"""Model-agnostic scoring of pair predictions on a benchmark."""

from __future__ import annotations

from typing import Callable, Dict, Hashable

import numpy as np

from src.data.pairs import Benchmark
from src.eval.metrics import auprc
from src.eval.participation import _safe_auroc


def evaluate_scorer(
    scorer: Callable[[Hashable, Hashable], float],
    bench: Benchmark,
    *,
    name: str = "model",
) -> Dict:
    """Score every pair and report AUROC/AUPRC, skipping missing scores."""
    raw = [scorer(a, b) for a, b in bench.pairs]
    idx = [i for i, score in enumerate(raw) if score is not None and not np.isnan(score)]
    if not idx:
        raise ValueError("scorer returned no valid scores")
    labels = bench.labels[idx]
    scores = np.asarray([raw[i] for i in idx], dtype=float)
    model_auroc = _safe_auroc(labels, scores)
    pos_rate = float(labels.mean()) if len(labels) else None
    have_two = len(labels) and len(np.unique(labels)) >= 2
    model_auprc = auprc(labels, scores) if have_two else None
    return {
        "scorer": name,
        "benchmark": bench.name,
        "n_scored": len(idx),
        "n_total": len(bench.pairs),
        "pos_rate": round(pos_rate, 4) if pos_rate is not None else None,
        "auroc": round(model_auroc, 4) if model_auroc is not None else None,
        "auprc": round(model_auprc, 4) if model_auprc is not None else None,
    }


__all__ = ["evaluate_scorer"]
