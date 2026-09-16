"""Pure scoring primitives for PPI evaluation (no model, no torch).

``participation_t`` computes the per-protein positive rate
``t(p) = (#positive pairs touching p) / (#pairs touching p)`` and degree count from a split's own
labels; it is used for participation/degree descriptive statistics and the negative-sampling bias
analysis. Protein identity is the explicit protein id (e.g. a UniProt accession).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

Pair = Tuple[Hashable, Hashable]


def pair_score_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Metric block for endpoint-additive pair scorers (probabilities in ``p``).

    Shared verbatim by the C3 / PRING endpoint-additive audits (EBM / MLP). The
    field set — including ``brier``, ``accuracy_at_0.5`` and the score-distribution
    summary — is part of those audits' on-disk JSON contract, so it is deliberately
    distinct from :func:`src.eval.classification.binary_classification_metrics`.
    """
    pred = (p >= 0.5).astype(np.int8)
    return {
        "n": int(y.size),
        "pos_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "accuracy_at_0.5": float((pred == y).mean()),
        "brier": float(brier_score_loss(y, p)),
        "score_mean": float(p.mean()),
        "score_std": float(p.std()),
    }


def participation_t(
    pairs: Sequence[Pair], labels: Sequence[int]
) -> Tuple[Dict[Hashable, float], Dict[Hashable, int]]:
    """Per-protein positive rate t(p) and degree count, from the benchmark's own labels.

    Returns (t, degree) dicts keyed by protein id. ``t[p]`` ∈ [0,1]; ``degree[p]`` = #pairs touching p
    (each pair contributes to BOTH endpoints, matching the audit)."""
    pos: Dict[Hashable, int] = defaultdict(int)
    cnt: Dict[Hashable, int] = defaultdict(int)
    for (a, b), y in zip(pairs, labels):
        yi = int(y)
        for p in (a, b):
            cnt[p] += 1
            pos[p] += yi
    t = {p: pos[p] / cnt[p] for p in cnt}
    return t, dict(cnt)


def safe_spearman(left, right):
    """Finite Spearman correlation rounded for experiment summaries.

    Returns ``None`` when fewer than two finite paired points remain (or the
    statistic is non-finite), else the correlation rounded to 4 decimals. Used by
    the sequence-participation oracle to compare predicted vs true ``t(p)``.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return None if not np.isfinite(value) else round(value, 4)
