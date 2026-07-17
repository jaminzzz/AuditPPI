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
from sklearn.metrics import average_precision_score, roc_auc_score

Pair = Tuple[Hashable, Hashable]


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return float(average_precision_score(np.asarray(labels), np.asarray(scores)))


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
