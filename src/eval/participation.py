"""Benchmark participation diagnostic — the honesty meter (DESIGN §7).

Given a benchmark's (pairs, labels) alone — **no model, no features** — quantify how separable it is by
per-protein participation. Reports the per-protein positive-rate distribution t(p): a wide, bimodal
t (large ``t_std``, high ``frac_t_gt_0.8`` / ``frac_t_lt_0.2``) ⇒ the benchmark is largely decided by
*which* proteins are present rather than interaction biology, so any model's absolute AUROC should be
read with that participation structure in mind.
"""

from __future__ import annotations

from typing import Dict, Hashable, Sequence, Tuple

import numpy as np

from src.eval.metrics import (
    participation_t,
    roc_auc,
)

Pair = Tuple[Hashable, Hashable]


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC, or None if a subset has only one class present (roc_auc undefined)."""
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return roc_auc(labels, scores)


def benchmark_diagnostic(pairs: Sequence[Pair], labels: Sequence[int]) -> Dict:
    """Participation diagnostic for one benchmark. Mirrors probe_participation_bias.py:analyze."""
    y = np.asarray(labels).astype(int)
    t, cnt = participation_t(pairs, y)
    tv = np.array(list(t.values()), dtype=float)
    degs = np.array(list(cnt.values()), dtype=float)

    return {
        "n_pairs": int(y.size),
        "n_proteins": int(len(cnt)),
        "pos_rate": round(float(y.mean()), 4),
        "degree_mean": round(float(degs.mean()), 2),
        "degree_median": float(np.median(degs)),
        "frac_deg1": round(float(np.mean(degs == 1)), 4),
        "t_mean": round(float(tv.mean()), 4),
        "t_std": round(float(tv.std()), 4),
        "frac_t_lt_0.2": round(float(np.mean(tv < 0.2)), 4),
        "frac_t_gt_0.8": round(float(np.mean(tv > 0.8)), 4),
    }
