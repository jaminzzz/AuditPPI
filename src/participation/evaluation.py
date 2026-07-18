"""Node-, pair-, and output-level evaluation for participation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from conf.audit import PARTICIPATION_QUANTILE
from src.eval.classification import safe_auprc, safe_auroc

Pair = Tuple[str, str]


def round_or_none(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    if value is None or np.isnan(value):
        return None
    return round(float(value), ndigits)


def participation_node_metrics(
    protein_ids: Sequence[str],
    degree: Mapping[str, int],
    predicted_log_degree: np.ndarray,
    predicted_degree: np.ndarray,
) -> dict:
    """Evaluate log-degree and calibrated degree predictions per protein."""
    true_degree = np.asarray([degree[protein_id] for protein_id in protein_ids], dtype=float)
    true_log_degree = np.log1p(true_degree)
    spearman_degree = (
        spearmanr(predicted_degree, true_degree).correlation if len(protein_ids) >= 3 else None
    )
    spearman_log = (
        spearmanr(predicted_log_degree, true_log_degree).correlation
        if len(protein_ids) >= 3
        else None
    )
    pearson = (
        np.corrcoef(predicted_degree, true_degree)[0, 1]
        if len(protein_ids) >= 3
        and np.std(predicted_degree) > 0
        and np.std(true_degree) > 0
        else None
    )
    threshold = float(np.quantile(true_degree, PARTICIPATION_QUANTILE)) if len(true_degree) else 0.0
    high_degree = (true_degree >= threshold).astype(int)
    return {
        "n": int(len(protein_ids)),
        "degree_mean": round_or_none(float(true_degree.mean()) if len(true_degree) else None),
        "degree_median": round_or_none(float(np.median(true_degree)) if len(true_degree) else None),
        "degree_p90": round_or_none(threshold),
        "degree_max": round_or_none(float(true_degree.max()) if len(true_degree) else None),
        "pred_degree_mean": round_or_none(float(predicted_degree.mean()) if len(predicted_degree) else None),
        "pred_degree_sum": round_or_none(float(predicted_degree.sum()) if len(predicted_degree) else None),
        "true_degree_sum": round_or_none(float(true_degree.sum()) if len(true_degree) else None),
        "spearman_pred_degree": round_or_none(spearman_degree),
        "spearman_pred_logdegree": round_or_none(spearman_log),
        "pearson_pred_degree": round_or_none(pearson),
        "mae_log1p_degree": round_or_none(
            float(np.mean(np.abs(predicted_log_degree - true_log_degree)))
            if len(true_log_degree)
            else None
        ),
        "high_degree_threshold_p90": round_or_none(threshold),
        "high_degree_auroc": round_or_none(safe_auroc(high_degree, predicted_degree)),
        "high_degree_auprc": round_or_none(safe_auprc(high_degree, predicted_degree)),
    }


def read_labeled_pairs(
    path: Path, *, drop_self_pairs: bool
) -> Tuple[List[Pair], np.ndarray]:
    pairs: List[Pair] = []
    labels: List[int] = []
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            endpoint_a, endpoint_b, label = parts[0], parts[1], int(parts[2])
            if drop_self_pairs and endpoint_a == endpoint_b:
                continue
            pairs.append((endpoint_a, endpoint_b))
            labels.append(label)
    return pairs, np.asarray(labels, dtype=int)


def participation_pair_metrics(
    pair_path: Path,
    *,
    pred_t: Mapping[str, float],
    true_t: Mapping[str, float],
    drop_self_pairs: bool,
) -> dict:
    """Evaluate endpoint-min participation scores on a labeled pair file."""
    pairs, labels = read_labeled_pairs(pair_path, drop_self_pairs=drop_self_pairs)
    predicted_scores: List[float] = []
    true_scores: List[float] = []
    kept_labels: List[int] = []
    skipped = 0
    for (endpoint_a, endpoint_b), label in zip(pairs, labels):
        predicted_a, predicted_b = pred_t.get(endpoint_a), pred_t.get(endpoint_b)
        true_a, true_b = true_t.get(endpoint_a), true_t.get(endpoint_b)
        if predicted_a is None or predicted_b is None or true_a is None or true_b is None:
            skipped += 1
            continue
        predicted_scores.append(min(predicted_a, predicted_b))
        true_scores.append(min(true_a, true_b))
        kept_labels.append(int(label))

    scored_labels = np.asarray(kept_labels, dtype=int)
    predicted = np.asarray(predicted_scores, dtype=float)
    oracle = np.asarray(true_scores, dtype=float)
    return {
        "path": str(pair_path),
        "n_total": int(len(pairs)),
        "n_scored": int(len(scored_labels)),
        "n_skipped": int(skipped),
        "pos_rate": round_or_none(
            float(scored_labels.mean()) if len(scored_labels) else None
        ),
        "pred_min_t_auroc": round_or_none(safe_auroc(scored_labels, predicted)),
        "pred_min_t_auprc": round_or_none(safe_auprc(scored_labels, predicted)),
        "true_full_min_t_auroc": round_or_none(safe_auroc(scored_labels, oracle)),
        "true_full_min_t_auprc": round_or_none(safe_auprc(scored_labels, oracle)),
    }


def write_participation_predictions(
    path: Path,
    *,
    split_ids: Mapping[str, Sequence[str]],
    degree: Mapping[str, int],
    true_t: Mapping[str, float],
    pred_log: Mapping[str, float],
    pred_degree: Mapping[str, float],
    pred_t: Mapping[str, float],
) -> None:
    with path.open("w") as handle:
        handle.write(
            "protein\tsplit\tdegree_full\tlog1p_degree_full\ttrue_t_full\t"
            "pred_log1p_degree\tpred_degree\tpred_t\n"
        )
        for split, protein_ids in split_ids.items():
            for protein_id in protein_ids:
                handle.write(
                    f"{protein_id}\t{split}\t{degree[protein_id]}\t"
                    f"{np.log1p(degree[protein_id]):.6g}\t{true_t[protein_id]:.6g}\t"
                    f"{pred_log[protein_id]:.6g}\t{pred_degree[protein_id]:.6g}\t"
                    f"{pred_t[protein_id]:.6g}\n"
                )


__all__ = [
    "participation_node_metrics",
    "participation_pair_metrics",
    "read_labeled_pairs",
    "round_or_none",
    "write_participation_predictions",
]
