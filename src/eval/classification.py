"""Reusable binary-classification metrics with safe single-class handling."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def best_f1_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[Optional[float], Optional[float]]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None, None
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return None, None
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_index]), float(f1[best_index])


def precision_at_k(
    labels: np.ndarray, scores: np.ndarray, k: int
) -> Optional[float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or k <= 0:
        return None
    k = min(k, labels.size)
    top_indices = np.argsort(scores)[::-1][:k]
    return float(labels[top_indices].mean())


def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    include_pred_prob_mean: bool = False,
) -> dict:
    """Return the common metric block used by protein classifiers."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    positives = int(labels.sum())
    threshold, best_f1 = best_f1_threshold(labels, scores)
    predicted_at_half = (scores >= 0.5).astype(np.int8)
    result = {
        "n": int(labels.size),
        "pos": positives,
        "neg": int(labels.size - positives),
        "pos_rate": round(float(labels.mean()), 6) if labels.size else None,
        "auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        "baseline_auprc": round(float(labels.mean()), 6) if labels.size else None,
        "precision_at_n_pos": precision_at_k(labels, scores, positives),
        "best_f1_threshold": threshold,
        "best_f1": best_f1,
        "f1_at_0.5": (
            float(f1_score(labels, predicted_at_half, zero_division=0))
            if labels.size
            else None
        ),
    }
    if include_pred_prob_mean:
        result["pred_prob_mean"] = (
            round(float(scores.mean()), 6) if labels.size else None
        )
    return result


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Binned expected calibration error for binary probabilities."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return error if len(labels) else float("nan")


def probe_classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict:
    """Pair-probe metric block (AUROC/AUPRC/Brier/ECE/F1/acc at 0.5).

    Used by compact SAE pair probes. Unlike :func:`binary_classification_metrics`,
    this assumes both classes are present (sklearn will raise otherwise) and
    reports calibration metrics that those protein classifiers do not need.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score as sklearn_f1_score,
        roc_auc_score,
    )

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": float(expected_calibration_error(labels, probabilities)),
        "f1": float(sklearn_f1_score(labels, predictions)),
        "acc": float(accuracy_score(labels, predictions)),
    }


__all__ = [
    "best_f1_threshold",
    "binary_classification_metrics",
    "expected_calibration_error",
    "precision_at_k",
    "probe_classification_metrics",
    "safe_auprc",
    "safe_auroc",
]
