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


__all__ = [
    "best_f1_threshold",
    "binary_classification_metrics",
    "precision_at_k",
    "safe_auprc",
    "safe_auroc",
]
