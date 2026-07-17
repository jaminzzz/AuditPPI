import numpy as np

from src.eval.classification import (
    best_f1_threshold,
    binary_classification_metrics,
    precision_at_k,
    safe_auprc,
    safe_auroc,
)


def test_binary_metrics_handle_ranked_scores():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=float)
    metrics = binary_classification_metrics(
        labels, scores, include_pred_prob_mean=True
    )
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["precision_at_n_pos"] == 1.0
    assert metrics["pred_prob_mean"] == 0.5
    assert best_f1_threshold(labels, scores)[0] is not None
    assert precision_at_k(labels, scores, 2) == 1.0


def test_safe_binary_metrics_return_none_for_single_class():
    labels = np.zeros(3, dtype=np.int8)
    scores = np.asarray([0.1, 0.2, 0.3])
    assert safe_auroc(labels, scores) is None
    assert safe_auprc(labels, scores) is None
    assert best_f1_threshold(labels, scores) == (None, None)
