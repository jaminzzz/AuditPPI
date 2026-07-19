"""Evaluation harness for AuditPPI.

Ported and cleaned from the SAE_PPI audit. The audit's central concern is that a PPI benchmark's
headline AUROC can reflect **per-protein participation bias** rather than interaction biology: each
protein `p` has a positive rate `t(p)`, quantified by the benchmark diagnostic (t(p) distribution).

This package is model- and dataset-agnostic:
  - `metrics`        — pure scoring primitives (participation_t, pair_score_metrics, safe_spearman).
  - `classification` — single-class-safe binary metrics (safe_auroc,
                       binary_classification_metrics) plus pair-probe metrics
                       (probe_classification_metrics, expected_calibration_error).
  - `participation`  — benchmark t(p) diagnostic + PRING participation-workflow evaluation.
  - `scoring`        — evaluate any pair-scorer (AUROC/AUPRC).

Benchmark objects and concrete loaders live in :mod:`src.data.pairs`.
"""

from src.data.pairs import Benchmark
from src.eval.metrics import (
    pair_score_metrics,
    participation_t,
    safe_spearman,
)
from src.eval.classification import (
    best_f1_threshold,
    binary_classification_metrics,
    expected_calibration_error,
    precision_at_k,
    probe_classification_metrics,
    safe_auprc,
    safe_auroc,
)
from src.eval.participation import benchmark_diagnostic
from src.eval.scoring import evaluate_scorer

__all__ = [
    "Benchmark",
    "evaluate_scorer",
    "benchmark_diagnostic",
    "participation_t",
    "pair_score_metrics",
    "safe_spearman",
    "best_f1_threshold",
    "binary_classification_metrics",
    "expected_calibration_error",
    "precision_at_k",
    "probe_classification_metrics",
    "safe_auprc",
    "safe_auroc",
]
