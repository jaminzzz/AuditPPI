"""Evaluation harness for AuditPPI.

Ported and cleaned from the SAE_PPI audit. The audit's central concern is that a PPI benchmark's
headline AUROC can reflect **per-protein participation bias** rather than interaction biology: each
protein `p` has a positive rate `t(p)`, quantified by the benchmark diagnostic (t(p) distribution).

This package is model- and dataset-agnostic:
  - `metrics`        — pure scoring primitives (roc_auc, auprc, per-protein participation stats).
  - `participation`  — benchmark diagnostic (t(p) stats); needs only (pairs, labels).
  - `scoring`        — evaluate any pair-scorer (AUROC/AUPRC).

Benchmark objects and concrete loaders live in :mod:`src.data.benchmarks`.
"""

from src.data.benchmarks import Benchmark
from src.eval.metrics import (
    auprc,
    participation_t,
    roc_auc,
)
from src.eval.classification import (
    best_f1_threshold,
    binary_classification_metrics,
    precision_at_k,
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
    "roc_auc",
    "auprc",
    "best_f1_threshold",
    "binary_classification_metrics",
    "precision_at_k",
    "safe_auprc",
    "safe_auroc",
]
