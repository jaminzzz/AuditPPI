"""PPI fingerprint feature assembly and baseline experiment protocol.

The package combines pooled per-protein representations into pair features,
then orchestrates XGBoost, TabPFN, MLP-pair, or TabM-pair evaluation. Shared data,
models, and general analyses live in their corresponding top-level packages.
"""

from src.ppi_fingerprint.config import (
    CACHE,
    MODEL_NAMES,
    NATIVE_TRAIN,
    OUT_DIR,
    REPRESENTATIONS,
)


def __getattr__(name: str):
    if name == "run_baseline":
        from src.ppi_fingerprint.baseline import run_baseline

        return run_baseline
    if name == "run_baseline_evals":
        from src.ppi_fingerprint.baseline import run_baseline_evals

        return run_baseline_evals
    raise AttributeError(name)

__all__ = [
    "run_baseline",
    "run_baseline_evals",
    "CACHE",
    "MODEL_NAMES",
    "NATIVE_TRAIN",
    "OUT_DIR",
    "REPRESENTATIONS",
]
