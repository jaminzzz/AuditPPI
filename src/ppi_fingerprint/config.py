"""Configuration for the pooled PPI fingerprint baseline protocol."""

from __future__ import annotations

from conf.model import REPRESENTATIONS
from conf.paths import POOLED_SEQ_CACHES as CACHE, RESULTS_MISC

MODEL_NAMES = ("xgb", "tabpfn", "dualtower")

# Benchmark -> pooled per-sequence cache. Centralized in
# conf.paths.POOLED_SEQ_CACHES and shared with the C3 / cross-species sequence
# participation oracle.

NATIVE_TRAIN = {
    "c3": "c3:train",
    "cross_species": "cross_species:human_train",
    "rf2ppi": "c3:train",
}

OUT_DIR = RESULTS_MISC / "ppi_fingerprint"

__all__ = [
    "CACHE",
    "MODEL_NAMES",
    "NATIVE_TRAIN",
    "OUT_DIR",
    "REPRESENTATIONS",
]
