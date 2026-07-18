"""Configuration for the pooled PPI fingerprint baseline protocol."""

from __future__ import annotations

from conf.model import REPRESENTATIONS
from conf.paths import (
    RESULTS_MISC,
    CROSS_SPECIES_SEQ_CACHE,
    ESMC_DEFAULT_SEQ_CACHE,
    ROSETTA_SEQ_CACHE,
)

MODEL_NAMES = ("xgb", "tabpfn", "dualtower")

CACHE = {
    "c3": ESMC_DEFAULT_SEQ_CACHE,
    "cross_species": CROSS_SPECIES_SEQ_CACHE,
    "rf2ppi": ROSETTA_SEQ_CACHE,
}

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
