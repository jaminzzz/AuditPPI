"""Configuration for the pooled PPI fingerprint baseline protocol."""

from __future__ import annotations

from conf.paths import (
    AUDIT,
    CROSS_SPECIES_SEQ_CACHE,
    ESMC_DEFAULT_SEQ_CACHE,
    ROSETTA_SEQ_CACHE,
)

MODEL_NAMES = ("xgb", "tabpfn", "dualtower")
REPRESENTATIONS = ("binary", "sae_max", "esmc_mean")

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

OUT_DIR = AUDIT / "ppi_fingerprint"

__all__ = [
    "CACHE",
    "MODEL_NAMES",
    "NATIVE_TRAIN",
    "OUT_DIR",
    "REPRESENTATIONS",
]
