"""Shared constants for participation experiment protocols."""

from conf.paths import (
    AUDIT,
    BERNETT_SEQ_CACHE,
    CROSS_SPECIES_SEQ_CACHE,
    ESMC_DEFAULT_SEQ_CACHE,
    ROSETTA_SEQ_CACHE,
)

OUT_DIR = AUDIT / "pring_participation"
DEFAULT_PRING_CACHE = OUT_DIR / "pring_human_esmc_sae_cache.pt"

FALLBACK_CACHE_CANDIDATES = (
    ROSETTA_SEQ_CACHE,
    CROSS_SPECIES_SEQ_CACHE,
    BERNETT_SEQ_CACHE,
    ESMC_DEFAULT_SEQ_CACHE,
)

TRAIN_SPECIES = "human"
CROSS_SPECIES = ("yeast", "ecoli", "arath")
PAIR_EVALS = ("none", "human_test", "all")

__all__ = [
    "CROSS_SPECIES",
    "DEFAULT_PRING_CACHE",
    "FALLBACK_CACHE_CANDIDATES",
    "OUT_DIR",
    "PAIR_EVALS",
    "TRAIN_SPECIES",
]
