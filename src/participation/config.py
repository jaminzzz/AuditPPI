"""Shared constants for participation experiment protocols."""

from conf.paths import (
    RESULTS_PROTEIN,
    BERNETT_SEQ_CACHE,
    CROSS_SPECIES_SEQ_CACHE,
    ESMC_DEFAULT_SEQ_CACHE,
    PRING_HUMAN_SAE_CACHE,
    PROTEIN_SAE_CACHES,
    ROSETTA_SEQ_CACHE,
)

# Experiment OUTPUT dir (xgboost tsv/json etc.). Reusable pooled caches no
# longer live here -- they moved to PROTEIN_SAE_CACHES under data/sae/.
OUT_DIR = RESULTS_PROTEIN / "pring_participation"
# Reusable pooled feature caches (input): human is a named constant; cross-
# species caches derive from PRING_CACHE_DIR (see cache.species_cache_path).
PRING_CACHE_DIR = PROTEIN_SAE_CACHES
DEFAULT_PRING_CACHE = PRING_HUMAN_SAE_CACHE

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
    "PRING_CACHE_DIR",
    "PAIR_EVALS",
    "TRAIN_SPECIES",
]
