"""Configuration for the v1 PPI fingerprint baseline protocol.

The Ladder-2 baseline reads the per-dataset ``auditppi_protein_features_v1``
protein caches (``conf.paths.PPI_PREDICTION_CACHES``): one cache per benchmark
family holds every endpoint sequence across that family's splits, with BOTH
backbone lines (ESM-C L60/L80 + ESM-2 L33) and all four channels in one payload.
Backbone/layer are selected within the cache at read time. PRING is per-species
(human train graph + yeast/ecoli/arath zero-shot test graphs), so it resolves
its cache by species via ``PRING_SPECIES_SAE_CACHES`` rather than one family key.
"""

from __future__ import annotations

from conf.model import REPRESENTATIONS
from conf.paths import (
    PPI_PREDICTION_CACHES as CACHE,
    PRING_SPECIES_SAE_CACHES,
    RESULTS_MISC,
)

MODEL_NAMES = ("xgb", "tabpfn", "mlp_pair", "tabm_pair")

# Benchmark family -> its own native train split. Each family trains on its own
# train set and is scored on its eval split (user decision). RF2-PPI has no train
# split, so it borrows C3's (zero-shot eval). PRING is handled specially in the
# runner: it trains on the human train graph of the SAME sampling method as the
# eval (default BFS for the zero-shot cross-species graphs), so it is not keyed
# here -- see ``PRING_DEFAULT_METHOD``.
NATIVE_TRAIN = {
    "c1": "c1:train",
    "c2": "c2:train",
    "c3": "c3:train",
    "cross_species": "cross_species:human_train",
    "bernett": "bernett:train",
    "rf2ppi": "c3:train",
}

# PRING human-graph sampling method used for the train side when the eval graph
# does not itself pin one (the zero-shot cross-species test graphs).
PRING_DEFAULT_METHOD = "BFS"

OUT_DIR = RESULTS_MISC / "ppi_fingerprint"

__all__ = [
    "CACHE",
    "MODEL_NAMES",
    "NATIVE_TRAIN",
    "OUT_DIR",
    "PRING_DEFAULT_METHOD",
    "PRING_SPECIES_SAE_CACHES",
    "REPRESENTATIONS",
]
