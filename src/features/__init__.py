"""Unified protein and protein-pair feature extraction for AuditPPI.

The package is intentionally independent of any particular benchmark. Dataset
scripts only need to provide protein identifiers/sequences; the extractors own
the ESM-C / ESM-2 / SAE implementation and the pair module owns the symmetric
feature construction.

Module map
----------
- ``extractors`` / ``pooling`` / ``manifest`` — write protein feature caches.
- ``pairs`` — pair construction + ``auditppi_protein_features_v1`` cache load
  (:func:`~src.features.pairs.load_protein_feature_cache`).
- ``protein_cache`` — seq-keyed pooled fingerprint read
  (:func:`~src.features.protein_cache.protein_feature_rows`).
- ``pooled_assembly`` — id-first pooled assembly + degree splits
  (:func:`~src.features.pooled_assembly.load_pooled_payload`).
- ``sequence_composition`` / ``feature_selection`` / ``sampling`` — composition
  features, column selection, row subsample.
- ``baseline_io`` — independent baseline pair CSV + cache write.

Pair-probe ranking / sym top-k / fit wrappers live in
:mod:`src.interp.pair_probe` (not here).
"""

from src.data.sequences import normalize_sequence

from .manifest import ProteinManifest, load_protein_manifest
from .pairs import (
    ORDER_SENSITIVE_MODES,
    PAIR_MODE_MULTIPLIER,
    PAIR_MODES,
    needs_abba,
    pair_features,
    pair_mode_dim,
)
from .protein_cache import (
    REPRESENTATIONS,
    pair_feature_rows,
    protein_feature_rows,
    rep_dim,
    representation_matrix,
)

__all__ = [
    "ORDER_SENSITIVE_MODES",
    "PAIR_MODE_MULTIPLIER",
    "PAIR_MODES",
    "ProteinManifest",
    "REPRESENTATIONS",
    "load_protein_manifest",
    "needs_abba",
    "normalize_sequence",
    "pair_features",
    "pair_feature_rows",
    "pair_mode_dim",
    "protein_feature_rows",
    "rep_dim",
    "representation_matrix",
]
