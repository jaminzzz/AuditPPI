"""Unified protein and protein-pair feature extraction for AuditPPI.

The package is intentionally independent of any particular benchmark. Dataset
scripts only need to provide protein identifiers/sequences; the extractors own
the ESM-C / ESM-2 / SAE implementation and the pair module owns the symmetric
feature construction.
"""

from src.data.sequences import normalize_sequence

from .manifest import ProteinManifest, load_protein_manifest
from .pairs import PAIR_MODES, pair_features
from .protein_cache import (
    REPRESENTATIONS,
    load_pooled_cache,
    pair_feature_rows,
    protein_feature_rows,
    rep_dim,
    representation_matrix,
)

__all__ = [
    "PAIR_MODES",
    "ProteinManifest",
    "REPRESENTATIONS",
    "load_pooled_cache",
    "load_protein_manifest",
    "normalize_sequence",
    "pair_features",
    "pair_feature_rows",
    "protein_feature_rows",
    "rep_dim",
    "representation_matrix",
]
