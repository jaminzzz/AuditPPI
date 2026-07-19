"""Sequence-composition features (length, AA / dipeptide composition) and
pooled-cache feature naming/normalization helpers.

These are the non-pLM, model-free protein features used by the participation
workflows' ``sequence_basic`` representation, plus the small helpers that name
and normalize the pooled-SAE feature kinds. They live here beside the ESM-C/SAE
extractors as the model-free end of the feature package.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA)}
FEATURE_KINDS = ("sequence_basic", "sae_max", "sae_mean", "binary", "esmc_mean")
FORMAL_FEATURE_KINDS = ("sae_max", "binary", "esmc_mean")
SAE_FEATURE_KINDS = ("sae_max", "sae_mean", "binary")

_FEATURE_ALIASES = {
    "pooled_sae": "sae_max",
    "sae": "sae_max",
    "binary_sae": "binary",
    "esmc": "esmc_mean",
}


def normalize_feature_kind(kind: str) -> str:
    normalized = _FEATURE_ALIASES.get(kind.lower(), kind.lower())
    if normalized not in FEATURE_KINDS:
        raise ValueError(f"feature_kind must be one of {FEATURE_KINDS}")
    return normalized


def sequence_feature_names(kmer: int) -> list[str]:
    names = ["length", "log1p_length", "frac_standard_aa"]
    names += [f"aa_{amino_acid}" for amino_acid in AA]
    if kmer >= 2:
        names += [f"di_{left}{right}" for left in AA for right in AA]
    return names


def cache_feature_names(feature_kind: str, dim: int) -> list[str]:
    if feature_kind == "esmc_mean":
        return [f"esmc_mean_{index:04d}" for index in range(dim)]
    if feature_kind == "binary":
        return [f"sae_binary_{index:05d}" for index in range(dim)]
    prefix = "sae_mean" if feature_kind == "sae_mean" else "sae_max"
    return [f"{prefix}_{index:05d}" for index in range(dim)]


def sequence_features(sequence: str, *, kmer: int = 2) -> np.ndarray:
    """Length, amino-acid composition, and optional dipeptide composition."""
    sequence = sequence.upper()
    length = len(sequence)
    features: list[float] = [float(length), float(np.log1p(length))]
    amino_acid_counts = np.zeros(len(AA), dtype=np.float32)
    standard_count = 0
    for residue in sequence:
        index = AA_TO_INDEX.get(residue)
        if index is not None:
            amino_acid_counts[index] += 1.0
            standard_count += 1
    denominator = max(1, length)
    features.append(float(standard_count / denominator))
    features.extend((amino_acid_counts / denominator).tolist())
    if kmer >= 2:
        dipeptides = np.zeros(len(AA) * len(AA), dtype=np.float32)
        total = 0
        for left, right in zip(sequence[:-1], sequence[1:]):
            left_index, right_index = AA_TO_INDEX.get(left), AA_TO_INDEX.get(right)
            if left_index is None or right_index is None:
                continue
            dipeptides[left_index * len(AA) + right_index] += 1.0
            total += 1
        features.extend((dipeptides / max(1, total)).tolist())
    return np.asarray(features, dtype=np.float32)


def sequence_matrix(
    protein_ids: Sequence[str],
    sequences: Mapping[str, str],
    *,
    kmer: int,
) -> np.ndarray:
    return np.vstack(
        [sequence_features(sequences[protein_id], kmer=kmer) for protein_id in protein_ids]
    ).astype(np.float32)


__all__ = [
    "FEATURE_KINDS",
    "FORMAL_FEATURE_KINDS",
    "SAE_FEATURE_KINDS",
    "cache_feature_names",
    "normalize_feature_kind",
    "sequence_feature_names",
    "sequence_features",
    "sequence_matrix",
]
