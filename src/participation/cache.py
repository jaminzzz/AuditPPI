"""Feature-cache access and split assembly for participation workflows."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.participation.config import (
    DEFAULT_PRING_CACHE,
    FALLBACK_CACHE_CANDIDATES,
    PRING_CACHE_DIR,
    TRAIN_SPECIES,
)
from src.participation.features import (
    cache_feature_names,
    normalize_feature_kind,
    sequence_feature_names,
    sequence_matrix,
)
from src.participation.labels import ParticipationLabels


@dataclass
class FeaturePack:
    Xtr: np.ndarray
    Xva: np.ndarray
    Xte: np.ndarray
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    feature_names: List[str]
    info: dict


def stratified_degree_split(
    ids: Sequence[str],
    degree: Mapping[str, int],
    *,
    val_frac: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Create an inner validation split stratified by degree quantiles."""
    rng = np.random.default_rng(seed)
    ids_array = np.asarray(list(ids), dtype=object)
    degrees = np.asarray([degree[protein_id] for protein_id in ids_array], dtype=float)
    if len(ids_array) < 5 or val_frac <= 0:
        return list(ids_array), []

    quantiles = np.unique(np.quantile(degrees, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(quantiles) <= 2:
        bins = np.zeros(len(ids_array), dtype=int)
    else:
        bins = np.digitize(degrees, quantiles[1:-1], right=True)

    validation_indices: List[int] = []
    for bin_index in np.unique(bins):
        indices = np.flatnonzero(bins == bin_index)
        rng.shuffle(indices)
        if len(indices) >= 10:
            n_take = max(1, int(round(len(indices) * val_frac)))
        else:
            n_take = int(round(len(indices) * val_frac))
        validation_indices.extend(indices[:n_take].tolist())

    validation_indices = sorted(set(validation_indices))
    if not validation_indices:
        validation_indices = [int(rng.integers(0, len(ids_array)))]
    train_mask = np.ones(len(ids_array), dtype=bool)
    train_mask[validation_indices] = False
    return ids_array[train_mask].tolist(), ids_array[validation_indices].tolist()


def load_feature_cache(path: Path) -> dict:
    """Load a pooled feature cache on CPU, using mmap when torch supports it."""
    import torch

    kwargs = {"map_location": "cpu", "weights_only": False}
    if "mmap" in inspect.signature(torch.load).parameters:
        kwargs["mmap"] = True
    return torch.load(path, **kwargs)


def best_existing_fallback_cache() -> Optional[Path]:
    """Return the first existing pooled-cache fallback candidate."""
    for path in FALLBACK_CACHE_CANDIDATES:
        if path.exists():
            return path
    return None


def cache_id_map(cache: Mapping) -> Mapping[str, int]:
    """Return the first supported protein-id lookup mapping in a cache."""
    for key in ("uniprotid2idx", "unprotid2idx", "protein_id2idx", "id2idx"):
        mapping = cache.get(key)
        if mapping is not None:
            return mapping
    return {}


def lookup_cache_index(
    protein_id: str,
    sequences: Mapping[str, str],
    id_to_index: Mapping[str, int],
    sequence_to_index: Mapping[str, int],
) -> Tuple[Optional[int], str]:
    if protein_id in id_to_index:
        return int(id_to_index[protein_id]), "id"
    sequence = sequences.get(protein_id)
    if sequence is not None and sequence in sequence_to_index:
        return int(sequence_to_index[sequence]), "sequence"
    return None, "missing"


def filter_cached_ids(
    protein_ids: Sequence[str],
    sequences: Mapping[str, str],
    cache: Mapping,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Partition protein IDs into cached and missing sets."""
    id_to_index = cache_id_map(cache)
    sequence_to_index = cache.get("seq2idx", {})
    kept: List[str] = []
    missing: List[str] = []
    source_counts = {"id": 0, "sequence": 0, "missing": 0}
    for protein_id in protein_ids:
        row, source = lookup_cache_index(
            protein_id, sequences, id_to_index, sequence_to_index
        )
        source_counts[source] += 1
        if row is None:
            missing.append(protein_id)
        else:
            kept.append(protein_id)
    return kept, missing, source_counts


def cached_feature_matrix(
    protein_ids: Sequence[str],
    sequences: Mapping[str, str],
    cache: Mapping,
    feature_kind: str,
) -> np.ndarray:
    """Assemble one pooled feature matrix for the requested proteins."""
    import torch

    id_to_index = cache_id_map(cache)
    sequence_to_index = cache.get("seq2idx", {})
    rows: List[int] = []
    for protein_id in protein_ids:
        row, _ = lookup_cache_index(
            protein_id, sequences, id_to_index, sequence_to_index
        )
        if row is None:
            raise KeyError(f"protein {protein_id!r} is not present in the feature cache")
        rows.append(row)

    if feature_kind in ("sae_max", "binary"):
        matrix = cache["esmc_sae_max"]
    elif feature_kind == "sae_mean":
        matrix = cache["esmc_sae_mean"]
    elif feature_kind == "esmc_mean":
        matrix = cache["esmc_mean"]
    else:
        raise ValueError(feature_kind)

    indices = torch.as_tensor(rows, dtype=torch.long)
    output = matrix.index_select(0, indices)
    if feature_kind == "binary":
        output = output > 0
    return output.float().numpy().astype(np.float32, copy=False)


def prepare_features(
    *,
    feature_kind: str,
    seqs: Mapping[str, str],
    labels: ParticipationLabels,
    candidate_train: Sequence[str],
    candidate_test: Sequence[str],
    kmer: int,
    val_frac: float,
    seed: int,
    cache_path: Optional[Path],
) -> FeaturePack:
    """Prepare aligned train/validation/test participation feature matrices."""
    feature_kind = normalize_feature_kind(feature_kind)
    if feature_kind == "sequence_basic":
        train_pool = sorted(candidate_train)
        test_ids = sorted(candidate_test)
        train_ids, val_ids = stratified_degree_split(
            train_pool, labels.degree, val_frac=val_frac, seed=seed
        )
        train_x = sequence_matrix(train_ids, seqs, kmer=kmer)
        val_x = sequence_matrix(val_ids, seqs, kmer=kmer)
        test_x = sequence_matrix(test_ids, seqs, kmer=kmer)
        names = sequence_feature_names(kmer)
        info = {
            "kind": feature_kind,
            "kmer": kmer,
            "dim": len(names),
            "cache_path": None,
            "cache_required": False,
            "n_train_pool_with_features": len(train_pool),
            "n_test_with_features": len(test_ids),
            "missing_train_features": 0,
            "missing_test_features": 0,
        }
        return FeaturePack(
            train_x, val_x, test_x, train_ids, val_ids, test_ids, names, info
        )

    resolved_cache = Path(cache_path) if cache_path is not None else DEFAULT_PRING_CACHE
    if not resolved_cache.exists():
        fallback = best_existing_fallback_cache()
        hint = (
            "Build a PRING-specific cache with "
            "scripts/cache/cache_pring_human_esmc_sae.py, or pass --cache-path "
            "to an existing pooled cache."
        )
        if fallback is not None:
            hint += f" Best existing fallback detected: {fallback}"
        raise FileNotFoundError(
            f"pooled feature cache not found: {resolved_cache}. {hint}"
        )

    cache = load_feature_cache(resolved_cache)
    train_pool, missing_train, train_sources = filter_cached_ids(
        candidate_train, seqs, cache
    )
    test_ids, missing_test, test_sources = filter_cached_ids(
        candidate_test, seqs, cache
    )
    train_pool = sorted(train_pool)
    test_ids = sorted(test_ids)
    train_ids, val_ids = stratified_degree_split(
        train_pool, labels.degree, val_frac=val_frac, seed=seed
    )
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError(
            f"empty split after cache intersection: train={len(train_ids)} "
            f"val={len(val_ids)} test={len(test_ids)}"
        )

    train_x = cached_feature_matrix(train_ids, seqs, cache, feature_kind)
    val_x = cached_feature_matrix(val_ids, seqs, cache, feature_kind)
    test_x = cached_feature_matrix(test_ids, seqs, cache, feature_kind)
    names = cache_feature_names(feature_kind, train_x.shape[1])
    info = {
        "kind": feature_kind,
        "dim": int(train_x.shape[1]),
        "cache_path": str(resolved_cache),
        "cache_required": True,
        "cache_keys": sorted(str(key) for key in cache.keys()),
        "n_cache_rows": int(cache["esmc_mean"].shape[0]) if "esmc_mean" in cache else None,
        "n_train_pool_with_features": len(train_pool),
        "n_test_with_features": len(test_ids),
        "missing_train_features": len(missing_train),
        "missing_test_features": len(missing_test),
        "train_cache_lookup_sources": train_sources,
        "test_cache_lookup_sources": test_sources,
        "missing_train_feature_examples": missing_train[:20],
        "missing_test_feature_examples": missing_test[:20],
    }
    return FeaturePack(
        train_x, val_x, test_x, train_ids, val_ids, test_ids, names, info
    )


def features_from_cache(
    protein_ids: Sequence[str],
    sequences: Mapping[str, str],
    *,
    feature_kind: str,
    kmer: int,
    cache: Optional[Mapping],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a matrix while dropping IDs absent from a pooled cache."""
    if feature_kind == "sequence_basic":
        kept = [protein_id for protein_id in protein_ids if protein_id in sequences]
        missing = [protein_id for protein_id in protein_ids if protein_id not in sequences]
        matrix = (
            sequence_matrix(kept, sequences, kmer=kmer)
            if kept
            else np.empty((0, 0), np.float32)
        )
        return matrix, kept, missing
    if cache is None:
        raise ValueError("cache required for non-sequence_basic features")
    kept, missing, _ = filter_cached_ids(protein_ids, sequences, cache)
    kept = sorted(kept)
    matrix = (
        cached_feature_matrix(kept, sequences, cache, feature_kind)
        if kept
        else np.empty((0, 0), np.float32)
    )
    return matrix, kept, missing


def species_cache_path(species: str) -> Path:
    """Return the default pooled feature-cache path for one PRING species."""
    species = species.lower()
    if species == TRAIN_SPECIES:
        return DEFAULT_PRING_CACHE
    return PRING_CACHE_DIR / f"pring_{species}_esmc_sae_cache.pt"


__all__ = [
    "FeaturePack",
    "best_existing_fallback_cache",
    "cache_id_map",
    "cached_feature_matrix",
    "features_from_cache",
    "filter_cached_ids",
    "load_feature_cache",
    "lookup_cache_index",
    "prepare_features",
    "species_cache_path",
    "stratified_degree_split",
]
