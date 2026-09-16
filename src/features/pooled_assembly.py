"""Pooled feature-cache assembly and split preparation for participation.

Id-first (UniProt-ID indexed) resolution into a pooled per-protein cache, the
degree-stratified inner split, and the train/val/test :class:`FeaturePack` the
PRING participation workflows train on. Complements
:mod:`src.features.protein_cache` (the sequence-keyed pair/protein row
primitives): both share the representation -> matrix switch via
:func:`~src.features.protein_cache.representation_matrix`.
:func:`load_pooled_payload` loads the *whole* payload (it needs the
``uniprotid2idx`` maps and uses mmap), so it stays distinct from
:func:`src.features.pairs.load_protein_feature_cache` (the formal
``auditppi_protein_features_v1`` writer format).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from conf.paths import (
    PRING_HUMAN_SAE_CACHE as DEFAULT_PRING_CACHE,
    PRING_FALLBACK_CACHES as FALLBACK_CACHE_CANDIDATES,
    PROTEIN_SAE_CACHES as PRING_CACHE_DIR,
    PRING_TRAIN_SPECIES as TRAIN_SPECIES,
)
from conf.model import (
    CACHE_MAX_RESIDUES_DEFAULT,
    DEFAULT_BACKBONE,
    cache_max_residues_tag,
    resolve_backbone_layer,
)
from src.data.pring_graph import ParticipationLabels
from src.features.protein_cache import representation_matrix
from src.features.sequence_composition import (
    cache_feature_names,
    normalize_feature_kind,
    sequence_feature_names,
    sequence_matrix,
)


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


def load_pooled_payload(path: Path) -> dict:
    """Load a full pooled fingerprint cache on CPU (mmap when torch supports it).

    Returns the entire payload so callers can use id maps (``uniprotid2idx``,
    …) in addition to the pooled tensors.
    """
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


def _cache_row_count(cache: Mapping) -> Optional[int]:
    """Row count of a pooled cache across v1 and legacy schemas.

    v1 caches carry a ``features`` subdict (any channel shares the row count);
    legacy flat caches expose ``esmc_mean`` directly.
    """
    features = cache.get("features")
    if features:
        first = next(iter(features.values()))
        return int(first.shape[0])
    if "esmc_mean" in cache:
        return int(cache["esmc_mean"].shape[0])
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
    layer: Optional[int] = None,
    backbone: str = DEFAULT_BACKBONE,
) -> np.ndarray:
    """Assemble one pooled feature matrix for the requested proteins.

    Row resolution is id-first (``uniprotid2idx`` etc.) then sequence-keyed; the
    representation -> matrix switch is shared with the pair/protein primitives via
    :func:`src.features.protein_cache.representation_matrix` (thresholding
    commutes with the row ``index_select``, so this stays bit-identical to the
    old per-kind switch). ``backbone``/``layer`` select the v1 channel family
    (``layer=None`` picks the backbone default).
    """
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

    matrix = representation_matrix(cache, feature_kind, layer, backbone)
    indices = torch.as_tensor(rows, dtype=torch.long)
    output = matrix.index_select(0, indices)
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
    layer: Optional[int] = None,
    backbone: str = DEFAULT_BACKBONE,
) -> FeaturePack:
    """Prepare aligned train/validation/test participation feature matrices.

    ``backbone`` selects the pLM line (``esmc`` / ``esm2``) and ``layer`` the
    layer within it (``None`` = that backbone's default: ESM-C 60, ESM-2 33) for
    v1 feature caches; both are ignored for ``sequence_basic`` and for legacy
    single-layer flat caches.
    """
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
            "Build a PRING v1 protein cache with "
            "scripts/prep/slice_dataset_protein_cache.py (from the pooled "
            "seq caches under data/sae/seq_caches), or pass --cache-path to "
            "an existing protein_features_max1022.pt cache."
        )
        if fallback is not None:
            hint += f" Best existing fallback detected: {fallback}"
        raise FileNotFoundError(
            f"pooled feature cache not found: {resolved_cache}. {hint}"
        )

    cache = load_pooled_payload(resolved_cache)
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

    train_x = cached_feature_matrix(train_ids, seqs, cache, feature_kind, layer, backbone)
    val_x = cached_feature_matrix(val_ids, seqs, cache, feature_kind, layer, backbone)
    test_x = cached_feature_matrix(test_ids, seqs, cache, feature_kind, layer, backbone)
    names = cache_feature_names(feature_kind, train_x.shape[1])
    info = {
        "kind": feature_kind,
        "backbone": backbone,
        "layer": resolve_backbone_layer(backbone, layer),
        "dim": int(train_x.shape[1]),
        "cache_path": str(resolved_cache),
        "cache_required": True,
        "cache_keys": sorted(str(key) for key in cache.keys()),
        "n_cache_rows": _cache_row_count(cache),
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
    layer: Optional[int] = None,
    backbone: str = DEFAULT_BACKBONE,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a matrix while dropping IDs absent from a pooled cache.

    ``layer``/``backbone`` select the v1 feature channel (``layer=None`` uses the
    backbone default: ESM-C 60, ESM-2 33).
    """
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
        cached_feature_matrix(kept, sequences, cache, feature_kind, layer, backbone)
        if kept
        else np.empty((0, 0), np.float32)
    )
    return matrix, kept, missing


def species_cache_path(
    species: str, max_residues: int = CACHE_MAX_RESIDUES_DEFAULT
) -> Path:
    """Return the v1 protein feature-cache path for one PRING species.

    ``max_residues`` selects the on-disk length variant (``max1022`` /
    ``max2046``); the default is the cross-backbone-comparable 1022 slice. Human
    reuses :data:`DEFAULT_PRING_CACHE` only at the default length; other lengths
    resolve by the shared ``pring_{species}_protein_features_{tag}.pt`` pattern.
    """
    species = species.lower()
    tag = cache_max_residues_tag(max_residues)
    if species == TRAIN_SPECIES and max_residues == CACHE_MAX_RESIDUES_DEFAULT:
        return DEFAULT_PRING_CACHE
    return PRING_CACHE_DIR / f"pring_{species}_protein_features_{tag}.pt"


__all__ = [
    "cache_id_map",
    "features_from_cache",
    "load_pooled_payload",
    "prepare_features",
    "species_cache_path",
    "stratified_degree_split",
]
