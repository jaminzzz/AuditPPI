"""Shared accessor for pooled per-protein feature caches.

A *pooled cache* (produced by ``scripts/cache/cache_esmc_fingerprints.py`` and
friends) is a ``torch.save`` payload with at least::

    seq2idx       dict[str, int]   sequence string -> row index
    esmc_sae_max  Tensor [N, ESMC_SAE_DIM]   pooled SAE max  (the SAE channel)
    esmc_mean     Tensor [N, ESMC_DIM]       pooled dense ESM-C mean

Older on-disk caches may still contain ``esmc_sae_mean``; that channel is no
longer part of the formal feature contract and is ignored by this reader.

Both the fingerprint baseline (``src.ppi_fingerprint``) and the participation
oracle scripts previously each re-implemented the cache load, the
representation -> matrix switch, and the id/sequence -> row assembly. That logic
now lives here once; the id-keyed pooled assembly in
``src.features.pooled_assembly`` reuses the representation switch too.

Import-time torch-free
----------------------
``torch`` is imported lazily *inside* the functions that need it. Importing this
module (and hence ``src.ppi_fingerprint.config``, which re-exports
``REPRESENTATIONS``) never pulls torch into a fresh process -- a property the
ppi_fingerprint package test pins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from conf.model import (
    BACKBONE_DENSE_DIM,
    BACKBONE_SAE_DIM,
    DEFAULT_BACKBONE,
    ESMC_DIM,
    ESMC_SAE_DIM,
    REPRESENTATIONS,
    SAE_BINARY_THRESHOLD,
    feature_cache_key,
    resolve_backbone_layer,
)

# Keys picked out of a pooled cache payload (the rest of the file is ignored).
# ``esmc_sae_mean`` is intentionally absent: new extracts do not write it, and
# consumers only use max / binary / dense-mean.
POOLED_CACHE_KEYS = ("seq2idx", "esmc_mean", "esmc_sae_max")


def load_pooled_cache(path: Path, *, keys: Tuple[str, ...] = POOLED_CACHE_KEYS) -> Dict:
    """Read a per-protein pooled cache -> ``{seq2idx, esmc_mean, esmc_sae_max}``."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {key: payload[key] for key in keys}


def rep_dim(rep: str, backbone: str = DEFAULT_BACKBONE) -> int:
    """Column count of a representation's per-protein matrix for a backbone."""
    if rep not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {rep!r}; choose from {REPRESENTATIONS}")
    return BACKBONE_DENSE_DIM[backbone] if rep == "esmc_mean" else BACKBONE_SAE_DIM[backbone]


# Representation -> v1 channel suffix. ``binary`` maps to the stored
# ``sae_binary`` channel, which is bit-identical to ``sae_max > 0`` (verified),
# so v1 and legacy flat-key caches yield the same boolean matrix. ``esmc_mean``
# is a historical alias for the dense layer-mean channel of *either* backbone.
_REP_TO_V1_CHANNEL = {
    "sae_max": "sae_max",
    "binary": "sae_binary",
    "esmc_mean": "dense_mean",
}


def representation_matrix(
    cache: Dict,
    rep: str,
    layer: Optional[int] = None,
    backbone: str = DEFAULT_BACKBONE,
):
    """The per-protein matrix for a representation (torch tensor; ``binary`` is bool).

    Reads the formal ``auditppi_protein_features_v1`` layout when the cache
    carries a ``features`` subdict, selecting
    ``features[f"{backbone}_l{layer}_{channel}"]`` (``layer=None`` picks the
    backbone's default layer). Falls back to the legacy flat-key schema
    (``esmc_sae_max`` / ``esmc_mean``, ESM-C single layer) for the archived
    ``*_old`` caches; ``layer``/``backbone`` are ignored there.
    """
    if rep not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {rep!r}; choose from {REPRESENTATIONS}")

    features = cache.get("features")
    if features is not None:
        resolved_layer = resolve_backbone_layer(backbone, layer)
        channel = _REP_TO_V1_CHANNEL[rep]
        key = feature_cache_key(backbone, resolved_layer, channel)
        if key not in features:
            raise KeyError(
                f"feature {key!r} absent from v1 cache; available: {sorted(features)}"
            )
        return features[key]

    # Legacy flat-key fallback (ESM-C single-layer archived caches only).
    if backbone != "esmc":
        raise KeyError(
            f"legacy flat-key cache holds only ESM-C; cannot serve backbone {backbone!r}"
        )
    if rep == "binary":
        return cache["esmc_sae_max"] > SAE_BINARY_THRESHOLD
    if rep == "sae_max":
        return cache["esmc_sae_max"]
    return cache["esmc_mean"]


def protein_feature_rows(
    protein_ids,
    sequences: Dict[str, str],
    cache: Dict,
    rep: str,
    layer: Optional[int] = None,
    backbone: str = DEFAULT_BACKBONE,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """Map protein IDs through their sequences into pooled fingerprint rows.

    Each id is resolved ``id -> sequences[id] -> cache["seq2idx"] -> row``; ids
    whose sequence is absent from the cache are skipped. Returns
    ``(rows, kept_ids)`` with ``rows`` a float32 numpy array ``(n_kept, rep_dim)``
    in ``kept_ids`` order, or ``(None, [])`` when nothing was cached.
    """
    import torch

    seq2idx = cache["seq2idx"]
    matrix = representation_matrix(cache, rep, layer, backbone)
    rows: List[int] = []
    kept: List[str] = []
    for protein_id in protein_ids:
        sequence = sequences.get(protein_id)
        row = seq2idx.get(sequence) if sequence is not None else None
        if row is None:
            continue
        rows.append(int(row))
        kept.append(protein_id)
    if not rows:
        return None, []
    index = torch.as_tensor(rows, dtype=torch.long)
    return matrix.index_select(0, index).float().numpy(), kept


def pair_feature_rows(
    bench,
    cache: Dict,
    rep: str,
    layer: Optional[int] = None,
    backbone: str = DEFAULT_BACKBONE,
):
    """Map benchmark pairs into per-endpoint pooled representation rows.

    Returns ``(A, B, y, kept_idx)`` with A/B float32 torch tensors
    ``(n_kept, rep_dim)``, ``y`` an int64 numpy array, and ``kept_idx`` the
    indices into ``bench.pairs`` whose *both* endpoints were cached (the rest are
    skipped). Returns ``None`` when no pair had both endpoints cached.
    """
    import torch

    seq2idx = cache["seq2idx"]
    matrix = representation_matrix(cache, rep, layer, backbone)
    rows_a: List[int] = []
    rows_b: List[int] = []
    ys: List[int] = []
    kept: List[int] = []
    for i, ((a, b), y) in enumerate(zip(bench.pairs, bench.labels)):
        sa = bench.seqs.get(a)
        sb = bench.seqs.get(b)
        ja = seq2idx.get(sa) if sa is not None else None
        jb = seq2idx.get(sb) if sb is not None else None
        if ja is None or jb is None:
            continue
        rows_a.append(ja)
        rows_b.append(jb)
        ys.append(int(y))
        kept.append(i)
    if not kept:
        return None
    ia = torch.as_tensor(rows_a, dtype=torch.long)
    ib = torch.as_tensor(rows_b, dtype=torch.long)
    A = matrix.index_select(0, ia).float()
    B = matrix.index_select(0, ib).float()
    return A, B, np.asarray(ys, dtype=np.int64), kept


__all__ = [
    "POOLED_CACHE_KEYS",
    "REPRESENTATIONS",
    "load_pooled_cache",
    "pair_feature_rows",
    "protein_feature_rows",
    "rep_dim",
    "representation_matrix",
]
