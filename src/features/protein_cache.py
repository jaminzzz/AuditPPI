"""Shared accessor for pooled per-protein feature caches.

Both the fingerprint baseline (``src.ppi_fingerprint``) and the participation
oracle scripts previously each re-implemented the representation -> matrix
switch and the id/sequence -> row assembly. That logic now lives here once; the
id-keyed pooled assembly in ``src.features.pooled_assembly`` reuses the
representation switch too.

Import-time torch-free
----------------------
``torch`` is imported lazily *inside* the functions that need it. Importing this
module (and hence ``src.ppi_fingerprint.config``, which re-exports
``REPRESENTATIONS``) never pulls torch into a fresh process -- a property the
ppi_fingerprint package test pins.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from conf.model import (
    BACKBONE_DENSE_DIM,
    BACKBONE_SAE_DIM,
    DEFAULT_BACKBONE,
    ESIG_DIM,
    REPRESENTATIONS,
    SAE_BINARY_THRESHOLD,
    feature_cache_key,
    resolve_backbone_layer,
)

# eSIG-Net 573-D is a backbone/layer-agnostic pure-sequence fingerprint served
# from one global cache (see ``_esig_matrix_for``), so it is not a member of the
# (backbone, layer)-coupled ``REPRESENTATIONS`` tuple. Accessors accept it as an
# extra representation name alongside the SAE views.
ESIG_REP = "esig"
_ACCEPTED_REPS = (*REPRESENTATIONS, ESIG_REP)

_ESIG_FORMAT = "auditppi_esig_features_v1"
_ESIG_CACHE_KEY = "_esig_matrix_573"  # memo slot on a protein cache dict
_ESIG_GLOBAL: Optional[Dict] = None   # process-wide global eSIG cache payload


def rep_dim(rep: str, backbone: str = DEFAULT_BACKBONE) -> int:
    """Column count of a representation's per-protein matrix for a backbone."""
    if rep not in _ACCEPTED_REPS:
        raise ValueError(f"unknown representation {rep!r}; choose from {_ACCEPTED_REPS}")
    if rep == ESIG_REP:
        return ESIG_DIM
    return BACKBONE_DENSE_DIM[backbone] if rep == "esmc_mean" else BACKBONE_SAE_DIM[backbone]


def _load_global_esig() -> Dict:
    """Load (once per process) the global eSIG-Net 573-D sequence cache."""
    global _ESIG_GLOBAL
    if _ESIG_GLOBAL is None:
        import torch

        from conf.paths import POOLED_ESIG_CACHE

        if not POOLED_ESIG_CACHE.exists():
            raise FileNotFoundError(
                f"global eSIG cache absent: {POOLED_ESIG_CACHE}; build it with "
                f"scripts/prep/build_esig_seq_cache.py"
            )
        payload = torch.load(POOLED_ESIG_CACHE, map_location="cpu", weights_only=False)
        if payload.get("format") != _ESIG_FORMAT:
            raise ValueError(f"{POOLED_ESIG_CACHE} is not an {_ESIG_FORMAT} cache")
        _ESIG_GLOBAL = payload
    return _ESIG_GLOBAL


def _esig_matrix_for(cache: Dict):
    """Per-cache eSIG-Net 573-D matrix, row-aligned to the cache's SAE channels.

    eSIG is backbone/layer-agnostic: one global pure-sequence cache
    (``POOLED_ESIG_CACHE``) holds every unique sequence's 573-D vector. Each
    protein cache's rows are produced by mapping its own ``seq2idx`` (the same
    seq→row map every SAE channel is indexed by) through the global cache, so the
    result lines up row-for-row with ``features[...]`` and can be gathered by the
    identical ``index_select`` path. Memoized on the cache dict so the gather is
    paid once per process.
    """
    cached = cache.get(_ESIG_CACHE_KEY)
    if cached is not None:
        return cached
    import torch

    esig = _load_global_esig()
    g_seq2idx = esig["seq2idx"]
    g_matrix = esig["esig_573"]
    local_seq2idx = cache["seq2idx"]
    rows = [0] * len(local_seq2idx)
    for seq, local_row in local_seq2idx.items():
        g_row = g_seq2idx.get(seq)
        if g_row is None:
            raise KeyError(
                "cache sequence absent from global eSIG cache; rebuild it with "
                "scripts/prep/build_esig_seq_cache.py to cover all sequences"
            )
        rows[int(local_row)] = int(g_row)
    out = g_matrix.index_select(0, torch.as_tensor(rows, dtype=torch.long)).float()
    cache[_ESIG_CACHE_KEY] = out
    return out


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
    if rep not in _ACCEPTED_REPS:
        raise ValueError(f"unknown representation {rep!r}; choose from {_ACCEPTED_REPS}")

    # eSIG is backbone/layer-agnostic -- served from the global sequence cache and
    # gathered to this cache's row order, ignoring backbone/layer entirely.
    if rep == ESIG_REP:
        return _esig_matrix_for(cache)

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


def pair_feature_row_indices(
    bench,
    cache: Dict,
    rep: str,
    layer: Optional[int] = None,
    backbone: str = DEFAULT_BACKBONE,
):
    """Resolve benchmark pairs to shared-matrix row indices (no materialization).

    Index-only twin of :func:`pair_feature_rows`. Returns
    ``(matrix, rows_a, rows_b, y, kept_idx)`` where ``matrix`` is the shared
    per-protein representation tensor (native dtype -- ``binary`` stays bool,
    ``sae_max``/``esmc_mean`` stay float16/float32 as stored), ``rows_a``/``rows_b``
    are int64 numpy arrays of row indices into ``matrix`` for each kept pair, ``y``
    is an int64 numpy label array, and ``kept_idx`` the indices into ``bench.pairs``
    whose *both* endpoints were cached. Returns ``None`` when no pair matched.

    Unlike :func:`pair_feature_rows` this never gathers the per-pair endpoint
    graph (which force-casts to float32 -- tens of GB for large pair counts). The
    caller decides when/whether to gather rows, so memory stays bounded by the
    unique-protein matrix rather than the pair count.
    """
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
    return (
        matrix,
        np.asarray(rows_a, dtype=np.int64),
        np.asarray(rows_b, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
        kept,
    )


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

    resolved = pair_feature_row_indices(bench, cache, rep, layer, backbone)
    if resolved is None:
        return None
    matrix, rows_a, rows_b, ys, kept = resolved
    ia = torch.as_tensor(rows_a, dtype=torch.long)
    ib = torch.as_tensor(rows_b, dtype=torch.long)
    A = matrix.index_select(0, ia).float()
    B = matrix.index_select(0, ib).float()
    return A, B, ys, kept


__all__ = [
    "REPRESENTATIONS",
    "pair_feature_rows",
    "pair_feature_row_indices",
    "protein_feature_rows",
    "rep_dim",
    "representation_matrix",
]
