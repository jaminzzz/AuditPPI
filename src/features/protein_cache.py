"""Shared accessor for pooled per-protein feature caches.

A *pooled cache* (produced by ``scripts/cache/cache_esmc_fingerprints.py`` and
friends) is a ``torch.save`` payload with at least::

    seq2idx       dict[str, int]   sequence string -> row index
    esmc_sae_max  Tensor [N, ESMC_SAE_DIM]   pooled SAE max  (the SAE channel)
    esmc_sae_mean Tensor [N, ESMC_SAE_DIM]   pooled SAE mean (optional)
    esmc_mean     Tensor [N, ESMC_DIM]       pooled dense ESM-C mean

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

from conf.model import ESMC_DIM, ESMC_SAE_DIM, REPRESENTATIONS, SAE_BINARY_THRESHOLD

# Keys picked out of a pooled cache payload (the rest of the file is ignored).
POOLED_CACHE_KEYS = ("seq2idx", "esmc_mean", "esmc_sae_max", "esmc_sae_mean")


def load_pooled_cache(path: Path, *, keys: Tuple[str, ...] = POOLED_CACHE_KEYS) -> Dict:
    """Read a per-protein pooled cache -> ``{seq2idx, esmc_mean, esmc_sae_max, esmc_sae_mean}``."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {key: payload[key] for key in keys}


def rep_dim(rep: str) -> int:
    """Column count of a representation's per-protein matrix."""
    if rep not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {rep!r}; choose from {REPRESENTATIONS}")
    return ESMC_DIM if rep == "esmc_mean" else ESMC_SAE_DIM


def representation_matrix(cache: Dict, rep: str):
    """The per-protein matrix for a representation (torch tensor; ``binary`` is bool).

    Handles the three baseline ``REPRESENTATIONS`` plus ``sae_mean`` -- the pooled
    SAE-mean channel used by the participation feature kinds. ``sae_mean`` is
    intentionally *not* in ``REPRESENTATIONS`` (that tuple is the fingerprint
    baseline's rep contract); it is accepted here only as a decodable channel.
    """
    if rep == "binary":
        return cache["esmc_sae_max"] > SAE_BINARY_THRESHOLD
    if rep == "sae_max":
        return cache["esmc_sae_max"]
    if rep == "sae_mean":
        return cache["esmc_sae_mean"]
    if rep == "esmc_mean":
        return cache["esmc_mean"]
    raise ValueError(f"unknown representation {rep!r}; choose from {(*REPRESENTATIONS, 'sae_mean')}")


def protein_feature_rows(
    protein_ids,
    sequences: Dict[str, str],
    cache: Dict,
    rep: str,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """Map protein IDs through their sequences into pooled fingerprint rows.

    Each id is resolved ``id -> sequences[id] -> cache["seq2idx"] -> row``; ids
    whose sequence is absent from the cache are skipped. Returns
    ``(rows, kept_ids)`` with ``rows`` a float32 numpy array ``(n_kept, rep_dim)``
    in ``kept_ids`` order, or ``(None, [])`` when nothing was cached.
    """
    import torch

    seq2idx = cache["seq2idx"]
    matrix = representation_matrix(cache, rep)
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


def pair_feature_rows(bench, cache: Dict, rep: str):
    """Map benchmark pairs into per-endpoint pooled representation rows.

    Returns ``(A, B, y, kept_idx)`` with A/B float32 torch tensors
    ``(n_kept, rep_dim)``, ``y`` an int64 numpy array, and ``kept_idx`` the
    indices into ``bench.pairs`` whose *both* endpoints were cached (the rest are
    skipped). Returns ``None`` when no pair had both endpoints cached.
    """
    import torch

    seq2idx = cache["seq2idx"]
    matrix = representation_matrix(cache, rep)
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
