#!/usr/bin/env python3
"""Build the global eSIG-Net 573-D physicochemical fingerprint cache.

eSIG-Net (:func:`src.features.h5file.extract_573_features`) is a pure per-sequence
function -- amino-acid composition + conjoint-triad + physicochemical
auto-correlation, no ML model, no backbone. Because it depends only on the
sequence string it can be computed ONCE for every unique benchmark sequence and
reused by every downstream experiment, exactly like the pooled ESM caches are
sliced per-dataset.

Unlike the ESM channels (which cap at 1022 residues), eSIG runs on the FULL
sequence -- that is its native behaviour, so this cache carries no ``max{N}``
length tag.

Input is the pooled sequence manifest parquet
(:data:`conf.paths.POOLED_SEQUENCE_MANIFEST`), whose ``sequence`` column already
holds every unique normalized benchmark sequence -- so this builder never loads
the 16 GB pooled ESM cache. The manifest normalization
(:func:`src.data.sequences.normalize_sequence`) is the same spelling the
per-dataset protein caches key their ``seq2idx`` on, so a consumer resolves a
row as ``cache_sequence -> esig seq2idx -> row`` with no re-normalization.

Output (:data:`conf.paths.POOLED_ESIG_CACHE`)::

    {
      "format": "auditppi_esig_features_v1",
      "seq2idx": {normalized_sequence: row},
      "esig_573": float32 tensor (n_unique, 573),
      "meta": {...},
    }

Example::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/prep/build_esig_seq_cache.py            # 16 workers, all sequences
    $PY scripts/prep/build_esig_seq_cache.py --workers 8 --limit 500   # smoke test
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conf.model import ESIG_DIM
from conf.paths import POOLED_ESIG_CACHE, POOLED_SEQUENCE_MANIFEST
from src.data.sequences import normalize_sequence
from src.features.h5file import PROPERTY_TABLE_VERSION, extract_573_features


def _row_features(sequence: str) -> np.ndarray:
    """573-D eSIG vector for one already-normalized sequence (worker entry point)."""
    return extract_573_features(sequence).astype(np.float32, copy=False)


def load_manifest_sequences(manifest_path: Path) -> list[str]:
    """Unique normalized sequences from the pooled manifest, in manifest order.

    The manifest ``sequence`` column is already normalized, but we re-normalize
    and de-duplicate defensively so the cache key is guaranteed to match the
    :func:`normalize_sequence` spelling every consumer looks up with.
    """
    frame = pd.read_parquet(manifest_path, columns=["sequence"])
    seen: dict[str, None] = {}
    for raw in frame["sequence"].tolist():
        seq = normalize_sequence(raw)
        if seq and seq not in seen:
            seen[seq] = None
    return list(seen)


def build_esig_cache(
    manifest_path: Path,
    output_path: Path,
    *,
    workers: int,
    limit: int | None,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_path}; pass --overwrite to replace it")

    sequences = load_manifest_sequences(manifest_path)
    if limit is not None:
        sequences = sequences[:limit]
    n = len(sequences)
    if n == 0:
        raise ValueError(f"no sequences found in {manifest_path}")
    print(f"[esig] {n} unique sequences from {manifest_path.name}", flush=True)

    matrix = np.empty((n, ESIG_DIM), dtype=np.float32)
    start = time.time()
    if workers <= 1:
        for i, seq in enumerate(sequences):
            matrix[i] = _row_features(seq)
            if (i + 1) % 10000 == 0:
                print(f"[esig] {i + 1}/{n} ({time.time() - start:.0f}s)", flush=True)
    else:
        # eSIG is CPU-bound pure Python (the auto-correlation is an O(L*lag) loop),
        # so process-level parallelism gives a near-linear speedup. chunksize keeps
        # the IPC overhead low across 137k short tasks.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = 0
            for i, vec in enumerate(pool.map(_row_features, sequences, chunksize=64)):
                matrix[i] = vec
                done += 1
                if done % 10000 == 0:
                    print(f"[esig] {done}/{n} ({time.time() - start:.0f}s)", flush=True)
    print(f"[esig] computed {n} vectors in {time.time() - start:.0f}s", flush=True)

    seq2idx = {seq: row for row, seq in enumerate(sequences)}
    payload = {
        "format": "auditppi_esig_features_v1",
        "seq2idx": seq2idx,
        "esig_573": torch.from_numpy(matrix),
        "meta": {
            "feature_dim": ESIG_DIM,
            "property_table_version": PROPERTY_TABLE_VERSION,
            "n_unique_sequences": n,
            "source_manifest": str(manifest_path),
            "full_sequence": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[esig] wrote {output_path} ({matrix.shape})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=POOLED_SEQUENCE_MANIFEST)
    p.add_argument("--output", type=Path, default=POOLED_ESIG_CACHE)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=None, help="cap sequences (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    build_esig_cache(
        args.manifest, args.output,
        workers=args.workers, limit=args.limit, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
