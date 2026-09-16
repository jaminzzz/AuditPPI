#!/usr/bin/env python3
"""Build a lightweight pair-index cache from a pair CSV + a protein feature cache.

A *pair-index cache* (``auditppi_pair_index_v1``) stores only the per-pair
endpoint *row indices* into an ``auditppi_protein_features_v1`` protein cache,
plus the labels and provenance -- never any materialized feature vectors. This
is the pair-scale analogue of the protein slicer: one small file replaces the
old per-rep ``{split}_embeddings.pt`` dumps (which duplicated every shared
endpoint's 16384-d vector across every pair it appeared in).

Because the indices point into a protein cache that already holds every
``(backbone, layer, rep)`` channel, *one* index cache serves every channel and
every pair mode. A consumer loads it, picks a channel matrix via
:func:`~src.features.protein_cache.representation_matrix`, then materializes
endpoints with ``matrix.index_select(0, rows_a/rows_b)``.

Endpoint resolution mirrors :func:`src.features.pairs.load_pair_indices`:
``--pair-key auto`` tries the protein cache ``id2idx`` first (PRING UniProt
ids), then falls back to ``seq2idx`` (C3 / cross-species raw sequences).
``kept_pair_indices`` preserves the original CSV row order, so downstream
row-aligned tables (e.g. the C3 pair-id alignment parquet) line up.

Examples
--------
C3 (raw sequences in query/text; resolves via seq2idx)::

    /data/wmzhu/anaconda3/envs/E1/bin/python \
      scripts/prep/build_pair_index_cache.py \
      --cache data/sae/protein_caches/c3_protein_features_max1022.pt \
      --pairs data/raw/rapppid_c3/c3.train.csv \
      --a-col query --b-col text --label-col label \
      --output data/sae/pair_caches/c3/train_pairs.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.features.pairs import (
    load_pair_indices,
    load_protein_feature_cache,
    save_pair_index_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, required=True, help="protein feature cache (auditppi_protein_features_v1)")
    parser.add_argument("--pairs", type=Path, required=True, help="pair CSV/TSV with a/b/label columns")
    parser.add_argument("--a-col", default="query")
    parser.add_argument("--b-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument(
        "--pair-key",
        choices=["auto", "id", "sequence"],
        default="auto",
        help="auto: id2idx then seq2idx; id: id2idx only; sequence: seq2idx only",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="drop pairs whose endpoint is absent from the cache instead of failing",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite to replace it")

    cache = load_protein_feature_cache(args.cache)
    rows_a, rows_b, labels, kept, total = load_pair_indices(
        args.pairs,
        cache=cache,
        a_col=args.a_col,
        b_col=args.b_col,
        label_col=args.label_col,
        pair_key=args.pair_key,
        skip_missing=args.skip_missing,
    )
    payload = save_pair_index_cache(
        args.output,
        rows_a=rows_a,
        rows_b=rows_b,
        labels=labels,
        kept_pair_indices=kept,
        n_total=total,
        source_cache=str(args.cache),
        source_pairs=str(args.pairs),
        pair_key=args.pair_key,
        overwrite=args.overwrite,
    )
    pos = int(labels.sum())
    n_kept = payload["n_kept"]
    print(
        f"[saved] {args.output} kept={n_kept}/{total} "
        f"pos={pos} neg={n_kept - pos} "
        f"pos_rate={pos / n_kept:.4f} pair_key={args.pair_key}",
        flush=True,
    )


if __name__ == "__main__":
    main()
