#!/usr/bin/env python3
"""Slice a per-dataset protein feature cache out of the pooled seq caches.

The pooled seq caches under ``data/sae/seq_caches`` hold the formal ESM-C
(layers 60/80) and ESM-2 (layer 33) features for *every* unique benchmark
sequence, keyed by sequence string. This script builds a dataset-specific
``auditppi_protein_features_v1`` cache by:

  1. loading the dataset's own (protein_id, sequence) manifest -- exactly the
     manifest ``extract_protein_features.py`` would build (shared code, so the
     id/sequence contract is identical);
  2. resolving each manifest sequence to its row in each pooled cache via the
     pooled ``seq2idx``;
  3. ``index_select``-ing every feature channel into a compact per-dataset
     matrix (ESM-C L60/L80 x {dense_mean,dense_max,sae_max,sae_binary} plus
     ESM-2 L33 x the same four channels when an ESM-2 pooled cache is given);
  4. writing the result with the dataset's *real* ``id2idx`` so downstream
     ID-keyed lookups (PRING UniProt ids, etc.) resolve directly.

This is a pure-CPU slice: no GPU, no model load. It replaces the old
``cache_pring_*`` / ``cache_*_esmc_fingerprints`` GPU builders now that the
pooled caches already contain every sequence's features.

Coverage is expected to be 100% -- the pooled manifest was collected from every
benchmark's sequences. A missing sequence is a hard error unless
``--allow-missing`` is passed (then the missing ids are dropped and recorded).

Example (PRING yeast):

    /data/wmzhu/anaconda3/envs/E1/bin/python \
      scripts/prep/slice_dataset_protein_cache.py \
      --input baselines/PRING/data_process/pring_dataset/yeast/yeast_simple.fasta \
      --esmc-pooled data/sae/seq_caches/pooled_esmc_l60_l80_max1022_features.pt \
      --esm2-pooled data/sae/seq_caches/pooled_esm2_l33_max1022_features.pt \
      --output data/sae/protein_caches/pring_yeast_protein_features_max1022.pt
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from src.features.manifest import ProteinManifest, load_protein_manifest


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_pooled(path: Path) -> dict[str, Any]:
    """Load a pooled ``auditppi_protein_features_v1`` cache (mmap when possible)."""
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if "mmap" in inspect.signature(torch.load).parameters:
        kwargs["mmap"] = True
    payload = torch.load(path, **kwargs)
    if payload.get("format") != "auditppi_protein_features_v1":
        raise ValueError(f"{path} is not an auditppi_protein_features_v1 pooled cache")
    return payload


def resolve_rows(
    manifest: ProteinManifest, pooled: dict[str, Any], label: str
) -> tuple[list[int], list[int], list[str]]:
    """Map each manifest sequence to its pooled row via the pooled seq2idx.

    Returns (manifest_positions_kept, pooled_rows, missing_sequences_ids).
    """
    seq2idx = pooled["seq2idx"]
    kept_positions: list[int] = []
    rows: list[int] = []
    missing: list[str] = []
    for position, sequence in enumerate(manifest.sequences):
        row = seq2idx.get(sequence)
        if row is None:
            missing.append(manifest.protein_ids[position])
            continue
        kept_positions.append(position)
        rows.append(int(row))
    print(
        f"[{label}] resolved {len(rows)}/{len(manifest.sequences)} sequences "
        f"(missing={len(missing)})",
        flush=True,
    )
    return kept_positions, rows, missing


def slice_channels(
    pooled: dict[str, Any], rows: torch.Tensor
) -> dict[str, torch.Tensor]:
    """index_select every feature channel of a pooled cache onto the kept rows."""
    out: dict[str, torch.Tensor] = {}
    for name, matrix in pooled["features"].items():
        out[name] = matrix.index_select(0, rows).clone()
    return out


def build_sub_manifest(
    manifest: ProteinManifest, kept_positions: list[int]
) -> ProteinManifest:
    """Restrict a manifest to kept sequence positions, reindexing to 0..k-1.

    ID aliases whose sequence survived are remapped to the new row; aliases to
    dropped sequences are dropped.
    """
    old_to_new = {old: new for new, old in enumerate(kept_positions)}
    protein_ids = [manifest.protein_ids[old] for old in kept_positions]
    sequences = [manifest.sequences[old] for old in kept_positions]
    seq2idx = {seq: i for i, seq in enumerate(sequences)}
    id2idx = {
        pid: old_to_new[old]
        for pid, old in manifest.id2idx.items()
        if old in old_to_new
    }
    return ProteinManifest(
        protein_ids=protein_ids,
        sequences=sequences,
        id2idx=id2idx,
        seq2idx=seq2idx,
        sources=manifest.sources,
    )


def verify_slice(
    features: dict[str, torch.Tensor],
    pooled: dict[str, Any],
    rows: list[int],
    n_check: int = 5,
) -> None:
    """Assert sliced rows are bit-identical to the pooled rows they came from."""
    check_rows = rows[:n_check]
    for name, sliced in features.items():
        source = pooled["features"][name]
        for local_i, pooled_row in enumerate(check_rows):
            a = sliced[local_i]
            b = source[pooled_row]
            if not torch.equal(a, b):
                raise AssertionError(
                    f"slice mismatch for {name} at local row {local_i} "
                    f"(pooled row {pooled_row})"
                )
    print(f"[verify] {len(features)} channels bit-identical on {len(check_rows)} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, action="append", required=True, help="repeat for multiple inputs")
    parser.add_argument("--input-format", choices=["auto", "fasta", "table"], default="auto")
    parser.add_argument("--sequence-cols", default="sequence", help="comma-separated; e.g. query,text")
    parser.add_argument("--id-cols", default="", help="optional comma-separated ID columns")
    parser.add_argument("--esmc-pooled", type=Path, default=None, help="pooled ESM-C L60/L80 feature cache")
    parser.add_argument("--esm2-pooled", type=Path, default=None, help="pooled ESM-2 L33 feature cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true", help="drop sequences absent from a pooled cache instead of failing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.esmc_pooled is None and args.esm2_pooled is None:
        parser.error("at least one of --esmc-pooled / --esm2-pooled is required")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite to replace it")

    sequence_cols = comma_list(args.sequence_cols)
    id_cols = comma_list(args.id_cols) or None
    manifest = load_protein_manifest(
        args.input,
        input_format=args.input_format,
        sequence_cols=sequence_cols,
        id_cols=id_cols,
    )
    print(
        f"[manifest] unique_sequences={len(manifest)} id_aliases={len(manifest.id2idx)} "
        f"inputs={len(args.input)}",
        flush=True,
    )

    # Determine which manifest sequences survive across all requested pooled caches.
    pooled_caches: list[tuple[str, Path, dict[str, Any]]] = []
    if args.esmc_pooled is not None:
        pooled_caches.append(("esmc", args.esmc_pooled, load_pooled(args.esmc_pooled)))
    if args.esm2_pooled is not None:
        pooled_caches.append(("esm2", args.esm2_pooled, load_pooled(args.esm2_pooled)))

    # Kept = sequences present in EVERY requested pooled cache (channels must align).
    kept_mask = [True] * len(manifest.sequences)
    per_cache_missing: dict[str, list[str]] = {}
    for label, _path, pooled in pooled_caches:
        seq2idx = pooled["seq2idx"]
        missing: list[str] = []
        for position, sequence in enumerate(manifest.sequences):
            if sequence not in seq2idx:
                kept_mask[position] = False
                missing.append(manifest.protein_ids[position])
        per_cache_missing[label] = missing
        print(f"[{label}] missing sequences: {len(missing)}", flush=True)

    kept_positions = [i for i, keep in enumerate(kept_mask) if keep]
    n_missing = len(manifest.sequences) - len(kept_positions)
    if n_missing and not args.allow_missing:
        examples = [
            manifest.protein_ids[i] for i in range(len(kept_mask)) if not kept_mask[i]
        ][:20]
        raise RuntimeError(
            f"{n_missing} sequences absent from a pooled cache; pass --allow-missing "
            f"to drop them. Examples: {examples}"
        )

    sub_manifest = build_sub_manifest(manifest, kept_positions)

    # Slice every channel from every pooled cache onto the kept rows.
    features: dict[str, torch.Tensor] = {}
    extractor_meta: dict[str, Any] = {"sliced_from": {}}
    for label, path, pooled in pooled_caches:
        seq2idx = pooled["seq2idx"]
        rows = [int(seq2idx[seq]) for seq in sub_manifest.sequences]
        row_index = torch.as_tensor(rows, dtype=torch.long)
        channels = slice_channels(pooled, row_index)
        verify_slice(channels, pooled, rows)
        features.update(channels)
        extractor_meta["sliced_from"][label] = {
            "pooled_cache": str(path),
            "extractor": pooled.get("meta", {}).get("extractor"),
            "channels": sorted(channels),
        }

    # Write with the shared v1 writer so the on-disk format has one definition.
    from src.features.extractors import save_feature_cache

    save_feature_cache(
        args.output,
        manifest=sub_manifest,
        features=features,
        extractor_meta=extractor_meta,
        overwrite=args.overwrite,
    )
    coverage = {
        "n_manifest_sequences": len(manifest.sequences),
        "n_kept_sequences": len(kept_positions),
        "n_dropped_sequences": n_missing,
        "per_cache_missing_counts": {k: len(v) for k, v in per_cache_missing.items()},
    }
    print(f"[coverage] {json.dumps(coverage)}", flush=True)
    print(f"[saved] {args.output} channels={sorted(features)}", flush=True)


if __name__ == "__main__":
    main()
