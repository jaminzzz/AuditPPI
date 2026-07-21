#!/usr/bin/env python3
"""Build the PIC human essentiality v1 protein cache (pure-CPU slice).

PIC (Protein Importance Calculator) predicts human essential proteins from
single sequences. Its data ships a legacy-numpy pickle keyed by Ensembl protein
ids (ENSP...) with a per-dataset binary label column. Every PIC human sequence
already lives in the pooled seq caches under ``data/sae/seq_caches`` (collected
from every benchmark's unique sequences), so this step no longer runs the model:
it *slices* a dataset-specific ``auditppi_protein_features_v1`` cache out of the
pooled caches, exactly like ``scripts/prep/slice_dataset_protein_cache.py``.

  1. read PIC human via ``src.data.proteins.load_pic`` (the single id/seq/label
     contract, incl. the numpy-core shim for the old pickle);
  2. build a de-duplicated manifest (identical sequences share one feature row,
     every ENSP id kept as an alias) via the shared manifest builder, so the
     id/sequence contract is identical to the FASTA/CSV slicer;
  3. resolve each unique sequence to its row in each pooled cache and
     ``index_select`` every ESM-C L60/L80 and ESM-2 L33 channel onto it;
  4. write the result with the real ENSP ``id2idx`` so the essentiality
     classifier resolves labels (from the PIC pickle) row-by-row.

Coverage is expected to be 100% -- the pooled manifest contains every PIC human
sequence. A missing sequence is a hard error unless ``--allow-missing``.

Run in the unified E1 conda environment:

  /data/wmzhu/anaconda3/envs/E1/bin/python \
      scripts/cache/cache_pic_human_esmc_sae.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conf.paths import (
    PIC_HUMAN_SAE_CACHE,
    POOLED_ESM2_SEQ_CACHE,
    POOLED_ESMC_SEQ_CACHE,
)
from src.data.proteins import load_pic
from src.features.extractors import save_feature_cache
from src.features.manifest import ProteinManifest, _ManifestBuilder
from scripts.prep.slice_dataset_protein_cache import (
    build_sub_manifest,
    load_pooled,
    slice_channels,
    verify_slice,
)


def build_pic_manifest(label_col: str) -> ProteinManifest:
    """Manifest of PIC human proteins (dedup by sequence, ENSP ids as aliases)."""
    dataset = load_pic("human", label_col=label_col)
    builder = _ManifestBuilder()
    for identifier in dataset.ids:
        builder.add(identifier, dataset.seqs[identifier])
    return ProteinManifest(
        protein_ids=builder.protein_ids,
        sequences=builder.sequences,
        id2idx=builder.id2idx,
        seq2idx=builder.seq2idx,
        sources=[f"pic:{label_col}"],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--label-col", type=str, default="human")
    p.add_argument("--esmc-pooled", type=Path, default=POOLED_ESMC_SEQ_CACHE)
    p.add_argument("--esm2-pooled", type=Path, default=POOLED_ESM2_SEQ_CACHE)
    p.add_argument("--out", type=Path, default=PIC_HUMAN_SAE_CACHE)
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="drop sequences absent from a pooled cache instead of failing",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists: {args.out}; pass --overwrite to replace it"
        )

    manifest = build_pic_manifest(args.label_col)
    print(
        f"[manifest] pic:{args.label_col} unique_sequences={len(manifest)} "
        f"id_aliases={len(manifest.id2idx)}",
        flush=True,
    )

    pooled_caches: list[tuple[str, Path, dict]] = []
    if args.esmc_pooled is not None:
        pooled_caches.append(("esmc", args.esmc_pooled, load_pooled(args.esmc_pooled)))
    if args.esm2_pooled is not None:
        pooled_caches.append(("esm2", args.esm2_pooled, load_pooled(args.esm2_pooled)))
    if not pooled_caches:
        raise SystemExit("at least one of --esmc-pooled / --esm2-pooled is required")

    # Kept = sequences present in EVERY requested pooled cache (channels align).
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

    features: dict[str, torch.Tensor] = {}
    extractor_meta: dict = {"sliced_from": {}, "dataset": f"PIC {args.label_col} essentiality"}
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

    save_feature_cache(
        args.out,
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
    print(f"[saved] {args.out} channels={sorted(features)}", flush=True)
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
