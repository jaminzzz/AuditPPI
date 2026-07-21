#!/usr/bin/env python3
"""Extract the formal per-protein ESM-C or ESM-2 feature set.

The script is dataset-neutral. Pass one or more FASTA/CSV/TSV inputs; identical
sequences across all inputs are encoded once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from conf.model import (
    ESM2_MAX_RESIDUES,
    ESMC_MAX_RESIDUES,
    ESMC_MAX_RESIDUES_COMPAT,
)
from conf.paths import (
    ESM2_650M_MODEL,
    ESM2_SAE_CKPT,
    ESMC_MODEL,
    ESMC_SAE,
    INTERPLM_ROOT,
)
from src.features.extractors import (
    extract_esmc_features,
    extract_esm2_features,
    save_feature_cache,
)
from src.features.manifest import load_protein_manifest
from src.runtime.device import pick_free_gpu


def pick_gpu() -> str:
    return str(pick_free_gpu())


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["esmc", "esm2"], required=True)
    parser.add_argument("--input", type=Path, action="append", required=True, help="repeat for multiple inputs")
    parser.add_argument("--input-format", choices=["auto", "fasta", "table"], default="auto")
    parser.add_argument("--sequence-cols", default="sequence", help="comma-separated; e.g. query,text")
    parser.add_argument("--id-cols", default="", help="optional comma-separated ID columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="encode only the first N unique sequences")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--max-residues",
        type=int,
        default=None,
        help=(
            "residue truncation cap (BOS/EOS extra). "
            f"Default: ESM-C={ESMC_MAX_RESIDUES} (use {ESMC_MAX_RESIDUES_COMPAT} for "
            f"legacy-aligned ESM-C), ESM-2={ESM2_MAX_RESIDUES}."
        ),
    )
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--sae-token-chunk", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--esmc-model", type=Path, default=ESMC_MODEL)
    parser.add_argument("--esmc-sae", type=Path, default=ESMC_SAE)
    parser.add_argument("--layers", default="60,80", help="ESM-C layers, comma-separated")

    parser.add_argument("--esm2-model", default=ESM2_650M_MODEL)
    parser.add_argument("--interplm-root", type=Path, default=INTERPLM_ROOT)
    parser.add_argument("--esm2-sae", type=Path, default=ESM2_SAE_CKPT)
    parser.add_argument(
        "--interplm-normalize-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.max_residues is None:
        args.max_residues = (
            ESMC_MAX_RESIDUES if args.backbone == "esmc" else ESM2_MAX_RESIDUES
        )

    if args.device.startswith("cuda"):
        device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"[device] CUDA_VISIBLE_DEVICES={device_id}", flush=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sequence_cols = comma_list(args.sequence_cols)
    id_cols = comma_list(args.id_cols) or None
    manifest = load_protein_manifest(
        args.input,
        input_format=args.input_format,
        sequence_cols=sequence_cols,
        id_cols=id_cols,
    )
    if args.limit:
        keep = min(args.limit, len(manifest))
        manifest.protein_ids = manifest.protein_ids[:keep]
        manifest.sequences = manifest.sequences[:keep]
        manifest.seq2idx = {sequence: i for i, sequence in enumerate(manifest.sequences)}
        manifest.id2idx = {pid: idx for pid, idx in manifest.id2idx.items() if idx < keep}
    print(
        f"[manifest] unique_sequences={len(manifest)} id_aliases={len(manifest.id2idx)} "
        f"inputs={len(args.input)} max_residues={args.max_residues} backbone={args.backbone}",
        flush=True,
    )

    if args.backbone == "esmc":
        layers = [int(value) for value in comma_list(args.layers)]
        features, meta = extract_esmc_features(
            manifest,
            model_path=args.esmc_model,
            sae_path=args.esmc_sae,
            layers=layers,
            max_residues=args.max_residues,
            token_budget=args.token_budget or 3072,
            sae_token_chunk=args.sae_token_chunk,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        features, meta = extract_esm2_features(
            manifest,
            model_name=args.esm2_model,
            interplm_root=args.interplm_root,
            sae_checkpoint=args.esm2_sae,
            max_residues=args.max_residues,
            token_budget=args.token_budget or 8192,
            sae_token_chunk=args.sae_token_chunk,
            normalize_sae_features=args.interplm_normalize_features,
            device=args.device,
            dtype=args.dtype,
        )

    save_feature_cache(
        args.output,
        manifest=manifest,
        features=features,
        extractor_meta=meta,
        overwrite=args.overwrite,
    )
    print(f"[saved] {args.output} features={list(features)}", flush=True)


if __name__ == "__main__":
    main()
