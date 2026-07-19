#!/usr/bin/env python3
"""Extract DeepNano mean/min/max features from a frozen PLM final layer.

This baseline-only implementation supports ESM-2-650M layer 33 and ESM-C-6B
layer 80. The historical layer-60 script remains unchanged at
``scripts/cache/cache_deepnano_embeddings.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from conf.paths import DEEPNANO_DIR, ESM2_650M_MODEL, ESMC_MODEL
from src.features.manifest import load_protein_manifest
from src.runtime.device import pick_free_gpu


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def pick_gpu() -> str:
    return str(pick_free_gpu())


def make_batches(sequences: list[str], max_residues: int, token_budget: int) -> list[list[int]]:
    order = sorted(range(len(sequences)), key=lambda i: min(len(sequences[i]), max_residues))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        length = min(len(sequences[idx]), max_residues) + 2
        proposed = max(current_max, length)
        if current and proposed * (len(current) + 1) > token_budget:
            batches.append(current)
            current, current_max = [], 0
        current.append(idx)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def residue_mask(attention_mask):
    import torch

    positions = attention_mask.bool().nonzero(as_tuple=True)[0]
    if positions.numel() <= 2:
        raise ValueError("protein has no residue tokens after removing BOS/EOS")
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    mask[positions[1:-1]] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["esm2", "esmc"], required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--input-format", choices=["auto", "fasta", "table"], default="auto")
    parser.add_argument("--sequence-cols", default="sequence")
    parser.add_argument("--id-cols", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default="", help="override the local model path/name")
    parser.add_argument("--max-residues", type=int, default=800, help="DeepNano uses 800 by default")
    parser.add_argument("--token-budget", type=int, default=3072)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"[device] CUDA_VISIBLE_DEVICES={device_id}", flush=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModel, AutoTokenizer, EsmModel

    from src.features.extractors import save_feature_cache

    manifest = load_protein_manifest(
        args.input,
        input_format=args.input_format,
        sequence_cols=comma_list(args.sequence_cols),
        id_cols=comma_list(args.id_cols) or None,
    )
    if args.limit:
        keep = min(args.limit, len(manifest))
        manifest.protein_ids = manifest.protein_ids[:keep]
        manifest.sequences = manifest.sequences[:keep]
        manifest.seq2idx = {seq: i for i, seq in enumerate(manifest.sequences)}
        manifest.id2idx = {pid: idx for pid, idx in manifest.id2idx.items() if idx < keep}

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.backbone == "esmc":
        model_name = args.model or str(ESMC_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(args.device).eval()
        layer = int(model.config.n_layers)
        hidden_dim = int(model.config.d_model)
        default_output = DEEPNANO_DIR / "deepnano_esmc_last" / "protein_features.pt"
    else:
        model_name = args.model or ESM2_650M_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(args.device).eval()
        layer = int(model.config.num_hidden_layers)
        hidden_dim = int(model.config.hidden_size)
        default_output = DEEPNANO_DIR / "deepnano_esm2_last" / "protein_features.pt"

    prefix = f"deepnano_{args.backbone}_l{layer}_"
    features = {
        prefix + "mean": torch.empty((len(manifest), hidden_dim), dtype=torch.float16),
        prefix + "min": torch.empty((len(manifest), hidden_dim), dtype=torch.float16),
        prefix + "max": torch.empty((len(manifest), hidden_dim), dtype=torch.float16),
    }
    batches = make_batches(manifest.sequences, args.max_residues, args.token_budget)
    device_type = torch.device(args.device).type
    use_amp = device_type == "cuda" and args.dtype != "fp32"
    start_time = time.time()
    done = 0
    with torch.inference_mode():
        for batch_no, indices in enumerate(batches, start=1):
            sequences = [manifest.sequences[i][: args.max_residues] for i in indices]
            encoded = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_residues + 2,
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            with torch.autocast(device_type=device_type, dtype=torch_dtype, enabled=use_amp):
                hidden = model(**encoded).last_hidden_state
            for batch_row, output_row in enumerate(indices):
                mask = residue_mask(encoded["attention_mask"][batch_row])
                residue = hidden[batch_row][mask].float()
                features[prefix + "mean"][output_row].copy_(residue.mean(0).half().cpu())
                features[prefix + "min"][output_row].copy_(residue.min(0).values.half().cpu())
                features[prefix + "max"][output_row].copy_(residue.max(0).values.half().cpu())
            done += len(indices)
            if batch_no == len(batches) or batch_no % 20 == 0:
                rate = done / max(time.time() - start_time, 1e-6)
                print(f"[DeepNano] {done}/{len(manifest)} proteins {rate:.2f}/s", flush=True)

    output = args.output or default_output
    save_feature_cache(
        output,
        manifest=manifest,
        features=features,
        extractor_meta={
            "baseline": "DeepNano",
            "backbone": args.backbone,
            "model": model_name,
            "layer": layer,
            "pooling": ["mean", "min", "max"],
            "max_residues": args.max_residues,
            "dtype": args.dtype,
        },
        overwrite=args.overwrite,
    )
    print(f"[saved] {output} features={list(features)}", flush=True)


if __name__ == "__main__":
    main()
