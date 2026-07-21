#!/usr/bin/env python3
"""Extract pair-conditioned MINT endpoint embeddings.

Each protein pair is jointly encoded with chain IDs. The final layer is then
mean-pooled separately over chain A and chain B, matching MINT's GeneralPPI
embedding wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

from conf.paths import BASELINES
from src.features.baseline_io import (
    count_pair_rows,
    iter_pair_rows,
    save_baseline_pair_cache,
    truncate_pair_balanced,
)
from src.runtime import ensure_on_sys_path
from src.runtime.device import pick_free_gpu


def pick_gpu() -> str:
    return str(pick_free_gpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--a-col", default="query")
    parser.add_argument("--b-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--mint-root", type=Path, default=BASELINES / "mint")
    parser.add_argument("--checkpoint", type=Path, default=BASELINES / "mint" / "mint.ckpt")
    parser.add_argument(
        "--config",
        type=Path,
        default=BASELINES / "mint" / "data" / "esm2_t33_650M_UR50D.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # MINT is a vendored upstream tree under baselines/, not an installed package.
    ensure_on_sys_path(args.mint_root)

    import torch
    from mint.data import Alphabet
    from mint.model.esm2 import ESM2

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[device] CUDA requested but unavailable; falling back to cpu", flush=True)
            args.device = "cpu"
        else:
            device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            print(f"[device] CUDA_VISIBLE_DEVICES={device_id}", flush=True)

    config = json.loads(args.config.read_text())
    layer = int(config["encoder_layers"])
    embed_dim = int(config["encoder_embed_dim"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = OrderedDict(
        (key.removeprefix("model."), value) for key, value in checkpoint["state_dict"].items()
    )
    with torch.device("meta"):
        model = ESM2(
            num_layers=layer,
            embed_dim=embed_dim,
            attention_heads=int(config["encoder_attention_heads"]),
            token_dropout=bool(config["token_dropout"]),
            use_multimer=True,
        )
    model.load_state_dict(state, strict=True, assign=True)
    del checkpoint, state
    model = model.to(args.device).eval()
    alphabet = Alphabet.from_architecture("ESM-1b")

    n = count_pair_rows(args.pairs, args.limit)
    features = {
        "mint_embed_a": torch.empty((n, embed_dim), dtype=torch.float16),
        "mint_embed_b": torch.empty((n, embed_dim), dtype=torch.float16),
    }
    labels = torch.empty(n, dtype=torch.float32)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device_type = torch.device(args.device).type
    use_amp = device_type == "cuda" and args.dtype != "fp32"
    start_time = time.time()

    with torch.inference_mode():
        for output_row, (_row_index, seq_a, seq_b, label) in enumerate(
            iter_pair_rows(
                args.pairs,
                a_col=args.a_col,
                b_col=args.b_col,
                label_col=args.label_col,
                limit=args.limit,
            )
        ):
            seq_a, seq_b = truncate_pair_balanced(seq_a, seq_b, args.max_total_tokens)
            token_a = torch.tensor(
                alphabet.encode("<cls>" + seq_a.replace("J", "L") + "<eos>"),
                dtype=torch.long,
            )
            token_b = torch.tensor(
                alphabet.encode("<cls>" + seq_b.replace("J", "L") + "<eos>"),
                dtype=torch.long,
            )
            tokens = torch.cat([token_a, token_b]).unsqueeze(0).to(args.device)
            chain_ids = torch.cat(
                [torch.zeros_like(token_a), torch.ones_like(token_b)]
            ).unsqueeze(0).to(args.device)
            with torch.autocast(device_type=device_type, dtype=torch_dtype, enabled=use_amp):
                representation = model(tokens, chain_ids, repr_layers=[layer])["representations"][layer][0]
            valid = (
                tokens[0].ne(model.cls_idx)
                & tokens[0].ne(model.eos_idx)
                & tokens[0].ne(model.padding_idx)
            )
            mask_a = valid & chain_ids[0].eq(0)
            mask_b = valid & chain_ids[0].eq(1)
            features["mint_embed_a"][output_row].copy_(
                representation[mask_a].mean(0).half().cpu()
            )
            features["mint_embed_b"][output_row].copy_(
                representation[mask_b].mean(0).half().cpu()
            )
            labels[output_row] = label
            if (output_row + 1) % args.progress_every == 0 or output_row + 1 == n:
                rate = (output_row + 1) / max(time.time() - start_time, 1e-6)
                print(f"[MINT] {output_row + 1}/{n} pairs {rate:.2f}/s", flush=True)

    save_baseline_pair_cache(
        args.output,
        baseline="MINT",
        labels=labels,
        features=features,
        meta={
            "source_pairs": str(args.pairs),
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "layer": layer,
            "embedding_dim": embed_dim,
            "max_total_tokens": args.max_total_tokens,
            "pooling": "mean over each jointly contextualized chain",
        },
        overwrite=args.overwrite,
    )
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
