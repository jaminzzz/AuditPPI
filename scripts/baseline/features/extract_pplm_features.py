#!/usr/bin/env python3
"""Extract the official pair-conditioned PPLM-PPI feature components.

For each pair this saves mean- and max-pooled inter-chain attention, both
intra-chain attentions, and both final-layer contextual embeddings.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / ".project-root").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conf.paths import BASELINES  # noqa: E402
from src.features.baseline_io import (  # noqa: E402
    count_pair_rows,
    iter_pair_rows,
    save_baseline_pair_cache,
    truncate_pair_balanced,
)


def pick_gpu() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
        ).decode()
        return sorted(
            ((int(line.split(",")[1]), line.split(",")[0].strip()) for line in output.splitlines()),
            reverse=True,
        )[0][1]
    except Exception:  # noqa: BLE001
        return "0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--a-col", default="query")
    parser.add_argument("--b-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--pplm-root", type=Path, default=BASELINES / "PPLM")
    parser.add_argument("--checkpoint", type=Path, default=BASELINES / "PPLM" / "pplm_t33_650M.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"[device] CUDA_VISIBLE_DEVICES={device_id}", flush=True)
    if str(args.pplm_root) not in sys.path:
        sys.path.insert(0, str(args.pplm_root))

    import torch
    from pplm import Alphabet, PPLM

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    params = checkpoint["param"]
    alphabet = Alphabet.from_architecture()
    with torch.device("meta"):
        model = PPLM(
            num_layers=int(params["encoder_layers"]),
            embed_dim=int(params["encoder_embed_dim"]),
            attention_heads=int(params["encoder_attention_heads"]),
            token_dropout=bool(params.get("token_dropout", True)),
            alphabet=alphabet,
        )
    model.load_state_dict(checkpoint["model"], strict=True, assign=True)
    del checkpoint
    model = model.to(args.device).eval()

    n = count_pair_rows(args.pairs, args.limit)
    attention_dim = model.num_layers * model.attention_heads
    embed_dim = model.embed_dim
    features = {}
    for pool in ("mean", "max"):
        features[f"pplm_{pool}_inter_attn"] = torch.empty((n, attention_dim), dtype=torch.float16)
        features[f"pplm_{pool}_intra_a"] = torch.empty((n, attention_dim), dtype=torch.float16)
        features[f"pplm_{pool}_intra_b"] = torch.empty((n, attention_dim), dtype=torch.float16)
        features[f"pplm_{pool}_embed_a"] = torch.empty((n, embed_dim), dtype=torch.float16)
        features[f"pplm_{pool}_embed_b"] = torch.empty((n, embed_dim), dtype=torch.float16)
    labels = torch.empty(n, dtype=torch.float32)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device_type = torch.device(args.device).type
    use_amp = device_type == "cuda" and args.dtype != "fp32"
    converter = alphabet.get_batch_converter()
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
            _, _, tokens_a = converter([("A", seq_a)])
            _, _, tokens_b = converter([("B", seq_b)])
            tokens = torch.cat([tokens_a, tokens_b], dim=-1).to(args.device)
            len_a_tokens = int(tokens_a.shape[1])
            len_b_tokens = int(tokens_b.shape[1])
            total_tokens = len_a_tokens + len_b_tokens
            inter_mask = torch.ones((total_tokens, total_tokens), device=args.device)
            inter_mask[:len_a_tokens, :len_a_tokens] = 0
            inter_mask[len_a_tokens:, len_a_tokens:] = 0

            with torch.autocast(device_type=device_type, dtype=torch_dtype, enabled=use_amp):
                result = model(
                    tokens,
                    inter_mask,
                    repr_layers=[model.num_layers],
                    need_head_weights=True,
                    return_contacts=False,
                )
            representation = result["representations"][model.num_layers][0]
            embed_a = representation[1 : len_a_tokens - 1]
            embed_b = representation[len_a_tokens + 1 : total_tokens - 1]
            attentions = result["attentions"][0].reshape(
                attention_dim, total_tokens, total_tokens
            )
            a_slice = slice(1, len_a_tokens - 1)
            b_slice = slice(len_a_tokens + 1, total_tokens - 1)
            attn_aa = attentions[:, a_slice, a_slice]
            attn_ab = attentions[:, a_slice, b_slice]
            attn_ba = attentions[:, b_slice, a_slice]
            attn_bb = attentions[:, b_slice, b_slice]
            inter = (attn_ab + attn_ba.transpose(1, 2)) / 2

            for pool, reducer in (("mean", torch.mean), ("max", torch.amax)):
                features[f"pplm_{pool}_inter_attn"][output_row].copy_(
                    reducer(inter, dim=(1, 2)).half().cpu()
                )
                features[f"pplm_{pool}_intra_a"][output_row].copy_(
                    reducer(attn_aa, dim=(1, 2)).half().cpu()
                )
                features[f"pplm_{pool}_intra_b"][output_row].copy_(
                    reducer(attn_bb, dim=(1, 2)).half().cpu()
                )
                features[f"pplm_{pool}_embed_a"][output_row].copy_(
                    reducer(embed_a, dim=0).half().cpu()
                )
                features[f"pplm_{pool}_embed_b"][output_row].copy_(
                    reducer(embed_b, dim=0).half().cpu()
                )
            labels[output_row] = label
            if (output_row + 1) % args.progress_every == 0 or output_row + 1 == n:
                rate = (output_row + 1) / max(time.time() - start_time, 1e-6)
                print(f"[PPLM] {output_row + 1}/{n} pairs {rate:.2f}/s", flush=True)

    save_baseline_pair_cache(
        args.output,
        baseline="PPLM",
        labels=labels,
        features=features,
        meta={
            "source_pairs": str(args.pairs),
            "checkpoint": str(args.checkpoint),
            "layers": model.num_layers,
            "heads": model.attention_heads,
            "embedding_dim": model.embed_dim,
            "max_total_tokens": args.max_total_tokens,
            "feature_contract": "official PPLM-PPI mean/max attention and embedding components",
        },
        overwrite=args.overwrite,
    )
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
