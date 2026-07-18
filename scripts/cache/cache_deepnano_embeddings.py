#!/usr/bin/env python3
"""Cache DeepNano-style pooled embeddings for RAPPPID-C3 (baseline provenance).

DeepNano-seq's architectural idea is not tied to its 8M checkpoint: one PLM
embedding is pooled three ways (mean/min/max), three MLP heads are trained, and
the three probabilities are averaged. This script builds the frozen-PLM embedding
cache needed for that baseline with the backbones used in the project:

  --backbone esmc_6b   : ESM-C-6B layer 60, dim=2560   (run in the E1 env)
  --backbone esm2_650m : ESM-2-650M layer 33, dim=1280 (run in the E1 env)

No PLM parameters are trained here. RAPPPID CSVs are read-only. Outputs are
written under DEEPNANO_DIR/deepnano_<backbone>/{seq_cache.pt,meta.json}. The
products are not consumed by the audit yet; kept for reproducibility.

  /data/wmzhu/anaconda3/envs/E1/bin/python \
      scripts/cache/cache_deepnano_embeddings.py --backbone esmc_6b
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.paths import (
    ESMC_MODEL, ESM2_650M_MODEL, DEEPNANO_DIR,
    C3_TRAIN_CSV, C3_VAL_CSV, C3_TEST_CSV,
)
from conf.model import ESMC_SAE_DEFAULT_LAYER, ESM2_LAYER, ESMC_DIM, ESM2_DIM, MAX_RESIDUES

CSV_SPLITS = {"train": C3_TRAIN_CSV, "val": C3_VAL_CSV, "test": C3_TEST_CSV}
COL_A, COL_B = "query", "text"


def pick_gpu() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    ).decode()
    return sorted(
        ((int(line.split(",")[1]), line.split(",")[0].strip()) for line in out.strip().splitlines()),
        reverse=True,
    )[0][1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache mean/min/max frozen PLM embeddings")
    p.add_argument("--backbone", choices=["esmc_6b", "esm2_650m"], default="esmc_6b")
    p.add_argument("--pretrained-model", default="",
                   help="Override model path/name. Useful for local ESM-2 weights.")
    p.add_argument("--layer", type=int, default=None,
                   help=f"Override hidden-state layer. Defaults: ESM-C {ESMC_SAE_DEFAULT_LAYER}, "
                        f"ESM-2 {ESM2_LAYER}.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--limit", type=int, default=0, help="Cap unique sequences for smoke tests.")
    p.add_argument("--max-residues", type=int, default=MAX_RESIDUES,
                   help="Residues kept before model special tokens.")
    p.add_argument("--token-budget", type=int, default=3072,
                   help="Approximate residues per length-sorted batch.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-id", type=int, default=None, help="physical GPU id; default=freest")
    return p.parse_args()


def default_out_dir(backbone: str, layer: int) -> Path:
    """Tag non-default layers so layer-80 caches do not clobber the default dir."""
    base = DEEPNANO_DIR / f"deepnano_{backbone}"
    default_layer = ESMC_SAE_DEFAULT_LAYER if backbone == "esmc_6b" else ESM2_LAYER
    if layer == default_layer:
        return base
    return DEEPNANO_DIR / f"deepnano_{backbone}_l{layer}"


def collect_unique_sequences(splits: list[str], limit: int) -> list[str]:
    import pandas as pd

    seen: set[str] = set()
    seqs: list[str] = []
    for split in splits:
        df = pd.read_csv(CSV_SPLITS[split])
        for col in (COL_A, COL_B):
            for seq in df[col].astype(str):
                if seq not in seen:
                    seen.add(seq)
                    seqs.append(seq)
    return seqs[:limit] if limit else seqs


def make_batches(seqs: list[str], max_residues: int, token_budget: int) -> list[list[int]]:
    order = sorted(range(len(seqs)), key=lambda i: min(len(seqs[i]), max_residues))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_tokens = 0
    for idx in order:
        n_tok = min(len(seqs[idx]), max_residues) + 2
        if cur and cur_tokens + n_tok > token_budget:
            batches.append(cur)
            cur = []
            cur_tokens = 0
        cur.append(idx)
        cur_tokens += n_tok
    if cur:
        batches.append(cur)
    return batches


def masked_pools(hidden, mask):
    """Return mask-aware mean/min/max over residue positions."""
    import torch

    keep = mask.bool().unsqueeze(-1)
    denom = keep.sum(dim=1).clamp_min(1)
    mean = (hidden * keep).sum(dim=1) / denom
    max_pool = hidden.masked_fill(~keep, -torch.inf).max(dim=1).values
    min_pool = hidden.masked_fill(~keep, torch.inf).min(dim=1).values
    return mean.cpu(), min_pool.cpu(), max_pool.cpu()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id) if args.device_id is not None else pick_gpu()
    print(f"[device] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (physical)", flush=True)

    import torch
    from transformers import AutoModel, AutoTokenizer, EsmModel

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.backbone == "esmc_6b":
        layer_idx = ESMC_SAE_DEFAULT_LAYER if args.layer is None else args.layer
    else:
        layer_idx = ESM2_LAYER if args.layer is None else args.layer
    out_dir = args.out_dir or default_out_dir(args.backbone, layer_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = collect_unique_sequences(splits, args.limit)
    batches = make_batches(seqs, args.max_residues, args.token_budget)
    print(f"[data] unique={len(seqs)} splits={splits} limit={args.limit or 'none'}", flush=True)
    print(f"[batch] {len(batches)} batches token_budget={args.token_budget}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    use_amp = dev.type == "cuda" and args.dtype != "fp32"

    if args.backbone == "esmc_6b":
        model_name = args.pretrained_model or str(ESMC_MODEL)
        print(f"[load] ESM-C model={model_name} layer={layer_idx}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=amp_dtype if args.dtype != "fp32" else torch.float32,
            trust_remote_code=True,
        ).to(dev).eval()

        @torch.inference_mode()
        def encode_batch(batch_seqs: list[str]):
            trunc = [s[:args.max_residues] for s in batch_seqs]
            enc = tok(trunc, return_tensors="pt", padding=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                out = model(**enc, output_hidden_states=True)
            hidden_all = out.hidden_states[layer_idx].float()
            residue_masks = []
            residues = []
            for row in range(hidden_all.size(0)):
                mask = enc["attention_mask"][row].bool().clone()
                idxs = mask.nonzero(as_tuple=True)[0]
                if idxs.numel() > 2:
                    mask[idxs[0]] = False
                    mask[idxs[-1]] = False
                residues.append(hidden_all[row])
                residue_masks.append(mask)
            return masked_pools(torch.stack(residues), torch.stack(residue_masks))

    else:
        model_name = args.pretrained_model or ESM2_650M_MODEL
        print(f"[load] ESM-2 model={model_name} layer={layer_idx}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(dev).eval()
        n_layers = int(model.config.num_hidden_layers)
        use_last = layer_idx == n_layers

        @torch.inference_mode()
        def encode_batch(batch_seqs: list[str]):
            trunc = [s[:args.max_residues] for s in batch_seqs]
            enc = tok(trunc, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_residues + 2)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                if use_last:
                    hidden_all = model(**enc).last_hidden_state
                else:
                    hidden_all = model(**enc, output_hidden_states=True).hidden_states[layer_idx]
            hidden_all = hidden_all.float()
            residue_masks = []
            residues = []
            for row in range(hidden_all.size(0)):
                mask = enc["attention_mask"][row].bool().clone()
                idxs = mask.nonzero(as_tuple=True)[0]
                if idxs.numel() > 2:
                    mask[idxs[0]] = False
                    mask[idxs[-1]] = False
                residues.append(hidden_all[row])
                residue_masks.append(mask)
            return masked_pools(torch.stack(residues), torch.stack(residue_masks))

    if hasattr(model.config, "hidden_size"):
        hidden_size = int(model.config.hidden_size)
    elif hasattr(model.config, "d_model"):
        hidden_size = int(model.config.d_model)
    else:
        raise AttributeError("model config has neither hidden_size nor d_model")
    expected_dim = ESMC_DIM if args.backbone == "esmc_6b" else ESM2_DIM
    assert hidden_size == expected_dim, (args.backbone, hidden_size, expected_dim)
    mean = torch.empty(len(seqs), hidden_size, dtype=torch.float16)
    min_pool = torch.empty_like(mean)
    max_pool = torch.empty_like(mean)

    done = 0
    t0 = time.time()
    for bi, idxs in enumerate(batches):
        b_mean, b_min, b_max = encode_batch([seqs[i] for i in idxs])
        for row, seq_idx in enumerate(idxs):
            mean[seq_idx] = b_mean[row].half()
            min_pool[seq_idx] = b_min[row].half()
            max_pool[seq_idx] = b_max[row].half()
        done += len(idxs)
        if bi == len(batches) - 1 or bi % 20 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  batch {bi + 1}/{len(batches)} seqs {done}/{len(seqs)} "
                  f"{rate:.1f} seq/s", flush=True)

    meta = {
        "baseline": "DeepNano-style frozen-embedding ensemble",
        "backbone": args.backbone,
        "pretrained_model": model_name,
        "layer": layer_idx,
        "hidden_size": hidden_size,
        "pools": ["mean", "min", "max"],
        "pooling": "mask-aware residue mean/min/max; model special tokens excluded",
        "max_residues": args.max_residues,
        "splits": splits,
        "n_unique": len(seqs),
    }
    payload = {
        "sequences": seqs,
        "seq2idx": {seq: i for i, seq in enumerate(seqs)},
        "mean": mean,
        "min": min_pool,
        "max": max_pool,
        "meta": meta,
    }
    cache_path = out_dir / "seq_cache.pt"
    torch.save(payload, cache_path)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[saved] {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("DEEPNANO_EMBED_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
