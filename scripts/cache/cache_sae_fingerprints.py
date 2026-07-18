#!/usr/bin/env python3
"""Cache pooled ESM-2 (650M) + InterPLM-SAE features for RAPPPID-C3 (legacy line).

Pipeline per unique sequence (ESM-2-650M / layer 33 -> InterPLM ReLU-SAE):
    seq --ESM2-650M--> residue embeddings (L, 1280)
        --SAE.encode(normalize_features=True)--> residue SAE features (L, 10240)
        --max over L--> sae_max (10240)   # source of the 0/1 fingerprint
        --mean over L--> sae_mean (10240) # length-normalized continuous baseline
    residue embeddings --mean over L--> esm_mean (1280)  # raw ESM baseline

Sequences are deduplicated across the three C3 splits so ESM+SAE runs once each.
This is the ESM-2 provenance for the pre-ESM-C fingerprint results; the ESM-C
line (cache_esmc_fingerprints.py) supersedes it. Writes ESM2_SEQ_CACHE.

Run with the E1 conda env (uses the InterPLM package under external/). The
ESM-2-650M weights are loaded from HuggingFace by name, so they must be cached
locally when HF_HUB_OFFLINE=1.

  /data/wmzhu/anaconda3/envs/E1/bin/python \
      scripts/cache/cache_sae_fingerprints.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.paths import (
    ESM2_650M_MODEL, ESM2_SAE_CKPT, INTERPLM_ROOT, ESM2_SEQ_CACHE,
    C3_TRAIN_CSV, C3_VAL_CSV, C3_TEST_CSV,
)
from conf.model import ESM2_LAYER, ESM2_DIM, ESM2_SAE_DIM, MAX_RESIDUES

# InterPLM package (interplm.sae.dictionary.ReLUSAE) lives under external/InterPLM.
sys.path.insert(0, str(INTERPLM_ROOT))

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
    p = argparse.ArgumentParser(description="Cache ESM-2 650M + InterPLM-SAE pooled features")
    p.add_argument("--sae-path", type=str, default=str(ESM2_SAE_CKPT))
    p.add_argument("--esm-model", type=str, default=ESM2_650M_MODEL)
    p.add_argument("--layer", type=int, default=ESM2_LAYER)
    p.add_argument("--max-length", type=int, default=MAX_RESIDUES, help="max residues kept (CLS/EOS extra)")
    p.add_argument("--max-batch-tokens", type=int, default=8192,
                   help="length-bucketed batching budget (residues per ESM batch)")
    p.add_argument("--out", type=str, default=str(ESM2_SEQ_CACHE))
    p.add_argument("--device-id", type=int, default=None, help="physical GPU id; default=freest")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast for ESM")
    p.add_argument("--limit", type=int, default=0, help="smoke test: only first N unique seqs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id) if args.device_id is not None else pick_gpu()
    print(f"[device] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (physical)", flush=True)

    import pandas as pd
    import torch
    from transformers import AutoTokenizer, EsmModel

    dev = "cuda"
    amp = not args.no_amp

    # ---- collect unique sequences across all splits (read-only CSVs) --------
    uniq: set[str] = set()
    for split, path in CSV_SPLITS.items():
        df = pd.read_csv(path)
        uniq |= set(df[COL_A].astype(str)) | set(df[COL_B].astype(str))
    sequences = sorted(uniq)  # deterministic order
    if args.limit and args.limit > 0:
        sequences = sequences[: args.limit]
    n = len(sequences)
    print(f"[data] unique sequences: {n} (limit={args.limit or 'none'})", flush=True)

    # ---- load InterPLM ReLU-SAE (package on sys.path via INTERPLM_ROOT) -----
    from interplm.sae.dictionary import ReLUSAE

    sae = ReLUSAE.from_pretrained(args.sae_path, device=dev)
    sae.eval()
    D = sae.dict_size
    assert D == ESM2_SAE_DIM, (D, ESM2_SAE_DIM)
    assert sae.activation_dim == ESM2_DIM, (sae.activation_dim, ESM2_DIM)
    assert not bool(sae.normalize_to_sqrt_d), "expected normalize_to_sqrt_d=False"
    print(f"[sae] {sae.__class__.__name__} dict={D} act={sae.activation_dim} "
          f"sqrt_d={bool(sae.normalize_to_sqrt_d)}", flush=True)

    # ---- load ESM-2 650M (cached, offline) ---------------------------------
    tok = AutoTokenizer.from_pretrained(args.esm_model)
    esm = EsmModel.from_pretrained(args.esm_model, add_pooling_layer=False).to(dev).eval()
    n_layers = esm.config.num_hidden_layers
    use_last = args.layer == n_layers  # layer 33 == last_hidden_state (cheaper)
    print(f"[esm] {args.esm_model} layers={n_layers} use_last_hidden={use_last} amp={amp}", flush=True)

    # ---- output buffers (fp16 to save disk) --------------------------------
    sae_max = torch.zeros((n, D), dtype=torch.float16)
    sae_mean = torch.zeros((n, D), dtype=torch.float16)
    esm_mean = torch.zeros((n, sae.activation_dim), dtype=torch.float16)

    # ---- length-bucketed batching ------------------------------------------
    trunc = [s[: args.max_length] for s in sequences]
    order = sorted(range(n), key=lambda i: len(trunc[i]))  # short->long
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    for i in order:
        L = len(trunc[i])
        if cur and (cur_len + L) > args.max_batch_tokens:
            batches.append(cur); cur = []; cur_len = 0
        cur.append(i); cur_len += L + 2
    if cur:
        batches.append(cur)
    print(f"[batch] {len(batches)} batches (token budget={args.max_batch_tokens})", flush=True)

    t0 = time.time()
    done = 0
    with torch.inference_mode():
        for bi, idxs in enumerate(batches):
            seqs = [trunc[i] for i in idxs]
            enc = tok(seqs, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length + 2)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                if use_last:
                    hidden = esm(**enc).last_hidden_state
                else:
                    hidden = esm(**enc, output_hidden_states=True).hidden_states[args.layer]
            hidden = hidden.float()  # (B, T, 1280)
            mask = enc["attention_mask"]  # (B, T)
            for row, i in enumerate(idxs):
                L = int(mask[row].sum().item()) - 2  # drop CLS(0) and EOS(L+1)
                if L <= 0:
                    continue
                res = hidden[row, 1:1 + L, :]                       # (L, 1280)
                feats = sae.encode(res, normalize_features=True)   # (L, 10240)
                sae_max[i] = feats.max(dim=0).values.half().cpu()
                sae_mean[i] = feats.mean(dim=0).half().cpu()
                esm_mean[i] = res.mean(dim=0).half().cpu()
            done += len(idxs)
            if bi % 20 == 0 or bi == len(batches) - 1:
                el = time.time() - t0
                print(f"  batch {bi+1}/{len(batches)}  seqs {done}/{n}  "
                      f"{done/max(el,1e-6):.1f} seq/s  elapsed {el:.0f}s", flush=True)

    save_path = Path(args.out)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequences": sequences,
        "seq2idx": {s: i for i, s in enumerate(sequences)},
        "sae_max": sae_max,
        "sae_mean": sae_mean,
        "esm_mean": esm_mean,
        "meta": {
            "esm_model": args.esm_model, "layer": args.layer,
            "max_length": args.max_length, "sae_path": args.sae_path,
            "dict_size": D, "act_dim": sae.activation_dim,
            "amp_bf16": amp, "n_unique": n,
            "splits": {k: str(v) for k, v in CSV_SPLITS.items()},
        },
    }
    torch.save(payload, save_path)
    el = time.time() - t0
    print(f"[saved] {save_path}  ({save_path.stat().st_size/1e6:.0f} MB)  in {el:.0f}s", flush=True)

    # ---- quick sanity on @0.5 sparsity -------------------------------------
    on = (sae_max.float() >= 0.5).sum(dim=1)
    print(f"[sanity] bits@0.5 per seq: min/med/max = "
          f"{int(on.min())}/{int(on.median())}/{int(on.max())} "
          f"({100*on.float().mean()/D:.2f}% density)", flush=True)
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
