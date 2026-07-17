#!/usr/bin/env python3
"""Cache ESM-C (6B) layer-60 representations + SAE fingerprints for RAPPPID-C3.

Run with the E1 conda env (Biohub transformers fork + xformers). For each unique
sequence across the three C3 splits we run ESMC-6B once and store:
  - esmc_mean    [2560]  : mean-pool of raw layer-60 hidden states (raw-rep baseline)
  - esmc_sae_max [16384] : max-pool of layer-60 SAE features  (ECFP-like fingerprint)
  - esmc_sae_mean[16384] : mean-pool of layer-60 SAE features

The SAE encode is replicated from transformers' _ESMCSAELayer.forward:
    x = zscore(h);  feat = TopK_k(ReLU((x - b_dec) @ W_enc))     # raw magnitudes

This writes the project's default pooled cache (ESMC_DEFAULT_SEQ_CACHE), which the
audit consumes via conf.paths (`ppi_fingerprint.baseline` /
`analysis.participation_predictor`). The
RAPPPID-C3 CSVs are read-only.

  PYTHONPATH=. /data/wmzhu/anaconda3/envs/E1/bin/python scripts/cache/cache_esmc_fingerprints.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from conf.paths import (
    ESMC_MODEL, ESMC_SAE, ESMC_DEFAULT_SEQ_CACHE,
    C3_TRAIN_CSV, C3_VAL_CSV, C3_TEST_CSV,
)

MODEL = ESMC_MODEL
SAE = ESMC_SAE
LAYER = 60
MAX_RESIDUES = 1022
SPLIT_CSVS = {"train": C3_TRAIN_CSV, "val": C3_VAL_CSV, "test": C3_TEST_CSV}


def pick_gpu() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    ).decode()
    return sorted(
        ((int(line.split(",")[1]), line.split(",")[0].strip()) for line in out.strip().splitlines()),
        reverse=True,
    )[0][1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cache ESM-C layer-60 pooled reps + SAE features (C3)")
    ap.add_argument("--limit", type=int, default=0, help="cap #sequences (smoke test)")
    ap.add_argument("--longest-first", action="store_true",
                    help="apply --limit to the longest sequences first for OOM smoke tests")
    ap.add_argument("--token-budget", type=int, default=3072)
    ap.add_argument("--device-id", type=int, default=None,
                    help="physical GPU id; default picks the GPU with most free memory")
    ap.add_argument("--col-a", default="query")
    ap.add_argument("--col-b", default="text")
    ap.add_argument("--max-residues", type=int, default=MAX_RESIDUES,
                    help="max residues kept before tokenizer adds BOS/EOS")
    ap.add_argument("--sae-token-chunk", type=int, default=512,
                    help="residue chunk size for SAE pooled encoding; lowers peak GPU memory")
    ap.add_argument("--out", type=Path, default=ESMC_DEFAULT_SEQ_CACHE)
    ap.add_argument("--store-all-hidden-states", action="store_true",
                    help="store all transformer hidden states and select hidden_states[60]; "
                         "default uses last_hidden_state to reduce GPU memory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer

    # ---- unique sequences from the 3 CSVs (read-only) ----
    seqs: list[str] = []
    seen: set[str] = set()
    row_counts: dict[str, int] = {}
    for split, path in SPLIT_CSVS.items():
        df = pd.read_csv(path, usecols=[args.col_a, args.col_b])
        row_counts[split] = len(df)
        for col in (args.col_a, args.col_b):
            for s in df[col].astype(str):
                if s not in seen:
                    seen.add(s)
                    seqs.append(s)
    if args.longest_first:
        seqs = sorted(seqs, key=len, reverse=True)
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"[device] CUDA_VISIBLE_DEVICES={device_id} (physical)", flush=True)
    print(f"[data] splits={row_counts} unique sequences: {len(seqs)} "
          f"(limit={args.limit or 'none'}) max_residues={args.max_residues}", flush=True)

    # ---- model + SAE weights ----
    print("[load] ESMC-6B ...", flush=True)
    model = AutoModel.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    sae = AutoModel.from_pretrained(str(SAE), trust_remote_code=True)
    sae.initialize_layers([LAYER])
    layer = sae.layers[str(LAYER)]
    w_enc = layer.W_enc.detach().float().cuda()      # [2560, 16384]
    b_dec = layer.b_dec.detach().float().cuda()      # [2560]
    k = int(layer.params.k)
    act_dim, dict_dim = w_enc.shape
    print(f"[sae] W_enc={tuple(w_enc.shape)} k={k}", flush=True)

    def sae_pool(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_max = torch.zeros(dict_dim, device=h.device, dtype=torch.float32)
        pooled_sum = torch.zeros(dict_dim, device=h.device, dtype=torch.float32)
        n_tokens = max(int(h.shape[0]), 1)
        chunk = max(1, args.sae_token_chunk)
        for start in range(0, n_tokens, chunk):
            hc = h[start:start + chunk]
            x = hc - hc.mean(-1, keepdim=True)
            x = x / (x.std(-1, keepdim=True) + 1e-5)
            pre = torch.relu((x - b_dec) @ w_enc)
            vals, idx = pre.topk(k, dim=-1)
            flat_idx = idx.reshape(-1)
            flat_vals = vals.reshape(-1)
            pooled_sum.scatter_add_(0, flat_idx, flat_vals)
            pooled_max.scatter_reduce_(0, flat_idx, flat_vals, reduce="amax", include_self=True)
        return pooled_max, pooled_sum / n_tokens

    @torch.inference_mode()
    def run_batch(batch_seqs: list[str]):
        trunc = [s[: args.max_residues] for s in batch_seqs]
        enc = tok(trunc, return_tensors="pt", padding=True)
        enc = {kk: v.to("cuda") for kk, v in enc.items()}
        out = model(**enc, output_hidden_states=args.store_all_hidden_states)
        h60 = out.hidden_states[LAYER] if args.store_all_hidden_states else out.last_hidden_state
        am = enc["attention_mask"].bool()
        em, smax, smean = [], [], []
        for i in range(h60.size(0)):
            mask = am[i].clone()
            idxs = mask.nonzero(as_tuple=True)[0]
            # strip BOS (first real token) and EOS (last real token)
            if idxs.numel() > 2:
                mask[idxs[0]] = False
                mask[idxs[-1]] = False
            hi = h60[i][mask].float()                        # [Lr, 2560]
            em.append(hi.mean(0).half().cpu())
            fmax, fmean = sae_pool(hi)
            smax.append(fmax.half().cpu())
            smean.append(fmean.half().cpu())
        return em, smax, smean

    # ---- length-sorted, token-budget batching ----
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    esmc_mean = [None] * len(seqs)
    sae_max = [None] * len(seqs)
    sae_mean = [None] * len(seqs)
    i = 0
    done = 0
    t0 = time.time()
    while i < len(order):
        seq_len = min(len(seqs[order[i]]), args.max_residues) + 2
        bs = max(1, args.token_budget // max(seq_len, 1))
        idxs = order[i:i + bs]
        i += bs
        em, smax, smean = run_batch([seqs[j] for j in idxs])
        for j, a, b, c in zip(idxs, em, smax, smean):
            esmc_mean[j] = a
            sae_max[j] = b
            sae_mean[j] = c
        done += len(idxs)
        if done % 512 < bs:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(seqs)}  {rate:.1f} seq/s  elapsed {time.time()-t0:.0f}s", flush=True)

    cache = {
        "sequences": seqs,
        "seq2idx": {s: i for i, s in enumerate(seqs)},
        "esmc_mean": torch.stack(esmc_mean),
        "esmc_sae_max": torch.stack(sae_max),
        "esmc_sae_mean": torch.stack(sae_mean),
        "meta": {
            "dataset": "rapppid-c3",
            "model": "ESMC-6B", "layer": LAYER, "k": k, "dict": int(dict_dim),
            "act_dim": int(act_dim), "max_residues": args.max_residues,
            "sae_token_chunk": args.sae_token_chunk,
            "splits": {name: str(p) for name, p in SPLIT_CSVS.items()},
            "row_counts": row_counts, "col_a": args.col_a, "col_b": args.col_b,
        },
    }
    torch.save(cache, args.out)
    size_mb = args.out.stat().st_size / 1e6
    fp = cache["esmc_sae_max"]
    dens = (fp > 0).float().mean().item()
    bits = (fp > 0).sum(1).float()
    print(f"[saved] {args.out}  ({size_mb:.0f} MB)  in {time.time()-t0:.0f}s", flush=True)
    print(f"[sanity] esmc_sae_max active density={dens*100:.2f}%  bits/seq min/med/max="
          f"{int(bits.min())}/{int(bits.median())}/{int(bits.max())}", flush=True)
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
