#!/usr/bin/env python3
"""Cache ESM-C SAE fingerprints for the Bernett gold-standard PPI task.

Consumes the MINT GeneralPPI ``Bernett`` split (via conf.paths.BERNETT_DIR):
  train = Intra1_seqs.csv, val = Intra0_seqs.csv, test = Intra2_seqs.csv

Each unique (cleaned) sequence is encoded once and written to BERNETT_SEQ_CACHE
in the pooled-cache contract shared by the audit:
  - sequences / seq2idx
  - esmc_mean / esmc_sae_max / esmc_sae_mean

Run with the E1 conda env (Biohub transformers fork). CSVs are read-only.

  PYTHONPATH=. /data/wmzhu/anaconda3/envs/E1/bin/python \
      scripts/cache/cache_bernett_esmc_fingerprints.py
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
    ESMC_MODEL, ESMC_SAE, BERNETT_DIR, BERNETT_SPLIT_CSVS, BERNETT_SEQ_CACHE,
)

MODEL = ESMC_MODEL
SAE = ESMC_SAE
LAYER = 60
MAX_RESIDUES = 1022


def pick_gpu() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    ).decode()
    return sorted(
        ((int(line.split(",")[1]), line.split(",")[0].strip()) for line in out.strip().splitlines()),
        reverse=True,
    )[0][1]


def clean_seq(seq: str) -> str:
    # Mirrors MINT task cleaning, with J mapped to L for PLM tokenizers.
    return str(seq).replace("*", "").replace("f", "").replace("J", "L")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache Bernett ESM-C pooled reps + SAE features")
    p.add_argument("--limit", type=int, default=0, help="cap #sequences (smoke test)")
    p.add_argument("--token-budget", type=int, default=3072)
    p.add_argument("--sae-token-chunk", type=int, default=512,
                   help="residue chunk size for SAE pooled encoding; lowers peak GPU memory")
    p.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    p.add_argument("--data-dir", type=Path, default=BERNETT_DIR)
    p.add_argument("--device-id", type=int, default=None,
                   help="physical GPU id to use; default picks the GPU with most free memory")
    p.add_argument("--out", type=Path, default=BERNETT_SEQ_CACHE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"[device] CUDA_VISIBLE_DEVICES={device_id} (physical)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer

    split_dfs: dict[str, "pd.DataFrame"] = {}
    seen: set[str] = set()
    seqs: list[str] = []
    for split, filename in BERNETT_SPLIT_CSVS.items():
        df = pd.read_csv(args.data_dir / filename)
        df["seq1"] = df["seq1"].map(clean_seq)
        df["seq2"] = df["seq2"].map(clean_seq)
        split_dfs[split] = df
        for col in ("seq1", "seq2"):
            for seq in df[col].astype(str):
                if seq not in seen:
                    seen.add(seq)
                    seqs.append(seq)
    if args.limit:
        seqs = seqs[: args.limit]
    seq2idx = {seq: i for i, seq in enumerate(seqs)}
    print(
        "[data] "
        + " ".join(f"{split}={len(df)} pos={int(df['labels'].sum())}"
                   for split, df in split_dfs.items())
        + f" unique_sequences={len(seqs)}",
        flush=True,
    )

    print("[load] ESMC-6B ...", flush=True)
    model = AutoModel.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(str(MODEL))

    sae = AutoModel.from_pretrained(str(SAE), trust_remote_code=True)
    sae.initialize_layers([LAYER])
    layer = sae.layers[str(LAYER)]
    w_enc = layer.W_enc.detach().float().cuda()
    b_dec = layer.b_dec.detach().float().cuda()
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
        out = model(**enc, output_hidden_states=True)
        h60 = out.hidden_states[LAYER]
        am = enc["attention_mask"].bool()
        em, smax, smean = [], [], []
        for i in range(h60.size(0)):
            mask = am[i].clone()
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() > 2:
                mask[idxs[0]] = False
                mask[idxs[-1]] = False
            hi = h60[i][mask].float()
            em.append(hi.mean(0).half().cpu())
            fmax, fmean = sae_pool(hi)
            smax.append(fmax.half().cpu())
            smean.append(fmean.half().cpu())
        return em, smax, smean

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
        if done % 512 < bs or done == len(seqs):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(seqs)}  {rate:.1f} seq/s  elapsed {time.time()-t0:.0f}s", flush=True)

    cache = {
        "sequences": seqs,
        "seq2idx": seq2idx,
        "esmc_mean": torch.stack(esmc_mean),
        "esmc_sae_max": torch.stack(sae_max),
        "esmc_sae_mean": torch.stack(sae_mean),
        "meta": {
            "dataset": "bernett",
            "model": "ESMC-6B", "layer": LAYER, "k": k, "dict": int(dict_dim),
            "act_dim": int(act_dim), "max_residues": args.max_residues,
            "sae_token_chunk": args.sae_token_chunk,
            "data_dir": str(args.data_dir), "splits": dict(BERNETT_SPLIT_CSVS),
        },
    }
    torch.save(cache, args.out)
    fp = cache["esmc_sae_max"]
    dens = (fp > 0).float().mean().item()
    bits = (fp > 0).sum(1).float()
    size_mb = args.out.stat().st_size / 1e6
    print(f"[saved] {args.out} ({size_mb:.0f} MB) in {time.time()-t0:.0f}s", flush=True)
    print(f"[sanity] active density={dens*100:.2f}% bits/seq min/med/max="
          f"{int(bits.min())}/{int(bits.median())}/{int(bits.max())}", flush=True)
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
