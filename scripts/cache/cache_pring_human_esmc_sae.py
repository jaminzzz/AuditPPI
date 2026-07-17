#!/usr/bin/env python3
"""Cache pooled ESM-C + SAE features for PRING Human proteins.

Run in the E1 conda env because it needs the local Biohub transformers fork:

  PYTHONPATH=. /data/wmzhu/anaconda3/envs/E1/bin/python scripts/cache/cache_pring_human_esmc_sae.py

Output format matches the ppi_fingerprint pooled cache:
  - seq2idx
  - esmc_mean
  - esmc_sae_max
  - esmc_sae_mean

This PRING-specific cache additionally stores:
  - protein_ids
  - uniprotid2idx
  - unprotid2idx  (typo-compatible alias)
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
sys.path.insert(0, str(_ROOT))
from conf.paths import ESMC_MODEL, ESMC_SAE, PRING_ROOT, AUDIT, ROSETTA_SEQ_CACHE  # noqa: E402
from src.data.sequences import read_fasta  # noqa: E402

MODEL = ESMC_MODEL
SAE = ESMC_SAE
FASTA = PRING_ROOT / "human" / "human_simple.fasta"
OUT = AUDIT / "pring_participation" / "pring_human_esmc_sae_cache.pt"
RF2PPI_CACHE = ROSETTA_SEQ_CACHE
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache PRING Human pooled ESM-C/SAE features")
    p.add_argument("--fasta", type=Path, default=FASTA)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument(
        "--seed-cache",
        type=Path,
        default=RF2PPI_CACHE,
        help="existing pooled cache used to prefill PRING rows by UniProt id or exact sequence",
    )
    p.add_argument("--no-seed-cache", action="store_true", help="do not prefill from --seed-cache")
    p.add_argument(
        "--resume-cache",
        type=Path,
        default=None,
        help="existing PRING cache to resume from; default uses --out if it already exists",
    )
    p.add_argument(
        "--prefill-only",
        action="store_true",
        help="write the rows copied from existing caches and skip GPU inference for missing proteins",
    )
    p.add_argument(
        "--force-infer-all",
        action="store_true",
        help="ignore existing caches and recompute every protein with ESM-C/SAE",
    )
    p.add_argument("--limit", type=int, default=0, help="cap number of proteins for smoke tests")
    p.add_argument("--longest-first", action="store_true", help="apply --limit to longest proteins first")
    p.add_argument("--token-budget", type=int, default=3072)
    p.add_argument("--device-id", type=int, default=None, help="physical GPU id; default picks most free")
    p.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    p.add_argument("--sae-token-chunk", type=int, default=512)
    p.add_argument(
        "--store-all-hidden-states",
        action="store_true",
        help="store all transformer hidden states and select hidden_states[60]; default uses last_hidden_state",
    )
    return p.parse_args()


def torch_load_cache(path: Path) -> dict:
    import inspect

    import torch

    kwargs = {"map_location": "cpu", "weights_only": False}
    if "mmap" in inspect.signature(torch.load).parameters:
        kwargs["mmap"] = True
    return torch.load(path, **kwargs)


def cache_id_map(cache: dict) -> dict:
    for key in ("uniprotid2idx", "unprotid2idx", "protein_id2idx", "id2idx"):
        if key in cache:
            return cache[key]
    return {}


def lookup_cache_row(cache: dict, protein_id: str, sequence: str) -> int | None:
    id2idx = cache_id_map(cache)
    if protein_id in id2idx:
        return int(id2idx[protein_id])
    seq2idx = cache.get("seq2idx", {})
    if sequence in seq2idx:
        return int(seq2idx[sequence])
    return None


def clone_cache_row(cache: dict, row: int):
    return (
        cache["esmc_mean"][row].detach().cpu().clone(),
        cache["esmc_sae_max"][row].detach().cpu().clone(),
        cache["esmc_sae_mean"][row].detach().cpu().clone(),
    )


def save_cache(
    *,
    out: Path,
    ids: list[str],
    seqs: list[str],
    esmc_mean: list,
    sae_max: list,
    sae_mean: list,
    meta: dict,
) -> None:
    import torch

    uniprotid2idx = {pid: i for i, pid in enumerate(ids)}
    seq2idx = {}
    for i, seq in enumerate(seqs):
        seq2idx.setdefault(seq, i)
    cache = {
        "protein_ids": ids,
        "sequences": seqs,
        "uniprotid2idx": uniprotid2idx,
        "unprotid2idx": dict(uniprotid2idx),
        "seq2idx": seq2idx,
        "esmc_mean": torch.stack(esmc_mean),
        "esmc_sae_max": torch.stack(sae_max),
        "esmc_sae_mean": torch.stack(sae_mean),
        "meta": meta,
    }
    torch.save(cache, out)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch

    seq_map = read_fasta(args.fasta)
    ids = [pid for pid, seq in seq_map.items() if seq]
    if args.longest_first:
        ids = sorted(ids, key=lambda pid: len(seq_map[pid]), reverse=True)
    if args.limit:
        ids = ids[: args.limit]
    seqs = [seq_map[pid] for pid in ids]
    print(
        f"[data] fasta={args.fasta} proteins={len(ids)} "
        f"(limit={args.limit or 'none'}) max_residues={args.max_residues}",
        flush=True,
    )

    esmc_mean = [None] * len(seqs)
    sae_max = [None] * len(seqs)
    sae_mean = [None] * len(seqs)
    source = ["missing"] * len(seqs)

    copied_resume = 0
    copied_seed = 0
    resume_path = args.resume_cache
    if resume_path is None and args.out.exists():
        resume_path = args.out

    if not args.force_infer_all and resume_path is not None and resume_path.exists():
        print(f"[prefill] resume cache: {resume_path}", flush=True)
        resume_cache = torch_load_cache(resume_path)
        for i, (pid, seq) in enumerate(zip(ids, seqs)):
            row = lookup_cache_row(resume_cache, pid, seq)
            if row is None:
                continue
            esmc_mean[i], sae_max[i], sae_mean[i] = clone_cache_row(resume_cache, row)
            source[i] = "resume"
            copied_resume += 1
        print(f"[prefill] copied from resume: {copied_resume}/{len(ids)}", flush=True)

    seed_path = None if args.no_seed_cache else args.seed_cache
    if not args.force_infer_all and seed_path is not None and seed_path.exists():
        print(f"[prefill] seed cache: {seed_path}", flush=True)
        seed_cache = torch_load_cache(seed_path)
        for i, (pid, seq) in enumerate(zip(ids, seqs)):
            if esmc_mean[i] is not None:
                continue
            row = lookup_cache_row(seed_cache, pid, seq)
            if row is None:
                continue
            esmc_mean[i], sae_max[i], sae_mean[i] = clone_cache_row(seed_cache, row)
            source[i] = "seed"
            copied_seed += 1
        print(f"[prefill] copied from seed: {copied_seed}/{len(ids)}", flush=True)
    elif not args.force_infer_all and seed_path is not None:
        print(f"[prefill] seed cache missing: {seed_path}", flush=True)

    missing = [i for i, val in enumerate(esmc_mean) if val is None]
    print(f"[prefill] total copied={len(ids) - len(missing)} missing={len(missing)}", flush=True)

    if args.prefill_only:
        keep = [i for i, val in enumerate(esmc_mean) if val is not None]
        if not keep:
            raise RuntimeError("prefill-only requested, but no proteins were found in existing caches")
        out_ids = [ids[i] for i in keep]
        out_seqs = [seqs[i] for i in keep]
        meta = {
            "dataset": "PRING human",
            "fasta": str(args.fasta),
            "complete": False,
            "prefill_only": True,
            "seed_cache": str(seed_path) if seed_path is not None else None,
            "resume_cache": str(resume_path) if resume_path is not None else None,
            "n_pring_proteins_requested": len(ids),
            "n_cached_proteins": len(keep),
            "n_missing_proteins": len(missing),
            "missing_protein_ids": [ids[i] for i in missing],
            "source_counts": {name: source.count(name) for name in sorted(set(source))},
        }
        save_cache(
            out=args.out,
            ids=out_ids,
            seqs=out_seqs,
            esmc_mean=[esmc_mean[i] for i in keep],
            sae_max=[sae_max[i] for i in keep],
            sae_mean=[sae_mean[i] for i in keep],
            meta=meta,
        )
        size_mb = args.out.stat().st_size / 1e6
        print(f"[saved] {args.out} ({size_mb:.0f} MB) prefilled rows={len(keep)} missing={len(missing)}", flush=True)
        print("PREFILL_DONE", flush=True)
        return

    if not missing:
        meta = {
            "dataset": "PRING human",
            "fasta": str(args.fasta),
            "complete": True,
            "prefill_only": False,
            "seed_cache": str(seed_path) if seed_path is not None else None,
            "resume_cache": str(resume_path) if resume_path is not None else None,
            "model": "ESMC-6B",
            "layer": LAYER,
            "max_residues": args.max_residues,
            "n_proteins": len(ids),
            "source_counts": {name: source.count(name) for name in sorted(set(source))},
        }
        save_cache(
            out=args.out,
            ids=ids,
            seqs=seqs,
            esmc_mean=esmc_mean,
            sae_max=sae_max,
            sae_mean=sae_mean,
            meta=meta,
        )
        print(f"[saved] {args.out} complete from existing caches; no inference needed", flush=True)
        print("CACHE_DONE", flush=True)
        return

    device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"[device] CUDA_VISIBLE_DEVICES={device_id} (physical)", flush=True)

    from transformers import AutoModel, AutoTokenizer

    print("[load] ESMC-6B + SAE ...", flush=True)
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
            hc = h[start : start + chunk]
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
        trunc = [seq[: args.max_residues] for seq in batch_seqs]
        enc = tok(trunc, return_tensors="pt", padding=True)
        enc = {key: val.to("cuda") for key, val in enc.items()}
        out = model(**enc, output_hidden_states=args.store_all_hidden_states)
        h60 = out.hidden_states[LAYER] if args.store_all_hidden_states else out.last_hidden_state
        am = enc["attention_mask"].bool()
        esmc_mean, sae_max, sae_mean = [], [], []
        for i in range(h60.size(0)):
            mask = am[i].clone()
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() > 2:
                mask[idxs[0]] = False
                mask[idxs[-1]] = False
            hi = h60[i][mask].float()
            esmc_mean.append(hi.mean(0).half().cpu())
            fmax, fmean = sae_pool(hi)
            sae_max.append(fmax.half().cpu())
            sae_mean.append(fmean.half().cpu())
        return esmc_mean, sae_max, sae_mean

    order = sorted(missing, key=lambda i: len(seqs[i]))
    i = 0
    done = 0
    t0 = time.time()
    while i < len(order):
        seq_len = min(len(seqs[order[i]]), args.max_residues) + 2
        bs = max(1, args.token_budget // max(seq_len, 1))
        idxs = order[i : i + bs]
        i += bs
        em, smx, smn = run_batch([seqs[j] for j in idxs])
        for j, a, b, c in zip(idxs, em, smx, smn):
            esmc_mean[j] = a
            sae_max[j] = b
            sae_mean[j] = c
            source[j] = "inferred"
        done += len(idxs)
        if done % 512 < bs:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(missing)}  {rate:.1f} seq/s  elapsed {time.time()-t0:.0f}s", flush=True)

    meta = {
        "dataset": "PRING human",
        "fasta": str(args.fasta),
        "complete": True,
        "prefill_only": False,
        "seed_cache": str(seed_path) if seed_path is not None else None,
        "resume_cache": str(resume_path) if resume_path is not None else None,
        "model": "ESMC-6B",
        "layer": LAYER,
        "k": k,
        "dict": int(dict_dim),
        "act_dim": int(act_dim),
        "max_residues": args.max_residues,
        "sae_token_chunk": args.sae_token_chunk,
        "n_proteins": len(ids),
        "n_prefilled_from_resume": copied_resume,
        "n_prefilled_from_seed": copied_seed,
        "n_inferred": len(missing),
        "source_counts": {name: source.count(name) for name in sorted(set(source))},
    }
    save_cache(
        out=args.out,
        ids=ids,
        seqs=seqs,
        esmc_mean=esmc_mean,
        sae_max=sae_max,
        sae_mean=sae_mean,
        meta=meta,
    )
    fp = torch.stack(sae_max)
    dens = (fp > 0).float().mean().item()
    bits = (fp > 0).sum(1).float()
    size_mb = args.out.stat().st_size / 1e6
    print(f"[saved] {args.out} ({size_mb:.0f} MB) in {time.time()-t0:.0f}s", flush=True)
    print(
        f"[sanity] active density={dens*100:.2f}% bits/seq "
        f"min/med/max={int(bits.min())}/{int(bits.median())}/{int(bits.max())}",
        flush=True,
    )
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
