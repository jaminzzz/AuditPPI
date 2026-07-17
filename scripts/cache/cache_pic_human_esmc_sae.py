#!/usr/bin/env python3
"""Cache pooled ESM-C + SAE features for PIC human essentiality proteins.

PIC (Protein Importance Calculator) predicts human essential proteins from
single sequences. Its data uses Ensembl protein ids (ENSP...), which do not
map to the UniProt ids in our existing pooled caches, so proteins are matched
to cached features by *exact sequence* (truncated to --max-residues) only.

Run in the primenet conda env (needs the local transformers fork for ESM-C):

  PYTHONPATH=. /data/wmzhu/anaconda3/envs/primenet/bin/python \
      scripts/cache/cache_pic_human_esmc_sae.py --prefill-only

Output format matches the ppi_fingerprint pooled cache:
  - seq2idx / sequences
  - esmc_mean / esmc_sae_max / esmc_sae_mean
PIC-specific rows additionally stored:
  - protein_ids (ENSP), id2idx, labels (essentiality 0/1)
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
sys.path.insert(0, str(_ROOT))
from conf.paths import (  # noqa: E402
    ESMC_MODEL, ESMC_SAE, PIC_DATA, AUDIT,
    ROSETTA_SEQ_CACHE, CROSS_SPECIES_SEQ_CACHE, BERNETT_SEQ_CACHE,
)

MODEL = ESMC_MODEL
SAE = ESMC_SAE
OUT = AUDIT / "pic_essentiality" / "pic_human_esmc_sae_cache.pt"

# Existing pooled caches to prefill from, all ESMC-6B / layer60 / k64 / dict16384.
SEED_CACHES = (
    ROSETTA_SEQ_CACHE,
    CROSS_SPECIES_SEQ_CACHE,
    BERNETT_SEQ_CACHE,
    AUDIT / "pring_participation" / "pring_human_esmc_sae_cache.pt",
)

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
    p = argparse.ArgumentParser(description="Cache PIC human pooled ESM-C/SAE features")
    p.add_argument("--data", type=Path, default=PIC_DATA)
    p.add_argument("--label-col", type=str, default="human")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    p.add_argument(
        "--prefill-only",
        action="store_true",
        help="only keep proteins matched by sequence in existing caches; skip GPU inference",
    )
    p.add_argument("--force-infer-all", action="store_true", help="ignore caches; recompute every protein")
    p.add_argument("--limit", type=int, default=0, help="cap number of proteins for smoke tests")
    p.add_argument("--token-budget", type=int, default=3072)
    p.add_argument("--device-id", type=int, default=None, help="physical GPU id; default picks most free")
    p.add_argument("--sae-token-chunk", type=int, default=512)
    p.add_argument(
        "--store-all-hidden-states",
        action="store_true",
        help="store all hidden states and select hidden_states[60]; default uses last_hidden_state",
    )
    return p.parse_args()


def torch_load_cache(path: Path) -> dict:
    import inspect

    import torch

    kwargs = {"map_location": "cpu", "weights_only": False}
    if "mmap" in inspect.signature(torch.load).parameters:
        kwargs["mmap"] = True
    return torch.load(str(path), **kwargs)


def clone_cache_row(cache: dict, row: int):
    return (
        cache["esmc_mean"][row].detach().cpu().clone(),
        cache["esmc_sae_max"][row].detach().cpu().clone(),
        cache["esmc_sae_mean"][row].detach().cpu().clone(),
    )


def load_pic(data_path: Path, label_col: str) -> tuple[list[str], list[str], list[int]]:
    with data_path.open("rb") as f:
        df = pickle.load(f)
    ids = [str(x) for x in df["ID"].tolist()]
    seqs = [str(s).upper() for s in df["sequence"].tolist()]
    labels = [int(v) for v in df[label_col].tolist()]
    return ids, seqs, labels


def save_cache(
    *,
    out: Path,
    ids: list[str],
    seqs: list[str],
    labels: list[int],
    esmc_mean: list,
    sae_max: list,
    sae_mean: list,
    meta: dict,
) -> None:
    import torch

    id2idx = {pid: i for i, pid in enumerate(ids)}
    seq2idx: dict[str, int] = {}
    for i, seq in enumerate(seqs):
        seq2idx.setdefault(seq, i)
    cache = {
        "protein_ids": ids,
        "sequences": seqs,
        "labels": torch.tensor(labels, dtype=torch.int8),
        "id2idx": id2idx,
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

    ids, seqs_full, labels = load_pic(args.data, args.label_col)
    # PIC truncates to truncation_seq_length=1024 tokens (~1022 residues + BOS/EOS).
    seqs = [s[: args.max_residues] for s in seqs_full]
    if args.limit:
        ids, seqs, labels = ids[: args.limit], seqs[: args.limit], labels[: args.limit]
    pos = sum(labels)
    print(
        f"[data] {args.data.name} proteins={len(ids)} label='{args.label_col}' "
        f"pos={pos} neg={len(ids) - pos} pos_rate={pos / max(1, len(ids)):.4f} "
        f"max_residues={args.max_residues}",
        flush=True,
    )

    esmc_mean = [None] * len(seqs)
    sae_max = [None] * len(seqs)
    sae_mean = [None] * len(seqs)
    source = ["missing"] * len(seqs)

    # Resume from our own output if present.
    resume_paths = []
    if not args.force_infer_all and args.out.exists():
        resume_paths.append(args.out)
    seed_paths = [] if args.force_infer_all else [p for p in SEED_CACHES if p.exists()]

    for path in resume_paths + seed_paths:
        cache = torch_load_cache(path)
        seq2idx = cache.get("seq2idx", {})
        hit = 0
        for i, seq in enumerate(seqs):
            if esmc_mean[i] is not None:
                continue
            row = seq2idx.get(seq)
            if row is None:
                continue
            esmc_mean[i], sae_max[i], sae_mean[i] = clone_cache_row(cache, int(row))
            source[i] = "resume" if path in resume_paths else "seed"
            hit += 1
        print(f"[prefill] {path.name}: matched {hit} by sequence", flush=True)

    missing = [i for i, v in enumerate(esmc_mean) if v is None]
    matched = len(ids) - len(missing)
    print(f"[prefill] total matched={matched}/{len(ids)} missing={len(missing)}", flush=True)

    def meta_base(complete: bool, prefill_only: bool) -> dict:
        return {
            "dataset": "PIC human essentiality",
            "data_path": str(args.data),
            "label_col": args.label_col,
            "model": "ESMC-6B",
            "layer": LAYER,
            "k": 64,
            "dict": 16384,
            "act_dim": 2560,
            "max_residues": args.max_residues,
            "complete": complete,
            "prefill_only": prefill_only,
            "id_type": "ensembl_protein (ENSP)",
            "match_strategy": "exact truncated sequence",
            "n_pic_proteins_requested": len(ids),
            "seed_caches": [str(p) for p in seed_paths],
            "source_counts": {name: source.count(name) for name in sorted(set(source))},
        }

    if args.prefill_only or not missing:
        keep = [i for i, v in enumerate(esmc_mean) if v is not None]
        if not keep:
            raise RuntimeError("no PIC proteins matched any existing cache by sequence")
        meta = meta_base(complete=not bool(missing), prefill_only=args.prefill_only)
        meta["n_cached_proteins"] = len(keep)
        meta["n_missing_proteins"] = len(missing)
        meta["missing_protein_ids"] = [ids[i] for i in missing][:5000]
        save_cache(
            out=args.out,
            ids=[ids[i] for i in keep],
            seqs=[seqs[i] for i in keep],
            labels=[labels[i] for i in keep],
            esmc_mean=[esmc_mean[i] for i in keep],
            sae_max=[sae_max[i] for i in keep],
            sae_mean=[sae_mean[i] for i in keep],
            meta=meta,
        )
        size_mb = args.out.stat().st_size / 1e6
        kept_pos = sum(labels[i] for i in keep)
        print(
            f"[saved] {args.out} ({size_mb:.0f} MB) rows={len(keep)} "
            f"pos={kept_pos} pos_rate={kept_pos / len(keep):.4f} missing={len(missing)}",
            flush=True,
        )
        print("PREFILL_DONE" if args.prefill_only and missing else "CACHE_DONE", flush=True)
        return

    # --- GPU inference for the missing proteins ---
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
        enc = tok(batch_seqs, return_tensors="pt", padding=True)
        enc = {key: val.to("cuda") for key, val in enc.items()}
        out = model(**enc, output_hidden_states=args.store_all_hidden_states)
        h60 = out.hidden_states[LAYER] if args.store_all_hidden_states else out.last_hidden_state
        am = enc["attention_mask"].bool()
        em, smx, smn = [], [], []
        for i in range(h60.size(0)):
            mask = am[i].clone()
            idxs = mask.nonzero(as_tuple=True)[0]
            if idxs.numel() > 2:
                mask[idxs[0]] = False
                mask[idxs[-1]] = False
            hi = h60[i][mask].float()
            em.append(hi.mean(0).half().cpu())
            fmax, fmean = sae_pool(hi)
            smx.append(fmax.half().cpu())
            smn.append(fmean.half().cpu())
        return em, smx, smn

    order = sorted(missing, key=lambda i: len(seqs[i]))
    i = 0
    done = 0
    t0 = time.time()
    while i < len(order):
        seq_len = max(len(seqs[order[i]]), 1) + 2
        bs = max(1, args.token_budget // seq_len)
        idxs = order[i : i + bs]
        i += bs
        em, smx, smn = run_batch([seqs[j] for j in idxs])
        for j, a, b, c in zip(idxs, em, smx, smn):
            esmc_mean[j], sae_max[j], sae_mean[j] = a, b, c
            source[j] = "inferred"
        done += len(idxs)
        if done % 512 < bs:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(missing)}  {rate:.1f} seq/s  elapsed {time.time()-t0:.0f}s", flush=True)

    meta = meta_base(complete=True, prefill_only=False)
    meta["n_inferred"] = len(missing)
    save_cache(
        out=args.out,
        ids=ids,
        seqs=seqs,
        labels=labels,
        esmc_mean=esmc_mean,
        sae_max=sae_max,
        sae_mean=sae_mean,
        meta=meta,
    )
    fp = torch.stack(sae_max)
    dens = (fp > 0).float().mean().item()
    size_mb = args.out.stat().st_size / 1e6
    print(f"[saved] {args.out} ({size_mb:.0f} MB) in {time.time()-t0:.0f}s active_density={dens*100:.2f}%", flush=True)
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
