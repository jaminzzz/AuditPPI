#!/usr/bin/env python3
"""FlashPPI zero-training **ContactHead** baseline over a PPI family.

Unlike :mod:`eval_flashppi_baseline` (which scores pairs with the CLIP retrieval
similarity ``q(A)·k(B)`` -- FlashPPI's *stage-1 recall* head), this replicates
the score FlashPPI actually reports in ``predict_proteome.py`` /
``predict_cross_proteome.py``: the **stage-2 ContactHead** applied to the two
proteins' residue embeddings, with the pair score being the maximum contact
probability over the (masked) residue-residue contact map::

    contact_logits, valid = contact_head(res(A), res(B), maskA, maskB)   # (B,L1,L2)
    score(A,B) = max( sigmoid(contact_logits) masked by valid )

The ContactHead is a cross-attention module that concatenates both proteins'
residues (+ segment embeddings), runs a 2-layer transformer so residues of A
attend to residues of B, then reads a multi-head Q/K contact map -- so it is a
genuinely *pairwise* operation and cannot be reduced to a per-protein pooled
vector. That is why this baseline runs the real published checkpoint rather than
reusing the pooled per-protein feature cache.

Two-level compute (mirrors FlashPPI's own two-stage inference):

1. **Backbone, once per unique sequence.** Every unique endpoint sequence in the
   eval benchmark is run through the gLM2 backbone once (truncated to
   ``--max-length``, default 512, matching the training config) to obtain
   residue embeddings ``[L, plm_dim]``; these are cached in RAM (fp16) so a
   sequence appearing in many pairs is encoded only once.
2. **ContactHead, per pair.** Each eval pair gathers its two residue tensors and
   is scored through ``predict_contacts``. Because benchmark pairs are undirected
   and the ContactHead is order-sensitive (asymmetric Q/K + segment embeddings),
   the score is symmetrised: ``0.5 * ( s(A,B) + s(B,A) )`` -- consistent with the
   CLIP baseline's undirected symmetrisation.

There is no head to train and no train/val split (the checkpoint is frozen), so
``--seed`` is a label only and results are deterministic. Only the summary/cell
JSONs are written; no residue embeddings are persisted to disk.

Results land under ``results/main/baselines/flashppi/{family}/seed_{S}/`` with
``feat_tag=flashppi_glm2_contact`` (distinct from the CLIP baseline's
``flashppi_glm2_clip``).

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/baseline/runs/eval_flashppi_contact_baseline.py --family c3 --device-id 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from conf.model import DEFAULT_SEED  # noqa: E402
from conf.paths import BASELINES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    from scripts.baseline.runs._protocol import FAMILIES

    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument(
        "--model",
        type=Path,
        default=BASELINES / "FlashPPI" / "FlashPPI-weights",
    )
    p.add_argument("--max-length", type=int, default=512,
                   help="residue truncation (matches FlashPPI training max_len)")
    p.add_argument("--encode-batch-size", type=int, default=16,
                   help="unique-sequence batch for the gLM2 backbone pass")
    p.add_argument("--pair-batch-size", type=int, default=16,
                   help="pair batch for the ContactHead pass")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp16")
    p.add_argument("--no-symmetrize", action="store_true",
                   help="score only s(A,B) instead of 0.5*(s(A,B)+s(B,A))")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="label only; the ContactHead scorer is deterministic")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda"):
        if args.device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
            print(f"[device] CUDA_VISIBLE_DEVICES={args.device_id}", flush=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from src.data.pairs import load_benchmark
    from scripts.baseline.runs._protocol import family_evals, run_scoring_baseline

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = args.device
    device_type = torch.device(device).type
    use_amp = device_type == "cuda" and args.dtype != "fp32"

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch_dtype,
    ).to(device).eval()
    plm_dim = int(model.config.plm_dim)
    print(f"[model] FlashPPI gLM2 backbone + ContactHead loaded (plm_dim={plm_dim})", flush=True)

    def encode_unique(sequences: list[str]) -> dict[str, "torch.Tensor"]:
        """Run the gLM2 backbone once per unique (truncated) sequence.

        Returns seq -> residue embeddings ``[L, plm_dim]`` (fp16, on CPU), trimmed
        to the true (post-truncation) length so ContactHead sees no padding.
        """
        uniq = sorted({s[: args.max_length] for s in sequences})
        cache: dict[str, "torch.Tensor"] = {}
        start_time = time.time()
        with torch.inference_mode():
            for start in range(0, len(uniq), args.encode_batch_size):
                batch = uniq[start:start + args.encode_batch_size]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_length,
                )
                input_ids = inputs["input_ids"].to(device)
                attn = inputs["attention_mask"].to(device)
                with torch.autocast(device_type=device_type, dtype=torch_dtype, enabled=use_amp):
                    res = model.encode_protein(input_ids, attn)  # (b, L, D)
                res = res.to(torch.float16).cpu()
                lens = attn.sum(dim=1).tolist()
                for j, seq in enumerate(batch):
                    cache[seq] = res[j, : int(lens[j]), :].clone()
                if (start // args.encode_batch_size + 1) % 20 == 0 or start + args.encode_batch_size >= len(uniq):
                    rate = min(start + args.encode_batch_size, len(uniq)) / max(time.time() - start_time, 1e-6)
                    print(f"[encode] {min(start + args.encode_batch_size, len(uniq))}/{len(uniq)} "
                          f"unique seqs {rate:.1f}/s", flush=True)
        return cache

    def _pad_stack(res_list: list["torch.Tensor"]):
        """Pad a list of ``[Li, D]`` tensors to ``[B, Lmax, D]`` + bool mask."""
        lens = [r.shape[0] for r in res_list]
        lmax = max(lens)
        b = len(res_list)
        out = torch.zeros((b, lmax, plm_dim), dtype=torch_dtype)
        mask = torch.zeros((b, lmax), dtype=torch.long)
        for i, r in enumerate(res_list):
            li = r.shape[0]
            out[i, :li, :] = r.to(torch_dtype)
            mask[i, :li] = 1
        return out, mask

    def _contact_scores(a_res: list, b_res: list) -> "np.ndarray":
        """Batched ContactHead scoring: max masked sigmoid over the contact map."""
        e1, m1 = _pad_stack(a_res)
        e2, m2 = _pad_stack(b_res)
        e1, m1 = e1.to(device), m1.to(device)
        e2, m2 = e2.to(device), m2.to(device)
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch_dtype, enabled=use_amp):
            logits, valid = model.predict_contacts(e1, e2, m1, m2)  # (b,L1,L2)
            cmap = torch.sigmoid(logits.float())
            cmap = cmap.masked_fill(~valid, 0.0)
            scores = cmap.flatten(1).max(dim=-1).values
        return scores.float().cpu().numpy()

    def score_eval(name: str):
        bench = load_benchmark(name, attach_seqs=True)
        pairs = bench.pairs
        labels = np.asarray(bench.labels, dtype=np.int64)

        # residue-embedding cache over the benchmark's unique sequences
        all_seqs = [s for s in bench.seqs.values() if s is not None]
        res_cache = encode_unique(all_seqs)

        def res_of(pid: str):
            seq = bench.seqs.get(pid)
            if seq is None:
                return None
            return res_cache.get(seq[: args.max_length])

        rows_a, rows_b, ys, kept = [], [], [], []
        for i, ((a, b), y) in enumerate(zip(pairs, labels)):
            ra, rb = res_of(a), res_of(b)
            if ra is None or rb is None:
                continue
            rows_a.append(ra)
            rows_b.append(rb)
            ys.append(int(y))
            kept.append(i)
        n_total = len(pairs)
        n_scored = len(kept)
        if n_scored == 0:
            raise RuntimeError(f"no scorable pairs for {name!r}")

        scores = np.empty(n_scored, dtype=np.float64)
        start_time = time.time()
        for start in range(0, n_scored, args.pair_batch_size):
            end = min(start + args.pair_batch_size, n_scored)
            a_batch = rows_a[start:end]
            b_batch = rows_b[start:end]
            s_ab = _contact_scores(a_batch, b_batch)
            if args.no_symmetrize:
                batch_scores = s_ab
            else:
                s_ba = _contact_scores(b_batch, a_batch)
                batch_scores = 0.5 * (s_ab + s_ba)
            scores[start:end] = batch_scores.astype(np.float64)
            if (start // args.pair_batch_size + 1) % 25 == 0 or end == n_scored:
                rate = end / max(time.time() - start_time, 1e-6)
                print(f"[contact] {end}/{n_scored} pairs {rate:.1f}/s", flush=True)

        return scores, np.asarray(ys, dtype=np.int64), n_total, n_scored

    print(f"[plan] baseline=flashppi(contact) family={args.family} "
          f"symmetrize={not args.no_symmetrize} max_len={args.max_length} "
          f"evals={family_evals(args.family)}", flush=True)
    run_scoring_baseline(
        baseline="flashppi",
        family=args.family,
        score_eval=score_eval,
        feat_tag="flashppi_glm2_contact",
        seed=args.seed,
        hparams={
            "backbone": "gLM2-650M", "head": "ContactHead",
            "contact_dim": 1280, "contact_heads": 8, "contact_depth": 2,
            "score": "max sigmoid(contact_map)"
                     + ("" if args.no_symmetrize else ", symmetrised 0.5*(AB+BA)"),
            "max_length": args.max_length, "symmetrize": not args.no_symmetrize,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
