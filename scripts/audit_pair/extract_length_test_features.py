#!/usr/bin/env python3
"""Re-extract a benchmark's TEST endpoint features at one truncation length.

This is the variable side of the test-length robustness sweep. For one family's
test split it collects the unique endpoint sequences straight from the benchmark
loader (``load_benchmark(...).seqs`` -- C3 lives in HDF5, bernett in CSV, so we
never touch on-disk column names), builds a :class:`ProteinManifest` keyed by the
*full* normalized sequence, and runs ESM-C at ``--max-residues L``.

Key alignment invariant: the manifest ``seq2idx`` is keyed by the full sequence,
while :func:`extract_esmc_features` truncates ``seq[:max_residues]`` only for the
forward pass. So the cache is ``full-sequence key -> truncated-length feature``.
At eval time ``pair_feature_rows`` looks up ``normalize_sequence(bench.seqs[pid])``
-- the same full sequence -- and gets the truncated feature. The kept-pair set is
therefore identical across every length (truncation changes feature *values*, not
which key is present).

Only ESM-C L60 is extracted (``--layers 60``); combined with the extractor's
early-exit at the deepest requested layer this skips the upper 20 blocks entirely
(less compute, lower GPU memory floor).

Product::

    results/audit_pair/length_robustness/{family}/test_features_max{L}.pt
      (auditppi_protein_features_v1; features = esmc_l60_{dense_mean,dense_max,
       sae_max,sae_binary})

Run (pins one GPU; skips an existing cache unless --overwrite)::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/extract_length_test_features.py \
        --family c3 --max-residues 400 --device-id 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from conf.paths import ESMC_MODEL, ESMC_SAE, RESULTS_PAIR

FAMILIES = ("c1", "c2", "c3", "bernett")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--split", default="test")
    p.add_argument("--max-residues", type=int, required=True)
    p.add_argument("--layers", default="60", help="ESM-C layers, comma-separated (default just 60).")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--token-budget", type=int, default=3072)
    p.add_argument("--sae-token-chunk", type=int, default=256)
    p.add_argument("--out-root", type=Path, default=RESULTS_PAIR / "length_robustness")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_root / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_features_max{args.max_residues}.pt"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}", flush=True)
        return

    if args.device.startswith("cuda") and args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        print(f"[device] CUDA_VISIBLE_DEVICES={args.device_id}", flush=True)

    # Import after CUDA_VISIBLE_DEVICES is set so the right GPU is bound.
    from src.data import pairs as D
    from src.features.extractors import extract_esmc_features, save_feature_cache
    from src.features.manifest import ProteinManifest
    from src.data.sequences import normalize_sequence

    bench = D.load_benchmark(f"{args.family}:{args.split}", attach_seqs=True)

    # Unique endpoint sequences over this split's pairs, keyed by the FULL
    # normalized sequence -- the same key pair_feature_rows uses at eval time.
    seq2idx: dict[str, int] = {}
    sequences: list[str] = []
    protein_ids: list[str] = []
    id2idx: dict[str, int] = {}
    for pid, raw in bench.seqs.items():
        seq = normalize_sequence(raw)
        if not seq:
            continue
        idx = seq2idx.get(seq)
        if idx is None:
            idx = len(sequences)
            seq2idx[seq] = idx
            sequences.append(seq)
            protein_ids.append(str(pid))
        id2idx[str(pid)] = idx

    manifest = ProteinManifest(
        protein_ids=protein_ids,
        sequences=sequences,
        id2idx=id2idx,
        seq2idx=seq2idx,
        sources=[f"load_benchmark({args.family}:{args.split})"],
    )
    n_trunc = sum(1 for s in sequences if len(s) > args.max_residues)
    print(f"[manifest] {args.family}:{args.split} unique_seqs={len(manifest)} "
          f"id_aliases={len(id2idx)} truncated@{args.max_residues}={n_trunc} "
          f"({100 * n_trunc / max(len(sequences), 1):.1f}%)", flush=True)

    layers = [int(v) for v in args.layers.split(",") if v.strip()]
    features, meta = extract_esmc_features(
        manifest,
        model_path=ESMC_MODEL,
        sae_path=ESMC_SAE,
        layers=layers,
        max_residues=args.max_residues,
        token_budget=args.token_budget,
        sae_token_chunk=args.sae_token_chunk,
        device=args.device,
        dtype=args.dtype,
    )
    meta["length_robustness"] = {
        "family": args.family,
        "split": args.split,
        "max_residues": args.max_residues,
        "n_unique_sequences": len(manifest),
        "n_truncated": int(n_trunc),
    }
    save_feature_cache(
        out_path, manifest=manifest, features=features,
        extractor_meta=meta, overwrite=args.overwrite,
    )
    print(f"[saved] {out_path} features={list(features)}", flush=True)


if __name__ == "__main__":
    main()
