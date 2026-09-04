#!/usr/bin/env python3
"""Top-feature activation coverage vs test truncation length.

Companion to ``run_length_robustness_sweep.py``. The sweep shows *whether* test
AUROC drops when the test sequence is truncated; this script explains *why* by
asking, for each truncation length ``L``, how often the frozen model's most
important features are still "seen" in the truncated features.

The frozen booster's ranking (``feature_ranking_sae_max_sym.csv``, produced by
``train_length_baseline_model.py``) ranks every ``sym = [A*B, |A-B|]`` flat
column by mean|TreeSHAP|. Each flat column maps to an SAE feature via
``flat % sae_dim`` (the ``sae_feature`` column). Taking the top-K flat rows and
de-duplicating on ``sae_feature`` gives the distinct SAE features the model
leans on. A feature is "active" in a sequence when its ``sae_max`` value is > 0.

For each (length ``L``, top-K):
  * ``activation_rate``  -- mean over (unique test seq x top-K feat) of (max>0),
    reported overall and split by whether the sequence is truncated at ``L``
    (``len(seq) > L``); truncation can only lower activation on truncated seqs.
  * ``retention_vs_full`` -- ``activation_rate(L) / activation_rate(full)``, the
    fraction of the full-length top-feature coverage that survives truncation.
  * ``mean_per_seq_coverage`` -- per sequence, fraction of the top-K features
    that are active; mean/median across unique sequences.

Product::

    results/audit_pair/length_robustness/{family}/top_feature_coverage.json
      (dump_experiment spine; payload.cells = one row per (length, top_k))

Run (CPU; caches must already exist)::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/analyze_length_feature_coverage.py --family c3
    $PY scripts/audit_pair/analyze_length_feature_coverage.py --family bernett
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.model import DEFAULT_BACKBONE, resolve_backbone_layer
from conf.paths import RESULTS_PAIR
from src.experiments.results import dump_experiment
from src.features.pairs import load_protein_feature_cache
from src.features.protein_cache import representation_matrix
from src.interp.pair_probe import read_feature_ranking

FAMILIES = ("c1", "c2", "c3", "bernett")
REP = "sae_max"
DEFAULT_LENGTHS = (100, 200, 400, 600, 800, 1000, 1500, 2046)
DEFAULT_K_GRID = (10, 25, 50, 100, 250, 500, 1000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    p.add_argument("--k-grid", type=int, nargs="+", default=list(DEFAULT_K_GRID))
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="SAE layer; defaults to the backbone's default layer (ESM-C L60).")
    p.add_argument("--out-root", type=Path, default=RESULTS_PAIR / "length_robustness")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def distinct_top_sae_features(ranking: list[dict], top_k: int) -> list[int]:
    """Distinct SAE feature ids among the top-K ranked flat columns (rank order)."""
    seen: set[int] = set()
    out: list[int] = []
    for row in ranking[:top_k]:
        fid = int(row["sae_feature"])
        if fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out


def coverage_for_length(
    active: np.ndarray, truncated_mask: np.ndarray
) -> dict:
    """Activation-coverage stats for one (length, feature-set) active matrix.

    ``active`` is a bool matrix (n_seq x n_feat) of (sae_max > 0) over the chosen
    top features; ``truncated_mask`` flags sequences with len > L.
    """
    n_seq, n_feat = active.shape
    overall = float(active.mean()) if active.size else 0.0
    per_seq = active.mean(axis=1) if n_feat else np.zeros(n_seq)
    n_trunc = int(truncated_mask.sum())
    n_kept = n_seq - n_trunc
    rate_trunc = float(active[truncated_mask].mean()) if n_trunc else None
    rate_kept = float(active[~truncated_mask].mean()) if n_kept else None
    return {
        "activation_rate": round(overall, 6),
        "activation_rate_truncated_seqs": round(rate_trunc, 6) if rate_trunc is not None else None,
        "activation_rate_untruncated_seqs": round(rate_kept, 6) if rate_kept is not None else None,
        "mean_per_seq_coverage": round(float(per_seq.mean()), 6),
        "median_per_seq_coverage": round(float(np.median(per_seq)), 6),
        "n_truncated_seqs": n_trunc,
        "n_untruncated_seqs": n_kept,
    }


def main() -> None:
    args = parse_args()
    layer = resolve_backbone_layer(args.backbone, args.layer)
    fam_dir = args.out_root / args.family
    ranking_csv = fam_dir / f"feature_ranking_{REP}_sym.csv"
    if not ranking_csv.exists():
        raise FileNotFoundError(
            f"missing ranking: {ranking_csv}\n"
            f"run train_length_baseline_model.py --family {args.family} first."
        )
    out_path = fam_dir / "top_feature_coverage.json"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {out_path}; pass --overwrite to replace it")

    ranking = read_feature_ranking(ranking_csv)
    top_feats_by_k = {k: distinct_top_sae_features(ranking, k) for k in sorted(args.k_grid)}
    max_full = max(args.lengths)
    print(f"[coverage] {args.family} {args.backbone}L{layer} rep={REP} "
          f"K={sorted(args.k_grid)} full_ref_L={max_full}", flush=True)

    # active_by_length[L][k] = bool matrix over unique test seqs x distinct top-K feats.
    # Cache the full sae_max slice once per length, then subselect per K.
    per_length: dict[int, dict] = {}
    for length in sorted(args.lengths):
        cache_path = fam_dir / f"test_features_max{length}.pt"
        if not cache_path.exists():
            print(f"[skip] missing cache L={length}: {cache_path}", flush=True)
            continue
        cache = load_protein_feature_cache(cache_path)
        matrix = representation_matrix(cache, REP, layer, args.backbone)  # (n_seq, sae_dim) fp16
        seqs = cache["sequences"]
        seq_len = np.asarray([len(s) for s in seqs], dtype=np.int64)
        truncated_mask = seq_len > length
        per_length[length] = {
            "matrix": np.asarray(matrix, dtype=np.float32),
            "truncated_mask": truncated_mask,
            "n_seq": len(seqs),
        }
        print(f"[L={length:>4}] unique_seqs={len(seqs)} truncated={int(truncated_mask.sum())}",
              flush=True)

    if not per_length:
        raise RuntimeError("no length caches found; run extract_length_test_features.py first.")

    # Full-length reference activation rate per K (denominator for retention).
    full_len = max(per_length)
    full_rate: dict[int, float] = {}
    for k, feats in top_feats_by_k.items():
        m = per_length[full_len]["matrix"][:, feats]
        full_rate[k] = float((m > 0).mean()) if m.size else 0.0

    cells = []
    for length in sorted(per_length):
        entry = per_length[length]
        matrix = entry["matrix"]
        tmask = entry["truncated_mask"]
        for k in sorted(args.k_grid):
            feats = top_feats_by_k[k]
            active = matrix[:, feats] > 0
            cov = coverage_for_length(active, tmask)
            denom = full_rate[k]
            cov["retention_vs_full"] = (
                round(cov["activation_rate"] / denom, 6) if denom > 0 else None
            )
            cov["max_residues"] = int(length)
            cov["top_k"] = int(k)
            cov["n_distinct_top_features"] = len(feats)
            cells.append(cov)

    # Console summary: activation_rate grid (rows=L, cols=K).
    ks = sorted(args.k_grid)
    print("\n[activation_rate]  rows=L  cols=K", flush=True)
    print("    L  " + "".join(f"{('K=' + str(k)):>10}" for k in ks), flush=True)
    for length in sorted(per_length):
        row = {c["top_k"]: c for c in cells if c["max_residues"] == length}
        print(f"{length:>5}  " + "".join(f"{row[k]['activation_rate']:>10.4f}" for k in ks),
              flush=True)
    print("\n[retention_vs_full]  rows=L  cols=K", flush=True)
    print("    L  " + "".join(f"{('K=' + str(k)):>10}" for k in ks), flush=True)
    for length in sorted(per_length):
        row = {c["top_k"]: c for c in cells if c["max_residues"] == length}
        print(f"{length:>5}  " + "".join(
            f"{(row[k]['retention_vs_full'] if row[k]['retention_vs_full'] is not None else float('nan')):>10.4f}"
            for k in ks), flush=True)

    payload = {
        "family": args.family,
        "rep": REP,
        "backbone": args.backbone,
        "layer": int(layer),
        "lengths": sorted(per_length),
        "k_grid": ks,
        "full_ref_length": int(full_len),
        "n_distinct_top_features": {str(k): len(top_feats_by_k[k]) for k in ks},
        "cells": cells,
        "ranking_csv": str(ranking_csv),
        "note": ("top-K distinct SAE features from the frozen booster's sym ranking; "
                 "activation = sae_max>0 over unique test sequences. retention is "
                 "activation_rate(L)/activation_rate(full). Truncation lowers activation "
                 "only on sequences with len>L."),
    }
    dump_experiment(
        out_path,
        task="length_feature_coverage",
        dataset=args.family,
        features=REP,
        split="test",
        model="xgb_frozen_ranking",
        seed=-1,
        payload=payload,
        metrics=None,
        hyperparameters={"rep": REP, "backbone": args.backbone, "layer": int(layer)},
    )
    print(f"\n[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
