#!/usr/bin/env python3
"""Minimal-interpretable-fingerprint K-sweep for Fig 1 panel e.

For one family (default C3) on ESM-C L60, sweeps the number of top ranked ``sym``
features kept (``K``) and, for each ``K``, fits the canonical XGBoost pair
classifier on exactly those ``K`` flat columns and scores test AUROC. Two
reps are swept independently -- continuous ``sae_max`` and the binary
``0/1`` fingerprint -- each consuming its own rep-tagged ranking CSV produced by
``build_minimal_fingerprint_ranking.py``. A ``full`` point (all 2*sae_dim
columns, ``cols=None``) is evaluated under the *same* protocol so the sweep and
its reference line are strictly comparable (never lifted from the aggregate main
table, which pools six families).

Every (K, rep) cell is fit over ``--seeds`` (default 42,43,44) and reported as
mean +/- std, matching the fingerprint-baseline aggregate convention.

Protocol per rep (matrix built once, sliced per K):
  A. Assemble train/val/test endpoints via the fingerprint baseline path
     (``_assemble`` -> seq2idx gather). Train is class-stratified subsampled to
     ``--train-subsample`` (C3 train < that, so a no-op there).
  B. Build the full ``sym = [A*B, |A-B|]`` matrix (2*sae_dim cols) for each split
     once. Reused across every K and seed by column slicing.
  C. For each K in the grid: cols = top-K flat features from the ranking; fit
     XGBoost per seed on the sliced train (official val for early stopping),
     score test. K = "full" uses the whole matrix (cols=None).

Cross-species special case
--------------------------
``cross_species`` has no official val. Matching ``run_cross_species_tabpfn_topk.py``:

  1. Load full ``cross_species:human_train``.
  2. Carve stratified val (``val_frac``, seed+1).
  3. Cap remaining train (``train_subsample``, seed).
  4. Default test = ``cross_species:human_test``; override with
     ``--test-benchmark cross_species:yeast`` (etc.) for zero-shot species.

Product (one JSON, multi-cell sweep spine)::

    results/audit_pair/minimal_fingerprint/{family}/sweep.json

Run (GPU 0 default; ranking CSVs must already exist)::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py --family c3 --device-id 0
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.model import DEFAULT_BACKBONE, DEFAULT_SEED, resolve_backbone_layer
from conf.paths import (
    CROSS_SPECIES_PAIR_INDEX_CACHES,
    CROSS_SPECIES_SAE_CACHE,
    RESULTS_PAIR,
)
from src.runtime import setup_device
from src.eval.classification import probe_classification_metrics
from src.experiments.results import dump_experiment
from src.features.pairs import load_protein_feature_cache, sym_features
from src.features.sampling import stratified_subsample
from src.interp.pair_probe import (
    carve_cross_species_train_val,
    predict_proba_chunked,
    read_feature_ranking,
    sae_dim_for_backbone,
    select_top_features,
)
from src.models.estimators.xgboost import fit_xgb

FAMILIES = ("c1", "c2", "c3", "bernett", "pring", "cross_species")
REPS = ("sae_max", "binary")
DEFAULT_K_GRID = (10, 25, 50, 100, 250, 500, 1000, 2000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--reps", nargs="+", choices=REPS, default=list(REPS))
    p.add_argument(
        "--method",
        choices=["BFS", "DFS", "RANDOM_WALK"],
        default="BFS",
        help="PRING human sampling method for train/val/test (ignored for non-PRING).",
    )
    p.add_argument(
        "--test-benchmark",
        default=None,
        help="Optional override for the TEST split only. Examples: "
             "'pring:yeast:test' (PRING zero-shot), "
             "'cross_species:yeast' (cross-species zero-shot). "
             "Train/val still use --family/--method. Ranking is reused from "
             "the human ranking dir unless --ranking-dir is set.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="cross_species only: fraction of human_train carved as val "
             "(stratified, seed+1). Matches run_cross_species_tabpfn_topk.",
    )
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="SAE layer; defaults to the backbone's default layer.")
    p.add_argument("--k-grid", type=int, nargs="+", default=list(DEFAULT_K_GRID),
                   help="top-K values to sweep (flat columns, 0..2*sae_dim).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ranking-dir", type=Path, default=None,
                   help="dir holding feature_ranking_{rep}_sym.csv "
                        "(default: RESULTS_PAIR/minimal_fingerprint/{ranking_tag})")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override product dir "
                        "(default: RESULTS_PAIR/minimal_fingerprint/{family_tag})")
    p.add_argument("--train-subsample", type=int, default=100000,
                   help="class-stratified train cap (C3 train < this, so a no-op there).")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--predict-batch-size", type=int, default=20000)
    # Eval XGB hyperparameters (canonical fingerprint-baseline classifier).
    p.add_argument("--xgb-trees", type=int, default=1000)
    p.add_argument("--xgb-depth", type=int, default=4)
    p.add_argument("--xgb-lr", type=float, default=0.05)
    p.add_argument("--xgb-cpu", action="store_true")
    return p.parse_args()


def family_tag(family: str, method: str = "BFS", test_benchmark: str | None = None) -> str:
    """Product-dir tag. BFS keeps bare 'pring' (legacy); other methods and
    species-test overrides get distinct tags so they never clobber each other."""
    if test_benchmark is not None:
        # e.g. pring:yeast:test -> pring_yeast
        #      cross_species:yeast -> cross_species_yeast
        parts = test_benchmark.split(":")
        if parts[0] == "cross_species":
            sp = parts[1] if len(parts) > 1 else "test"
            return f"cross_species_{sp}"
        if parts[0] == "pring":
            sp = parts[1] if len(parts) > 1 else "test"
            return f"pring_{sp}"
        return test_benchmark.replace(":", "_")
    if family != "pring" or method == "BFS":
        return family
    return f"pring_{method.lower()}"


def ranking_tag(family: str, method: str = "BFS") -> str:
    """Ranking is always built on human train/val of the given method (no species)."""
    if family != "pring" or method == "BFS":
        return family
    return f"pring_{method.lower()}"


def assemble_split(
    family: str, split: str, rep: str, *, backbone: str, layer: int,
    method: str = "BFS", name_override: str | None = None,
):
    """train/val/test endpoints for a family via the fingerprint baseline path."""
    from src.ppi_fingerprint.baseline import _assemble

    if name_override is not None:
        name = name_override
    elif family == "pring":
        # PRING eval-name grammar: pring:species:split:method
        name = f"pring:human:{split}:{method}"
    elif family == "cross_species":
        # Default ID test graph; train/val go through assemble_cross_species_train_val.
        name = "cross_species:human_test" if split == "test" else f"cross_species:{split}"
    else:
        name = f"{family}:{split}"
    _, A, B, y, _ = _assemble(name, rep, backbone=backbone, layer=layer)
    return A, B, y


def assemble_cross_species_train_val(
    rep: str, *, backbone: str, layer: int,
    train_subsample: int, val_frac: float, seed: int,
):
    """TabPFN-aligned cross_species protocol: carve val first, then cap train.

    Delegates to :func:`carve_cross_species_train_val`, which selects train/val
    ROW INDICES from the cheap pair-index labels FIRST and gathers only the
    selected endpoint rows at the protein cache's native dtype -- never the full
    421k-pair float32 matrix. Order (val carve seed+1, then train cap seed) and
    selection are identical to ``run_cross_species_tabpfn_topk``.
    """
    protein_cache = load_protein_feature_cache(CROSS_SPECIES_SAE_CACHE)
    return carve_cross_species_train_val(
        CROSS_SPECIES_PAIR_INDEX_CACHES["human_train"], protein_cache,
        rep=rep, backbone=backbone, layer=layer,
        train_subsample=train_subsample, val_frac=val_frac, seed=seed,
    )


def eval_cell(
    Xtr, ytr, Xva, yva, Xte, yte,
    *, seeds, trees, depth, lr, cpu, predict_batch_size,
) -> dict:
    """Fit XGB on the given (already column-sliced) matrices over seeds; test metrics."""
    aurocs, auprcs = [], []
    for seed in seeds:
        clf = fit_xgb(
            Xtr, ytr, Xva, yva,
            trees=trees, depth=depth, lr=lr, seed=seed, cpu=cpu, verbose=False,
        )
        proba = predict_proba_chunked(clf, Xte, predict_batch_size)
        m = probe_classification_metrics(yte, proba)
        aurocs.append(m["auroc"])
        auprcs.append(m["auprc"])
        del clf, proba
        gc.collect()
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "auroc_seeds": [float(v) for v in aurocs],
        "seeds": list(seeds),
    }


def sweep_rep(
    rep: str, ranking_dir: Path, family: str, *,
    backbone: str, layer: int, sae_dim: int, k_grid, seeds,
    train_subsample, trees, depth, lr, cpu, predict_batch_size,
    method: str = "BFS",
    test_benchmark: str | None = None,
    val_frac: float = 0.1,
) -> list[dict]:
    ranking_csv = ranking_dir / f"feature_ranking_{rep}_sym.csv"
    if not ranking_csv.exists():
        raise FileNotFoundError(
            f"missing ranking for rep={rep}: {ranking_csv}\n"
            f"run build_minimal_fingerprint_ranking.py --rep {rep} first."
        )
    ranking = read_feature_ranking(ranking_csv)

    import torch

    if family == "cross_species":
        train_a, train_b, train_y, val_a, val_b, val_y = assemble_cross_species_train_val(
            rep, backbone=backbone, layer=layer,
            train_subsample=train_subsample, val_frac=val_frac, seed=DEFAULT_SEED,
        )
    else:
        train_a, train_b, train_y = assemble_split(
            family, "train", rep, backbone=backbone, layer=layer, method=method,
        )
        sub = stratified_subsample(train_y, train_subsample, DEFAULT_SEED)
        if sub is not None:
            ti = torch.as_tensor(sub, dtype=torch.long)
            train_a = train_a.index_select(0, ti).contiguous()
            train_b = train_b.index_select(0, ti).contiguous()
            train_y = train_y[sub]
        val_a, val_b, val_y = assemble_split(
            family, "val", rep, backbone=backbone, layer=layer, method=method,
        )

    # test_benchmark overrides ONLY the test split (species transfer).
    test_override = test_benchmark if test_benchmark is not None else None
    test_a, test_b, test_y = assemble_split(
        family, "test", rep, backbone=backbone, layer=layer, method=method,
        name_override=test_override,
    )
    ytr = np.asarray(train_y)
    yva = np.asarray(val_y)
    yte = np.asarray(test_y)
    print(
        f"[data:{rep}] train={len(ytr)} val={len(yva)} test={len(yte)}"
        + (f" test_bench={test_benchmark}" if test_benchmark else ""),
        flush=True,
    )

    # Build the full sym matrix once per split; slice columns per K.
    full_tr = sym_features(train_a, train_b)
    full_va = sym_features(val_a, val_b)
    full_te = sym_features(test_a, test_b)
    del train_a, train_b, val_a, val_b, test_a, test_b
    gc.collect()
    feature_dim = full_tr.shape[1]
    print(f"[matrix:{rep}] full sym feature_dim={feature_dim}", flush=True)

    rows = []
    # K grid (capped to feature_dim) + a same-protocol full-dim reference point.
    grid = [k for k in k_grid if k < feature_dim]
    for k in grid:
        flat_features, _ = select_top_features(ranking, k, "flat", sae_dim=sae_dim)
        cols = np.asarray(flat_features, dtype=np.int64)
        t0 = time.time()
        cell = eval_cell(
            full_tr[:, cols], ytr, full_va[:, cols], yva, full_te[:, cols], yte,
            seeds=seeds, trees=trees, depth=depth, lr=lr, cpu=cpu,
            predict_batch_size=predict_batch_size,
        )
        cell.update(rep=rep, k=int(k), is_full=False, input_dim=int(len(cols)))
        rows.append(cell)
        print(f"  [{rep}] K={k:<5d} test AUROC={cell['auroc_mean']:.4f}"
              f"+/-{cell['auroc_std']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # full-dim reference under the identical protocol (cols=None -> whole matrix)
    t0 = time.time()
    cell = eval_cell(
        full_tr, ytr, full_va, yva, full_te, yte,
        seeds=seeds, trees=trees, depth=depth, lr=lr, cpu=cpu,
        predict_batch_size=predict_batch_size,
    )
    cell.update(rep=rep, k=int(feature_dim), is_full=True, input_dim=int(feature_dim))
    rows.append(cell)
    print(f"  [{rep}] K=full ({feature_dim}) test AUROC={cell['auroc_mean']:.4f}"
          f"+/-{cell['auroc_std']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    del full_tr, full_va, full_te
    gc.collect()
    return rows


def main() -> None:
    args = parse_args()
    setup_device(args.device_id)
    layer = resolve_backbone_layer(args.backbone, args.layer)
    sae_dim = sae_dim_for_backbone(args.backbone)

    tag = family_tag(args.family, args.method, args.test_benchmark)
    rtag = ranking_tag(args.family, args.method)
    # Species transfer reuses the human ranking for the chosen method (default BFS).
    ranking_dir = args.ranking_dir or (RESULTS_PAIR / "minimal_fingerprint" / rtag)
    out_dir = args.out_dir or (RESULTS_PAIR / "minimal_fingerprint" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[plan] family={args.family} method={args.method} tag={tag} "
        f"reps={args.reps} {args.backbone}L{layer} "
        f"sae_dim={sae_dim} k_grid={args.k_grid} seeds={args.seeds} "
        f"ranking_dir={ranking_dir} out={out_dir}"
        + (f" test_benchmark={args.test_benchmark}" if args.test_benchmark else ""),
        flush=True,
    )

    all_rows = []
    for rep in args.reps:
        all_rows.extend(sweep_rep(
            rep, ranking_dir, args.family,
            backbone=args.backbone, layer=layer, sae_dim=sae_dim,
            k_grid=args.k_grid, seeds=args.seeds,
            train_subsample=args.train_subsample,
            trees=args.xgb_trees, depth=args.xgb_depth, lr=args.xgb_lr,
            cpu=args.xgb_cpu, predict_batch_size=args.predict_batch_size,
            method=args.method,
            test_benchmark=args.test_benchmark,
            val_frac=args.val_frac,
        ))

    payload = {
        "dataset": args.family,
        "method": args.method,
        "tag": tag,
        "test_benchmark": args.test_benchmark,
        "val_frac": args.val_frac if args.family == "cross_species" else None,
        "backbone": args.backbone,
        "layer": layer,
        "sae_dim": sae_dim,
        "pair_mode": "sym",
        "feature_dim": 2 * sae_dim,
        "k_grid": list(args.k_grid),
        "seeds": list(args.seeds),
        "ranking_source": f"{rtag}_val_mean_abs_shap (per-rep, seed {DEFAULT_SEED})",
        "ranking_dir": str(ranking_dir),
        "cells": all_rows,
    }
    out_json = out_dir / "sweep.json"
    dump_experiment(
        out_json,
        task="analysis.minimal_fingerprint_sweep",
        dataset=tag,
        features="+".join(args.reps),
        split="multi",
        model="xgb",
        seed=DEFAULT_SEED,
        payload=payload,
        hyperparameters={"xgb_trees": args.xgb_trees, "xgb_depth": args.xgb_depth,
                         "xgb_lr": args.xgb_lr, "top_k_mode": "flat",
                         "method": args.method, "test_benchmark": args.test_benchmark,
                         "val_frac": args.val_frac if args.family == "cross_species" else None},
    )
    print(f"[done] wrote {out_json} ({len(all_rows)} cells)", flush=True)


if __name__ == "__main__":
    main()
