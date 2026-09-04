#!/usr/bin/env python3
"""C1/C2/C3 TabPFN Top-K SAE probe audit (recomputes each family's own ranking).

Reproduces the ``backups/results/audit_pair/tabpfn`` protocol for the RAPPPID
leakage levels, but re-derives the ``binary``/``sym`` feature ranking **per
family** instead of reusing the single C3-derived ranking the old (deleted)
SAE_PPI repo shipped.

Pipeline (one family per run):

  A. Assemble train/val/test endpoints from the family's v1 protein cache via
     the fingerprint baseline's on-the-fly path (``_assemble`` -> seq2idx gather;
     ``binary`` rep, ESM-C L60 default axis). Train is class-stratified
     subsampled (default 100k); val/test kept whole.
  B. Recompute the ranking: fit XGBoost on the full 32768-dim sym feature matrix
     (``[A*B, |A-B|]``, 2x16384), score every feature by mean|TreeSHAP| on val,
     and write ``feature_ranking_binary_sym.csv`` in the backup's column format.
  C. TabPFN Top-K=200 (sae-id mode): expand each of the 200 top SAE ids into its
     ``AND(a*b)`` + ``|a-b|`` pair columns (400-dim), fit TabPFN, score val+test.

Products land under a per-family subdir so the three runs never overwrite each
other::

    results/audit_pair/tabpfn/{family}/tabpfn_topk/
        feature_ranking_binary_sym.csv
        selected_features_sae-id_k200.json
        tabpfn_sae-id_k200.json
        summary_sae-id.csv

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/run_clevel_tabpfn_topk.py --family c3
    $PY scripts/audit_pair/run_clevel_tabpfn_topk.py --family c1
    $PY scripts/audit_pair/run_clevel_tabpfn_topk.py --family c2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.model import DEFAULT_BACKBONE, DEFAULT_SEED, resolve_backbone_layer
from conf.paths import RESULTS_PAIR
from src.runtime import setup_device
from src.eval.classification import probe_classification_metrics
from src.experiments.results import dump_experiment
from src.features.pairs import sym_features
from src.features.sampling import stratified_subsample
from src.interp.pair_probe import (
    BLOCK_ABSDIFF,
    BLOCK_PRODUCT,
    build_dense_sym_topk,
    fit_tabpfn_probe,
    predict_proba_chunked,
    read_feature_ranking,
    sae_dim_for_backbone,
    select_top_features,
)
from src.models.estimators.xgboost import fit_xgb

FAMILIES = ("c1", "c2", "c3", "bernett", "pring")


def _assemble_name(family: str, split: str, *, pring_method: str) -> str:
    """Colon-spec benchmark name for ``_assemble``.

    C-levels and Bernett use ``<family>:<split>``; PRING trains/evaluates on the
    human graph of a fixed sampling method (``pring:human:<split>:<method>``).
    """
    if family == "pring":
        return f"pring:human:{split}:{pring_method}"
    return f"{family}:{split}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--rep", choices=["binary", "sae_max"], default="binary")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="SAE layer; defaults to the backbone's default layer.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override product dir (default: RESULTS_PAIR/tabpfn/{family}/tabpfn_topk)")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--top-k-mode", choices=["sae-id", "flat"], default="sae-id")
    p.add_argument("--train-subsample", type=int, default=100000)
    p.add_argument("--pring-method", default="BFS",
                   help="PRING human-graph sampling method (only used when --family pring)")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--predict-batch-size", type=int, default=5000)
    # Ranking XGB hyperparameters (match the fingerprint-baseline classifier).
    p.add_argument("--xgb-trees", type=int, default=1000)
    p.add_argument("--xgb-depth", type=int, default=4)
    p.add_argument("--xgb-lr", type=float, default=0.05)
    p.add_argument("--xgb-cpu", action="store_true")
    # TabPFN probe knobs (match the cross-species top-k runner).
    p.add_argument("--tabpfn-n-estimators", type=int, default=4)
    p.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    p.add_argument("--ignore-pretraining-limits", action="store_true", default=True)
    return p.parse_args()


def assemble_split(family: str, split: str, rep: str, *, backbone: str, layer: int,
                   pring_method: str = "BFS"):
    """train/val/test endpoints for a benchmark family via the fingerprint path."""
    from src.ppi_fingerprint.baseline import _assemble

    name = _assemble_name(family, split, pring_method=pring_method)
    _, A, B, y, _ = _assemble(name, rep, backbone=backbone, layer=layer)
    return A, B, y


def compute_ranking(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    sae_dim: int,
    trees: int,
    depth: int,
    lr: float,
    seed: int,
    cpu: bool,
) -> list[dict]:
    """Fit XGB on full sym features, rank by mean|TreeSHAP| on the val split.

    Returns backup-format ranking rows sorted by descending mean|shap|:
    ``rank, flat_feature, sae_feature, block, rank_score, xgb_importance,
    <split>_mean_abs_shap, <split>_mean_signed_shap``.
    """
    import xgboost as xgb

    feature_dim = train_x.shape[1]  # 2 * sae_dim
    clf = fit_xgb(
        train_x, train_y, val_x, val_y,
        trees=trees, depth=depth, lr=lr, seed=seed, cpu=cpu, verbose=50,
    )
    booster = clf.get_booster()

    # TreeSHAP on val. pred_contribs returns (n, feature_dim + 1); last col is the
    # bias term. booster was re-homed to CPU by fit_xgb, so this runs on CPU.
    dval = xgb.DMatrix(val_x)
    contribs = booster.predict(dval, pred_contribs=True)
    shap = contribs[:, :feature_dim]  # drop bias column
    mean_abs = np.abs(shap).mean(axis=0)
    mean_signed = shap.mean(axis=0)
    del contribs, shap, dval
    gc.collect()

    # XGB gain per flat feature (booster keys like "f123").
    gain = np.zeros(feature_dim, dtype=np.float64)
    for key, value in booster.get_score(importance_type="gain").items():
        if key.startswith("f"):
            idx = int(key[1:])
            if 0 <= idx < feature_dim:
                gain[idx] = float(value)

    order = np.argsort(mean_abs)[::-1]
    rows: list[dict] = []
    for rank, flat in enumerate(order, start=1):
        flat = int(flat)
        block = BLOCK_PRODUCT if flat < sae_dim else BLOCK_ABSDIFF
        rows.append({
            "rank": rank,
            "flat_feature": flat,
            "sae_feature": flat % sae_dim,
            "block": block,
            "rank_score": float(mean_abs[flat]),
            "xgb_importance": float(gain[flat]),
            "mean_abs_shap": float(mean_abs[flat]),
            "mean_signed_shap": float(mean_signed[flat]),
        })
    return rows


def write_ranking_csv(path: Path, rows: list[dict], split_tag: str) -> None:
    """Write the ranking in the backup column layout (split-tagged shap columns)."""
    fields = [
        "rank", "flat_feature", "sae_feature", "block", "rank_score",
        "xgb_importance",
        f"{split_tag}_mean_abs_shap", f"{split_tag}_mean_signed_shap",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for r in rows:
            writer.writerow([
                r["rank"], r["flat_feature"], r["sae_feature"], r["block"],
                r["rank_score"], r["xgb_importance"],
                r["mean_abs_shap"], r["mean_signed_shap"],
            ])


def main() -> None:
    args = parse_args()
    setup_device(args.device_id)
    layer = resolve_backbone_layer(args.backbone, args.layer)
    sae_dim = sae_dim_for_backbone(args.backbone)

    # PRING evaluates on the human graph of a fixed sampling method; keep its
    # products in a method-tagged dir and its shap columns tagged the same way so
    # they never collide with a future non-BFS run.
    family_tag = (f"pring_human_{args.pring_method.lower()}"
                  if args.family == "pring" else args.family)
    out_dir = args.out_dir or (RESULTS_PAIR / "tabpfn" / family_tag / "tabpfn_topk")
    out_dir.mkdir(parents=True, exist_ok=True)
    split_tag = f"{family_tag}_val"

    print(
        f"[plan] family={args.family} tag={family_tag} rep={args.rep} "
        f"{args.backbone}L{layer} sae_dim={sae_dim} top_k={args.top_k} out={out_dir}",
        flush=True,
    )

    # --- A. assemble endpoints -------------------------------------------------
    import torch

    train_a, train_b, train_y = assemble_split(
        args.family, "train", args.rep, backbone=args.backbone, layer=layer,
        pring_method=args.pring_method,
    )
    sub = stratified_subsample(train_y, args.train_subsample, args.seed)
    if sub is not None:
        ti = torch.as_tensor(sub, dtype=torch.long)
        train_a = train_a.index_select(0, ti).contiguous()
        train_b = train_b.index_select(0, ti).contiguous()
        train_y = train_y[sub]
    val_a, val_b, val_y = assemble_split(
        args.family, "val", args.rep, backbone=args.backbone, layer=layer,
        pring_method=args.pring_method,
    )
    test_a, test_b, test_y = assemble_split(
        args.family, "test", args.rep, backbone=args.backbone, layer=layer,
        pring_method=args.pring_method,
    )
    print(
        f"[data] train={len(train_y)} (pos={train_y.mean():.3f}) "
        f"val={len(val_y)} test={len(test_y)}",
        flush=True,
    )

    # --- B. recompute ranking (full 32768-dim sym features) --------------------
    print("[rank] fitting XGBoost on full sym features + TreeSHAP on val", flush=True)
    t0 = time.time()
    full_train_x = sym_features(train_a, train_b)
    full_val_x = sym_features(val_a, val_b)
    ranking_rows = compute_ranking(
        full_train_x, train_y, full_val_x, val_y,
        sae_dim=sae_dim,
        trees=args.xgb_trees, depth=args.xgb_depth, lr=args.xgb_lr,
        seed=args.seed, cpu=args.xgb_cpu,
    )
    del full_train_x, full_val_x
    gc.collect()
    ranking_csv = out_dir / "feature_ranking_binary_sym.csv"
    write_ranking_csv(ranking_csv, ranking_rows, split_tag)
    print(
        f"[rank] wrote {ranking_csv} ({len(ranking_rows)} features, "
        f"{time.time() - t0:.1f}s)",
        flush=True,
    )

    # --- C. TabPFN Top-K probe -------------------------------------------------
    ranking = read_feature_ranking(ranking_csv)
    flat_features, feature_metadata = select_top_features(
        ranking, args.top_k, args.top_k_mode, sae_dim=sae_dim
    )
    (out_dir / f"selected_features_{args.top_k_mode}_k{args.top_k}.json").write_text(
        json.dumps(feature_metadata, indent=2)
    )
    print(
        f"[topk] mode={args.top_k_mode} k={args.top_k} input_dim={len(flat_features)}",
        flush=True,
    )

    train_x = build_dense_sym_topk(train_a, train_b, flat_features, sae_dim=sae_dim)
    val_x = build_dense_sym_topk(val_a, val_b, flat_features, sae_dim=sae_dim)
    test_x = build_dense_sym_topk(test_a, test_b, flat_features, sae_dim=sae_dim)
    del train_a, train_b, val_a, val_b, test_a, test_b
    gc.collect()

    print("[fit] TabPFN", flush=True)
    t1 = time.time()
    model = fit_tabpfn_probe(
        train_x, train_y,
        n_estimators=args.tabpfn_n_estimators,
        subsample_samples=args.tabpfn_subsample_samples,
        ignore_limits=args.ignore_pretraining_limits,
        seed=args.seed,
    )
    val_proba = predict_proba_chunked(model, val_x, args.predict_batch_size)
    test_proba = predict_proba_chunked(model, test_x, args.predict_batch_size)
    val_metrics = probe_classification_metrics(val_y, val_proba)
    test_metrics = probe_classification_metrics(test_y, test_proba)
    elapsed = round(time.time() - t1, 3)

    result = {
        "dataset": args.family,
        "arch": "tabpfn",
        "rep": args.rep,
        "backbone": args.backbone,
        "layer": layer,
        "sae_dim": sae_dim,
        "pair_mode": "sym",
        "top_k_mode": args.top_k_mode,
        "top_k": args.top_k,
        "input_dim": len(flat_features),
        "train_n": int(len(train_y)),
        "val_n": int(len(val_y)),
        "test_n": int(len(test_y)),
        "ranking_csv": str(ranking_csv),
        "ranking_source": f"{split_tag}_mean_abs_shap",
        "val": val_metrics,
        "test": test_metrics,
        "elapsed_sec": elapsed,
    }
    result_path = out_dir / f"tabpfn_{args.top_k_mode}_k{args.top_k}.json"
    dump_experiment(
        result_path,
        task="analysis.clevel_tabpfn_topk",
        dataset=args.family,
        features=f"{args.rep}_sym_{args.top_k_mode}_k{args.top_k}",
        split="multi",
        model="tabpfn",
        seed=args.seed,
        payload=result,
        metrics={
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
        },
        hyperparameters={
            "top_k_mode": args.top_k_mode,
            "top_k": args.top_k,
            "input_dim": len(flat_features),
            "backbone": args.backbone,
            "layer": layer,
        },
    )

    summary_path = out_dir / f"summary_{args.top_k_mode}.csv"
    fields = [
        "arch", "top_k_mode", "top_k", "input_dim", "train_n", "val_n", "test_n",
        "val_auroc", "val_auprc", "test_auroc", "test_auprc",
        "test_f1", "test_acc", "test_ece", "elapsed_sec",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "arch": "tabpfn",
            "top_k_mode": args.top_k_mode,
            "top_k": args.top_k,
            "input_dim": len(flat_features),
            "train_n": int(len(train_y)),
            "val_n": int(len(val_y)),
            "test_n": int(len(test_y)),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_acc": test_metrics["acc"],
            "test_ece": test_metrics["ece"],
            "elapsed_sec": elapsed,
        })

    print(
        f"[result] {args.family} tabpfn k={args.top_k} "
        f"val_auroc={val_metrics['auroc']:.4f} val_auprc={val_metrics['auprc']:.4f} "
        f"test_auroc={test_metrics['auroc']:.4f} test_auprc={test_metrics['auprc']:.4f} "
        f"({elapsed}s)",
        flush=True,
    )
    print(f"[done] {result_path}", flush=True)
    print(f"[summary] {summary_path}", flush=True)
    print("CLEVEL_TABPFN_TOPK_DONE", flush=True)


if __name__ == "__main__":
    main()
