#!/usr/bin/env python3
"""Cross-species TabPFN Top-K SAE probe audit (in-distribution + zero-shot).

Aligned with ``run_clevel_tabpfn_topk.py``: TabPFN only, K=200 (sae-id mode).
Trains on the cross-species human_train graph (a stratified val is carved from
it in-memory, since no native val CSV exists), then scores TWO eval groups
separately:

  * ``human_test`` -- in-distribution held-out human graph (same species as
    train), reported on its own so it is not blended into the zero-shot number.
  * the 5 held-out species (ecoli/fly/mouse/worm/yeast) -- zero-shot transfer,
    each scored individually plus a macro mean.

Feature ranking is recomputed here (XGB + TreeSHAP on human_train/val, like the
c-level runner) rather than borrowed from C3, so the downstream attention audit
selects the same Top-K SAE ids this cross-species probe used. The Top-200 SAE
ids expand into 400 pair columns (``AND(a*b)`` + ``|a-b|``). Endpoints are
gathered from the shared cross-species v1 protein cache via the per-graph
pair-index caches.

Products::

    results/audit_pair/tabpfn/cross_species_tabpfn_topk/
        feature_ranking_binary_sym.csv
        selected_features_sae-id_k200.json
        tabpfn_sae-id_k200.json
        summary_sae-id.csv

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/run_cross_species_tabpfn_topk.py
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
from conf.paths import (
    CROSS_SPECIES_PAIR_INDEX_CACHES,
    CROSS_SPECIES_SAE_CACHE,
    CROSS_SPECIES_TABPFN_RANKING,
    RESULTS_PAIR,
)
from src.experiments.results import dump_experiment
from src.runtime import setup_device
from src.eval.classification import probe_classification_metrics
from src.features.pairs import load_protein_feature_cache, sym_features
from src.features.sampling import stratified_subsample
from src.interp.pair_probe import (
    build_dense_sym_topk,
    compute_sym_shap_ranking,
    evaluate_species_v1,
    fit_tabpfn_probe,
    materialize_pair_split,
    predict_proba_chunked,
    read_feature_ranking,
    sae_dim_for_backbone,
    select_top_features,
    write_feature_ranking,
)

DEFAULT_OUT = RESULTS_PAIR / "tabpfn" / "cross_species_tabpfn_topk"
# In-distribution held-out human graph, kept separate from the zero-shot group.
IN_DIST_TEST = "human_test"
# Zero-shot transfer graphs (each scored individually + macro-averaged).
ZERO_SHOT_SPECIES = ["ecoli", "fly", "mouse", "worm", "yeast"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ranking-csv", type=Path, default=None,
                        help="override ranking output path; default is the "
                        "cross-species tabpfn_topk dir. The ranking is recomputed "
                        "(XGB + TreeSHAP on human_train/val), not borrowed from C3.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rep", choices=["binary", "sae_max"], default="binary")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--layer", type=int, default=None,
                        help="SAE layer; defaults to the backbone's default layer.")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="fraction of human_train carved off as val (stratified).")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-k-mode", choices=["sae-id", "flat"], default="sae-id")
    parser.add_argument("--train-subsample", type=int, default=100000)
    parser.add_argument("--test-subsample", type=int, default=None)
    parser.add_argument("--zero-shot-species", nargs="+", default=ZERO_SHOT_SPECIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--predict-batch-size", type=int, default=5000)
    # Ranking XGB hyperparameters (match run_clevel_tabpfn_topk.py).
    parser.add_argument("--xgb-trees", type=int, default=1000)
    parser.add_argument("--xgb-depth", type=int, default=4)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-cpu", action="store_true")
    parser.add_argument("--tabpfn-n-estimators", type=int, default=4)
    parser.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    parser.add_argument("--ignore-pretraining-limits", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_device(args.device_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layer = resolve_backbone_layer(args.backbone, args.layer)
    sae_dim = sae_dim_for_backbone(args.backbone)
    ranking_csv = args.ranking_csv or CROSS_SPECIES_TABPFN_RANKING
    ranking_csv.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[load] cross-species human_train via pair-index cache "
        f"({args.rep} {args.backbone}L{layer} sae_dim={sae_dim})",
        flush=True,
    )
    protein_cache = load_protein_feature_cache(CROSS_SPECIES_SAE_CACHE)

    # --- train/val: carve a stratified val from human_train (no native val CSV) --
    import torch

    full_a, full_b, full_y = materialize_pair_split(
        CROSS_SPECIES_PAIR_INDEX_CACHES["human_train"], protein_cache,
        rep=args.rep, backbone=args.backbone, layer=layer, max_rows=None,
        seed=args.seed,
    )
    val_idx = stratified_subsample(full_y, max(1, int(len(full_y) * args.val_frac)), args.seed + 1)
    val_mask = np.zeros(len(full_y), dtype=bool)
    val_mask[val_idx] = True
    train_rows = torch.as_tensor(np.flatnonzero(~val_mask), dtype=torch.long)
    val_rows = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.long)
    # Cap the train side after the val carve (stratified), keeping val disjoint.
    if args.train_subsample is not None and args.train_subsample < len(train_rows):
        sub = stratified_subsample(full_y[train_rows.numpy()], args.train_subsample, args.seed)
        train_rows = train_rows.index_select(0, torch.as_tensor(sub, dtype=torch.long))
    train_a = full_a.index_select(0, train_rows).contiguous()
    train_b = full_b.index_select(0, train_rows).contiguous()
    train_y = full_y[train_rows.numpy()]
    val_a = full_a.index_select(0, val_rows).contiguous()
    val_b = full_b.index_select(0, val_rows).contiguous()
    val_y = full_y[val_rows.numpy()]
    del full_a, full_b
    gc.collect()
    print(
        f"[data] train={len(train_y)} val={len(val_y)} pos_rate={train_y.mean():.3f}",
        flush=True,
    )

    # --- recompute ranking (full 32768-dim sym features, XGB + TreeSHAP) ------
    # Self-produced from human_train/val (not borrowed from C3) so the audit that
    # follows selects the same Top-K SAE ids this cross-species probe used.
    print("[rank] fitting XGBoost on full sym features + TreeSHAP on val", flush=True)
    t_rank = time.time()
    full_train_x = sym_features(train_a, train_b)
    full_val_x = sym_features(val_a, val_b)
    ranking_rows = compute_sym_shap_ranking(
        full_train_x, train_y, full_val_x, val_y,
        sae_dim=sae_dim,
        trees=args.xgb_trees, depth=args.xgb_depth, lr=args.xgb_lr,
        seed=args.seed, cpu=args.xgb_cpu,
    )
    del full_train_x, full_val_x
    gc.collect()
    write_feature_ranking(ranking_csv, ranking_rows, "human_val")
    ranking = read_feature_ranking(ranking_csv)
    print(
        f"[rank] wrote {ranking_csv} ({len(ranking_rows)} features, "
        f"{time.time() - t_rank:.1f}s)",
        flush=True,
    )

    # --- Top-K feature selection + train matrices -----------------------------
    flat_features, feature_metadata = select_top_features(
        ranking, args.top_k, args.top_k_mode, sae_dim=sae_dim
    )
    print(
        f"[topk] mode={args.top_k_mode} k={args.top_k} input_dim={len(flat_features)} "
        f"sae_dim={sae_dim}",
        flush=True,
    )
    train_x = build_dense_sym_topk(train_a, train_b, flat_features, sae_dim=sae_dim)
    val_x = build_dense_sym_topk(val_a, val_b, flat_features, sae_dim=sae_dim)
    del train_a, train_b, val_a, val_b
    gc.collect()
    feature_file = args.out_dir / f"selected_features_{args.top_k_mode}_k{args.top_k}.json"
    feature_file.write_text(json.dumps(feature_metadata, indent=2))

    # --- fit TabPFN -----------------------------------------------------------
    print(f"[fit] tabpfn k={args.top_k}", flush=True)
    start = time.time()
    model = fit_tabpfn_probe(
        train_x,
        train_y,
        n_estimators=args.tabpfn_n_estimators,
        subsample_samples=args.tabpfn_subsample_samples,
        ignore_limits=args.ignore_pretraining_limits,
        seed=args.seed,
    )
    val_probabilities = predict_proba_chunked(model, val_x, args.predict_batch_size)
    val_metrics = probe_classification_metrics(val_y, val_probabilities)
    del val_x, val_probabilities
    gc.collect()

    # --- in-distribution eval: human_test -------------------------------------
    print(f"[test] in-distribution split={IN_DIST_TEST}", flush=True)
    human_test_metrics = evaluate_species_v1(
        model,
        CROSS_SPECIES_PAIR_INDEX_CACHES[IN_DIST_TEST],
        protein_cache,
        flat_features,
        rep=args.rep,
        backbone=args.backbone,
        layer=layer,
        test_subsample=args.test_subsample,
        seed=args.seed,
        predict_batch_size=args.predict_batch_size,
        sae_dim=sae_dim,
    )

    # --- zero-shot eval: 5 held-out species -----------------------------------
    species_metrics = {}
    for species in args.zero_shot_species:
        print(f"[test] zero-shot species={species}", flush=True)
        species_metrics[species] = evaluate_species_v1(
            model,
            CROSS_SPECIES_PAIR_INDEX_CACHES[species],
            protein_cache,
            flat_features,
            rep=args.rep,
            backbone=args.backbone,
            layer=layer,
            test_subsample=args.test_subsample,
            seed=args.seed,
            predict_batch_size=args.predict_batch_size,
            sae_dim=sae_dim,
        )
    mean_auroc = float(np.mean([item["auroc"] for item in species_metrics.values()]))
    mean_auprc = float(np.mean([item["auprc"] for item in species_metrics.values()]))
    elapsed = round(time.time() - start, 3)

    result = {
        "dataset": "cross_species",
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
        "ranking_csv": str(ranking_csv),
        "ranking_source": "human_val_mean_abs_shap",
        "selected_features": str(feature_file),
        "human_val": val_metrics,
        "human_test": human_test_metrics,
        "species": species_metrics,
        "mean_species_auroc": mean_auroc,
        "mean_species_auprc": mean_auprc,
        "elapsed_sec": elapsed,
    }
    result_path = args.out_dir / f"tabpfn_{args.top_k_mode}_k{args.top_k}.json"
    dump_experiment(
        result_path,
        task="analysis.cross_species_tabpfn_topk",
        dataset="cross_species",
        features=f"{args.rep}_sym_{args.top_k_mode}_k{args.top_k}",
        split="multi",
        model="tabpfn",
        seed=args.seed,
        payload=result,
        metrics={
            "human_val_auroc": val_metrics["auroc"],
            "human_val_auprc": val_metrics["auprc"],
            "human_test_auroc": human_test_metrics["auroc"],
            "human_test_auprc": human_test_metrics["auprc"],
            "mean_species_auroc": mean_auroc,
            "mean_species_auprc": mean_auprc,
        },
        hyperparameters={
            "top_k_mode": args.top_k_mode,
            "top_k": args.top_k,
            "input_dim": len(flat_features),
            "backbone": args.backbone,
            "layer": layer,
        },
    )

    # --- per-eval summary CSV (one row per eval graph) ------------------------
    summary_path = args.out_dir / f"summary_{args.top_k_mode}.csv"
    fields = [
        "eval", "group", "arch", "top_k_mode", "top_k", "input_dim", "train_n",
        "n", "pos_rate", "auroc", "auprc", "f1", "acc", "ece",
    ]

    def _row(eval_name: str, group: str, m: dict) -> dict:
        return {
            "eval": eval_name, "group": group, "arch": "tabpfn",
            "top_k_mode": args.top_k_mode, "top_k": args.top_k,
            "input_dim": len(flat_features), "train_n": int(len(train_y)),
            "n": m.get("n"), "pos_rate": m.get("pos_rate"),
            "auroc": m["auroc"], "auprc": m["auprc"],
            "f1": m.get("f1"), "acc": m.get("acc"), "ece": m.get("ece"),
        }

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_row("human_val", "val", val_metrics))
        writer.writerow(_row("human_test", "in_distribution", human_test_metrics))
        for species, m in species_metrics.items():
            writer.writerow(_row(species, "zero_shot", m))

    print(
        f"[result] tabpfn k={args.top_k} "
        f"human_val_auroc={val_metrics['auroc']:.4f} "
        f"human_test_auroc={human_test_metrics['auroc']:.4f} "
        f"mean_species_auroc={mean_auroc:.4f} mean_species_auprc={mean_auprc:.4f} "
        f"({elapsed}s)",
        flush=True,
    )
    print(f"[done] {result_path}", flush=True)
    print(f"[summary] {summary_path}", flush=True)
    print("CROSS_SPECIES_TABPFN_TOPK_DONE", flush=True)


if __name__ == "__main__":
    main()
