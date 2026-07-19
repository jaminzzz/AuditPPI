#!/usr/bin/env python3
"""Train compact Top-K probes on cross-species SAE pair features."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np

from conf.model import DEFAULT_SEED
from conf.paths import ROOT
from src.experiments.results import dump_experiment
from src.runtime import setup_device
from src.eval.classification import probe_classification_metrics
from src.interpretability.pair_probe import (
    build_dense_sym_topk,
    evaluate_species,
    fit_logistic_probe,
    fit_tabpfn_probe,
    fit_xgb_probe,
    load_pair_embedding_split,
    predict_proba_chunked,
    read_feature_ranking,
    select_top_features,
)

DEFAULT_EMBEDDING_DIR = ROOT / "outputs/cross_species/esmc/reps/binary_thr0"
DEFAULT_RANKING = ROOT / "outputs/cross_species/esmc/tabpfn_topk/feature_ranking_binary_sym.csv"
DEFAULT_OUT = ROOT / "outputs/cross_species/esmc/tabpfn_topk"
DEFAULT_TESTS = ["ecoli", "fly", "mouse", "worm", "yeast"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--ranking-csv", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device-id", type=int, default=2)
    parser.add_argument("--top-k", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    parser.add_argument("--top-k-mode", choices=["sae-id", "flat"], default="sae-id")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tabpfn", "xgb", "logreg"],
        default=["tabpfn", "xgb", "logreg"],
    )
    parser.add_argument("--train-subsample", type=int, default=100000)
    parser.add_argument("--val-subsample", type=int, default=None)
    parser.add_argument("--test-subsample", type=int, default=None)
    parser.add_argument("--test-species", nargs="+", default=DEFAULT_TESTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--predict-batch-size", type=int, default=5000)
    parser.add_argument("--xgb-trees", type=int, default=1000)
    parser.add_argument("--xgb-depth", type=int, default=4)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-cpu", action="store_true")
    parser.add_argument("--tabpfn-n-estimators", type=int, default=4)
    parser.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    parser.add_argument("--ignore-pretraining-limits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_device(args.device_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranking = read_feature_ranking(args.ranking_csv)
    all_results = []

    print(f"[ranking] {args.ranking_csv}", flush=True)
    print(f"[load] train/val from {args.embedding_dir}", flush=True)
    train_a, train_b, train_y = load_pair_embedding_split(
        args.embedding_dir / "train_embeddings.pt", args.train_subsample, args.seed
    )
    val_a, val_b, val_y = load_pair_embedding_split(
        args.embedding_dir / "val_embeddings.pt", args.val_subsample, args.seed + 1
    )
    print(
        f"[data] train={len(train_y)} val={len(val_y)} pos_rate={train_y.mean():.3f}",
        flush=True,
    )

    for top_k in args.top_k:
        flat_features, feature_metadata = select_top_features(
            ranking, top_k, args.top_k_mode
        )
        print(
            f"[topk] mode={args.top_k_mode} k={top_k} input_dim={len(flat_features)}",
            flush=True,
        )
        train_x = build_dense_sym_topk(train_a, train_b, flat_features)
        val_x = build_dense_sym_topk(val_a, val_b, flat_features)
        feature_file = args.out_dir / f"selected_features_{args.top_k_mode}_k{top_k}.json"
        feature_file.write_text(json.dumps(feature_metadata, indent=2))

        for model_name in args.models:
            print(f"[fit] model={model_name} k={top_k}", flush=True)
            start = time.time()
            if model_name == "tabpfn":
                model = fit_tabpfn_probe(
                    train_x,
                    train_y,
                    n_estimators=args.tabpfn_n_estimators,
                    subsample_samples=args.tabpfn_subsample_samples,
                    ignore_limits=args.ignore_pretraining_limits,
                    seed=args.seed,
                )
            elif model_name == "xgb":
                model = fit_xgb_probe(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    trees=args.xgb_trees,
                    depth=args.xgb_depth,
                    learning_rate=args.xgb_lr,
                    seed=args.seed,
                    cpu=args.xgb_cpu,
                )
            else:
                model = fit_logistic_probe(train_x, train_y, seed=args.seed)

            val_probabilities = predict_proba_chunked(
                model, val_x, args.predict_batch_size
            )
            val_metrics = probe_classification_metrics(val_y, val_probabilities)
            species_metrics = {}
            for species in args.test_species:
                print(f"[test] model={model_name} k={top_k} species={species}", flush=True)
                species_metrics[species] = evaluate_species(
                    model,
                    species,
                    args.embedding_dir,
                    flat_features,
                    test_subsample=args.test_subsample,
                    seed=args.seed,
                    predict_batch_size=args.predict_batch_size,
                )
            mean_auroc = float(np.mean([item["auroc"] for item in species_metrics.values()]))
            mean_auprc = float(np.mean([item["auprc"] for item in species_metrics.values()]))
            result = {
                "dataset": "cross-species",
                "arch": model_name,
                "rep": "binary_thr0",
                "pair_mode": "sym",
                "top_k_mode": args.top_k_mode,
                "top_k": top_k,
                "input_dim": len(flat_features),
                "train_n": int(len(train_y)),
                "val_n": int(len(val_y)),
                "ranking_csv": str(args.ranking_csv),
                "selected_features": str(feature_file),
                "human_val": val_metrics,
                "species": species_metrics,
                "mean_species_auroc": mean_auroc,
                "mean_species_auprc": mean_auprc,
                "elapsed_sec": round(time.time() - start, 3),
            }
            dump_experiment(
                args.out_dir / f"{model_name}_{args.top_k_mode}_k{top_k}.json",
                task="analysis.cross_species_tabpfn_topk",
                dataset="cross_species",
                features=f"binary_sym_{args.top_k_mode}_k{top_k}",
                split="multi",
                model=model_name,
                seed=args.seed,
                payload=result,
                metrics={
                    "human_val_auroc": val_metrics["auroc"],
                    "human_val_auprc": val_metrics["auprc"],
                    "mean_species_auroc": mean_auroc,
                    "mean_species_auprc": mean_auprc,
                },
                hyperparameters={
                    "top_k_mode": args.top_k_mode,
                    "top_k": top_k,
                    "input_dim": len(flat_features),
                },
            )
            all_results.append(result)
            print(
                f"[result] {model_name} k={top_k} val_auroc={val_metrics['auroc']:.4f} "
                f"mean_species_auroc={mean_auroc:.4f} mean_species_auprc={mean_auprc:.4f}",
                flush=True,
            )
            del model, val_probabilities
            gc.collect()
        del train_x, val_x
        gc.collect()

    summary_path = args.out_dir / f"summary_{args.top_k_mode}.csv"
    fields = [
        "arch",
        "top_k_mode",
        "top_k",
        "input_dim",
        "train_n",
        "human_val_auroc",
        "human_val_auprc",
        "mean_species_auroc",
        "mean_species_auprc",
        "elapsed_sec",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in all_results:
            writer.writerow(
                {
                    "arch": result["arch"],
                    "top_k_mode": result["top_k_mode"],
                    "top_k": result["top_k"],
                    "input_dim": result["input_dim"],
                    "train_n": result["train_n"],
                    "human_val_auroc": result["human_val"]["auroc"],
                    "human_val_auprc": result["human_val"]["auprc"],
                    "mean_species_auroc": result["mean_species_auroc"],
                    "mean_species_auprc": result["mean_species_auprc"],
                    "elapsed_sec": result["elapsed_sec"],
                }
            )
    print(f"[summary] {summary_path}", flush=True)
    print("CROSS_SPECIES_TABPFN_TOPK_DONE", flush=True)


if __name__ == "__main__":
    main()
