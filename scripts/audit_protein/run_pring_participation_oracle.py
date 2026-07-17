#!/usr/bin/env python3
"""Run PRING full-graph sequence -> participation prediction.

Definition:
  Use PRING's full Human reference graph only to define protein-level labels
  (full-graph degree / participation), then train and evaluate under PRING's
  protein-disjoint Human BFS/DFS/RANDOM_WALK splits.

Examples:
  PYTHONPATH=. /data/wmzhu/anaconda3/envs/E1/bin/python scripts/run_pring_participation_oracle.py \
    --method BFS --feature-kind all --cache-path data/pring_participation/pring_human_esmc_sae_cache.pt

  PYTHONPATH=. /data/wmzhu/anaconda3/envs/E1/bin/python scripts/run_pring_participation_oracle.py \
    --method all --feature-kind sae_max --model-kind xgboost --pair-eval human_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.participation import (
    DEFAULT_PRING_CACHE,
    FEATURE_KINDS,
    FORMAL_FEATURE_KINDS,
    METHODS,
    MODEL_KINDS,
    OUT_DIR,
    PAIR_EVALS,
    PRING_ROOT,
    run_pring_participation_oracle,
)


def _metric(res: dict, path: list[str]):
    cur = res
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="BFS", choices=[*METHODS, "all"])
    p.add_argument("--root", type=Path, default=PRING_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--self-loop-mode", choices=["drop", "once"], default="drop")
    p.add_argument("--keep-self-pairs", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kmer", type=int, default=2, choices=[1, 2])
    p.add_argument(
        "--feature-kind",
        default="sae_max",
        choices=[*FEATURE_KINDS, "pooled_sae", "binary_sae", "esmc", "all"],
        help="'all' runs the formal inputs: sae_max, binary, esmc_mean",
    )
    p.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_PRING_CACHE,
        help="pooled ESM-C/SAE cache with seq2idx and preferably uniprotid2idx",
    )
    p.add_argument("--model-kind", default="xgboost", choices=MODEL_KINDS)
    p.add_argument("--calibration", choices=["log_linear", "scale", "none"], default="log_linear")
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument(
        "--top-k-features",
        type=int,
        default=0,
        help="optional XGBoost top-k feature selection before the final regressor; useful for TabPFN",
    )
    p.add_argument("--no-importance", action="store_true", help="disable feature-importance output")
    p.add_argument("--importance-top-n", type=int, default=100)
    p.add_argument("--importance-estimators", type=int, default=200)
    p.add_argument("--tabpfn-estimators", type=int, default=8)
    p.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    p.add_argument("--pair-eval", choices=PAIR_EVALS, default="none")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    methods = METHODS if args.method == "all" else (args.method,)
    feature_kinds = FORMAL_FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)

    summary = {}
    for method in methods:
        summary[method] = {}
        for feature_kind in feature_kinds:
            res = run_pring_participation_oracle(
                method=method,
                root=args.root,
                out_dir=args.out_dir,
                self_loop_mode=args.self_loop_mode,
                drop_self_pairs=not args.keep_self_pairs,
                val_frac=args.val_frac,
                seed=args.seed,
                kmer=args.kmer,
                feature_kind=feature_kind,
                cache_path=args.cache_path,
                model_kind=args.model_kind,
                calibration=args.calibration,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                lr=args.lr,
                device=args.device,
                early_stopping_rounds=args.early_stopping_rounds,
                top_k_features=args.top_k_features,
                importance=not args.no_importance,
                importance_top_n=args.importance_top_n,
                importance_estimators=args.importance_estimators,
                tabpfn_estimators=args.tabpfn_estimators,
                tabpfn_subsample_samples=args.tabpfn_subsample_samples,
                pair_eval=args.pair_eval,
                write=not args.no_write,
            )
            summary[method][feature_kind] = {
                "test_spearman": _metric(res, ["node_metrics", "test", "spearman_pred_degree"]),
                "test_high_degree_auroc": _metric(res, ["node_metrics", "test", "high_degree_auroc"]),
                "n_train": _metric(res, ["split", "n_train_nodes"]),
                "n_val": _metric(res, ["split", "n_val_nodes"]),
                "n_test": _metric(res, ["split", "n_test_nodes"]),
                "missing_train_features": _metric(res, ["features", "missing_train_features"]),
                "missing_test_features": _metric(res, ["features", "missing_test_features"]),
                "pair_human_test_auroc": _metric(
                    res, ["pair_metrics", "human_test_ppi", "pred_min_t_auroc"]
                ),
                "pair_human_test_auprc": _metric(
                    res, ["pair_metrics", "human_test_ppi", "pred_min_t_auprc"]
                ),
            }

    print("\n=== PRING full-graph sequence -> participation summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"pring_human_{args.model_kind}_summary"
        (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[done] results in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
