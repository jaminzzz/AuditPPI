#!/usr/bin/env python3
"""Run PRING cross-species participation generalization.

Definition:
  Train the sequence -> full-graph participation oracle on ALL human PRING
  proteins (an inner stratified val split drives early stopping + degree
  calibration), then zero-shot test on each held-out PRING species' full graph
  (yeast / ecoli / arath). Each species' degree labels come from its OWN
  {species}_graph.pkl, so absolute degrees are not comparable across species --
  the reported cross-species signal is the rank-based test Spearman and
  high-degree AUROC.

Examples:
  # CPU smoke, no feature cache needed:
  PYTHONPATH=. /data/wmzhu/anaconda3/envs/genmol/bin/python \
    scripts/audit_protein/run_pring_cross_species_generalization.py \
    --feature-kind sequence_basic

  # SAE features (needs per-species caches built by cache_pring_species_esmc_sae.py):
  PYTHONPATH=. /data/wmzhu/anaconda3/envs/genmol/bin/python \
    scripts/audit_protein/run_pring_cross_species_generalization.py \
    --feature-kind sae_max --model-kind xgboost
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.participation import (  # noqa: E402
    CROSS_SPECIES,
    FEATURE_KINDS,
    FORMAL_FEATURE_KINDS,
    MODEL_KINDS,
    OUT_DIR,
    PRING_ROOT,
    run_pring_cross_species_generalization,
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
    p.add_argument("--test-species", nargs="+", default=list(CROSS_SPECIES))
    p.add_argument("--root", type=Path, default=PRING_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--feature-kind",
        default="sae_max",
        choices=[*FEATURE_KINDS, "pooled_sae", "binary_sae", "esmc", "all"],
        help="'all' runs the formal inputs: sae_max, binary, esmc_mean",
    )
    p.add_argument("--model-kind", default="xgboost", choices=MODEL_KINDS)
    p.add_argument("--train-cache-path", type=Path, default=None,
                   help="human pooled ESM-C/SAE cache; defaults to the standard PRING human cache")
    p.add_argument("--calibration", choices=["log_linear", "scale", "none"], default="log_linear")
    p.add_argument("--self-loop-mode", choices=["drop", "once"], default="drop")
    p.add_argument("--keep-self-pairs", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kmer", type=int, default=2, choices=[1, 2])
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--top-k-features", type=int, default=0)
    p.add_argument("--tabpfn-estimators", type=int, default=8)
    p.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    p.add_argument("--no-pair-eval", action="store_true")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    feature_kinds = FORMAL_FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)

    summary = {}
    for feature_kind in feature_kinds:
        res = run_pring_cross_species_generalization(
            root=args.root,
            out_dir=args.out_dir,
            test_species=args.test_species,
            self_loop_mode=args.self_loop_mode,
            drop_self_pairs=not args.keep_self_pairs,
            val_frac=args.val_frac,
            seed=args.seed,
            kmer=args.kmer,
            feature_kind=feature_kind,
            train_cache_path=args.train_cache_path,
            model_kind=args.model_kind,
            calibration=args.calibration,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            lr=args.lr,
            device=args.device,
            top_k_features=args.top_k_features,
            early_stopping_rounds=args.early_stopping_rounds,
            tabpfn_estimators=args.tabpfn_estimators,
            tabpfn_subsample_samples=args.tabpfn_subsample_samples,
            pair_eval=not args.no_pair_eval,
            write=not args.no_write,
        )
        summary[feature_kind] = {
            sp: {
                "test_spearman": _metric(spec, ["node_metrics", "spearman_pred_degree"]),
                "test_high_degree_auroc": _metric(spec, ["node_metrics", "high_degree_auroc"]),
                "n_test": _metric(spec, ["n_test_proteins"]),
                "missing_test_features": _metric(spec, ["missing_test_features"]),
            }
            for sp, spec in res.get("per_species", {}).items()
        }

    print("\n=== PRING cross-species participation generalization summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"pring_crossspecies_{args.model_kind}_summary"
        (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[done] results in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
