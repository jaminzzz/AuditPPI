#!/usr/bin/env python3
"""Evaluate saved PRING pred-oracle protein predictions as PPI pair scores.

This does not retrain the sequence -> participation model. It reads
``*_protein_predictions.tsv`` files, scores a pair by the previous pred-oracle
logic ``min(pred_t[a], pred_t[b])``, and evaluates PRING pair-level files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.participation import (
    FORMAL_FEATURE_KINDS,
    METHODS,
    OUT_DIR,
    PRING_ROOT,
    full_graph_participation_labels,
)
from src.eval.classification import safe_auprc, safe_auroc
from src.experiments.results import dump_experiment

PAIR_SETS = ("human_test", "all_test", "human_val", "human_train")
SCORE_MODES = ("min", "product", "mean")


def read_predictions(path: Path) -> dict[str, float]:
    pred_t: dict[str, float] = {}
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        protein_i = header.index("protein")
        pred_t_i = header.index("pred_t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= pred_t_i:
                continue
            pred_t[parts[protein_i]] = float(parts[pred_t_i])
    return pred_t


def pair_score(a: float, b: float, mode: str) -> float:
    if mode == "min":
        return min(a, b)
    if mode == "product":
        return a * b
    if mode == "mean":
        return 0.5 * (a + b)
    raise ValueError(mode)


def eval_pair_file(
    path: Path,
    *,
    pred_t: dict[str, float],
    true_t: dict[str, float],
    score_mode: str,
    drop_self_pairs: bool,
) -> dict:
    y: list[int] = []
    pred_scores: list[float] = []
    true_scores: list[float] = []
    n_total = 0
    n_self = 0
    n_skipped = 0
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            a, b, label = parts[0], parts[1], int(parts[2])
            if drop_self_pairs and a == b:
                n_self += 1
                continue
            n_total += 1
            pa, pb = pred_t.get(a), pred_t.get(b)
            ta, tb = true_t.get(a), true_t.get(b)
            if pa is None or pb is None or ta is None or tb is None:
                n_skipped += 1
                continue
            y.append(label)
            pred_scores.append(pair_score(pa, pb, score_mode))
            true_scores.append(pair_score(ta, tb, score_mode))

    yy = np.asarray(y, dtype=np.int8)
    ps = np.asarray(pred_scores, dtype=np.float32)
    ts = np.asarray(true_scores, dtype=np.float32)
    return {
        "path": str(path),
        "score_mode": score_mode,
        "n_total": int(n_total),
        "n_self_dropped": int(n_self),
        "n_scored": int(yy.size),
        "n_skipped": int(n_skipped),
        "pos_rate": round(float(yy.mean()), 6) if yy.size else None,
        "pred_oracle_auroc": safe_auroc(yy, ps),
        "pred_oracle_auprc": safe_auprc(yy, ps),
        "true_full_participation_auroc": safe_auroc(yy, ts),
        "true_full_participation_auprc": safe_auprc(yy, ts),
    }


def pred_path(pred_dir: Path, method: str, feature: str, model_kind: str) -> Path:
    return pred_dir / f"pring_human_{method.lower()}_{feature}_{model_kind}_protein_predictions.tsv"


def pair_path(root: Path, method: str, pair_set: str) -> Path:
    name = {
        "human_test": "human_test_ppi.txt",
        "all_test": "all_test_ppi.txt",
        "human_val": "human_val_ppi.txt",
        "human_train": "human_train_ppi.txt",
    }[pair_set]
    return root / "human" / method / name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, default=PRING_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--method", choices=[*METHODS, "all"], default="all")
    p.add_argument("--feature-kind", choices=[*FORMAL_FEATURE_KINDS, "all"], default="all")
    p.add_argument("--model-kind", default="xgboost")
    p.add_argument("--pair-set", choices=[*PAIR_SETS, "all"], nargs="+", default=["human_test"])
    p.add_argument("--score-mode", choices=SCORE_MODES, default="min")
    p.add_argument("--keep-self-pairs", action="store_true")
    args = p.parse_args()

    methods = METHODS if args.method == "all" else (args.method,)
    features = FORMAL_FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)
    pair_sets = PAIR_SETS if "all" in args.pair_set else tuple(args.pair_set)

    labels = full_graph_participation_labels(root=args.root, self_loop_mode="drop")
    summary: dict[str, dict] = {}
    for method in methods:
        summary[method] = {}
        for feature in features:
            pp = pred_path(args.pred_dir, method, feature, args.model_kind)
            pred_t = read_predictions(pp)
            summary[method][feature] = {}
            for pair_set_name in pair_sets:
                res = eval_pair_file(
                    pair_path(args.root, method, pair_set_name),
                    pred_t=pred_t,
                    true_t=labels.t,
                    score_mode=args.score_mode,
                    drop_self_pairs=not args.keep_self_pairs,
                )
                summary[method][feature][pair_set_name] = res
                print(
                    f"[{method}.{feature}.{pair_set_name}] "
                    f"AUROC={res['pred_oracle_auroc']:.4f} "
                    f"AUPRC={res['pred_oracle_auprc']:.4f} "
                    f"n={res['n_scored']}",
                    flush=True,
                )

    out = args.out
    if out is None:
        out = args.pred_dir / f"pring_pred_oracle_ppi_{args.score_mode}_summary.json"
    dump_experiment(
        out,
        task="pring_pred_oracle_ppi",
        dataset="pring",
        features="multi",
        split="multi",
        model=f"pred_oracle_{args.score_mode}",
        seed=-1,
        payload=summary,
    )
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
