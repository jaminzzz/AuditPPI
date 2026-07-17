#!/usr/bin/env python3
"""Train C3 SAE -> high-participation protein classifiers.

This is the C3 analogue of the PRING high-degree/hubness diagnostic, but the
protein-level target is C3 endpoint propensity:

    t(p) = #positive C3 pairs touching p / #C3 pairs touching p
    high(p) = 1[t(p) >= threshold]

The model sees only a single protein's representation, not pairs:

    SAE(p) -> high(p)

C3 train/val/test are protein-disjoint, so train labels are computed from the
train split, validation labels from the validation split, and test labels from
the test split. By default the threshold is a train-set quantile, matching the
PRING high-degree classifier. Use ``--threshold-t 0.5`` for the direct
"positive-endpoint-biased protein" label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from conf.paths import AUDIT
from src.eval.metrics import participation_t
from src.eval.classification import binary_classification_metrics
from src.models.estimators.xgboost import fit_xgb_classifier
from src.data import benchmarks as D
from src.ppi_fingerprint import features as FE
from src.participation.predictor import CACHE, assemble_protein_features


OUT_DIR = AUDIT / "c3_high_participation"


def node_metrics(ids: Sequence[str], y: np.ndarray, p: np.ndarray, t: Mapping[str, float], degree: Mapping[str, int]) -> dict:
    tt = np.asarray([t[x] for x in ids], dtype=float)
    deg = np.asarray([degree[x] for x in ids], dtype=float)
    metrics = binary_classification_metrics(y, p, include_pred_prob_mean=True)
    metrics.update({
        "t_mean": round(float(tt.mean()), 6) if y.size else None,
        "t_p50": round(float(np.quantile(tt, 0.5)), 6) if y.size else None,
        "t_p90": round(float(np.quantile(tt, 0.9)), 6) if y.size else None,
        "degree_mean": round(float(deg.mean()), 4) if y.size else None,
        "degree_p90": round(float(np.quantile(deg, 0.9)), 4) if y.size else None,
        "degree_max": int(deg.max()) if y.size else None,
    })
    return metrics


def split_tables(rep: str):
    cache = FE.load_pooled_cache(CACHE["c3"])
    out = {}
    for split in ("train", "val", "test"):
        bench = D.load_benchmark(f"c3:{split}", attach_seqs=True)
        t, degree = participation_t(bench.pairs, bench.labels)
        ids = sorted(t)
        X, kept = assemble_protein_features(ids, bench.seqs, cache, rep)
        if X is None:
            raise RuntimeError(f"no cached proteins for c3:{split} rep={rep}")
        out[split] = {
            "bench": bench,
            "t": t,
            "degree": degree,
            "ids": kept,
            "X": X.astype(np.float32, copy=False),
        }
    return out


def write_predictions(path: Path, *, rows: Mapping[str, dict], pred: Mapping[str, np.ndarray], threshold: float) -> None:
    with path.open("w") as f:
        f.write("protein\tsplit\tt\tdegree\thigh_participation\tpred_high_prob\tthreshold_t\n")
        for split in ("train", "val", "test"):
            ids = rows[split]["ids"]
            for pid, prob in zip(ids, pred[split]):
                t = rows[split]["t"][pid]
                degree = rows[split]["degree"][pid]
                high = int(t >= threshold)
                f.write(f"{pid}\t{split}\t{t:.8g}\t{degree}\t{high}\t{float(prob):.8g}\t{threshold:.8g}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=FE.REPS, default="sae_max")
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--threshold-t", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = split_tables(args.rep)
    train_t = np.asarray([rows["train"]["t"][pid] for pid in rows["train"]["ids"]], dtype=float)
    if args.threshold_t is None:
        threshold = float(np.quantile(train_t, args.quantile))
        label_mode = f"train_q{args.quantile:g}"
    else:
        threshold = float(args.threshold_t)
        label_mode = f"absolute_t_ge_{threshold:g}"

    y = {
        split: np.asarray([int(rows[split]["t"][pid] >= threshold) for pid in rows[split]["ids"]], dtype=np.int8)
        for split in ("train", "val", "test")
    }

    clf = fit_xgb_classifier(
        rows["train"]["X"],
        y["train"],
        rows["val"]["X"],
        y["val"],
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        device=args.device,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    pred = {split: clf.predict_proba(rows[split]["X"])[:, 1] for split in ("train", "val", "test")}

    metrics = {
        "task": "c3_high_participation_protein_classification",
        "target": "high_t = 1[t_split(p) >= threshold_t]",
        "rep": args.rep,
        "label_mode": label_mode,
        "quantile": args.quantile,
        "threshold_t": threshold,
        "splits": {
            split: {
                "n_pairs": int(len(rows[split]["bench"].pairs)),
                "n_proteins": int(len(rows[split]["ids"])),
                "pos_pair_rate": float(np.mean(rows[split]["bench"].labels)),
            }
            for split in ("train", "val", "test")
        },
        "model": {
            "kind": "xgboost_classifier",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.lr,
            "device": args.device,
            "early_stopping_rounds": args.early_stopping_rounds,
            "best_iteration": int(getattr(clf, "best_iteration", -1)),
        },
        "node_metrics": {
            split: node_metrics(rows[split]["ids"], y[split], pred[split], rows[split]["t"], rows[split]["degree"])
            for split in ("train", "val", "test")
        },
    }

    if args.threshold_t is None:
        suffix = f"q{int(args.quantile * 100):02d}"
    else:
        suffix = f"t_ge_{str(threshold).replace('.', 'p')}"
    stem = f"c3_{args.rep}_high_t_{suffix}_xgboost"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(metrics, indent=2))
    write_predictions(args.out_dir / f"{stem}_protein_predictions.tsv", rows=rows, pred=pred, threshold=threshold)
    print(
        f"[c3.{args.rep}.{label_mode}] threshold_t={threshold:.4g} "
        f"test AUROC={metrics['node_metrics']['test']['auroc']:.4f} "
        f"AUPRC={metrics['node_metrics']['test']['auprc']:.4f} "
        f"pos_rate={metrics['node_metrics']['test']['pos_rate']:.4f}",
        flush=True,
    )
    print(f"[done] wrote {args.out_dir / (stem + '.json')}", flush=True)


if __name__ == "__main__":
    main()
