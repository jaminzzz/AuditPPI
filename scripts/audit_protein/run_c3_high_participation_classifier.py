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
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from conf.paths import RESULTS_PROTEIN, PPI_PREDICTION_CACHES as CACHE
from conf.audit import PARTICIPATION_QUANTILE
from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    REPRESENTATIONS,
    resolve_backbone_layer,
)
from src.eval.metrics import participation_t
from src.eval.classification import binary_classification_metrics
from src.experiments.results import dump_experiment
from src.models.estimators.xgboost import fit_xgb_classifier
from src.data import pairs as D
from src.features.pairs import load_protein_feature_cache
from src.features.protein_cache import protein_feature_rows


OUT_DIR = RESULTS_PROTEIN / "c3_high_participation"


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


def split_tables(rep: str, *, backbone: str, layer: int):
    cache = load_protein_feature_cache(CACHE["c3"])
    out = {}
    for split in ("train", "val", "test"):
        bench = D.load_benchmark(f"c3:{split}", attach_seqs=True)
        t, degree = participation_t(bench.pairs, bench.labels)
        ids = sorted(t)
        X, kept = protein_feature_rows(ids, bench.seqs, cache, rep, layer=layer, backbone=backbone)
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
    ap.add_argument("--rep", choices=REPRESENTATIONS, default="sae_max")
    ap.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=DEFAULT_BACKBONE,
        help="Backbone family to read from the v1 feature cache (esmc or esm2).",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        choices=sorted({layer for layers in BACKBONE_LAYERS.values() for layer in layers}),
        help="Backbone layer; defaults to the backbone's default (ESM-C 60, ESM-2 33).",
    )
    ap.add_argument("--quantile", type=float, default=PARTICIPATION_QUANTILE)
    ap.add_argument("--threshold-t", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    args = ap.parse_args()
    args.layer = resolve_backbone_layer(args.backbone, args.layer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = split_tables(args.rep, backbone=args.backbone, layer=args.layer)
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
        "backbone": args.backbone,
        "layer": args.layer,
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
    stem = f"c3_{args.rep}_{args.backbone}_l{args.layer}_high_t_{suffix}_xgboost"
    result_path = args.out_dir / f"{stem}.json"
    dump_experiment(
        result_path,
        task="protein.high_participation",
        dataset="c3",
        features=args.rep,
        split="test",
        model="xgboost_classifier",
        seed=args.seed,
        payload=metrics,
        metrics=metrics["node_metrics"]["test"],
        hyperparameters=metrics.get("model"),
    )
    write_predictions(args.out_dir / f"{stem}_protein_predictions.tsv", rows=rows, pred=pred, threshold=threshold)
    print(
        f"[c3.{args.rep}.{label_mode}] threshold_t={threshold:.4g} "
        f"test AUROC={metrics['node_metrics']['test']['auroc']:.4f} "
        f"AUPRC={metrics['node_metrics']['test']['auprc']:.4f} "
        f"pos_rate={metrics['node_metrics']['test']['pos_rate']:.4f}",
        flush=True,
    )
    print(f"[done] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
