#!/usr/bin/env python3
"""Train PRING sequence -> high-participation classifiers.

Labels are defined once from the full PRING Human reference graph:

    high_q = 1[degree_full >= quantile_q(degree_full)]

Training and evaluation still use PRING protein-disjoint splits. Validation is
an inner protein split sampled from PRING train proteins, matching the current
pred-oracle logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from conf.audit import PARTICIPATION_QUANTILE
from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    resolve_backbone_layer,
)
from conf.paths import (
    PRING_HUMAN_SAE_CACHE as DEFAULT_PRING_CACHE,
    PRING_PARTICIPATION_DIR as OUT_DIR,
    PRING_ROOT,
)
from src.data.sequences import read_fasta
from src.data.pring_graph import (
    METHODS,
    full_graph_participation_labels,
    load_pring_human_split,
)
from src.eval.classification import (
    binary_classification_metrics,
    safe_auprc,
    safe_auroc,
)
from src.experiments.results import dump_experiment
from src.features.feature_selection import (
    build_importance_rows,
    extract_xgb_importance,
    write_feature_importance,
)
from src.features.pooled_assembly import prepare_features
from src.features.sequence_composition import FORMAL_FEATURE_KINDS
from src.eval.participation import read_labeled_pairs
from src.models.estimators.xgboost import fit_xgb_classifier


PAIR_EVALS = ("none", "human_test", "all")


def node_metrics(ids: Sequence[str], y: np.ndarray, p: np.ndarray, degree: Mapping[str, int]) -> dict:
    deg = np.asarray([degree[x] for x in ids], dtype=float)
    metrics = binary_classification_metrics(y, p, include_pred_prob_mean=True)
    metrics.update({
        "degree_mean": round(float(deg.mean()), 4) if y.size else None,
        "degree_p90_local": round(float(np.quantile(deg, 0.9)), 4) if y.size else None,
        "degree_max": int(deg.max()) if y.size else None,
    })
    return metrics


def write_labels(path: Path, *, ids: Sequence[str], degree: Mapping[str, int], t: Mapping[str, float], threshold: float, quantile: float) -> None:
    with path.open("w") as f:
        f.write("protein\tdegree_full\tlog1p_degree_full\ttrue_t_full\tquantile\tthreshold_degree\thigh_participation\n")
        for pid in ids:
            deg = degree[pid]
            f.write(
                f"{pid}\t{deg}\t{np.log1p(deg):.6g}\t{t[pid]:.6g}\t{quantile:.6g}\t"
                f"{threshold:.6g}\t{int(deg >= threshold)}\n"
            )


def write_predictions(
    path: Path,
    *,
    split_ids: Mapping[str, Sequence[str]],
    degree: Mapping[str, int],
    labels: Mapping[str, int],
    pred_prob: Mapping[str, float],
) -> None:
    with path.open("w") as f:
        f.write("protein\tsplit\tdegree_full\thigh_participation\tpred_high_prob\n")
        for split, ids in split_ids.items():
            for pid in ids:
                f.write(f"{pid}\t{split}\t{degree[pid]}\t{labels[pid]}\t{pred_prob[pid]:.8g}\n")


def pair_metrics(path: Path, *, pred_prob: Mapping[str, float], drop_self_pairs: bool) -> dict:
    pairs, y = read_labeled_pairs(path, drop_self_pairs=drop_self_pairs)
    scores: list[float] = []
    yy: list[int] = []
    skipped = 0
    for (a, b), label in zip(pairs, y):
        pa, pb = pred_prob.get(a), pred_prob.get(b)
        if pa is None or pb is None:
            skipped += 1
            continue
        scores.append(min(pa, pb))
        yy.append(int(label))
    y2 = np.asarray(yy, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float32)
    return {
        "path": str(path),
        "score": "min(pred_high_prob_a, pred_high_prob_b)",
        "n_total": int(len(pairs)),
        "n_scored": int(y2.size),
        "n_skipped": int(skipped),
        "pos_rate": round(float(y2.mean()), 6) if y2.size else None,
        "auroc": safe_auroc(y2, s),
        "auprc": safe_auprc(y2, s),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=PRING_ROOT)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_PRING_CACHE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR / "high_p90_xgboost")
    p.add_argument("--method", choices=[*METHODS, "all"], default="all")
    p.add_argument("--feature-kind", choices=[*FORMAL_FEATURE_KINDS, "all"], default="all")
    p.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=DEFAULT_BACKBONE,
        help="Backbone family to read from the v1 feature cache (esmc or esm2).",
    )
    p.add_argument(
        "--layer",
        type=int,
        choices=sorted({layer for layers in BACKBONE_LAYERS.values() for layer in layers}),
        default=None,
        help=(
            "Backbone layer to read from the v1 feature cache; defaults to the "
            "backbone's default layer (esmc=60, esm2=33). Ignored for sequence_basic."
        ),
    )
    p.add_argument("--quantile", type=float, default=PARTICIPATION_QUANTILE)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-estimators", type=int, default=3000)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--early-stopping-rounds", type=int, default=200)
    p.add_argument("--importance-top-n", type=int, default=200)
    p.add_argument("--pair-eval", choices=PAIR_EVALS, default="none")
    p.add_argument("--keep-self-pairs", action="store_true")
    p.add_argument("--write-labels-only", action="store_true")
    args = p.parse_args()
    args.layer = resolve_backbone_layer(args.backbone, args.layer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_labels = full_graph_participation_labels(root=args.root, self_loop_mode="drop")
    seqs = read_fasta(args.root / "human" / "human_simple.fasta")
    label_ids = sorted(pid for pid in graph_labels.degree if pid in seqs)
    degrees = np.asarray([graph_labels.degree[pid] for pid in label_ids], dtype=float)
    threshold = float(np.quantile(degrees, args.quantile))
    high = {pid: int(graph_labels.degree[pid] >= threshold) for pid in label_ids}

    label_path = args.out_dir / f"pring_human_high_q{int(args.quantile * 100):02d}_labels.tsv"
    write_labels(
        label_path,
        ids=label_ids,
        degree=graph_labels.degree,
        t=graph_labels.t,
        threshold=threshold,
        quantile=args.quantile,
    )
    label_summary = {
        "label": f"degree_full >= global_q{args.quantile}",
        "quantile": args.quantile,
        "threshold_degree": threshold,
        "n": len(label_ids),
        "pos": int(sum(high.values())),
        "neg": int(len(label_ids) - sum(high.values())),
        "pos_rate": float(np.mean([high[x] for x in label_ids])),
        "labels_tsv": str(label_path),
    }
    (args.out_dir / f"pring_human_high_q{int(args.quantile * 100):02d}_label_summary.json").write_text(
        json.dumps(label_summary, indent=2)
    )
    print(
        f"[labels] threshold={threshold:.4g} pos={label_summary['pos']} "
        f"neg={label_summary['neg']} pos_rate={label_summary['pos_rate']:.4f}",
        flush=True,
    )
    if args.write_labels_only:
        return

    methods = METHODS if args.method == "all" else (args.method,)
    features = FORMAL_FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)
    summary: dict[str, dict] = {}
    for method in methods:
        train_nodes, test_nodes = load_pring_human_split(method, root=args.root)
        usable_train = sorted(pid for pid in train_nodes if pid in seqs and pid in graph_labels.degree)
        usable_test = sorted(pid for pid in test_nodes if pid in seqs and pid in graph_labels.degree)
        summary[method] = {}
        for feature in features:
            pack = prepare_features(
                feature_kind=feature,
                seqs=seqs,
                labels=graph_labels,
                candidate_train=usable_train,
                candidate_test=usable_test,
                kmer=2,
                val_frac=args.val_frac,
                seed=args.seed,
                cache_path=args.cache_path,
                layer=args.layer,
                backbone=args.backbone,
            )
            ytr = np.asarray([high[pid] for pid in pack.train_ids], dtype=np.int8)
            yva = np.asarray([high[pid] for pid in pack.val_ids], dtype=np.int8)
            yte = np.asarray([high[pid] for pid in pack.test_ids], dtype=np.int8)

            clf = fit_xgb_classifier(
                pack.Xtr,
                ytr,
                pack.Xva,
                yva,
                seed=args.seed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.lr,
                device=args.device,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            ptr = clf.predict_proba(pack.Xtr)[:, 1]
            pva = clf.predict_proba(pack.Xva)[:, 1]
            pte = clf.predict_proba(pack.Xte)[:, 1]
            all_ids = pack.train_ids + pack.val_ids + pack.test_ids
            all_prob = np.concatenate([ptr, pva, pte])
            pred_prob = {pid: float(prob) for pid, prob in zip(all_ids, all_prob)}

            pair_eval = {}
            pair_dir = args.root / "human" / method
            if args.pair_eval in ("human_test", "all"):
                pair_eval["human_test_ppi"] = pair_metrics(
                    pair_dir / "human_test_ppi.txt",
                    pred_prob=pred_prob,
                    drop_self_pairs=not args.keep_self_pairs,
                )
            if args.pair_eval == "all":
                pair_eval["all_test_ppi"] = pair_metrics(
                    pair_dir / "all_test_ppi.txt",
                    pred_prob=pred_prob,
                    drop_self_pairs=not args.keep_self_pairs,
                )

            arrays = extract_xgb_importance(clf, len(pack.feature_names))
            importance_summary = None
            importance_rows = None
            if arrays is not None:
                importance_rows = build_importance_rows(
                    feature_names=pack.feature_names,
                    arrays=arrays,
                    selected_columns=None,
                )
                importance_summary = {
                    "top_n_in_json": min(args.importance_top_n, len(importance_rows)),
                    "top_features": importance_rows[: args.importance_top_n],
                }

            res = {
                "task": "pring_full_graph_high_participation_classification",
                "method": method,
                "feature_kind": feature,
                "label_definition": label_summary,
                "split": {
                    "source": f"human/{method}/human_{method}_split.pkl",
                    "val": "stratified inner protein split sampled from PRING train proteins",
                    "val_frac": args.val_frac,
                    "seed": args.seed,
                    "n_train_nodes": len(pack.train_ids),
                    "n_val_nodes": len(pack.val_ids),
                    "n_test_nodes": len(pack.test_ids),
                },
                "features": pack.info,
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
                    "train": node_metrics(pack.train_ids, ytr, ptr, graph_labels.degree),
                    "val": node_metrics(pack.val_ids, yva, pva, graph_labels.degree),
                    "test": node_metrics(pack.test_ids, yte, pte, graph_labels.degree),
                },
                "pair_eval": args.pair_eval,
                "pair_metrics": pair_eval,
                "feature_importance": importance_summary,
            }

            stem = (
                f"pring_human_{method.lower()}_{feature}_{args.backbone}_l{args.layer}"
                f"_high_q{int(args.quantile * 100):02d}_xgboost"
            )
            dump_experiment(
                args.out_dir / f"{stem}.json",
                task="protein.high_participation",
                dataset=f"pring_human_{method.lower()}",
                features=feature,
                split="test",
                model="xgboost_classifier",
                seed=args.seed,
                payload=res,
                metrics=res["node_metrics"]["test"],
                hyperparameters={**(res.get("model") or {}), "backbone": args.backbone, "layer": args.layer},
            )
            write_predictions(
                args.out_dir / f"{stem}_protein_predictions.tsv",
                split_ids={"train": pack.train_ids, "val": pack.val_ids, "test": pack.test_ids},
                degree=graph_labels.degree,
                labels=high,
                pred_prob=pred_prob,
            )
            if importance_rows is not None:
                write_feature_importance(args.out_dir / f"{stem}_feature_importance.tsv", importance_rows)

            summary[method][feature] = {
                "test_auroc": res["node_metrics"]["test"]["auroc"],
                "test_auprc": res["node_metrics"]["test"]["auprc"],
                "test_pos_rate": res["node_metrics"]["test"]["pos_rate"],
                "precision_at_n_pos": res["node_metrics"]["test"]["precision_at_n_pos"],
                "best_iteration": res["model"]["best_iteration"],
                "n_train": len(pack.train_ids),
                "n_val": len(pack.val_ids),
                "n_test": len(pack.test_ids),
                "pair_human_test_auroc": pair_eval.get("human_test_ppi", {}).get("auroc"),
                "pair_all_test_auroc": pair_eval.get("all_test_ppi", {}).get("auroc"),
            }
            print(
                f"[{method}.{feature}] test AUROC={summary[method][feature]['test_auroc']:.4f} "
                f"AUPRC={summary[method][feature]['test_auprc']:.4f} "
                f"pos_rate={summary[method][feature]['test_pos_rate']:.4f}",
                flush=True,
            )

    summary_path = args.out_dir / (
        f"pring_human_{args.backbone}_l{args.layer}"
        f"_high_q{int(args.quantile * 100):02d}_xgboost_summary.json"
    )
    dump_experiment(
        summary_path,
        task="protein.high_participation",
        dataset="pring_human",
        features="multi",
        split="multi",
        model="xgboost_classifier",
        seed=args.seed,
        payload=summary,
        metrics={
            method: {
                feature: {
                    "auroc": cell.get("test_auroc"),
                    "auprc": cell.get("test_auprc"),
                }
                for feature, cell in method_cells.items()
            }
            for method, method_cells in summary.items()
        },
        hyperparameters={
            "quantile": args.quantile,
            "backbone": args.backbone,
            "layer": args.layer,
        },
    )
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
