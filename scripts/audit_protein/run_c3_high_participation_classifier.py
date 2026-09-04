#!/usr/bin/env python3
"""Train C-level SAE -> high-participation protein classifiers.

This is the RAPPPID C1/C2/C3 analogue of the PRING high-degree/hubness
diagnostic, but the protein-level target is C-level endpoint propensity:

    t(p) = #positive C-level pairs touching p / #C-level pairs touching p
    high(p) = 1[t(p) >= threshold and degree(p) >= min_degree]

The model sees only a single protein's representation, not pairs:

    representation(p) -> high(p)

Labels are computed within each C-level split from that split's pair labels.
By default the threshold is the train-set quantile and is applied unchanged to
val/test, matching the original C3-only script and the PRING high-degree
classifier. Use ``--threshold-t 0.5`` for the direct
"positive-endpoint-biased protein" label. Use ``--min-degree`` to require a
minimum number of pair observations before a protein can be labelled high,
preventing degree-1 proteins with t=1 from dominating the target. The saved JSON
also reports a pair shortcut on each split:

    score(A, B) = min(p_high(A), p_high(B))

Runs on any RAPPPID C-level family (``--family c1|c2|c3``), routing to the
family-specific protein feature cache in ``PPI_PREDICTION_CACHES``.
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
from src.eval.classification import binary_classification_metrics, safe_auprc, safe_auroc
from src.experiments.results import dump_experiment
from src.models.estimators.xgboost import fit_xgb_classifier
from src.data import pairs as D
from src.features.pairs import load_protein_feature_cache
from src.features.protein_cache import protein_feature_rows


# C-level families use ``family:split``; PRING needs method-tagged human splits
# (``pring:human:{split}:{method}``). Bernett shares the C-level grammar.
# cross_species has no official val: train/val are carved from human_train pairs
# (val_frac, seed+1), test defaults to human_test -- matching the TabPFN /
# minimal-fingerprint cross_species protocol.
FAMILIES = ("c1", "c2", "c3", "bernett", "pring", "cross_species")
PRING_METHODS = ("BFS", "DFS", "RANDOM_WALK")


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


def _benchmark_name(family: str, split: str, *, method: str = "BFS") -> str:
    """Map (family, split) onto the load_benchmark grammar."""
    if family == "pring":
        return f"pring:human:{split}:{method}"
    if family == "cross_species":
        # Official graphs only; train/val carving is handled in split_tables.
        if split == "test":
            return "cross_species:human_test"
        return "cross_species:human_train"
    return f"{family}:{split}"


def _cache_for(family: str):
    """Resolve the v1 protein cache for a family (PRING -> human)."""
    if family not in CACHE:
        raise KeyError(f"no PPI_PREDICTION_CACHES entry for family {family!r}")
    return load_protein_feature_cache(CACHE[family])


def _bench_from_pair_subset(name: str, pairs, labels, seqs) -> D.Benchmark:
    """Build a Benchmark view over a carved pair subset (cross_species val)."""
    return D.Benchmark(name, list(pairs), np.asarray(labels, dtype=int), dict(seqs))


def _split_row(bench, cache, rep: str, *, backbone: str, layer: int) -> dict:
    t, degree = participation_t(bench.pairs, bench.labels)
    ids = sorted(t)
    X, kept = protein_feature_rows(
        ids, bench.seqs, cache, rep, layer=layer, backbone=backbone,
    )
    if X is None:
        raise RuntimeError(f"no cached proteins for {bench.name} rep={rep}")
    return {
        "bench": bench,
        "t": t,
        "degree": degree,
        "ids": kept,
        "X": X.astype(np.float32, copy=False),
    }


def split_tables(
    family: str, rep: str, *, backbone: str, layer: int,
    method: str = "BFS", val_frac: float = 0.1, seed: int = DEFAULT_SEED,
    test_benchmark: str | None = None,
):
    cache = _cache_for(family)
    out = {}

    if family == "cross_species":
        # No official val: carve stratified val pairs from human_train first
        # (seed+1), then derive within-split t/degree on each pair subset -- the
        # protein-level analogue of the TabPFN / minimal-fingerprint protocol.
        from src.features.sampling import stratified_subsample

        full = D.load_benchmark("cross_species:human_train", attach_seqs=True)
        y = np.asarray(full.labels, dtype=int)
        n_val = max(1, int(len(y) * val_frac))
        val_idx = stratified_subsample(y, n_val, seed + 1)
        if val_idx is None:
            raise ValueError(
                f"cross_species val carve failed: n={len(y)} val_frac={val_frac}"
            )
        val_mask = np.zeros(len(y), dtype=bool)
        val_mask[val_idx] = True
        train_idx = np.flatnonzero(~val_mask)
        train_pairs = [full.pairs[i] for i in train_idx]
        val_pairs = [full.pairs[i] for i in val_idx]
        train_bench = _bench_from_pair_subset(
            "cross_species_human_train_fit", train_pairs, y[train_idx], full.seqs,
        )
        val_bench = _bench_from_pair_subset(
            "cross_species_human_train_val", val_pairs, y[val_idx], full.seqs,
        )
        test_name = test_benchmark or "cross_species:human_test"
        test_bench = D.load_benchmark(test_name, attach_seqs=True)
        out["train"] = _split_row(train_bench, cache, rep, backbone=backbone, layer=layer)
        out["val"] = _split_row(val_bench, cache, rep, backbone=backbone, layer=layer)
        out["test"] = _split_row(test_bench, cache, rep, backbone=backbone, layer=layer)
        return out

    for split in ("train", "val", "test"):
        if split == "test" and test_benchmark is not None:
            name = test_benchmark
        else:
            name = _benchmark_name(family, split, method=method)
        bench = D.load_benchmark(name, attach_seqs=True)
        out[split] = _split_row(bench, cache, rep, backbone=backbone, layer=layer)
    return out


def endpoint_min_pair_metrics(bench, pred_high_prob: Mapping[str, float]) -> dict:
    scores = []
    labels = []
    skipped = 0
    for (a, b), label in zip(bench.pairs, bench.labels):
        pa, pb = pred_high_prob.get(a), pred_high_prob.get(b)
        if pa is None or pb is None:
            skipped += 1
            continue
        scores.append(min(pa, pb))
        labels.append(int(label))
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=float)
    return {
        "score": "min(pred_high_prob_a, pred_high_prob_b)",
        "n_total": int(len(bench.pairs)),
        "n_scored": int(y.size),
        "n_skipped": int(skipped),
        "pos_rate": round(float(y.mean()), 6) if y.size else None,
        "auroc": safe_auroc(y, p),
        "auprc": safe_auprc(y, p),
        "baseline_auprc": round(float(y.mean()), 6) if y.size else None,
    }


def write_predictions(path: Path, *, rows: Mapping[str, dict], pred: Mapping[str, np.ndarray], threshold: float) -> None:
    with path.open("w") as f:
        f.write("protein\tsplit\tt\tdegree\thigh_participation\tpred_high_prob\tthreshold_t\tmin_degree\n")
        for split in ("train", "val", "test"):
            ids = rows[split]["ids"]
            for pid, prob in zip(ids, pred[split]):
                t = rows[split]["t"][pid]
                degree = rows[split]["degree"][pid]
                high = int((t >= threshold) and (degree >= rows[split]["min_degree"]))
                f.write(
                    f"{pid}\t{split}\t{t:.8g}\t{degree}\t{high}\t"
                    f"{float(prob):.8g}\t{threshold:.8g}\t{rows[split]['min_degree']}\n"
                )


def run_one(
    args: argparse.Namespace, *, family: str, rep: str, out_dir: Path, method: str = "BFS",
) -> Path:
    rows = split_tables(
        family, rep, backbone=args.backbone, layer=args.layer, method=method,
        val_frac=getattr(args, "val_frac", 0.1), seed=args.seed,
        test_benchmark=getattr(args, "test_benchmark", None),
    )
    train_t = np.asarray([rows["train"]["t"][pid] for pid in rows["train"]["ids"]], dtype=float)
    if args.threshold_t is None:
        threshold = float(np.quantile(train_t, args.quantile))
        label_mode = f"train_q{args.quantile:g}"
    else:
        threshold = float(args.threshold_t)
        label_mode = f"absolute_t_ge_{threshold:g}"

    for split in ("train", "val", "test"):
        rows[split]["min_degree"] = args.min_degree
    y = {
        split: np.asarray([
            int((rows[split]["t"][pid] >= threshold)
                and (rows[split]["degree"][pid] >= args.min_degree))
            for pid in rows[split]["ids"]
        ], dtype=np.int8)
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
    pred_maps = {
        split: {pid: float(prob) for pid, prob in zip(rows[split]["ids"], pred[split])}
        for split in ("train", "val", "test")
    }

    # Product tag: bare family for non-PRING / PRING-BFS; method-suffixed otherwise.
    # cross_species species-test overrides get cross_species_{sp}.
    test_benchmark = getattr(args, "test_benchmark", None)
    if test_benchmark is not None and family == "cross_species":
        sp = test_benchmark.split(":")[-1]
        family_tag = f"cross_species_{sp}"
    elif family == "pring" and method != "BFS":
        family_tag = f"pring_{method.lower()}"
    else:
        family_tag = family

    metrics = {
        "task": "clevel_high_participation_protein_classification",
        "target": "high_t = 1[t_split(p) >= threshold_t]",
        "family": family,
        "family_tag": family_tag,
        "method": method if family == "pring" else None,
        "test_benchmark": test_benchmark,
        "val_frac": getattr(args, "val_frac", None) if family == "cross_species" else None,
        "rep": rep,
        "backbone": args.backbone,
        "layer": args.layer,
        "label_mode": label_mode,
        "quantile": args.quantile,
        "threshold_t": threshold,
        "min_degree": args.min_degree,
        "label_rule": "high = 1[t_split(p) >= threshold_t and degree_split(p) >= min_degree]",
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
        "pair_metrics": {
            split: endpoint_min_pair_metrics(rows[split]["bench"], pred_maps[split])
            for split in ("train", "val", "test")
        },
    }

    if args.threshold_t is None:
        suffix = f"q{int(args.quantile * 100):02d}"
    else:
        suffix = f"t_ge_{str(threshold).replace('.', 'p')}"
    bb_tag = f"{args.backbone}_l{args.layer}"
    stem = f"{family_tag}_{rep}_{bb_tag}_high_t_{suffix}_mindeg{args.min_degree}_xgboost"
    result_path = out_dir / f"{stem}.json"
    dump_experiment(
        result_path,
        task="protein.high_participation",
        dataset=family_tag,
        features=rep,
        split="test",
        model="xgboost_classifier",
        seed=args.seed,
        payload=metrics,
        metrics=metrics["node_metrics"]["test"],
        hyperparameters=metrics.get("model"),
    )
    write_predictions(out_dir / f"{stem}_protein_predictions.tsv", rows=rows, pred=pred, threshold=threshold)
    print(
        f"[{family_tag}.{rep}.{label_mode}] threshold_t={threshold:.4g} "
        f"test hub_AUROC={metrics['node_metrics']['test']['auroc']:.4f} "
        f"hub_AUPRC={metrics['node_metrics']['test']['auprc']:.4f} "
        f"ppi_min_AUROC={metrics['pair_metrics']['test']['auroc']:.4f} "
        f"ppi_min_AUPRC={metrics['pair_metrics']['test']['auprc']:.4f} "
        f"high_rate={metrics['node_metrics']['test']['pos_rate']:.4f}",
        flush=True,
    )
    print(f"[done] wrote {result_path}", flush=True)
    return result_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=[*FAMILIES, "all"], default="c3",
                    help="Family. Use all to run c1/c2/c3/bernett/pring/cross_species.")
    ap.add_argument("--rep", choices=[*REPRESENTATIONS, "all"], default="sae_max")
    ap.add_argument(
        "--method",
        choices=[*PRING_METHODS, "all"],
        default="BFS",
        help="PRING human sampling method (ignored for non-PRING). "
             "Use all to run BFS/DFS/RANDOM_WALK.",
    )
    ap.add_argument(
        "--test-benchmark",
        default=None,
        help="Optional test-split override (e.g. cross_species:yeast). "
             "Train/val still use the family's native protocol.",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="cross_species only: fraction of human_train pairs carved as val "
             "(stratified, seed+1). Matches run_cross_species_tabpfn_topk.",
    )
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
    ap.add_argument("--min-degree", type=int, default=1,
                    help="Minimum split degree required for a protein to be labelled high.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Defaults to RESULTS_PROTEIN/{family}_high_participation.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    args = ap.parse_args()
    args.layer = resolve_backbone_layer(args.backbone, args.layer)

    families = FAMILIES if args.family == "all" else (args.family,)
    reps = REPRESENTATIONS if args.rep == "all" else (args.rep,)
    methods = PRING_METHODS if args.method == "all" else (args.method,)
    multi_family = len(families) > 1
    for family in families:
        out_dir = (
            RESULTS_PROTEIN / f"{family}_high_participation"
            if args.out_dir is None
            else args.out_dir / family if multi_family else args.out_dir
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        fam_methods = methods if family == "pring" else ("BFS",)
        for method in fam_methods:
            for rep in reps:
                run_one(args, family=family, rep=rep, out_dir=out_dir, method=method)


if __name__ == "__main__":
    main()
