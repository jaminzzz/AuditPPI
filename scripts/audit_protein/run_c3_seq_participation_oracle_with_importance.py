#!/usr/bin/env python3
"""Run C3 sequence-predicted participation oracle with feature importance.

This is the same diagnostic as ``scripts/audit_protein/run_participation_oracle.py`` for C3:

    1. compute train+val protein target t(p) = pos(p) / degree(p)
    2. fit an XGBoost regressor t_hat(p) = f(SAE(p))
    3. score test pairs by min(t_hat(A), t_hat(B))

The original runner writes only the pair-level summary. This wrapper also writes
per-protein t_hat values and SAE feature importances from the regressor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from conf.model import DEFAULT_SEED
from conf.paths import RESULTS_PROTEIN, FEATURE_TABLE
from src.runtime import seed_all
from src.eval import evaluate_scorer
from src.eval.metrics import participation_t
from src.models.estimators.xgboost import fit_xgb_regressor
from src.data import pairs as D
from src.ppi_fingerprint import features as FE
from src.participation.predictor import (
    CACHE,
    TEST,
    TRAINVAL,
    assemble_protein_features,
    safe_spearman,
    train_target_t,
)

OUT_DIR = RESULTS_PROTEIN / "c3_seq_participation_oracle"


def feature_names(rep: str, dim: int) -> list[str]:
    if rep == "esmc_mean":
        return [f"esmc_mean_{i:04d}" for i in range(dim)]
    if rep == "binary":
        return [f"sae_binary_{i:05d}" for i in range(dim)]
    return [f"sae_max_{i:05d}" for i in range(dim)]


def weighted_corr_columns(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted Pearson corr between every X column and y."""
    w = w.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    sw = float(w.sum())
    if sw <= 0:
        return np.full(X.shape[1], np.nan, dtype=np.float32)
    y_centered = y - float(np.dot(w, y) / sw)
    wy = w * y_centered
    cov = X.astype(np.float64, copy=False).T @ wy / sw
    x_mean = (w @ X.astype(np.float64, copy=False)) / sw
    x2_mean = (w @ (X.astype(np.float64, copy=False) ** 2)) / sw
    x_var = np.maximum(x2_mean - x_mean**2, 0.0)
    y_var = max(float(np.dot(w, y_centered**2) / sw), 0.0)
    denom = np.sqrt(x_var * y_var)
    corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
    return corr.astype(np.float32)


def booster_importance(reg, dim: int, names: list[str]) -> pd.DataFrame:
    booster = reg.get_booster()
    rows = pd.DataFrame({"feature_id": np.arange(dim, dtype=int), "feature_name": names})
    for kind in ("weight", "gain", "cover", "total_gain", "total_cover"):
        score = booster.get_score(importance_type=kind)
        values = np.zeros(dim, dtype=np.float64)
        for key, val in score.items():
            if not key.startswith("f"):
                continue
            idx = int(key[1:])
            if 0 <= idx < dim:
                values[idx] = float(val)
        rows[kind] = values
    return rows


def add_annotations(df: pd.DataFrame, rep: str) -> pd.DataFrame:
    if rep not in {"sae_max", "binary"} or not FEATURE_TABLE.exists():
        return df
    ann = pd.read_parquet(
        FEATURE_TABLE,
        columns=["feature_id", "category", "summary", "activation_pattern", "uniref90_frequency"],
    )
    ann["category"] = ann["category"].fillna("Unclassified")
    return df.merge(ann, on="feature_id", how="left")


def write_protein_predictions(path: Path, *, ids, t_true, degree, t_hat) -> None:
    with path.open("w") as f:
        f.write("protein\ttrue_t\tdegree\tt_hat\n")
        for pid in ids:
            f.write(f"{pid}\t{t_true[pid]:.8g}\t{degree[pid]}\t{t_hat[pid]:.8g}\n")


def write_pair_predictions(path: Path, *, bench, t_hat) -> None:
    with path.open("w") as f:
        f.write("protein_a\tprotein_b\tlabel\tscore_min_t_hat\tt_hat_a\tt_hat_b\n")
        for (a, b), y in zip(bench.pairs, bench.labels):
            ha = t_hat.get(a)
            hb = t_hat.get(b)
            score = np.nan if ha is None or hb is None else min(ha, hb)
            f.write(f"{a}\t{b}\t{int(y)}\t{score:.8g}\t{ha if ha is not None else 'NA'}\t{hb if hb is not None else 'NA'}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=FE.REPS, default="sae_max")
    ap.add_argument("--family", choices=["c3"], default="c3")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    # Global RNG seed for the whole run; the holdout permutation below draws from
    # its own local default_rng(args.seed) stream and is unaffected.
    seed_all(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = FE.load_pooled_cache(CACHE[args.family])

    t_train, deg_train, train_seqs = train_target_t(args.family)
    train_ids = list(t_train.keys())
    X, kept = assemble_protein_features(train_ids, train_seqs, cache, args.rep)
    if X is None:
        raise RuntimeError("no cached train+val proteins")
    y = np.asarray([t_train[p] for p in kept], dtype=np.float32)
    w = np.asarray([deg_train[p] for p in kept], dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(kept))
    n_hold = max(1, int(len(perm) * args.holdout_frac))
    hold, tr = perm[:n_hold], perm[n_hold:]
    reg = fit_xgb_regressor(
        X[tr],
        y[tr],
        X[hold],
        y[hold],
        sample_weight=w[tr],
        sample_weight_eval=w[hold],
        seed=args.seed,
    )

    test_bench = D.load_benchmark(TEST[args.family], attach_seqs=True)
    test_ids = sorted({p for pair in test_bench.pairs for p in pair})
    Xte, kept_te = assemble_protein_features(test_ids, test_bench.seqs, cache, args.rep)
    if Xte is None:
        raise RuntimeError("no cached test proteins")
    pred = np.clip(reg.predict(Xte), 0.0, 1.0)
    pred_map = {pid: float(v) for pid, v in zip(kept_te, pred)}

    def scorer(a, b):
        ha = pred_map.get(a)
        hb = pred_map.get(b)
        return None if ha is None or hb is None else min(ha, hb)

    pair_metrics = evaluate_scorer(scorer, test_bench, name=f"seq_participation_oracle_{args.rep}")
    t_test, deg_test = participation_t(test_bench.pairs, test_bench.labels)
    common = [p for p in kept_te if p in t_test]
    true_test = np.asarray([t_test[p] for p in common], dtype=np.float32)
    pred_test = np.asarray([pred_map[p] for p in common], dtype=np.float32)
    deg_test_arr = np.asarray([deg_test[p] for p in common], dtype=np.float32)
    deg3 = [p for p in common if deg_test[p] >= 3]

    protein_metrics = {
        "n_test_proteins": int(len(common)),
        "spearman_that_vs_test_t": safe_spearman(pred_test, true_test),
        "spearman_that_vs_test_t_deg_ge3": safe_spearman(
            np.asarray([pred_map[p] for p in deg3], dtype=np.float32),
            np.asarray([t_test[p] for p in deg3], dtype=np.float32),
        ),
        "n_deg_ge3": int(len(deg3)),
        "mae": float(mean_absolute_error(true_test, pred_test)),
        "rmse": float(mean_squared_error(true_test, pred_test, squared=False)),
        "degree_weighted_mae": float(np.average(np.abs(true_test - pred_test), weights=deg_test_arr)),
        "true_t_mean": float(true_test.mean()),
        "that_mean": float(pred_test.mean()),
    }

    dim = X.shape[1]
    names = feature_names(args.rep, dim)
    imp = booster_importance(reg, dim, names)
    imp["train_weighted_corr_with_t"] = weighted_corr_columns(X, y, w)
    imp = add_annotations(imp, args.rep)
    imp["abs_train_weighted_corr_with_t"] = imp["train_weighted_corr_with_t"].abs()
    imp = imp.sort_values(["gain", "total_gain", "weight"], ascending=False)

    stem = f"seq_participation_oracle_{args.rep}_{args.family}"
    summary = {
        **pair_metrics,
        "family": args.family,
        "rep": args.rep,
        "train_splits": TRAINVAL[args.family],
        "test_split": TEST[args.family],
        "n_train_proteins": int(len(tr)),
        "n_holdout_proteins": int(len(hold)),
        "n_test_proteins_scored": int(len(kept_te)),
        "protein_t_prediction": protein_metrics,
        "model": {
            "kind": "xgboost_regressor",
            "objective": "reg:logistic",
            "eval_metric": "logloss",
            "best_iteration": int(getattr(reg, "best_iteration", -1)),
        },
        "outputs": {
            "feature_importance_tsv": str(args.out_dir / f"{stem}_feature_importance.tsv"),
            "top_features_annotated_tsv": str(args.out_dir / f"{stem}_top_features_annotated.tsv"),
            "test_protein_predictions_tsv": str(args.out_dir / f"{stem}_test_protein_predictions.tsv"),
            "test_pair_predictions_tsv": str(args.out_dir / f"{stem}_test_pair_predictions.tsv"),
        },
    }

    (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    imp.to_csv(args.out_dir / f"{stem}_feature_importance.tsv", sep="\t", index=False)
    imp.head(200).to_csv(args.out_dir / f"{stem}_top_features_annotated.tsv", sep="\t", index=False)
    write_protein_predictions(
        args.out_dir / f"{stem}_test_protein_predictions.tsv",
        ids=common,
        t_true=t_test,
        degree=deg_test,
        t_hat=pred_map,
    )
    write_pair_predictions(args.out_dir / f"{stem}_test_pair_predictions.tsv", bench=test_bench, t_hat=pred_map)

    print(
        f"[seq_oracle.{args.rep}.{args.family}] pair AUROC={pair_metrics['auroc']} "
        f"AUPRC={pair_metrics['auprc']} | spearman(t_hat,test_t)="
        f"{protein_metrics['spearman_that_vs_test_t']} "
        f"(deg>=3 {protein_metrics['spearman_that_vs_test_t_deg_ge3']})",
        flush=True,
    )
    print(f"[done] wrote {args.out_dir / (stem + '.json')}", flush=True)


if __name__ == "__main__":
    main()
