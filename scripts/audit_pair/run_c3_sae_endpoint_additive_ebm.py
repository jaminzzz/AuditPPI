#!/usr/bin/env python3
"""Train a no-interaction endpoint EBM for C3.

This uses InterpretML's Explainable Boosting Machine, but keeps the endpoint
diagnostic restriction:

    alpha(p) = EBM_no_interactions(SAE_p) - EBM_intercept
    logit PPI(A, B) = alpha(A) + alpha(B)

The EBM is trained on endpoint occurrences: each pair contributes two protein
rows with the pair label. At pair scoring time the two endpoint scores are
added. We set ``interactions=0`` in EBM, so alpha(p) is an additive sum of
single-feature shape functions. The final pair score has no global bias.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import numpy as np
import pandas as pd
import torch

from conf.model import DEFAULT_SEED
from conf.paths import RESULTS_PAIR, PAIR_CACHES as SAE_REP_ROOT
from src.eval.metrics import pair_score_metrics as metrics
from src.experiments.results import dump_experiment
from src.runtime import seed_all
from src.interpretability.annotations import add_sae_annotations
from src.interpretability.ebm_effects import (
    ebm_feature_tables,
    endpoint_matrix,
)
from src.models.estimators.ebm import (
    endpoint_pair_predictions,
    make_endpoint_ebm,
)

OUT_DIR = RESULTS_PAIR / "c3_endpoint_additive_ebm_sae"

# rep name -> subdirectory under a pair-cache root. A pair-cache root holds
# one {split}_embeddings.pt per rep subdir; --cache-root repoints to another
# dataset's cache built with the same layout.
REP_SUBDIR = {
    "sae_max": "sae_max",
    "binary": "binary_thr0",
}


def load_split(rep: str, split: str, cache_root: Path) -> dict[str, np.ndarray]:
    d = torch.load(cache_root / REP_SUBDIR[rep] / f"{split}_embeddings.pt", map_location="cpu", weights_only=False)
    return {
        "a": d["emb_a"].numpy(),
        "b": d["emb_b"].numpy(),
        "y": d["label"].float().numpy().astype(np.int8),
    }


def endpoint_corr_feature_selection(train: dict[str, np.ndarray], top_k: int, chunk_size: int) -> pd.DataFrame:
    a = train["a"]
    b = train["b"]
    y_pair = train["y"].astype(np.float64)
    y = np.concatenate([y_pair, y_pair])
    yc = y - y.mean()
    y_std = y.std()
    rows = []
    for start in range(0, a.shape[1], chunk_size):
        end = min(start + chunk_size, a.shape[1])
        x = np.concatenate([a[:, start:end], b[:, start:end]], axis=0).astype(np.float64, copy=False)
        xm = x.mean(axis=0)
        xs = x.std(axis=0)
        cov = ((x - xm) * yc[:, None]).mean(axis=0)
        corr = np.divide(cov, xs * y_std, out=np.zeros_like(cov), where=((xs > 0) & (y_std > 0)))
        for j, c in enumerate(corr):
            rows.append((start + j, float(c), float(abs(c))))
    out = pd.DataFrame(rows, columns=["feature_id", "endpoint_label_corr", "abs_endpoint_label_corr"])
    return out.sort_values("abs_endpoint_label_corr", ascending=False).head(top_k).reset_index(drop=True)


def pair_predictions(
    ebm, split: dict[str, np.ndarray], feature_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xa = split["a"][:, feature_ids].astype(np.float32, copy=False)
    xb = split["b"][:, feature_ids].astype(np.float32, copy=False)
    return endpoint_pair_predictions(ebm, xa, xb)


def write_pair_predictions(path: Path, split: dict[str, np.ndarray], prob: np.ndarray, alpha_a: np.ndarray, alpha_b: np.ndarray) -> None:
    pd.DataFrame(
        {
            "pair_index": np.arange(split["y"].size),
            "label": split["y"],
            "prob": prob,
            "alpha_a": alpha_a,
            "alpha_b": alpha_b,
            "logit_no_bias": alpha_a + alpha_b,
        }
    ).to_csv(path, sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=sorted(REP_SUBDIR), default="sae_max")
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=SAE_REP_ROOT,
        help="Pair-cache root holding {rep}/{split}_embeddings.pt. Repoint to "
        "another dataset's cache built with the same layout.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--selection-chunk-size", type=int, default=512)
    ap.add_argument("--max-bins", type=int, default=256)
    ap.add_argument("--outer-bags", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--max-rounds", type=int, default=5000)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--min-samples-leaf", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[load] embeddings", flush=True)
    train = load_split(args.rep, "train", args.cache_root)
    val = load_split(args.rep, "val", args.cache_root)
    test = load_split(args.rep, "test", args.cache_root)
    print(
        f"[data] train={train['y'].size:,} val={val['y'].size:,} test={test['y'].size:,} "
        f"dim={train['a'].shape[1]:,}",
        flush=True,
    )

    selected = endpoint_corr_feature_selection(train, args.top_k, args.selection_chunk_size)
    feature_ids = selected["feature_id"].to_numpy(dtype=np.int32)
    x_endpoint, y_endpoint = endpoint_matrix(train, feature_ids)
    feature_names = [f"sae_{int(fid):05d}" for fid in feature_ids]
    print(f"[fit] EBM endpoint rows={x_endpoint.shape[0]:,} features={x_endpoint.shape[1]:,} interactions=0", flush=True)
    ebm = make_endpoint_ebm(
        feature_names=feature_names,
        validation_size=0.15,
        outer_bags=args.outer_bags,
        learning_rate=args.learning_rate,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        min_samples_leaf=args.min_samples_leaf,
        max_bins=args.max_bins,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    ebm.fit(x_endpoint, y_endpoint)

    pred_train, alpha_train_a, alpha_train_b = pair_predictions(ebm, train, feature_ids)
    pred_val, alpha_val_a, alpha_val_b = pair_predictions(ebm, val, feature_ids)
    pred_test, alpha_test_a, alpha_test_b = pair_predictions(ebm, test, feature_ids)

    summary_df, effects_df = ebm_feature_tables(
        ebm, test, feature_ids, selected, args.rep
    )
    stem = (
        f"c3_endpoint_additive_ebm_{args.rep}_top{args.top_k}_bins{args.max_bins}_"
        f"rounds{args.max_rounds}_nobias_nointer"
    )
    result = {
        "model": "endpoint_additive_ebm_no_interactions_no_pair_bias",
        "formula": "logit(PPI(A,B)) = EBM_no_interactions(SAE_A)-intercept + EBM_no_interactions(SAE_B)-intercept",
        "rep": args.rep,
        "seed": args.seed,
        "top_k": int(args.top_k),
        "interactions": 0,
        "pair_global_bias": False,
        "ebm_endpoint_intercept_removed_at_pair_scoring": True,
        "train": metrics(train["y"], pred_train),
        "val": metrics(val["y"], pred_val),
        "test": metrics(test["y"], pred_test),
        "alpha_summary": {
            "train_alpha_a_mean": float(alpha_train_a.mean()),
            "train_alpha_b_mean": float(alpha_train_b.mean()),
            "val_alpha_a_mean": float(alpha_val_a.mean()),
            "val_alpha_b_mean": float(alpha_val_b.mean()),
            "test_alpha_a_mean": float(alpha_test_a.mean()),
            "test_alpha_b_mean": float(alpha_test_b.mean()),
        },
        "hyperparameters": {
            "max_bins": args.max_bins,
            "outer_bags": args.outer_bags,
            "learning_rate": args.learning_rate,
            "max_rounds": args.max_rounds,
            "early_stopping_rounds": args.early_stopping_rounds,
            "min_samples_leaf": args.min_samples_leaf,
        },
        "outputs": {
            "selected_features_tsv": str(args.out_dir / f"{stem}_selected_features.tsv"),
            "feature_summary_tsv": str(args.out_dir / f"{stem}_feature_summary_annotated.tsv"),
            "feature_effects_tsv": str(args.out_dir / f"{stem}_feature_effects_annotated.tsv"),
            "test_pair_predictions_tsv": str(args.out_dir / f"{stem}_test_pair_predictions.tsv"),
        },
    }
    result_path = args.out_dir / f"{stem}_metrics.json"
    dump_experiment(
        result_path,
        task="pair.endpoint_additive_ebm",
        dataset="c3",
        features=args.rep,
        split="test",
        model="endpoint_additive_ebm",
        seed=args.seed,
        payload=result,
        metrics=result["test"],
        hyperparameters=result.get("hyperparameters"),
    )
    add_sae_annotations(selected, args.rep).to_csv(
        args.out_dir / f"{stem}_selected_features.tsv", sep="\t", index=False
    )
    summary_df.to_csv(args.out_dir / f"{stem}_feature_summary_annotated.tsv", sep="\t", index=False)
    effects_df.to_csv(args.out_dir / f"{stem}_feature_effects_annotated.tsv", sep="\t", index=False)
    write_pair_predictions(args.out_dir / f"{stem}_test_pair_predictions.tsv", test, pred_test, alpha_test_a, alpha_test_b)

    print("[done] wrote", result_path, flush=True)
    print(
        f"[test] AUROC={result['test']['auroc']:.4f} "
        f"AUPRC={result['test']['auprc']:.4f} Brier={result['test']['brier']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
