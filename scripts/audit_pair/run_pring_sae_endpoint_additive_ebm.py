#!/usr/bin/env python3
"""Train no-interaction endpoint EBM baselines on PRING pair labels.

EBM analogue of ``run_pring_sae_endpoint_additive_mlp.py``. It uses InterpretML's
Explainable Boosting Machine but keeps the endpoint diagnostic restriction:

    alpha(p) = EBM_no_interactions(SAE_p) - EBM_intercept
    logit PPI(A, B) = alpha(A) + alpha(B)

No pair interaction features are used (``interactions=0``), no protein-ID
parameters, and no global pair bias. The EBM is trained on endpoint occurrences:
each pair contributes two protein rows carrying the pair label.

Training is fixed to PRING human (per graph-sampling method BFS/DFS/RANDOM_WALK,
each with native train/val/test splits). After the human model is fit, it scores
TWO eval groups, kept separate:

  * ``human_test`` -- in-distribution held-out human graph (same species as
    train), reported on its own so it is not blended into the zero-shot number.
  * the held-out PRING species (yeast/ecoli/arath) -- zero-shot transfer, each
    scored on its own ``{sp}_test_ppi.txt`` plus a macro mean. Endpoints are read
    from each species' own v1 protein cache (id2idx keyed); the human EBM is
    applied unchanged since every cache shares the same SAE feature space.

Feature selection (top-K by |endpoint-label correlation|) is fit on the human
train endpoints only; the selected columns are then applied unchanged to every
eval group so scores stay comparable across species.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import numpy as np
import pandas as pd
import torch

from conf.model import BACKBONES, DEFAULT_BACKBONE, DEFAULT_SEED, resolve_backbone_layer
from conf.paths import (
    RESULTS_PAIR,
    PRING_ROOT,
    PRING_HUMAN_SAE_CACHE,
    PRING_CROSS_SPECIES,
    PRING_SPECIES_SAE_CACHES,
)
from src.data.pring_graph import METHODS
from src.eval.metrics import pair_score_metrics as metrics
from src.experiments.results import dump_experiment
from src.features.pairs import load_protein_feature_cache
from src.features.protein_cache import representation_matrix
from src.features.sampling import stratified_subsample
from src.runtime import seed_all
from src.interp.annotations import add_sae_annotations
from src.interp.ebm_effects import ebm_feature_tables, endpoint_matrix
from src.models.estimators.ebm import endpoint_pair_predictions, make_endpoint_ebm

HUMAN_CACHE = PRING_HUMAN_SAE_CACHE
OUT_DIR = RESULTS_PAIR / "pring_endpoint_additive_ebm_sae"

REP_CHOICES = ("sae_max", "binary")

# Held-out PRING species scored zero-shot (test-only graphs, no train/val).
ZERO_SHOT_SPECIES = list(PRING_CROSS_SPECIES)


def read_pair_file(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    a_ids: list[str] = []
    b_ids: list[str] = []
    labels: list[int] = []
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            a_ids.append(parts[0])
            b_ids.append(parts[1])
            labels.append(int(parts[2]))
    return a_ids, b_ids, np.asarray(labels, dtype=np.int64)


def load_cache(
    rep: str, cache_path: Path, *, backbone: str = DEFAULT_BACKBONE, layer: int | None = None
) -> tuple[torch.Tensor, dict[str, int]]:
    """Load the endpoint feature matrix + id->row map from a v1 protein cache.

    Reads the ``auditppi_protein_features_v1`` layout, selecting the
    ``(backbone, layer, rep)`` channel via the shared
    :func:`~src.features.protein_cache.representation_matrix`. The matrix stays at
    native dtype here (``binary`` bool / ``sae_max`` float); per-pair endpoint
    rows are cast to float32 when materialized in :func:`load_pairs`.
    """
    cache = load_protein_feature_cache(cache_path)
    mat = representation_matrix(cache, rep, layer, backbone)
    return mat, cache["id2idx"]


def load_pairs(
    path: Path, matrix: torch.Tensor, id_to_idx: dict[str, int]
) -> dict[str, np.ndarray]:
    """Resolve a PRING edge list to dense endpoint arrays via a cache ``id2idx``.

    Works for both the human ``human_{split}_ppi.txt`` files and the held-out
    species ``{sp}_test_ppi.txt`` files: each line is ``id_a id_b label`` and
    every id is looked up in ``id_to_idx``. Pairs whose endpoint id is absent are
    skipped and counted. Returns ``{a, b, y, n_raw, n_skipped, path}`` with
    ``a``/``b`` float32 endpoint arrays (rows gathered once from ``matrix``).
    """
    a_ids, b_ids, y = read_pair_file(path)
    rows_a: list[int] = []
    rows_b: list[int] = []
    kept_y: list[int] = []
    n_skipped = 0
    for a, b, label in zip(a_ids, b_ids, y):
        ia = id_to_idx.get(a)
        ib = id_to_idx.get(b)
        if ia is None or ib is None:
            n_skipped += 1
            continue
        rows_a.append(int(ia))
        rows_b.append(int(ib))
        kept_y.append(int(label))
    ia_t = torch.as_tensor(rows_a, dtype=torch.long)
    ib_t = torch.as_tensor(rows_b, dtype=torch.long)
    emb_a = matrix.index_select(0, ia_t).float().numpy()
    emb_b = matrix.index_select(0, ib_t).float().numpy()
    return {
        "a": emb_a,
        "b": emb_b,
        "y": np.asarray(kept_y, dtype=np.int8),
        "path": str(path),
        "n_raw": int(len(y)),
        "n_skipped": int(n_skipped),
    }


def subsample_split(split: dict[str, np.ndarray], max_rows: int | None, seed: int) -> dict[str, np.ndarray]:
    """Class-stratified pair subsample (keeps a/b/y row-aligned)."""
    if max_rows is None or max_rows >= split["y"].size:
        return split
    idx = stratified_subsample(split["y"].astype(np.int64), max_rows, seed)
    if idx is None:
        return split
    out = dict(split)
    out["a"] = split["a"][idx]
    out["b"] = split["b"][idx]
    out["y"] = split["y"][idx]
    return out


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


def human_split_path(method: str, split: str) -> Path:
    return PRING_ROOT / "human" / method / f"human_{split}_ppi.txt"


def fit_method(
    args: argparse.Namespace,
    method: str,
    matrix: torch.Tensor,
    id_to_idx: dict[str, int],
    layer: int,
) -> tuple[object, np.ndarray, pd.DataFrame, dict, dict]:
    """Fit one human graph-sampling method's endpoint EBM; return model + tables."""
    seed_all(args.seed)
    train = subsample_split(
        load_pairs(human_split_path(method, "train"), matrix, id_to_idx),
        args.train_subsample,
        args.seed,
    )
    val = load_pairs(human_split_path(method, "val"), matrix, id_to_idx)
    test = load_pairs(human_split_path(method, "test"), matrix, id_to_idx)
    print(
        f"[data] {method} rep={args.rep} backbone={args.backbone} layer={layer} "
        f"train={train['y'].size:,} val={val['y'].size:,} test={test['y'].size:,} "
        f"dim={train['a'].shape[1]:,} skipped="
        f"{train['n_skipped']}/{val['n_skipped']}/{test['n_skipped']}",
        flush=True,
    )

    selected = endpoint_corr_feature_selection(train, args.top_k, args.selection_chunk_size)
    feature_ids = selected["feature_id"].to_numpy(dtype=np.int32)
    x_endpoint, y_endpoint = endpoint_matrix(train, feature_ids)
    feature_names = [f"sae_{int(fid):05d}" for fid in feature_ids]
    print(
        f"[fit {method}] EBM endpoint rows={x_endpoint.shape[0]:,} "
        f"features={x_endpoint.shape[1]:,} interactions=0",
        flush=True,
    )
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

    pred_train, at_a, at_b = pair_predictions(ebm, train, feature_ids)
    pred_val, av_a, av_b = pair_predictions(ebm, val, feature_ids)
    pred_test, ate_a, ate_b = pair_predictions(ebm, test, feature_ids)
    summary_df, effects_df = ebm_feature_tables(ebm, test, feature_ids, selected, args.rep)

    result = {
        "model": "pring_endpoint_additive_ebm_no_interactions_no_pair_bias",
        "formula": "logit(PPI(A,B)) = EBM_no_interactions(SAE_A)-intercept + EBM_no_interactions(SAE_B)-intercept",
        "method": method,
        "rep": args.rep,
        "backbone": args.backbone,
        "layer": layer,
        "seed": args.seed,
        "top_k": int(args.top_k),
        "interactions": 0,
        "pair_global_bias": False,
        "ebm_endpoint_intercept_removed_at_pair_scoring": True,
        "train": metrics(train["y"], pred_train),
        "val": metrics(val["y"], pred_val),
        "test": metrics(test["y"], pred_test),
        "alpha_summary": {
            "train_alpha_a_mean": float(at_a.mean()),
            "train_alpha_b_mean": float(at_b.mean()),
            "val_alpha_a_mean": float(av_a.mean()),
            "val_alpha_b_mean": float(av_b.mean()),
            "test_alpha_a_mean": float(ate_a.mean()),
            "test_alpha_b_mean": float(ate_b.mean()),
        },
        "input_files": {
            "cache": str(args.cache_path),
            "train": train["path"],
            "val": val["path"],
            "test": test["path"],
        },
        "n_skipped": {
            "train": int(train["n_skipped"]),
            "val": int(val["n_skipped"]),
            "test": int(test["n_skipped"]),
        },
        "hyperparameters": {
            "max_bins": args.max_bins,
            "outer_bags": args.outer_bags,
            "learning_rate": args.learning_rate,
            "max_rounds": args.max_rounds,
            "early_stopping_rounds": args.early_stopping_rounds,
            "min_samples_leaf": args.min_samples_leaf,
            "train_subsample": args.train_subsample,
        },
    }
    aux = {
        "selected": selected,
        "feature_ids": feature_ids,
        "summary_df": summary_df,
        "effects_df": effects_df,
        "test_split": test,
        "test_pred": (pred_test, ate_a, ate_b),
    }
    return ebm, feature_ids, selected, result, aux


def evaluate_species(
    args: argparse.Namespace,
    ebm,
    feature_ids: np.ndarray,
    species: str,
    layer: int,
) -> tuple[dict, dict, dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Zero-shot score the human EBM on a held-out species' ``{sp}_test_ppi.txt``.

    Endpoints are read from the species' own v1 protein cache (id2idx keyed) and
    reduced to the same selected feature columns; the human-trained EBM is applied
    unchanged since every PRING cache shares the SAE feature space.
    """
    matrix, id_to_idx = load_cache(
        args.rep, PRING_SPECIES_SAE_CACHES[species], backbone=args.backbone, layer=layer
    )
    test_path = PRING_ROOT / species / f"{species}_test_ppi.txt"
    split = load_pairs(test_path, matrix, id_to_idx)
    pack = pair_predictions(ebm, split, feature_ids)
    m = metrics(split["y"], pack[0])
    info = {
        "species": species,
        "test_path": str(test_path),
        "cache": str(PRING_SPECIES_SAE_CACHES[species]),
        "n_raw": int(split["n_raw"]),
        "n_skipped": int(split["n_skipped"]),
        "metrics": m,
    }
    return m, info, split, pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=[*METHODS, "all"], default="all")
    ap.add_argument("--rep", choices=REP_CHOICES, default="sae_max")
    ap.add_argument("--backbone", choices=BACKBONES, default=DEFAULT_BACKBONE)
    ap.add_argument("--layer", type=int, default=None,
                    help="SAE layer; defaults to the backbone's default layer.")
    ap.add_argument(
        "--cache-path",
        type=Path,
        default=HUMAN_CACHE,
        help="PRING human v1 protein cache (id2idx keyed). Training is fixed to "
        "PRING human; species are a held-out test concern.",
    )
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for metrics/tables/summary outputs. Defaults to "
                    "RESULTS_PAIR/pring_endpoint_additive_ebm_sae/seed_{seed}.")
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
    ap.add_argument("--train-subsample", type=int, default=None,
                    help="Cap human train pairs (class-stratified) before "
                    "materializing endpoints. Default: keep all.")
    ap.add_argument("--write-predictions", action="store_true")
    ap.add_argument(
        "--zero-shot-species",
        nargs="+",
        default=list(PRING_CROSS_SPECIES),
        help="Held-out PRING species scored zero-shot with the human model "
        "(default: yeast/ecoli/arath). Pass an empty list to skip.",
    )
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = OUT_DIR / f"seed_{args.seed}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layer = resolve_backbone_layer(args.backbone, args.layer)
    bb_tag = f"{args.backbone}_l{layer}"
    matrix, id_to_idx = load_cache(
        args.rep, args.cache_path, backbone=args.backbone, layer=layer
    )
    methods = METHODS if args.method == "all" else (args.method,)
    zero_shot_species = list(args.zero_shot_species)
    all_results = {}
    rows = []
    for method in methods:
        ebm, feature_ids, selected, result, aux = fit_method(
            args, method, matrix, id_to_idx, layer
        )

        # --- zero-shot: score the human EBM on each held-out species ----------
        species_results = {}
        if zero_shot_species:
            for species in zero_shot_species:
                m, info, split, pack = evaluate_species(args, ebm, feature_ids, species, layer)
                species_results[species] = info
                print(
                    f"[{method} zero-shot {species}] "
                    f"AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                    f"(n={info['n_raw']} skipped={info['n_skipped']})",
                    flush=True,
                )
                if args.write_predictions:
                    write_pair_predictions(
                        args.out_dir
                        / f"pring_{method.lower()}_zeroshot_{species}_pair_predictions.tsv",
                        split, pack[0], pack[1], pack[2],
                    )
            if species_results:
                result["mean_species_auroc"] = float(
                    np.mean([r["metrics"]["auroc"] for r in species_results.values()])
                )
                result["mean_species_auprc"] = float(
                    np.mean([r["metrics"]["auprc"] for r in species_results.values()])
                )
        result["zero_shot_species"] = species_results
        all_results[method] = result

        stem = (
            f"pring_{method.lower()}_endpoint_additive_ebm_{args.rep}_{bb_tag}_"
            f"top{args.top_k}_bins{args.max_bins}_rounds{args.max_rounds}_nobias_nointer"
        )
        result_path = args.out_dir / f"{stem}_metrics.json"
        dump_experiment(
            result_path,
            task="pair.endpoint_additive_ebm",
            dataset=f"pring_human_{method.lower()}",
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
        aux["summary_df"].to_csv(args.out_dir / f"{stem}_feature_summary_annotated.tsv", sep="\t", index=False)
        aux["effects_df"].to_csv(args.out_dir / f"{stem}_feature_effects_annotated.tsv", sep="\t", index=False)
        if args.write_predictions:
            pred_test, ate_a, ate_b = aux["test_pred"]
            write_pair_predictions(
                args.out_dir / f"{stem}_test_pair_predictions.tsv",
                aux["test_split"], pred_test, ate_a, ate_b,
            )

        # human train/val/test rows (group=in_distribution for the held-out test)
        for split_name in ("train", "val", "test"):
            m = result[split_name]
            rows.append(
                {
                    "rep": args.rep,
                    "method": method,
                    "eval": f"human_{split_name}",
                    "group": "in_distribution" if split_name == "test" else split_name,
                    "n": m["n"],
                    "pos_rate": m["pos_rate"],
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "accuracy_at_0.5": m["accuracy_at_0.5"],
                    "brier": m["brier"],
                }
            )
        for species, info in species_results.items():
            m = info["metrics"]
            rows.append(
                {
                    "rep": args.rep,
                    "method": method,
                    "eval": species,
                    "group": "zero_shot",
                    "n": m["n"],
                    "pos_rate": m["pos_rate"],
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "accuracy_at_0.5": m["accuracy_at_0.5"],
                    "brier": m["brier"],
                }
            )
        mean_msg = (
            f" mean_species_auroc={result['mean_species_auroc']:.4f}"
            if "mean_species_auroc" in result
            else ""
        )
        print(
            f"[{method} test] AUROC={result['test']['auroc']:.4f} "
            f"AUPRC={result['test']['auprc']:.4f}{mean_msg} -> {result_path}",
            flush=True,
        )

    summary_path = args.out_dir / f"pring_endpoint_additive_ebm_{args.rep}_{bb_tag}_top{args.top_k}_nobias_nointer_summary.json"
    tsv_path = args.out_dir / f"pring_endpoint_additive_ebm_{args.rep}_{bb_tag}_top{args.top_k}_nobias_nointer_metrics_summary.tsv"
    dump_experiment(
        summary_path,
        task="pair.endpoint_additive_ebm",
        dataset="pring_human",
        features=args.rep,
        split="multi",
        model="endpoint_additive_ebm",
        seed=args.seed,
        payload=all_results,
        metrics={
            method: {
                "human_test_auroc": res["test"]["auroc"],
                "human_test_auprc": res["test"]["auprc"],
                "mean_species_auroc": res.get("mean_species_auroc"),
                "mean_species_auprc": res.get("mean_species_auprc"),
            }
            for method, res in all_results.items()
        },
        hyperparameters={
            "top_k": args.top_k,
            "methods": list(methods),
            "zero_shot_species": zero_shot_species,
        },
    )
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print("[done] wrote", summary_path, flush=True)
    print("[done] wrote", tsv_path, flush=True)


if __name__ == "__main__":
    main()
