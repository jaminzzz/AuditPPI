#!/usr/bin/env python3
"""Train a no-interaction endpoint EBM for a benchmark family.

This uses InterpretML's Explainable Boosting Machine, but keeps the endpoint
diagnostic restriction:

    alpha(p) = EBM_no_interactions(SAE_p) - EBM_intercept
    logit PPI(A, B) = alpha(A) + alpha(B)

The EBM is trained on endpoint occurrences: each pair contributes two protein
rows with the pair label. At pair scoring time the two endpoint scores are
added. We set ``interactions=0`` in EBM, so alpha(p) is an additive sum of
single-feature shape functions. The final pair score has no global bias.

The family is selected by ``--family``:

  * c1/c2/c3      -- lightweight pair-index caches (seq2idx keyed, pre-filtered);
                     endpoints gathered via ``materialize_pair_endpoints``.
  * bernett       -- no pair-index cache; pairs resolved on the fly through the
                     shared protein cache's seq2idx via ``pair_feature_row_indices``
                     (row indices only), then the kept endpoint rows are gathered.
  * cross_species -- pair-index caches exist but ship no official val, so a
                     stratified val is carved from ``human_train`` and the train
                     side is capped (``--train-subsample``, default 100k -- EBM
                     must materialize the full endpoint arrays for feature
                     selection, unlike the batch-gather MLP, so the 421k-pair
                     graph is subsampled to stay inside the memory budget). The
                     in-distribution ``human_test`` graph is the test set.

Unlike the MLP additive script this materializes the full endpoint feature
arrays (EBM's correlation feature-selection and fit need dense numpy), so large
graphs are subsampled via ``--train-subsample``. PRING is per-method with
zero-shot species and keeps its own driver
(``run_pring_sae_endpoint_additive_ebm.py``).
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
    CLEVEL_PAIR_INDEX_CACHES,
    CROSS_SPECIES_PAIR_INDEX_CACHES,
    PPI_PREDICTION_CACHES,
)
from src.data.pairs import load_benchmark
from src.eval.metrics import pair_score_metrics as metrics
from src.experiments.results import dump_experiment
from src.features.pairs import (
    load_pair_index_cache,
    load_protein_feature_cache,
    materialize_pair_endpoints,
)
from src.features.protein_cache import pair_feature_row_indices, representation_matrix
from src.interp.pair_probe import carve_cross_species_train_val
from src.runtime import seed_all
from src.interp.annotations import add_sae_annotations
from src.interp.ebm_effects import (
    ebm_feature_tables,
    endpoint_matrix,
)
from src.models.estimators.ebm import (
    endpoint_pair_predictions,
    make_endpoint_ebm,
)

# Families with a native (train, val, test) triple this script can fit. c1/c2/c3
# and cross_species carry pair-index caches; bernett resolves pairs on the fly.
# cross_species has no official val (carved from human_train) and its train side
# is capped since EBM must materialize dense endpoint arrays.
FAMILIES = ("c1", "c2", "c3", "bernett", "cross_species")

# rep choices exposed on the CLI; both resolve to a v1 protein-cache channel via
# materialize_pair_endpoints (sae_max -> sae_max, binary -> sae_binary).
REP_CHOICES = ("sae_max", "binary")

# Default cross_species carve: matches the fingerprint-baseline / tabpfn-topk
# convention (stratified 10% val off human_train; train capped at 100k).
CROSS_SPECIES_VAL_FRAC = 0.1
CROSS_SPECIES_TRAIN_SUBSAMPLE = 100_000


def _endpoints_to_split(emb_a, emb_b, labels) -> dict[str, np.ndarray]:
    """Pack gathered endpoint tensors/arrays into the dense EBM split dict."""
    import torch

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.float().numpy()
        return np.asarray(x, dtype=np.float32)

    if isinstance(labels, torch.Tensor):
        y = labels.float().numpy().astype(np.int8)
    else:
        y = np.asarray(labels).astype(np.int8)
    return {"a": _to_np(emb_a), "b": _to_np(emb_b), "y": y}


def _clevel_split(
    family: str, rep: str, split: str, protein_cache: dict, *, backbone: str, layer: int | None
) -> dict[str, np.ndarray]:
    index_cache = load_pair_index_cache(CLEVEL_PAIR_INDEX_CACHES[family][split])
    emb_a, emb_b, labels = materialize_pair_endpoints(
        index_cache, protein_cache, rep=rep, backbone=backbone, layer=layer
    )
    return _endpoints_to_split(emb_a, emb_b, labels)


def _bernett_split(
    rep: str, split: str, protein_cache: dict, *, backbone: str, layer: int | None
) -> dict[str, np.ndarray]:
    """Bernett endpoints resolved on the fly (no pair-index cache).

    Pairs are keyed by sequence hash; ``pair_feature_row_indices`` resolves them
    through the shared protein cache's ``seq2idx`` and returns row indices only
    (never materializes the full pair graph). The kept endpoint rows are then
    gathered once at native dtype and cast to float for the EBM.
    """
    import torch

    bench = load_benchmark(f"bernett:{split}", attach_seqs=True)
    resolved = pair_feature_row_indices(bench, protein_cache, rep, layer, backbone)
    if resolved is None:
        raise RuntimeError(f"bernett:{split} had no pair with both endpoints cached")
    matrix, rows_a, rows_b, ys, _kept = resolved
    ia = torch.as_tensor(rows_a, dtype=torch.long)
    ib = torch.as_tensor(rows_b, dtype=torch.long)
    emb_a = matrix.index_select(0, ia)
    emb_b = matrix.index_select(0, ib)
    return _endpoints_to_split(emb_a, emb_b, ys)


def _cross_species_splits(
    rep: str, protein_cache: dict, *, backbone: str, layer: int | None,
    seed: int, val_frac: float, train_subsample: int | None,
) -> dict[str, dict[str, np.ndarray]]:
    """All three cross_species splits in one carve (memory-safe).

    ``train``/``val`` come from :func:`carve_cross_species_train_val`, which
    selects the row indices from the cheap pair-index labels FIRST and gathers
    only the kept endpoint rows (never the 421k-pair float32 graph). ``test`` is
    the in-distribution ``human_test`` graph. Returns a dict keyed by split.
    """
    train_a, train_b, train_y, val_a, val_b, val_y = carve_cross_species_train_val(
        CROSS_SPECIES_PAIR_INDEX_CACHES["human_train"],
        protein_cache,
        rep=rep,
        backbone=backbone,
        layer=layer,
        train_subsample=train_subsample,
        val_frac=val_frac,
        seed=seed,
    )
    test_index = load_pair_index_cache(CROSS_SPECIES_PAIR_INDEX_CACHES["human_test"])
    test_a, test_b, test_labels = materialize_pair_endpoints(
        test_index, protein_cache, rep=rep, backbone=backbone, layer=layer
    )
    return {
        "train": _endpoints_to_split(train_a, train_b, train_y),
        "val": _endpoints_to_split(val_a, val_b, val_y),
        "test": _endpoints_to_split(test_a, test_b, test_labels),
    }


def load_splits(
    family: str,
    rep: str,
    protein_cache: dict,
    *,
    backbone: str,
    layer: int | None,
    seed: int,
    val_frac: float,
    train_subsample: int | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Family-agnostic loader returning dense ``{train, val, test}`` EBM splits.

    Each split dict is ``{a, b, y}`` with ``a``/``b`` float32 endpoint arrays and
    ``y`` an int8 label array. Dispatches by family: pair-index route for
    c1/c2/c3, on-the-fly seq2idx resolution for bernett, and a memory-safe
    stratified carve for cross_species.
    """
    if family in CLEVEL_PAIR_INDEX_CACHES:
        return {
            split: _clevel_split(family, rep, split, protein_cache, backbone=backbone, layer=layer)
            for split in ("train", "val", "test")
        }
    if family == "bernett":
        return {
            split: _bernett_split(rep, split, protein_cache, backbone=backbone, layer=layer)
            for split in ("train", "val", "test")
        }
    if family == "cross_species":
        return _cross_species_splits(
            rep, protein_cache, backbone=backbone, layer=layer,
            seed=seed, val_frac=val_frac, train_subsample=train_subsample,
        )
    raise ValueError(f"unknown family {family!r}")


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
    ap.add_argument("--family", choices=FAMILIES, default="c3",
                    help="Benchmark family. c1/c2/c3 and cross_species route "
                    "pair-index caches; bernett resolves pairs on the fly. "
                    "cross_species carves val from human_train and caps train.")
    ap.add_argument("--rep", choices=REP_CHOICES, default="sae_max")
    ap.add_argument("--backbone", choices=BACKBONES, default=DEFAULT_BACKBONE)
    ap.add_argument("--layer", type=int, default=None,
                    help="SAE layer; defaults to the backbone's default layer.")
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
    ap.add_argument("--val-frac", type=float, default=CROSS_SPECIES_VAL_FRAC,
                    help="cross_species only: stratified val fraction carved from "
                    "human_train (ignored for families with a native val split).")
    ap.add_argument("--train-subsample", type=int, default=CROSS_SPECIES_TRAIN_SUBSAMPLE,
                    help="cross_species only: cap on the carved train rows (EBM "
                    "materializes dense endpoint arrays; the 421k-pair human_train "
                    "graph is subsampled to stay inside the memory budget). Pass 0 "
                    "to keep the full graph.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir. Defaults to "
                    "RESULTS_PAIR/{family}_endpoint_additive_ebm_sae.")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = (
            RESULTS_PAIR
            / f"{args.family}_endpoint_additive_ebm_sae"
            / f"seed_{args.seed}"
        )
    layer = resolve_backbone_layer(args.backbone, args.layer)
    seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[load] embeddings", flush=True)
    protein_cache = load_protein_feature_cache(PPI_PREDICTION_CACHES[args.family])
    train_subsample = args.train_subsample if args.train_subsample and args.train_subsample > 0 else None
    splits = load_splits(
        args.family,
        args.rep,
        protein_cache,
        backbone=args.backbone,
        layer=layer,
        seed=args.seed,
        val_frac=args.val_frac,
        train_subsample=train_subsample,
    )
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(
        f"[data] family={args.family} rep={args.rep} backbone={args.backbone} layer={layer} "
        f"train={train['y'].size:,} val={val['y'].size:,} test={test['y'].size:,} "
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
        f"{args.family}_endpoint_additive_ebm_{args.rep}_{args.backbone}_l{layer}_"
        f"top{args.top_k}_bins{args.max_bins}_rounds{args.max_rounds}_nobias_nointer"
    )
    result = {
        "model": "endpoint_additive_ebm_no_interactions_no_pair_bias",
        "formula": "logit(PPI(A,B)) = EBM_no_interactions(SAE_A)-intercept + EBM_no_interactions(SAE_B)-intercept",
        "family": args.family,
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
        dataset=args.family,
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
