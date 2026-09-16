#!/usr/bin/env python3
"""Train a stronger no-bias endpoint-additive SAE model for a C-level family.

The model has more capacity than the linear endpoint-additive baseline, but it
keeps the same diagnostic restriction: it cannot use pair-interaction features.

    alpha(p) = MLP_no_bias(SAE_p)
    logit PPI(A, B) = alpha(A) + alpha(B)

There is no global/pair bias term, and all Linear layers default to
``bias=False``. Feature explanations are computed for the single-protein
endpoint score with gradient*input attribution:

    attr_i(p) = SAE_p_i * d alpha(p) / d SAE_p_i

The global feature table aggregates attribution over endpoint occurrences.

Runs on any RAPPPID C-level family (``--family c1|c2|c3``). Endpoint rows come
from that family's lightweight ``auditppi_pair_index_v1`` pair-index cache
(``rows_a``/``rows_b``/``labels`` already pre-filtered to cached endpoints), which
indexes the family's ``auditppi_protein_features_v1`` protein cache. The desired
``(backbone, layer, rep)`` channel is selected by
:func:`~src.features.protein_cache.representation_matrix`. Shared endpoints are
stored once and referenced by row index, so one cache serves every
backbone/layer/rep and no per-pair ``emb_a``/``emb_b`` are materialized.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    resolve_backbone_layer,
)
from conf.paths import (
    RESULTS_PAIR,
    CLEVEL_PAIR_INDEX_CACHES,
    CROSS_SPECIES_PAIR_INDEX_CACHES,
    PPI_PREDICTION_CACHES,
)
from src.data.pairs import load_benchmark
from src.eval.metrics import pair_score_metrics as metrics
from src.experiments.results import dump_experiment
from src.features.pairs import load_pair_index_cache, load_protein_feature_cache
from src.features.protein_cache import pair_feature_row_indices, representation_matrix
from src.ppi_fingerprint.baseline import stratified_subsample
from src.runtime import seed_all
from src.interp.annotations import add_sae_annotations
from src.interp.attribution import (
    endpoint_gradient_input_attribution,
)
from src.models.architectures.mlp_endpoint import MLPEndpoint

# Families with a native (train, val, test) triple this script can fit:
#   c1/c2/c3       -- lightweight pair-index caches (seq2idx keyed, pre-filtered).
#   bernett        -- no pair-index cache; pairs resolved on the fly through the
#                     shared protein cache's seq2idx via pair_feature_row_indices.
#   cross_species  -- pair-index caches exist (human_train / human_test / species)
#                     but ship no official val, so val is carved stratified from
#                     human_train and the in-distribution human_test is the "test".
# PRING is per-species (human train graph + zero-shot species) and keeps its own
# driver (run_pring_sae_endpoint_additive_mlp.py).
FAMILIES = ("c1", "c2", "c3", "bernett", "cross_species")
REPS = ("sae_max", "binary")
# Default cross_species val carve: matches the fingerprint-baseline / tabpfn-topk
# convention (stratified 10% off human_train). Training uses memory-safe
# batch-gather, so the full human_train graph is kept (no train subsample).
CROSS_SPECIES_VAL_FRAC = 0.1


def matrix_from_cache(
    cache: dict, rep: str, *, backbone: str = DEFAULT_BACKBONE, layer: int | None = None
) -> torch.Tensor:
    """Select + cast the endpoint feature matrix from a loaded v1 protein cache.

    Reads the ``auditppi_protein_features_v1`` layout, selecting the
    ``(backbone, layer, rep)`` channel via the shared
    :func:`~src.features.protein_cache.representation_matrix`. ``binary`` arrives
    as bool and is cast to uint8 (the MLP's ``float()`` upcast is applied per
    batch), ``sae_max`` stays float. Rows are indexed by the pair splits.
    """
    mat = representation_matrix(cache, rep, layer, backbone)
    if rep == "binary":
        mat = mat.to(torch.uint8)
    elif rep == "sae_max":
        mat = mat.float()
    else:
        raise ValueError(rep)
    return mat


def _clevel_split(family: str, split: str) -> dict:
    """C-level split endpoint rows straight out of the pair-index cache.

    Reads ``rows_a``/``rows_b``/``labels`` from the family's
    ``auditppi_pair_index_v1`` cache. The cache is pre-filtered to pairs whose
    endpoints are present in the protein cache, so nothing is skipped; both row
    indices reference the shared protein-feature matrix.
    """
    index_cache = load_pair_index_cache(CLEVEL_PAIR_INDEX_CACHES[family][split])
    labels = index_cache["labels"].to(torch.float32)
    return {
        "rows_a": index_cache["rows_a"].to(torch.long),
        "rows_b": index_cache["rows_b"].to(torch.long),
        "y": labels,
        "split": split,
        "n_raw": int(labels.numel()),
        "n_skipped": 0,
    }


def _bernett_split(
    split: str, cache: dict, *, rep: str, backbone: str, layer: int | None
) -> dict:
    """Bernett split endpoint rows resolved on the fly (no pair-index cache).

    Bernett pairs are keyed by sequence hash and have no pre-built pair-index
    cache, so pairs are resolved through the shared protein cache's ``seq2idx``
    with :func:`~src.features.protein_cache.pair_feature_row_indices`, which
    returns row indices only (never materializes the pair-endpoint graph). Pairs
    whose either endpoint is absent from the cache are skipped and counted.
    """
    bench = load_benchmark(f"bernett:{split}", attach_seqs=True)
    n_raw = len(bench.pairs)
    resolved = pair_feature_row_indices(bench, cache, rep, layer, backbone)
    if resolved is None:
        raise RuntimeError(f"bernett:{split} had no pair with both endpoints cached")
    _matrix, rows_a, rows_b, ys, kept = resolved
    return {
        "rows_a": torch.as_tensor(rows_a, dtype=torch.long),
        "rows_b": torch.as_tensor(rows_b, dtype=torch.long),
        "y": torch.as_tensor(ys, dtype=torch.float32),
        "split": split,
        "n_raw": int(n_raw),
        "n_skipped": int(n_raw - len(kept)),
    }


def _cross_species_split(split: str, *, seed: int, val_frac: float) -> dict:
    """Cross-species split endpoint rows from the pair-index caches.

    ``train``/``val`` are carved from the ``human_train`` graph (no official val):
    a stratified ``val_frac`` slice (``seed+1``) is held out and the remainder is
    train -- the same seeds/order as the fingerprint baseline and tabpfn-topk, so
    the two calls yield disjoint, deterministic slices. ``test`` is the
    in-distribution ``human_test`` graph. All three index the shared cross_species
    protein-feature matrix; training keeps the full train graph (batch-gather is
    memory-safe, so no subsample is needed).
    """
    if split == "test":
        index_cache = load_pair_index_cache(CROSS_SPECIES_PAIR_INDEX_CACHES["human_test"])
        labels = index_cache["labels"].to(torch.float32)
        return {
            "rows_a": index_cache["rows_a"].to(torch.long),
            "rows_b": index_cache["rows_b"].to(torch.long),
            "y": labels,
            "split": split,
            "n_raw": int(labels.numel()),
            "n_skipped": 0,
        }

    index_cache = load_pair_index_cache(CROSS_SPECIES_PAIR_INDEX_CACHES["human_train"])
    labels_full = index_cache["labels"].numpy().astype(np.int64, copy=False)
    n = len(labels_full)
    n_val = max(1, int(n * val_frac))
    val_idx = stratified_subsample(labels_full, n_val, seed + 1)
    if val_idx is None:
        raise ValueError(f"cross_species val carve failed: n={n} val_frac={val_frac}")
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    sel = val_idx if split == "val" else np.flatnonzero(~val_mask)

    sel_t = torch.as_tensor(sel, dtype=torch.long)
    rows_a = index_cache["rows_a"].to(torch.long).index_select(0, sel_t)
    rows_b = index_cache["rows_b"].to(torch.long).index_select(0, sel_t)
    return {
        "rows_a": rows_a,
        "rows_b": rows_b,
        "y": torch.as_tensor(labels_full[sel], dtype=torch.float32),
        "split": split,
        "n_raw": int(sel.size),
        "n_skipped": 0,
    }


def load_split(
    family: str,
    split: str,
    *,
    cache: dict,
    rep: str,
    backbone: str,
    layer: int | None,
    seed: int,
    val_frac: float,
) -> dict:
    """Family-agnostic endpoint-row loader.

    Returns ``{rows_a, rows_b, y, split, n_raw, n_skipped}`` with ``rows_a``/
    ``rows_b`` long tensors indexing the shared protein-feature matrix and ``y``
    a float32 label tensor. Dispatches by family: pair-index route for c1/c2/c3,
    on-the-fly seq2idx resolution for bernett, and a stratified train/val carve
    (plus in-distribution human_test) for cross_species.
    """
    if family in CLEVEL_PAIR_INDEX_CACHES:
        return _clevel_split(family, split)
    if family == "bernett":
        return _bernett_split(split, cache, rep=rep, backbone=backbone, layer=layer)
    if family == "cross_species":
        return _cross_species_split(split, seed=seed, val_frac=val_frac)
    raise ValueError(f"unknown family {family!r}")


def batch_vectors(mat: torch.Tensor, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    rows = rows.to(mat.device, non_blocking=True)
    return mat.index_select(0, rows).to(device, non_blocking=True)


def materialize_endpoints(mat: torch.Tensor, split: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the per-pair endpoint vectors for attribution (float, on CPU)."""
    a = mat.index_select(0, split["rows_a"]).float()
    b = mat.index_select(0, split["rows_b"]).float()
    return a, b


@torch.no_grad()
def predict(
    model: MLPEndpoint,
    mat: torch.Tensor,
    split: dict,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    alpha_a = []
    alpha_b = []
    n = int(split["y"].numel())
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        a = batch_vectors(mat, split["rows_a"][sl], device)
        b = batch_vectors(mat, split["rows_b"][sl], device)
        aa = model.alpha(a)
        bb = model.alpha(b)
        logits = aa + bb
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        alpha_a.append(aa.detach().cpu().numpy())
        alpha_b.append(bb.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(alpha_a), np.concatenate(alpha_b)


def train(args: argparse.Namespace) -> tuple[MLPEndpoint, dict, dict]:
    seed_all(args.seed)
    resolved_layer = resolve_backbone_layer(args.backbone, args.layer)
    cache = load_protein_feature_cache(args.cache_path)
    mat = matrix_from_cache(
        cache, args.rep, backbone=args.backbone, layer=resolved_layer
    )
    split_kw = dict(
        cache=cache,
        rep=args.rep,
        backbone=args.backbone,
        layer=resolved_layer,
        seed=args.seed,
        val_frac=args.val_frac,
    )
    train_split = load_split(args.family, "train", **split_kw)
    val_split = load_split(args.family, "val", **split_kw)
    test_split = load_split(args.family, "test", **split_kw)

    dim = int(mat.shape[1])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MLPEndpoint(
        dim=dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        layer_bias=args.layer_bias,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    mat_device = mat.to(device) if args.cache_on_device and device.type == "cuda" else mat

    y_train = train_split["y"].numpy().astype(np.int8)
    y_val = val_split["y"].numpy().astype(np.int8)
    y_test = test_split["y"].numpy().astype(np.int8)
    rng = np.random.default_rng(args.seed)
    best = {"epoch": 0, "val_auroc": -np.inf, "state": None}
    history = []
    patience_left = args.patience
    n = int(train_split["y"].numel())
    print(
        f"[data] model=no_bias_mlp_endpoint rep={args.rep} backbone={args.backbone} "
        f"layer={resolved_layer} train={n:,} val={val_split['y'].numel():,} "
        f"test={test_split['y'].numel():,} dim={dim:,} hidden={args.hidden} "
        f"layers={args.layers} device={device} skipped="
        f"{train_split['n_skipped']}/{val_split['n_skipped']}/{test_split['n_skipped']}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, args.batch_size):
            idx = torch.as_tensor(order[start : start + args.batch_size], dtype=torch.long)
            a = batch_vectors(mat_device, train_split["rows_a"].index_select(0, idx), device)
            b = batch_vectors(mat_device, train_split["rows_b"].index_select(0, idx), device)
            y = train_split["y"].index_select(0, idx).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(a, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred, _, _ = predict(model, mat_device, val_split, device, args.eval_batch_size)
        val_m = metrics(y_val, val_pred)
        train_auc = None
        if epoch == 1 or epoch % args.report_every == 0:
            train_pred, _, _ = predict(model, mat_device, train_split, device, args.eval_batch_size)
            train_auc = float(roc_auc_score(y_train, train_pred))
            print(
                f"[epoch {epoch:03d}] loss={np.mean(losses):.4f} "
                f"train_auc={train_auc:.4f} val_auc={val_m['auroc']:.4f} "
                f"val_auprc={val_m['auprc']:.4f}",
                flush=True,
            )

        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "train_auroc": train_auc,
                "val_auroc": val_m["auroc"],
                "val_auprc": val_m["auprc"],
                "val_brier": val_m["brier"],
            }
        )

        if val_m["auroc"] > best["val_auroc"] + args.min_delta:
            best = {
                "epoch": epoch,
                "val_auroc": val_m["auroc"],
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] epoch={epoch} best_epoch={best['epoch']}", flush=True)
                break

    if best["state"] is None:
        raise RuntimeError("training did not produce a best state")
    model.load_state_dict(best["state"])
    train_pred, train_alpha_a, train_alpha_b = predict(model, mat_device, train_split, device, args.eval_batch_size)
    val_pred, val_alpha_a, val_alpha_b = predict(model, mat_device, val_split, device, args.eval_batch_size)
    test_pred, test_alpha_a, test_alpha_b = predict(model, mat_device, test_split, device, args.eval_batch_size)

    result = {
        "model": "endpoint_additive_mlp_sae_no_global_bias",
        "formula": "logit(PPI(A,B)) = MLP_no_bias(SAE_A) + MLP_no_bias(SAE_B)",
        "family": args.family,
        "rep": args.rep,
        "backbone": args.backbone,
        "layer": resolved_layer,
        "seed": args.seed,
        "best_epoch": int(best["epoch"]),
        "dim": dim,
        "train": metrics(y_train, train_pred),
        "val": metrics(y_val, val_pred),
        "test": metrics(y_test, test_pred),
        "alpha_summary": {
            "train_alpha_a_mean": float(train_alpha_a.mean()),
            "train_alpha_b_mean": float(train_alpha_b.mean()),
            "val_alpha_a_mean": float(val_alpha_a.mean()),
            "val_alpha_b_mean": float(val_alpha_b.mean()),
            "test_alpha_a_mean": float(test_alpha_a.mean()),
            "test_alpha_b_mean": float(test_alpha_b.mean()),
        },
        "input_files": {
            "cache": str(args.cache_path),
        },
        "n_skipped": {
            "train": int(train_split["n_skipped"]),
            "val": int(val_split["n_skipped"]),
            "test": int(test_split["n_skipped"]),
        },
        "hyperparameters": {
            "backbone": args.backbone,
            "layer": resolved_layer,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "layer_bias": bool(args.layer_bias),
            "global_bias": False,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "cache_on_device": bool(args.cache_on_device),
        },
    }
    aux = {
        "history": history,
        "mat": mat,
        "splits": {"train": train_split, "val": val_split, "test": test_split},
        "predictions": {
            "train": (train_pred, train_alpha_a, train_alpha_b),
            "val": (val_pred, val_alpha_a, val_alpha_b),
            "test": (test_pred, test_alpha_a, test_alpha_b),
        },
    }
    return model.cpu(), result, aux


def write_pair_predictions(path: Path, split: dict, pred_pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    prob, alpha_a, alpha_b = pred_pack
    y = split["y"].numpy().astype(np.int8)
    df = pd.DataFrame(
        {
            "pair_index": np.arange(y.size),
            "label": y,
            "prob": prob,
            "alpha_a": alpha_a,
            "alpha_b": alpha_b,
            "logit": alpha_a + alpha_b,
        }
    )
    df.to_csv(path, sep="\t", index=False)


def write_attribution_tables(model: MLPEndpoint, args: argparse.Namespace, aux: dict, stem: str) -> None:
    device = torch.device(args.attr_device if args.attr_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    split = aux["splits"][args.attr_split]
    endpoint_a, endpoint_b = materialize_endpoints(aux["mat"], split)
    df = endpoint_gradient_input_attribution(
        model,
        endpoint_a,
        endpoint_b,
        device=device,
        batch_size=args.attr_batch_size,
        max_endpoints=args.attr_max_endpoints,
        seed=args.seed,
    )
    df = add_sae_annotations(df, args.rep)
    df["abs_mean_signed_attr"] = df["mean_signed_attr"].abs()
    df = df.sort_values("mean_abs_attr", ascending=False)
    attr_path = args.out_dir / f"{stem}_{args.attr_split}_feature_attribution.tsv"
    df.to_csv(attr_path, sep="\t", index=False)

    pos = df.sort_values("mean_signed_attr", ascending=False).head(args.top_n).copy()
    neg = df.sort_values("mean_signed_attr", ascending=True).head(args.top_n).copy()
    abs_top = df.sort_values("mean_abs_attr", ascending=False).head(args.top_n).copy()
    pos["direction"] = "positive_endpoint_score"
    neg["direction"] = "negative_endpoint_score"
    abs_top["direction"] = "largest_abs_attribution"
    pd.concat([pos, neg, abs_top], ignore_index=True).to_csv(
        args.out_dir / f"{stem}_{args.attr_split}_top_features_annotated.tsv",
        sep="\t",
        index=False,
    )
    print("[done] wrote", attr_path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=FAMILIES, default="c3",
                   help="Benchmark family (c1/c2/c3/bernett/cross_species). "
                   "Routes both the protein feature cache and how splits are "
                   "resolved (pair-index caches for c1/c2/c3 and cross_species; "
                   "on-the-fly seq2idx for bernett; cross_species carves val "
                   "from human_train and tests on human_test).")
    p.add_argument("--rep", choices=REPS, default="sae_max")
    p.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=DEFAULT_BACKBONE,
        help="Backbone family to read from the formal feature cache (esmc or esm2).",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        choices=sorted({layer for layers in BACKBONE_LAYERS.values() for layer in layers}),
        help="Backbone layer; defaults to the backbone's default (ESM-C 60, ESM-2 33).",
    )
    p.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Protein feature cache (auditppi_protein_features_v1). Defaults to "
        "the selected family's cache (PPI_PREDICTION_CACHES[family]).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for metrics/predictions/attribution outputs. Defaults to "
        "RESULTS_PAIR/{family}_endpoint_additive_mlp_sae.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--val-frac",
        type=float,
        default=CROSS_SPECIES_VAL_FRAC,
        help="cross_species only: stratified fraction carved off human_train as "
        "the val split (seed+1). Ignored for families with a native val split.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--layer-bias", action="store_true", help="Allow biases inside MLP layers. Default: no biases.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--report-every", type=int, default=5)
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    p.add_argument("--cache-on-device", action="store_true")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--attr-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--attr-batch-size", type=int, default=256)
    p.add_argument("--attr-max-endpoints", type=int, default=0, help="0 means use all endpoint occurrences.")
    p.add_argument("--attr-device", default=None, help="cuda, cpu, or omitted for auto")
    args = p.parse_args()

    if args.cache_path is None:
        args.cache_path = PPI_PREDICTION_CACHES[args.family]
    if args.out_dir is None:
        args.out_dir = (
            RESULTS_PAIR
            / f"{args.family}_endpoint_additive_mlp_sae"
            / f"seed_{args.seed}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolved_layer = resolve_backbone_layer(args.backbone, args.layer)
    bb_tag = f"{args.backbone}L{resolved_layer}"
    model, result, aux = train(args)
    stem = f"{args.family}_endpoint_additive_mlp_{args.rep}_{bb_tag}_h{args.hidden}_l{args.layers}_nobias"

    result_path = args.out_dir / f"{stem}_metrics.json"
    history_path = args.out_dir / f"{stem}_history.tsv"
    dump_experiment(
        result_path,
        task="pair.endpoint_additive_mlp",
        dataset=args.family,
        features=args.rep,
        split="test",
        model="endpoint_additive_mlp",
        seed=args.seed,
        payload=result,
        metrics=result["test"],
        hyperparameters=result.get("hyperparameters"),
    )
    pd.DataFrame(aux["history"]).to_csv(history_path, sep="\t", index=False)
    write_pair_predictions(args.out_dir / f"{stem}_test_pair_predictions.tsv", aux["splits"]["test"], aux["predictions"]["test"])
    write_attribution_tables(model, args, aux, stem)

    print("[done] wrote", result_path, flush=True)
    print(
        f"[test] AUROC={result['test']['auroc']:.4f} "
        f"AUPRC={result['test']['auprc']:.4f} Brier={result['test']['brier']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
