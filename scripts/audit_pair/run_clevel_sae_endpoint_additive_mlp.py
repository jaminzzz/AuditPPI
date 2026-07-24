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
from conf.paths import RESULTS_PAIR, CLEVEL_PAIR_INDEX_CACHES, CLEVEL_SAE_CACHES
from src.eval.metrics import pair_score_metrics as metrics
from src.experiments.results import dump_experiment
from src.features.pairs import load_pair_index_cache, load_protein_feature_cache
from src.features.protein_cache import representation_matrix
from src.runtime import seed_all
from src.interp.annotations import add_sae_annotations
from src.interp.attribution import (
    endpoint_gradient_input_attribution,
)
from src.models.architectures.mlp_endpoint import MLPEndpoint

FAMILIES = ("c1", "c2", "c3")
REPS = ("sae_max", "binary")


def load_cache(
    rep: str, cache_path: Path, *, backbone: str = DEFAULT_BACKBONE, layer: int | None = None
) -> torch.Tensor:
    """Load the endpoint feature matrix from a v1 protein cache.

    Reads the ``auditppi_protein_features_v1`` layout, selecting the
    ``(backbone, layer, rep)`` channel via the shared
    :func:`~src.features.protein_cache.representation_matrix`. ``binary`` arrives
    as bool and is cast to uint8 (the MLP's ``float()`` upcast is applied per
    batch), ``sae_max`` stays float. Rows are indexed by the pair-index cache.
    """
    cache = load_protein_feature_cache(cache_path)
    mat = representation_matrix(cache, rep, layer, backbone)
    if rep == "binary":
        mat = mat.to(torch.uint8)
    elif rep == "sae_max":
        mat = mat.float()
    else:
        raise ValueError(rep)
    return mat


def load_split(family: str, split: str) -> dict:
    """Load a C-level split's endpoint rows from its pair-index cache.

    Reads ``rows_a``/``rows_b``/``labels`` straight out of the family's
    ``auditppi_pair_index_v1`` cache. The cache is already pre-filtered to pairs
    whose endpoints are present in the protein cache, so there is nothing to skip
    here; the two endpoint row indices reference the shared protein-feature matrix
    loaded by :func:`load_cache`.
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
    mat = load_cache(
        args.rep, args.cache_path, backbone=args.backbone, layer=resolved_layer
    )
    train_split = load_split(args.family, "train")
    val_split = load_split(args.family, "val")
    test_split = load_split(args.family, "test")

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
                   help="C-level leakage family (c1/c2/c3). Routes both the "
                   "protein feature cache and the pair-index caches.")
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
        "the selected family's cache (CLEVEL_SAE_CACHES[family]).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for metrics/predictions/attribution outputs. Defaults to "
        "RESULTS_PAIR/{family}_endpoint_additive_mlp_sae.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
        args.cache_path = CLEVEL_SAE_CACHES[args.family]
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
