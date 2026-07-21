#!/usr/bin/env python3
"""Train a stronger no-bias endpoint-additive SAE model for C3.

The model has more capacity than the linear endpoint-additive baseline, but it
keeps the same diagnostic restriction: it cannot use pair-interaction features.

    alpha(p) = MLP_no_bias(SAE_p)
    logit PPI(A, B) = alpha(A) + alpha(B)

There is no global/pair bias term, and all Linear layers default to
``bias=False``. Feature explanations are computed for the single-protein
endpoint score with gradient*input attribution:

    attr_i(p) = SAE_p_i * d alpha(p) / d SAE_p_i

The global feature table aggregates attribution over endpoint occurrences.

Endpoints are assembled on the fly from a per-dataset protein feature cache
(``auditppi_protein_features_v1``): the C3 RAPPPID splits carry raw endpoint
sequences (no protein ids), so each endpoint is resolved sequence -> row via the
cache ``seq2idx`` and the desired ``(backbone, layer, rep)`` channel is selected
by :func:`~src.features.protein_cache.representation_matrix`. No pair
embeddings are materialized: shared endpoints are stored once and referenced by
row index, which is both leaner than the old ``emb_a``/``emb_b`` dumps and
lets the same cache serve every backbone/layer/rep.
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
from conf.paths import RESULTS_PAIR, C3_SAE_CACHE
from src.data.pairs import load_c3
from src.data.sequences import normalize_sequence
from src.eval.metrics import pair_score_metrics as metrics
from src.experiments.results import dump_experiment
from src.features.pairs import load_protein_feature_cache
from src.features.protein_cache import representation_matrix
from src.runtime import seed_all
from src.interpretability.annotations import add_sae_annotations
from src.interpretability.attribution import (
    endpoint_gradient_input_attribution,
)
from src.models.architectures.endpoint_mlp import EndpointMLP

OUT_DIR = RESULTS_PAIR / "c3_endpoint_additive_mlp_sae"

REPS = ("sae_max", "binary")


def load_cache(
    rep: str, cache_path: Path, *, backbone: str = DEFAULT_BACKBONE, layer: int | None = None
) -> tuple[torch.Tensor, dict[str, int]]:
    """Load the endpoint feature matrix + sequence->row map from a v1 cache.

    Reads the ``auditppi_protein_features_v1`` layout, selecting the
    ``(backbone, layer, rep)`` channel via the shared
    :func:`~src.features.protein_cache.representation_matrix`. ``binary`` arrives
    as bool and is cast to uint8 (the MLP's ``float()`` upcast is applied per
    batch), ``sae_max`` stays float. C3 endpoints carry no protein ids, so the
    row map returned is the cache ``seq2idx`` (normalized sequence -> row).
    """
    cache = load_protein_feature_cache(cache_path)
    mat = representation_matrix(cache, rep, layer, backbone)
    if rep == "binary":
        mat = mat.to(torch.uint8)
    elif rep == "sae_max":
        mat = mat.float()
    else:
        raise ValueError(rep)
    return mat, cache["seq2idx"]


def load_split(split: str, seq_to_idx: dict[str, int]) -> dict:
    """Resolve a C3 split's pairs to endpoint rows via the cache ``seq2idx``.

    C3 pairs come from the RAPPPID HDF5 keyed by STRING protein id, with the
    endpoint sequence attached. Each endpoint is resolved
    ``protein_id -> attached sequence -> normalize -> seq2idx row``. Pairs whose
    endpoint sequence is absent from the cache are skipped and counted.
    """
    bench = load_c3(split=split, attach_seqs=True)
    rows_a: list[int] = []
    rows_b: list[int] = []
    kept_y: list[int] = []
    n_skipped = 0
    for (pid_a, pid_b), label in zip(bench.pairs, bench.labels):
        seq_a = bench.seqs.get(pid_a)
        seq_b = bench.seqs.get(pid_b)
        if seq_a is None or seq_b is None:
            n_skipped += 1
            continue
        ia = seq_to_idx.get(normalize_sequence(seq_a))
        ib = seq_to_idx.get(normalize_sequence(seq_b))
        if ia is None or ib is None:
            n_skipped += 1
            continue
        rows_a.append(int(ia))
        rows_b.append(int(ib))
        kept_y.append(int(label))
    return {
        "rows_a": torch.as_tensor(rows_a, dtype=torch.long),
        "rows_b": torch.as_tensor(rows_b, dtype=torch.long),
        "y": torch.as_tensor(kept_y, dtype=torch.float32),
        "split": split,
        "n_raw": int(bench.labels.size),
        "n_skipped": int(n_skipped),
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
    model: EndpointMLP,
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


def train(args: argparse.Namespace) -> tuple[EndpointMLP, dict, dict]:
    seed_all(args.seed)
    resolved_layer = resolve_backbone_layer(args.backbone, args.layer)
    mat, seq_to_idx = load_cache(
        args.rep, args.cache_path, backbone=args.backbone, layer=resolved_layer
    )
    train_split = load_split("train", seq_to_idx)
    val_split = load_split("val", seq_to_idx)
    test_split = load_split("test", seq_to_idx)

    dim = int(mat.shape[1])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = EndpointMLP(
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
        f"[data] model=no_bias_endpoint_mlp rep={args.rep} backbone={args.backbone} "
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


def write_attribution_tables(model: EndpointMLP, args: argparse.Namespace, aux: dict, stem: str) -> None:
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
        default=C3_SAE_CACHE,
        help="C3 protein feature cache (auditppi_protein_features_v1). Endpoints "
        "are resolved sequence -> row via the cache seq2idx.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for metrics/predictions/attribution outputs.",
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolved_layer = resolve_backbone_layer(args.backbone, args.layer)
    bb_tag = f"{args.backbone}L{resolved_layer}"
    model, result, aux = train(args)
    stem = f"c3_endpoint_additive_mlp_{args.rep}_{bb_tag}_h{args.hidden}_l{args.layers}_nobias"

    result_path = args.out_dir / f"{stem}_metrics.json"
    history_path = args.out_dir / f"{stem}_history.tsv"
    dump_experiment(
        result_path,
        task="pair.endpoint_additive_mlp",
        dataset="c3",
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
