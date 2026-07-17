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
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from conf.paths import AUDIT, SAE_REPS as SAE_REP_ROOT
from src.interpretability.annotations import add_sae_annotations
from src.interpretability.attribution import (
    endpoint_gradient_input_attribution,
)
from src.models.architectures.endpoint_mlp import EndpointMLP

OUT_DIR = AUDIT / "c3_endpoint_additive_mlp_sae"

REP_DIR = {
    "sae_max": SAE_REP_ROOT / "sae_max",
    "binary": SAE_REP_ROOT / "binary_thr0",
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(rep: str, split: str) -> dict[str, torch.Tensor]:
    path = REP_DIR[rep] / f"{split}_embeddings.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "a": d["emb_a"],
        "b": d["emb_b"],
        "y": d["label"].float(),
    }


@torch.no_grad()
def predict(
    model: EndpointMLP, split: dict[str, torch.Tensor], device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    alpha_a = []
    alpha_b = []
    n = int(split["y"].numel())
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a = split["a"][start:end].to(device, non_blocking=True)
        b = split["b"][start:end].to(device, non_blocking=True)
        aa = model.alpha(a)
        bb = model.alpha(b)
        logits = aa + bb
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        alpha_a.append(aa.detach().cpu().numpy())
        alpha_b.append(bb.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(alpha_a), np.concatenate(alpha_b)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(np.int8)
    return {
        "n": int(y.size),
        "pos_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "accuracy_at_0.5": float((pred == y).mean()),
        "brier": float(brier_score_loss(y, p)),
        "score_mean": float(p.mean()),
        "score_std": float(p.std()),
    }


def train(args: argparse.Namespace) -> tuple[EndpointMLP, dict, dict]:
    seed_all(args.seed)
    train_split = load_split(args.rep, "train")
    val_split = load_split(args.rep, "val")
    test_split = load_split(args.rep, "test")

    dim = int(train_split["a"].shape[1])
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

    y_train = train_split["y"].numpy().astype(np.int8)
    y_val = val_split["y"].numpy().astype(np.int8)
    y_test = test_split["y"].numpy().astype(np.int8)
    rng = np.random.default_rng(args.seed)
    best = {"epoch": 0, "val_auroc": -np.inf, "state": None}
    history = []
    patience_left = args.patience
    n = int(train_split["y"].numel())
    print(
        f"[data] model=no_bias_endpoint_mlp rep={args.rep} train={n:,} "
        f"val={val_split['y'].numel():,} test={test_split['y'].numel():,} "
        f"dim={dim:,} hidden={args.hidden} layers={args.layers} device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, args.batch_size):
            idx = torch.as_tensor(order[start : start + args.batch_size], dtype=torch.long)
            a = train_split["a"].index_select(0, idx).to(device, non_blocking=True)
            b = train_split["b"].index_select(0, idx).to(device, non_blocking=True)
            y = train_split["y"].index_select(0, idx).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(a, b), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred, _, _ = predict(model, val_split, device, args.eval_batch_size)
        val_m = metrics(y_val, val_pred)
        train_auc = None
        if epoch == 1 or epoch % args.report_every == 0:
            train_pred, _, _ = predict(model, train_split, device, args.eval_batch_size)
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
    train_pred, train_alpha_a, train_alpha_b = predict(model, train_split, device, args.eval_batch_size)
    val_pred, val_alpha_a, val_alpha_b = predict(model, val_split, device, args.eval_batch_size)
    test_pred, test_alpha_a, test_alpha_b = predict(model, test_split, device, args.eval_batch_size)

    result = {
        "model": "endpoint_additive_mlp_sae_no_global_bias",
        "formula": "logit(PPI(A,B)) = MLP_no_bias(SAE_A) + MLP_no_bias(SAE_B)",
        "rep": args.rep,
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
        "hyperparameters": {
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
        },
    }
    aux = {
        "history": history,
        "splits": {"train": train_split, "val": val_split, "test": test_split},
        "predictions": {
            "train": (train_pred, train_alpha_a, train_alpha_b),
            "val": (val_pred, val_alpha_a, val_alpha_b),
            "test": (test_pred, test_alpha_a, test_alpha_b),
        },
    }
    return model.cpu(), result, aux


def write_pair_predictions(path: Path, split: dict[str, torch.Tensor], pred_pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
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
    df = endpoint_gradient_input_attribution(
        model,
        split["a"],
        split["b"],
        device=device,
        batch_size=args.attr_batch_size,
        max_endpoints=args.attr_max_endpoints,
        seed=args.seed,
    )
    df = add_sae_annotations(df, args.rep)
    df["abs_mean_signed_attr"] = df["mean_signed_attr"].abs()
    df = df.sort_values("mean_abs_attr", ascending=False)
    attr_path = OUT_DIR / f"{stem}_{args.attr_split}_feature_attribution.tsv"
    df.to_csv(attr_path, sep="\t", index=False)

    pos = df.sort_values("mean_signed_attr", ascending=False).head(args.top_n).copy()
    neg = df.sort_values("mean_signed_attr", ascending=True).head(args.top_n).copy()
    abs_top = df.sort_values("mean_abs_attr", ascending=False).head(args.top_n).copy()
    pos["direction"] = "positive_endpoint_score"
    neg["direction"] = "negative_endpoint_score"
    abs_top["direction"] = "largest_abs_attribution"
    pd.concat([pos, neg, abs_top], ignore_index=True).to_csv(
        OUT_DIR / f"{stem}_{args.attr_split}_top_features_annotated.tsv",
        sep="\t",
        index=False,
    )
    print("[done] wrote", attr_path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rep", choices=sorted(REP_DIR), default="sae_max")
    p.add_argument("--seed", type=int, default=7)
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
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--attr-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--attr-batch-size", type=int, default=256)
    p.add_argument("--attr-max-endpoints", type=int, default=0, help="0 means use all endpoint occurrences.")
    p.add_argument("--attr-device", default=None, help="cuda, cpu, or omitted for auto")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, result, aux = train(args)
    stem = f"c3_endpoint_additive_mlp_{args.rep}_h{args.hidden}_l{args.layers}_nobias"

    result_path = OUT_DIR / f"{stem}_metrics.json"
    history_path = OUT_DIR / f"{stem}_history.tsv"
    result_path.write_text(json.dumps(result, indent=2))
    pd.DataFrame(aux["history"]).to_csv(history_path, sep="\t", index=False)
    write_pair_predictions(OUT_DIR / f"{stem}_test_pair_predictions.tsv", aux["splits"]["test"], aux["predictions"]["test"])
    write_attribution_tables(model, args, aux, stem)

    print("[done] wrote", result_path, flush=True)
    print(
        f"[test] AUROC={result['test']['auroc']:.4f} "
        f"AUPRC={result['test']['auprc']:.4f} Brier={result['test']['brier']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
