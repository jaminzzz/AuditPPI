#!/usr/bin/env python3
"""Train no-bias endpoint-additive MLP baselines on PRING Human pair labels.

The model is restricted to endpoint terms:

    alpha(p) = MLP_no_bias(SAE_p)
    logit PPI(A, B) = alpha(A) + alpha(B)

No pair interaction features are used: no SAE_A * SAE_B, no |SAE_A - SAE_B|,
and no protein-ID parameters. There is also no global bias term.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from conf.model import DEFAULT_SEED
from conf.paths import RESULTS_PAIR, PRING_ROOT, PRING_HUMAN_SAE_CACHE
from src.eval.metrics import pair_score_metrics as metrics
from src.runtime import seed_all
from src.models.architectures.endpoint_mlp import EndpointMLP

HUMAN_CACHE = PRING_HUMAN_SAE_CACHE
OUT_DIR = RESULTS_PAIR / "pring_endpoint_additive_mlp_sae"

METHODS = ("BFS", "DFS", "RANDOM_WALK")
REPS = ("sae_max", "binary")


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


def load_cache(rep: str) -> tuple[torch.Tensor, dict[str, int]]:
    cache = torch.load(HUMAN_CACHE, map_location="cpu", weights_only=False)
    mat = cache["esmc_sae_max"]
    if rep == "binary":
        mat = (mat > 0).to(torch.uint8)
    elif rep == "sae_max":
        mat = mat.float()
    else:
        raise ValueError(rep)
    return mat, cache["uniprotid2idx"]


def load_split(method: str, split: str, id_to_idx: dict[str, int]) -> dict:
    path = PRING_ROOT / "human" / method / f"human_{split}_ppi.txt"
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
    return {
        "rows_a": torch.as_tensor(rows_a, dtype=torch.long),
        "rows_b": torch.as_tensor(rows_b, dtype=torch.long),
        "y": torch.as_tensor(kept_y, dtype=torch.float32),
        "path": str(path),
        "n_raw": int(len(y)),
        "n_skipped": int(n_skipped),
    }


def batch_vectors(mat: torch.Tensor, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    rows = rows.to(mat.device, non_blocking=True)
    return mat.index_select(0, rows).to(device, non_blocking=True)


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


def write_pair_predictions(path: Path, split: dict, pred_pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    prob, alpha_a, alpha_b = pred_pack
    y = split["y"].numpy().astype(np.int8)
    pd.DataFrame(
        {
            "pair_index": np.arange(y.size),
            "label": y,
            "prob": prob,
            "alpha_a": alpha_a,
            "alpha_b": alpha_b,
            "logit": alpha_a + alpha_b,
        }
    ).to_csv(path, sep="\t", index=False)


def train_one(args: argparse.Namespace, method: str, mat: torch.Tensor, id_to_idx: dict[str, int]) -> tuple[dict, dict]:
    seed_all(args.seed)
    train_split = load_split(method, "train", id_to_idx)
    val_split = load_split(method, "val", id_to_idx)
    test_split = load_split(method, "test", id_to_idx)

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

    y_train_np = train_split["y"].numpy().astype(np.int8)
    y_val_np = val_split["y"].numpy().astype(np.int8)
    y_test_np = test_split["y"].numpy().astype(np.int8)
    rng = np.random.default_rng(args.seed)
    n = int(train_split["y"].numel())
    best = {"epoch": 0, "val_auroc": -np.inf, "state": None}
    patience_left = args.patience
    history = []

    print(
        f"[data] {method} rep={args.rep} train={n:,} val={val_split['y'].numel():,} "
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
        val_m = metrics(y_val_np, val_pred)
        train_auc = None
        if epoch == 1 or epoch % args.report_every == 0:
            train_pred, _, _ = predict(model, mat_device, train_split, device, args.eval_batch_size)
            train_auc = float(roc_auc_score(y_train_np, train_pred))
            print(
                f"[{method} epoch {epoch:03d}] loss={np.mean(losses):.4f} "
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
                print(f"[{method} early-stop] epoch={epoch} best_epoch={best['epoch']}", flush=True)
                break

    if best["state"] is None:
        raise RuntimeError(f"{method}: no best state")
    model.load_state_dict(best["state"])
    train_pack = predict(model, mat_device, train_split, device, args.eval_batch_size)
    val_pack = predict(model, mat_device, val_split, device, args.eval_batch_size)
    test_pack = predict(model, mat_device, test_split, device, args.eval_batch_size)
    result = {
        "model": "pring_endpoint_additive_mlp_sae_no_global_bias",
        "formula": "logit(PPI(A,B)) = MLP_no_bias(SAE_A) + MLP_no_bias(SAE_B)",
        "method": method,
        "rep": args.rep,
        "seed": args.seed,
        "best_epoch": int(best["epoch"]),
        "dim": dim,
        "train": metrics(y_train_np, train_pack[0]),
        "val": metrics(y_val_np, val_pack[0]),
        "test": metrics(y_test_np, test_pack[0]),
        "alpha_summary": {
            "train_alpha_a_mean": float(train_pack[1].mean()),
            "train_alpha_b_mean": float(train_pack[2].mean()),
            "val_alpha_a_mean": float(val_pack[1].mean()),
            "val_alpha_b_mean": float(val_pack[2].mean()),
            "test_alpha_a_mean": float(test_pack[1].mean()),
            "test_alpha_b_mean": float(test_pack[2].mean()),
        },
        "input_files": {
            "cache": str(HUMAN_CACHE),
            "train": train_split["path"],
            "val": val_split["path"],
            "test": test_split["path"],
        },
        "n_skipped": {
            "train": int(train_split["n_skipped"]),
            "val": int(val_split["n_skipped"]),
            "test": int(test_split["n_skipped"]),
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
            "cache_on_device": bool(args.cache_on_device),
        },
    }
    aux = {
        "history": history,
        "splits": {"train": train_split, "val": val_split, "test": test_split},
        "predictions": {"train": train_pack, "val": val_pack, "test": test_pack},
    }
    return result, aux


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=[*METHODS, "all"], default="all")
    p.add_argument("--rep", choices=REPS, default="sae_max")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--layer-bias", action="store_true", help="Allow biases inside MLP layers. Default: no biases.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--report-every", type=int, default=5)
    p.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto")
    p.add_argument("--cache-on-device", action="store_true")
    p.add_argument("--write-predictions", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mat, id_to_idx = load_cache(args.rep)
    methods = METHODS if args.method == "all" else (args.method,)
    all_results = {}
    rows = []
    for method in methods:
        result, aux = train_one(args, method, mat, id_to_idx)
        all_results[method] = result
        stem = f"pring_{method.lower()}_endpoint_additive_mlp_{args.rep}_h{args.hidden}_l{args.layers}_nobias"
        result_path = OUT_DIR / f"{stem}_metrics.json"
        history_path = OUT_DIR / f"{stem}_history.tsv"
        result_path.write_text(json.dumps(result, indent=2))
        pd.DataFrame(aux["history"]).to_csv(history_path, sep="\t", index=False)
        if args.write_predictions:
            write_pair_predictions(OUT_DIR / f"{stem}_test_pair_predictions.tsv", aux["splits"]["test"], aux["predictions"]["test"])
        for split in ("train", "val", "test"):
            m = result[split]
            rows.append(
                {
                    "rep": args.rep,
                    "method": method,
                    "split": split,
                    "n": m["n"],
                    "pos_rate": m["pos_rate"],
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "accuracy_at_0.5": m["accuracy_at_0.5"],
                    "brier": m["brier"],
                    "best_epoch": result["best_epoch"],
                }
            )
        print(
            f"[{method} test] AUROC={result['test']['auroc']:.4f} "
            f"AUPRC={result['test']['auprc']:.4f} -> {result_path}",
            flush=True,
        )

    summary_path = OUT_DIR / f"pring_endpoint_additive_mlp_{args.rep}_h{args.hidden}_l{args.layers}_nobias_summary.json"
    tsv_path = OUT_DIR / f"pring_endpoint_additive_mlp_{args.rep}_h{args.hidden}_l{args.layers}_nobias_metrics_summary.tsv"
    summary_path.write_text(json.dumps(all_results, indent=2))
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print("[done] wrote", summary_path, flush=True)
    print("[done] wrote", tsv_path, flush=True)


if __name__ == "__main__":
    main()
