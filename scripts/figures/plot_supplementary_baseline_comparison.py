#!/usr/bin/env python3
"""Supplementary baseline comparison for C3 PPI prediction.

This standalone figure compares the ESM-C SAE fingerprint with previously run
DeepNano, MINT, PPLM, FlashPPI, and the earlier ESM-2/InterPLM-SAE phase.

The InterPLM-SAE row is loaded from
``data/ppi_fingerprint/interplm_sae_c3_result.json`` if present. Otherwise, the
script uses the verified ESM-2 phase value recorded in
``SAE_PPI/ppi_fingerprint/make_report_figures.py`` and marks AUPRC/val metrics
as unavailable.

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/plot_supplementary_baseline_comparison.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

from conf.paths import ROOT
from conf.paths import AUDIT, FIGURES, SAE_SUPP_INPUTS, C3_TEST_CSV

# precomputed results JSONs + tabm ckpts (was SAE_PPI/ppi_fingerprint/outputs/)
SAE_OUT = SAE_SUPP_INPUTS
OUT_FIG = FIGURES
OUT_DATA = AUDIT / "supplementary_audit_figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)

P = {
    "ink": "#2B2B2B",
    "muted": "#6F6F77",
    "grid": "#DADDE5",
    "light": "#F4F6FA",
    "blue": "#2F5C99",
    "blue_soft": "#B8CBE8",
    "teal": "#3D9A9E",
    "teal_soft": "#BFE3E1",
    "rose": "#C7647A",
    "rose_soft": "#EBC6CF",
    "gold": "#C99532",
    "gold_soft": "#F2D89B",
    "green": "#4E9D64",
    "green_soft": "#CDE8D4",
    "violet": "#7664A8",
    "violet_soft": "#D9D2EC",
    "brown": "#8E6C4A",
    "grey_soft": "#E9EAEE",
}

FAMILY_COLOR = {
    "SAE fingerprint": P["blue"],
    "Dense PLM": P["gold"],
    "Published head": P["green"],
    "Pair-aware PLM": P["violet"],
    "Zero-shot transfer": P["muted"],
}

CURVE_SPECS = [
    {
        "name": "ESM-C SAE-max TabM",
        "short": "SAE-max TabM",
        "family": "SAE fingerprint",
        "kind": "tabm",
        "embedding_dir": SAE_OUT / "esmc" / "reps" / "sae_max",
        "ckpt": SAE_OUT / "esmc" / "reps" / "sae_max" / "tabm_sym_ckpt" / "best.ckpt",
        "color": P["blue"],
    },
    {
        "name": "ESM-C binary SAE TabM",
        "short": "Binary SAE",
        "family": "SAE fingerprint",
        "kind": "tabm",
        "embedding_dir": SAE_OUT / "esmc" / "reps" / "binary_thr0",
        "ckpt": SAE_OUT / "esmc" / "reps" / "binary_thr0" / "tabm_sym_ckpt" / "best.ckpt",
        "color": "#6F88BF",
    },
    {
        "name": "ESM-C dense mean",
        "short": "Dense mean",
        "family": "Dense PLM",
        "kind": "tabm",
        "embedding_dir": SAE_OUT / "esmc" / "reps" / "esmc_mean",
        "ckpt": SAE_OUT / "esmc" / "reps" / "esmc_mean" / "tabm_concat_ckpt" / "best.ckpt",
        "color": P["gold"],
    },
    {
        "name": "DeepNano ESM-C",
        "short": "DeepNano ESM-C",
        "family": "Published head",
        "kind": "deepnano",
        "cache": SAE_OUT / "baselines" / "deepnano_esmc_6b" / "seq_cache.pt",
        "ckpt": SAE_OUT / "baselines" / "deepnano_esmc_6b" / "deepnano_concat_ckpt" / "best.ckpt",
        "color": P["green"],
    },
    {
        "name": "MINT",
        "short": "MINT",
        "family": "Pair-aware PLM",
        "kind": "tabm",
        "embedding_dir": SAE_OUT / "baselines" / "mint",
        "ckpt": SAE_OUT / "baselines" / "mint" / "tabm_concat_ckpt" / "best.ckpt",
        "color": P["violet"],
    },
    {
        "name": "PPLM",
        "short": "PPLM",
        "family": "Pair-aware PLM",
        "kind": "tabm",
        "embedding_dir": SAE_OUT / "baselines" / "pplm",
        "ckpt": SAE_OUT / "baselines" / "pplm" / "tabm_concat_ckpt" / "best.ckpt",
        "color": "#9A79B8",
    },
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.8,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=P["ink"],
    )


def add_grid_x(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=P["grid"], lw=0.55, alpha=0.85)
    ax.set_axisbelow(True)


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem} -> {OUT_FIG}")


def metric(row: dict, *names: str) -> float:
    for name in names:
        if name in row and row[name] is not None:
            return float(row[name])
    return float("nan")


def add_row(
    rows: list[dict],
    name: str,
    family: str,
    source: str,
    setup: str,
    dim: int,
    c3_trained: bool,
    pair_aware: bool,
    interpretable: bool,
    published_method: bool,
    val_auroc: float,
    test_auroc: float,
    test_auprc: float,
    test_f1: float = float("nan"),
    notes: str = "",
) -> None:
    rows.append(
        {
            "name": name,
            "family": family,
            "source": source,
            "setup": setup,
            "dim_per_protein": dim,
            "c3_trained": c3_trained,
            "pair_aware_extractor": pair_aware,
            "interpretable_bits": interpretable,
            "published_method": published_method,
            "val_auroc": val_auroc,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc,
            "test_f1": test_f1,
            "val_minus_test_auroc": val_auroc - test_auroc
            if np.isfinite(val_auroc)
            else float("nan"),
            "notes": notes,
        }
    )


def load_baseline_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []

    esmc_res = SAE_OUT / "esmc" / "results"
    sae = read_json(esmc_res / "xgb_sae_max_sym.json")
    binary = read_json(esmc_res / "xgb_binary_sym.json")
    dense = read_json(esmc_res / "tabm_esmc_mean_concat.json")

    add_row(
        rows,
        "ESM-C SAE-max",
        "SAE fingerprint",
        "outputs/esmc/results/xgb_sae_max_sym.json",
        "XGB sym",
        int(sae["feat_dim"]),
        True,
        False,
        True,
        False,
        metric(sae, "best_val_auroc"),
        metric(sae, "test_auroc"),
        metric(sae, "test_auprc"),
        metric(sae, "test_f1"),
        "project best continuous SAE fingerprint",
    )
    add_row(
        rows,
        "ESM-C binary SAE",
        "SAE fingerprint",
        "outputs/esmc/results/xgb_binary_sym.json",
        "XGB sym",
        int(binary["feat_dim"]),
        True,
        False,
        True,
        False,
        metric(binary, "best_val_auroc"),
        metric(binary, "test_auroc"),
        metric(binary, "test_auprc"),
        metric(binary, "test_f1"),
        "fully interpretable 0/1 fingerprint",
    )

    inter_path = AUDIT / "ppi_fingerprint" / "interplm_sae_c3_result.json"
    if inter_path.exists():
        inter = read_json(inter_path)
        add_row(
            rows,
            inter.get("name", "InterPLM-SAE"),
            "SAE fingerprint",
            str(inter_path.relative_to(ROOT)),
            inter.get("setup", "ESM-2 SAE, XGB"),
            int(inter.get("feat_dim", inter.get("dim_per_protein", 10240))),
            bool(inter.get("c3_trained", True)),
            bool(inter.get("pair_aware_extractor", False)),
            bool(inter.get("interpretable_bits", True)),
            bool(inter.get("published_method", False)),
            metric(inter, "val_auroc", "best_val_auroc"),
            metric(inter, "test_auroc", "auroc"),
            metric(inter, "test_auprc", "auprc"),
            metric(inter, "test_f1", "f1"),
            inter.get("notes", "loaded optional InterPLM result file"),
        )
    else:
        add_row(
            rows,
            "InterPLM ESM-2 SAE",
            "SAE fingerprint",
            "make_report_figures.py verified ESM-2 phase constants",
            "XGB sym",
            10240,
            True,
            False,
            True,
            False,
            float("nan"),
            0.9339,
            float("nan"),
            float("nan"),
            "AUPRC/val metrics not available in on-disk JSON; replace with data/ppi_fingerprint/interplm_sae_c3_result.json when available",
        )

    add_row(
        rows,
        "ESM-C dense mean",
        "Dense PLM",
        "outputs/esmc/results/tabm_esmc_mean_concat.json",
        "TabM concat",
        int(dense["feat_dim"]),
        True,
        False,
        False,
        False,
        metric(dense, "best_val_auroc"),
        metric(dense, "test_auroc"),
        metric(dense, "test_auprc"),
        metric(dense, "test_f1"),
        "dense mean-pool baseline on same ESM-C backbone",
    )

    base_res = SAE_OUT / "baselines" / "results"
    for filename, label, dim in [
        ("deepnano_esmc_6b_concat.json", "DeepNano ESM-C", 2560),
        ("deepnano_esm2_650m_concat.json", "DeepNano ESM-2", 1280),
    ]:
        d = read_json(base_res / filename)
        add_row(
            rows,
            label,
            "Published head",
            f"outputs/baselines/results/{filename}",
            "3-pool MLP concat",
            int(d.get("feat_dim", dim)),
            True,
            False,
            False,
            True,
            metric(d, "best_val_auroc"),
            metric(d, "test_auroc"),
            metric(d, "test_auprc"),
            metric(d, "test_f1"),
            "faithful DeepNano embedding-head reproduction",
        )

    for rep, label in [("mint", "MINT"), ("pplm", "PPLM")]:
        configs = []
        for arch in ["tabm", "xgb"]:
            for pair_mode in ["concat", "sym"]:
                p = base_res / f"{arch}_{rep}_{pair_mode}.json"
                d = read_json(p)
                configs.append((d["test_auroc"], p, d))
        _, best_path, best = max(configs, key=lambda x: x[0])
        add_row(
            rows,
            label,
            "Pair-aware PLM",
            f"outputs/baselines/results/{best_path.name}",
            f"{best['arch'].upper()} {best['pair_mode']}",
            int(best["feat_dim"]),
            True,
            True,
            False,
            False,
            metric(best, "best_val_auroc"),
            metric(best, "test_auroc"),
            metric(best, "test_auprc"),
            metric(best, "test_f1"),
            "frozen pair-aware extractor under matched probe",
        )

    flash_path = base_res / "flashppi_glm2_650m.json"
    if flash_path.exists():
        d = read_json(flash_path)
        add_row(
            rows,
            "FlashPPI",
            "Zero-shot transfer",
            "outputs/baselines/results/flashppi_glm2_650m.json",
            "clip retrieval",
            1024,
            False,
            False,
            False,
            True,
            metric(d, "val_auroc"),
            metric(d, "test_auroc"),
            metric(d, "test_auprc"),
            metric(d, "test_f1"),
            "published checkpoint; no RAPPPID training",
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("test_auroc", ascending=False).reset_index(drop=True)

    grid_rows = []
    for rep in ["mint", "pplm"]:
        for arch in ["tabm", "xgb"]:
            for pair_mode in ["concat", "sym"]:
                d = read_json(base_res / f"{arch}_{rep}_{pair_mode}.json")
                grid_rows.append(
                    {
                        "rep": rep.upper(),
                        "arch": arch.upper(),
                        "pair_mode": pair_mode,
                        "config": f"{arch.upper()} {pair_mode}",
                        "test_auroc": float(d["test_auroc"]),
                        "test_auprc": float(d["test_auprc"]),
                        "val_auroc": float(d["best_val_auroc"]),
                    }
                )
    grid = pd.DataFrame(grid_rows)

    df.to_csv(OUT_DATA / "s5_external_baseline_summary.tsv", sep="\t", index=False)
    grid.to_csv(OUT_DATA / "s5_mint_pplm_probe_grid.tsv", sep="\t", index=False)
    return df, grid


def build_pair_features(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    if mode == "concat":
        return np.concatenate([a, b], axis=1)
    if mode == "sym":
        return np.concatenate([a * b, np.abs(a - b)], axis=1)
    if mode == "rich":
        return np.concatenate([a, b, a * b, np.abs(a - b)], axis=1)
    raise ValueError(mode)


def build_pair_features_sparse(a: np.ndarray, b: np.ndarray, mode: str):
    from scipy import sparse

    def to_csr_float32(x: np.ndarray):
        if x.dtype != np.float16:
            return sparse.csr_matrix(x).astype(np.float32, copy=False)
        chunk = int(os.environ.get("S5_SPARSE_CHUNK_ROWS", "1024"))
        parts = [
            sparse.csr_matrix(x[start:start + chunk].astype(np.float32, copy=False))
            for start in range(0, x.shape[0], chunk)
        ]
        return sparse.vstack(parts, format="csr", dtype=np.float32)

    a_sp = to_csr_float32(a)
    b_sp = to_csr_float32(b)
    if mode == "concat":
        return sparse.hstack([a_sp, b_sp], format="csr", dtype=np.float32)
    if mode == "sym":
        prod = a_sp.multiply(b_sp)
        diff = a_sp - b_sp
        diff.data = np.abs(diff.data)
        diff.eliminate_zeros()
        return sparse.hstack([prod, diff], format="csr", dtype=np.float32)
    if mode == "rich":
        prod = a_sp.multiply(b_sp)
        diff = a_sp - b_sp
        diff.data = np.abs(diff.data)
        diff.eliminate_zeros()
        return sparse.hstack([a_sp, b_sp, prod, diff], format="csr", dtype=np.float32)
    raise ValueError(mode)


def load_embedding_split(embedding_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    d = torch.load(embedding_dir / f"{split}_embeddings.pt", map_location="cpu", weights_only=False)
    return (
        d["emb_a"].numpy(),
        d["emb_b"].numpy(),
        d["label"].numpy().astype(np.int64),
    )


def load_sparse_pair_split(embedding_dir: Path, split: str, mode: str):
    a, b, y = load_embedding_split(embedding_dir, split)
    X = build_pair_features_sparse(a, b, mode)
    del a, b
    return X, y


def predict_xgb_test_scores(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Re-train the original XGB probe from saved pair embeddings and export test scores."""
    import xgboost as xgb

    ed = Path(spec["embedding_dir"])
    pair_mode = spec["pair_mode"]
    Xtr, ytr = load_sparse_pair_split(ed, "train", pair_mode)
    Xva, yva = load_sparse_pair_split(ed, "val", pair_mode)
    Xte, yte = load_sparse_pair_split(ed, "test", pair_mode)

    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "tree_method": "hist",
        "device": "cpu",
        "seed": 42,
        "nthread": max(1, os.cpu_count() or 1),
    }
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=int(os.environ.get("S5_XGB_TREES", "3000")),
        evals=[(dva, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    limit = getattr(booster, "best_iteration", None)
    iteration_range = (0, int(limit) + 1) if limit is not None else (0, 0)
    score = booster.predict(dte, iteration_range=iteration_range).astype(np.float32)
    return yte, score


def predict_tabm_test_scores(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from src.models.architectures.tabm_pair import TabMPair

    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    model = TabMPair.from_hparams(hp)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    d = torch.load(Path(spec["embedding_dir"]) / "test_embeddings.pt", map_location="cpu", weights_only=False)
    a = d["emb_a"].float()
    b = d["emb_b"].float()
    y = d["label"].numpy().astype(np.int64)
    batch = int(os.environ.get("S5_PRED_BATCH", "512"))
    parts = []
    with torch.no_grad():
        for start in range(0, y.shape[0], batch):
            logits = model(a[start:start + batch], b[start:start + batch])
            parts.append(torch.sigmoid(logits).cpu())
    return y, torch.cat(parts).numpy().astype(np.float32)


def predict_deepnano_test_scores(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn as nn

    c3_test = C3_TEST_CSV

    def combine(a, b, mode: str):
        if mode == "concat":
            return torch.cat([a, b], dim=1)
        if mode == "sym":
            return torch.cat([a * b, (a - b).abs()], dim=1)
        if mode == "rich":
            return torch.cat([a, b, a * b, (a - b).abs()], dim=1)
        raise ValueError(mode)

    def make_head(in_dim: int, hidden1: int, hidden2: int):
        return nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1), nn.Sigmoid(),
        )

    class DeepNanoEmbeddingMLP(nn.Module):
        def __init__(self, feat_dim: int, hp: dict):
            super().__init__()
            self.pair_mode = hp["pair_mode"]
            in_dim = {"concat": 2, "sym": 2, "rich": 4}[self.pair_mode] * feat_dim
            self.head_mean = make_head(in_dim, int(hp["hidden1"]), int(hp["hidden2"]))
            self.head_min = make_head(in_dim, int(hp["hidden1"]), int(hp["hidden2"]))
            self.head_max = make_head(in_dim, int(hp["hidden1"]), int(hp["hidden2"]))

        def forward(self, batch: dict[str, "torch.Tensor"]):
            p_mean = self.head_mean(combine(batch["a_mean"], batch["b_mean"], self.pair_mode)).squeeze(1)
            p_min = self.head_min(combine(batch["a_min"], batch["b_min"], self.pair_mode)).squeeze(1)
            p_max = self.head_max(combine(batch["a_max"], batch["b_max"], self.pair_mode)).squeeze(1)
            return (p_mean + p_min + p_max) / 3.0

    cache = torch.load(spec["cache"], map_location="cpu", weights_only=False)
    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    pools = {name: cache[name].float() for name in ("mean", "min", "max")}
    seq2idx = cache["seq2idx"]
    feat_dim = int(pools["mean"].shape[1])
    model = DeepNanoEmbeddingMLP(feat_dim, hp)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pairs = pd.read_csv(c3_test)
    ia = torch.tensor([seq2idx[str(s)] for s in pairs["query"].astype(str)], dtype=torch.long)
    ib = torch.tensor([seq2idx[str(s)] for s in pairs["text"].astype(str)], dtype=torch.long)
    y = pairs["label"].to_numpy(dtype=np.int64)
    batch_size = int(os.environ.get("S5_PRED_BATCH", "512"))
    parts = []
    with torch.no_grad():
        for start in range(0, y.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            batch = {}
            for pool_name in ("mean", "min", "max"):
                batch[f"a_{pool_name}"] = pools[pool_name][ia[sl]]
                batch[f"b_{pool_name}"] = pools[pool_name][ib[sl]]
            parts.append(model(batch).cpu())
    return y, torch.cat(parts).numpy().astype(np.float32)


def make_prediction_cache() -> pd.DataFrame:
    rows = []
    for spec in CURVE_SPECS:
        print(f"[s5 curves] scoring {spec['name']} ({spec['kind']})", flush=True)
        if spec["kind"] == "xgb":
            y, score = predict_xgb_test_scores(spec)
        elif spec["kind"] == "tabm":
            y, score = predict_tabm_test_scores(spec)
        elif spec["kind"] == "deepnano":
            y, score = predict_deepnano_test_scores(spec)
        else:
            raise ValueError(spec["kind"])
        for i, (yy, ss) in enumerate(zip(y, score)):
            rows.append(
                {
                    "method": spec["name"],
                    "short": spec["short"],
                    "family": spec["family"],
                    "sample_index": i,
                    "label": int(yy),
                    "score": float(ss),
                }
            )
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT_DATA / "s5_c3_test_predictions.tsv", sep="\t", index=False)
    return pred


def load_curve_predictions() -> pd.DataFrame:
    path = OUT_DATA / "s5_c3_test_predictions.tsv"
    required = {s["name"] for s in CURVE_SPECS}
    refresh = os.environ.get("S5_REFRESH_PREDICTIONS", "0") == "1"
    if path.exists() and not refresh:
        pred = pd.read_csv(path, sep="\t")
        if required.issubset(set(pred["method"])):
            return pred[pred["method"].isin(required)].copy()
    return make_prediction_cache()


def curve_metric_table(pred: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for spec in CURVE_SPECS:
        sub = pred[pred["method"] == spec["name"]]
        y = sub["label"].to_numpy(dtype=int)
        score = sub["score"].to_numpy(dtype=float)
        rows.append(
            {
                "method": spec["name"],
                "test_auroc_from_scores": float(roc_auc_score(y, score)),
                "test_auprc_from_scores": float(average_precision_score(y, score)),
                "n": int(len(sub)),
                "positive_rate": float(y.mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / "s5_curve_score_summary.tsv", sep="\t", index=False)
    return out


def plot_roc_pr_curves(fig: plt.Figure, subspec, pred: pd.DataFrame) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec, wspace=0.34)
    ax_roc = fig.add_subplot(inner[0, 0])
    ax_pr = fig.add_subplot(inner[0, 1])
    panel_label(ax_roc, "b", x=-0.23, y=1.08)

    for spec in CURVE_SPECS:
        sub = pred[pred["method"] == spec["name"]]
        y = sub["label"].to_numpy(dtype=int)
        score = sub["score"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, score)
        prec, rec, _ = precision_recall_curve(y, score)
        auroc = roc_auc_score(y, score)
        auprc = average_precision_score(y, score)
        lw = 1.65 if spec["name"] == "ESM-C SAE-max" else 1.05
        z = 5 if spec["name"] == "ESM-C SAE-max" else 3
        ax_roc.plot(fpr, tpr, color=spec["color"], lw=lw, label=spec["short"], zorder=z)
        ax_pr.plot(rec, prec, color=spec["color"], lw=lw, label=f"{spec['short']} ({auroc:.3f}/{auprc:.3f})", zorder=z)

    pos_rate = float(pred.drop_duplicates(["method", "sample_index"])["label"].mean())
    ax_roc.plot([0, 1], [0, 1], color=P["grid"], lw=0.75, ls="--", zorder=1)
    ax_pr.axhline(pos_rate, color=P["grid"], lw=0.75, ls="--", zorder=1)
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(color=P["grid"], lw=0.45, alpha=0.65)
        ax.set_axisbelow(True)
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title("ROC curves", loc="left", fontweight="bold")
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision-recall curves", loc="left", fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=4.4, handlelength=1.4, borderaxespad=0.2)


def make_figure() -> None:
    df, grid = load_baseline_table()
    curve_pred = load_curve_predictions()
    curve_metric_table(curve_pred)
    fig = plt.figure(figsize=(7.2, 4.85))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.58,
        wspace=0.62,
        width_ratios=[1.12, 0.92, 0.82],
    )

    ax = fig.add_subplot(gs[:, 0])
    panel_label(ax, "a", x=-0.18, y=1.06)
    rank = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rank))
    for yi, row in zip(y, rank.itertuples()):
        color = FAMILY_COLOR[row.family]
        ax.plot([0.82, row.test_auroc], [yi, yi], color=P["grid"], lw=1.5, zorder=1)
        marker = "D" if not row.c3_trained else "o"
        face = "white" if row.name.startswith("InterPLM") and not np.isfinite(row.test_auprc) else color
        ax.scatter(
            [row.test_auroc],
            [yi],
            s=58,
            marker=marker,
            color=face,
            edgecolor=color,
            lw=1.0,
            zorder=3,
        )
        ax.text(row.test_auroc + 0.004, yi, f"{row.test_auroc:.3f}", va="center", fontsize=5.9, color=P["ink"])
    ax.axvline(0.5, color=P["grid"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(rank["name"])
    ax.set_xlim(0.82, 0.948)
    ax.set_xlabel("C3 test AUROC")
    ax.set_title("External baseline ranking", loc="left", fontweight="bold")
    add_grid_x(ax)

    plot_roc_pr_curves(fig, gs[0, 1:], curve_pred)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c")
    gap = df[np.isfinite(df["val_auroc"])].copy()
    gap = gap.sort_values("val_minus_test_auroc", ascending=True).reset_index(drop=True)
    y = np.arange(len(gap))
    for yi, row in zip(y, gap.itertuples()):
        color = FAMILY_COLOR[row.family]
        ax.plot([row.test_auroc, row.val_auroc], [yi, yi], color=P["grid"], lw=2.1, solid_capstyle="round", zorder=1)
        ax.scatter([row.test_auroc], [yi], s=34, color=color, edgecolor="white", lw=0.7, zorder=3)
        ax.scatter([row.val_auroc], [yi], s=34, color="white", edgecolor=color, lw=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(gap["name"])
    ax.set_xlim(0.84, 0.965)
    ax.set_xlabel("AUROC (filled=test, open=validation)")
    ax.set_title("Validation-to-test movement", loc="left", fontweight="bold")
    add_grid_x(ax)

    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "d")
    configs = ["TABM concat", "TABM sym", "XGB concat", "XGB sym"]
    reps = ["MINT", "PPLM"]
    mat = np.full((len(reps), len(configs)), np.nan)
    for i, rep in enumerate(reps):
        for j, cfg in enumerate(configs):
            v = grid.loc[(grid["rep"] == rep) & (grid["config"] == cfg), "test_auroc"]
            if not v.empty:
                mat[i, j] = float(v.iloc[0])
    cmap = LinearSegmentedColormap.from_list("mint_pplm", [P["rose_soft"], "#FFFFFF", P["violet"]])
    ax.imshow(mat, cmap=cmap, vmin=0.82, vmax=0.89, aspect="auto")
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.5, color=P["ink"])
    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in configs], fontsize=5.3)
    ax.set_yticks(np.arange(len(reps)))
    ax.set_yticklabels(reps, fontsize=5.8)
    ax.tick_params(length=0)
    ax.set_title("MINT/PPLM matched probes", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.51,
        0.006,
        "Panel b uses regenerated C3 test scores from saved TabM/DeepNano checkpoints; scalar-only InterPLM-SAE and FlashPPI remain in panel a.",
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=P["muted"],
    )

    save_fig(fig, "supplementary_s5_external_baseline_comparison")


def main() -> None:
    setup_style()
    make_figure()


if __name__ == "__main__":
    main()
