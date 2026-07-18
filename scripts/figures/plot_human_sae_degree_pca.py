#!/usr/bin/env python3
"""Plot PRING Human pooled SAE PCA/SVD coordinates colored by graph degree.

Each point is one PRING Human protein. Coordinates are computed from the
16,384-dimensional pooled ESM-C SAE max-activation vector (``esmc_sae_max``),
after matching proteins to the PRING full-graph degree table.

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/figures/plot_human_sae_degree_pca.py
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
import torch
from sklearn.decomposition import PCA

from conf.model import ESMC_SAE_DIM as DIM
from conf.paths import RESULTS_PROTEIN, FIGURES, PRING_HUMAN_SAE_CACHE

HUMAN_CACHE = PRING_HUMAN_SAE_CACHE
DEGREE_TABLE = (
    RESULTS_PROTEIN
    / "pring_participation"
    / "high_p90_xgboost"
    / "pring_human_bfs_binary_high_q90_xgboost_protein_predictions.tsv"
)
OUT_FIG = FIGURES
OUT_DIR = RESULTS_PROTEIN / "sae_degree_pca"

RANDOM_STATE = 7

P = {
    "ink": "#2B2B2B",
    "muted": "#6F6F77",
    "grid": "#DADDE5",
    "blue_soft": "#B8CBE8",
    "gold_soft": "#F2D89B",
    "rose": "#C7647A",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_inputs() -> tuple[torch.Tensor, list[str], pd.DataFrame]:
    cache = torch.load(HUMAN_CACHE, map_location="cpu", weights_only=False)
    mat = cache["esmc_sae_max"]
    if tuple(mat.shape) != (len(cache["protein_ids"]), DIM):
        raise ValueError(f"unexpected esmc_sae_max shape: {tuple(mat.shape)}")

    degree_df = pd.read_csv(DEGREE_TABLE, sep="\t")
    degree_df = degree_df.drop_duplicates("protein").copy()
    cache_ids = set(cache["uniprotid2idx"].keys())
    degree_df = degree_df[degree_df["protein"].isin(cache_ids)].copy()
    degree_df = degree_df.sort_values("protein").reset_index(drop=True)

    rows = [int(cache["uniprotid2idx"][p]) for p in degree_df["protein"]]
    return mat[rows].float(), rows, degree_df


def compute_pca(mat: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = mat.numpy().astype(np.float32, copy=False)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=RANDOM_STATE)
    coords = pca.fit_transform(x)
    return coords, pca.explained_variance_ratio_, pca.singular_values_


def plot(coords: np.ndarray, var: np.ndarray, degree_df: pd.DataFrame) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    degree = degree_df["degree_full"].to_numpy(dtype=float)
    degree_plus = degree + 1.0
    hub = degree_df["high_participation"].to_numpy(dtype=int) == 1
    hub_cutoff = int(degree_df.loc[hub, "degree_full"].min()) if hub.any() else None

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "degree", [P["blue_soft"], "#FFFFFF", P["gold_soft"], P["rose"]]
    )
    norm = mpl.colors.LogNorm(vmin=max(1.0, degree_plus.min()), vmax=degree_plus.max())

    fig, ax = plt.subplots(figsize=(4.7, 4.05))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=degree_plus,
        cmap=cmap,
        norm=norm,
        s=5,
        alpha=0.62,
        linewidth=0,
        rasterized=True,
    )
    if hub.any():
        ax.scatter(
            coords[hub, 0],
            coords[hub, 1],
            s=9,
            facecolor="none",
            edgecolor=P["ink"],
            linewidth=0.25,
            alpha=0.20,
            rasterized=True,
        )

    ax.axhline(0, color=P["grid"], lw=0.5, zorder=0)
    ax.axvline(0, color=P["grid"], lw=0.5, zorder=0)
    ax.set_xlabel(f"SAE PC1 ({var[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"SAE PC2 ({var[1] * 100:.1f}% var.)")
    ax.set_title("PRING Human proteins in pooled SAE space", loc="left", fontweight="bold")
    ax.grid(color=P["grid"], lw=0.35, alpha=0.45)

    cb = fig.colorbar(sc, ax=ax, fraction=0.050, pad=0.03)
    cb.set_label("Full-graph degree", fontsize=6.2)
    tick_candidates = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500], dtype=float)
    ticks = tick_candidates[(tick_candidates >= degree_plus.min()) & (tick_candidates <= degree_plus.max())]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(int(t - 1)) for t in ticks])
    cb.ax.tick_params(labelsize=5.6)

    note = f"n = {len(degree):,} proteins\ninput: esmc_sae_max ({DIM:,} features)"
    if hub_cutoff is not None:
        note += f"\noutlined: degree >= {hub_cutoff}"
    ax.text(
        0.03,
        0.04,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=P["ink"],
    )

    fig.suptitle(
        "Protein-level PRING degree over raw ESM-C SAE PCA/SVD coordinates",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    for ext in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(OUT_FIG / f"human_sae_degree_pca.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load]", HUMAN_CACHE, flush=True)
    print("[load]", DEGREE_TABLE, flush=True)
    mat, rows, degree_df = load_inputs()
    print(f"[pca] proteins={mat.shape[0]:,} features={mat.shape[1]:,}", flush=True)
    coords, var, singular_values = compute_pca(mat)

    out = degree_df[["protein", "split", "degree_full", "high_participation"]].copy()
    out["cache_row"] = rows
    out["sae_pc1"] = coords[:, 0]
    out["sae_pc2"] = coords[:, 1]
    out["log10_degree_plus1"] = np.log10(out["degree_full"].to_numpy(dtype=float) + 1.0)
    out.to_csv(OUT_DIR / "human_sae_degree_pca_coordinates.tsv", sep="\t", index=False)

    meta = {
        "human_cache": str(HUMAN_CACHE),
        "degree_table": str(DEGREE_TABLE),
        "n_proteins_used": int(mat.shape[0]),
        "n_sae_features": int(mat.shape[1]),
        "representation": "esmc_sae_max",
        "pca_solver": "sklearn PCA randomized",
        "random_state": RANDOM_STATE,
        "explained_variance_ratio": [float(v) for v in var],
        "singular_values": [float(v) for v in singular_values],
        "degree_min": int(out["degree_full"].min()),
        "degree_median": float(out["degree_full"].median()),
        "degree_max": int(out["degree_full"].max()),
    }
    with (OUT_DIR / "human_sae_degree_pca_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    plot(coords, var, degree_df)
    print("[done] wrote", OUT_FIG / "human_sae_degree_pca.png", flush=True)
    print("[done] wrote", OUT_DIR / "human_sae_degree_pca_coordinates.tsv", flush=True)


if __name__ == "__main__":
    main()
