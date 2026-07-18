#!/usr/bin/env python3
"""Plot how PRING Human hubness relates to ESM-C SAE category composition.

This standalone figure combines:
  - PRING Human full-graph degree / high-participation labels,
  - pooled per-protein ESM-C SAE activations,
  - ESM-C SAE feature annotations from the public feature table.

The SAE representation is summarized as each protein's top-256 pooled SAE
features by ``esmc_sae_max``. Category shares are computed within those top
features, which avoids the pooled-max matrix's dense low-level activation tail.

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/figures/plot_human_sae_hub_category_association.py
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
from scipy.stats import spearmanr

from conf.paths import RESULTS_PROTEIN, FEATURE_TABLE, FIGURES, PRING_HUMAN_SAE_CACHE

HUMAN_CACHE = PRING_HUMAN_SAE_CACHE
DEGREE_TABLE = (
    RESULTS_PROTEIN
    / "pring_participation"
    / "high_p90_xgboost"
    / "pring_human_bfs_binary_high_q90_xgboost_protein_predictions.tsv"
)
OUT_FIG = FIGURES
OUT_DIR = RESULTS_PROTEIN / "sae_hub_category"

DIM = 16384
TOP_K = 256
EPS = 1e-12

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
    "grey_soft": "#E9EAEE",
}

CAT_COLORS = [
    P["blue"],
    P["teal"],
    P["gold"],
    P["violet"],
    P["rose"],
    P["green"],
    "#8E6C4A",
    "#4A8FBA",
    "#A8679B",
    "#B9803D",
    "#7A9E3D",
    P["muted"],
    "#9C9CA5",
    "#C4C7CF",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.8,
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


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.11,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=P["ink"],
    )


def load_inputs():
    feature_df = pd.read_parquet(FEATURE_TABLE, columns=["feature_id", "category"]).sort_values("feature_id")
    if feature_df.shape[0] != DIM:
        raise ValueError(f"expected {DIM} features, got {feature_df.shape[0]}")
    if not np.array_equal(feature_df["feature_id"].to_numpy(), np.arange(DIM)):
        raise ValueError("feature_id must be contiguous 0..16383")
    feature_df["category"] = feature_df["category"].fillna("Unclassified")

    cache = torch.load(HUMAN_CACHE, map_location="cpu", weights_only=False)
    degree_df = pd.read_csv(DEGREE_TABLE, sep="\t")
    return feature_df, cache, degree_df


def topk_category_matrix(feature_df: pd.DataFrame, cache: dict, protein_ids: list[str], chunk_size: int = 256):
    categories = feature_df["category"].value_counts().sort_values(ascending=False).index.tolist()
    cat_to_i = {c: i for i, c in enumerate(categories)}
    cat_idx = np.array([cat_to_i[c] for c in feature_df["category"]], dtype=np.int64)
    cat_t = torch.from_numpy(cat_idx)

    cache_id = cache["uniprotid2idx"]
    rows = [int(cache_id[p]) for p in protein_ids]
    mat = cache["esmc_sae_max"][rows].float()

    counts = []
    for start in range(0, mat.shape[0], chunk_size):
        chunk = mat[start : start + chunk_size]
        _, idx = torch.topk(chunk, k=TOP_K, dim=1)
        cats = cat_t[idx]
        onehot = torch.nn.functional.one_hot(cats, num_classes=len(categories)).sum(dim=1)
        counts.append(onehot.cpu().numpy())
    counts = np.concatenate(counts, axis=0).astype(float)
    props = counts / float(TOP_K)
    return categories, counts, props


def prepare_data():
    feature_df, cache, degree_df = load_inputs()
    cache_ids = set(cache["uniprotid2idx"].keys())
    degree_df = degree_df[degree_df["protein"].isin(cache_ids)].copy()
    degree_df = degree_df.sort_values("protein").reset_index(drop=True)
    categories, counts, props = topk_category_matrix(feature_df, cache, degree_df["protein"].tolist())

    q25, q50 = np.percentile(degree_df["degree_full"], [25, 50])
    hub_cutoff = int(degree_df.loc[degree_df["high_participation"] == 1, "degree_full"].min())
    degree = degree_df["degree_full"].to_numpy(dtype=float)
    groups = np.select(
        [
            degree <= q25,
            (degree > q25) & (degree <= q50),
            (degree > q50) & (degree < hub_cutoff),
            degree >= hub_cutoff,
        ],
        ["low", "mid", "high", "hub"],
        default="high",
    )
    group_order = ["low", "mid", "high", "hub"]
    group_labels = {
        "low": f"low\n(deg <= {int(q25)})",
        "mid": f"mid\n({int(q25) + 1}-{int(q50)})",
        "high": f"high\n({int(q50) + 1}-{hub_cutoff - 1})",
        "hub": f"hub\n(deg >= {hub_cutoff})",
    }

    overall = props.mean(axis=0)
    enrich = []
    for g in group_order:
        share = props[groups == g].mean(axis=0)
        enrich.append(np.log2((share + EPS) / (overall + EPS)))
    enrich = np.vstack(enrich)

    comp_idx = categories.index("Compositional bias")
    structural_idx = categories.index("Structural motif")
    stat_rows = []
    for i, cat in enumerate(categories):
        rho, pval = spearmanr(np.log1p(degree), props[:, i])
        stat_rows.append(
            {
                "category": cat,
                "overall_top256_share": float(overall[i]),
                "spearman_log1p_degree": float(rho),
                "spearman_p": float(pval),
            }
        )
        for g in group_order:
            stat_rows[-1][f"{g}_mean_share"] = float(props[groups == g, i].mean())
        stat_rows[-1]["hub_vs_nonhub_delta"] = float(
            props[groups == "hub", i].mean() - props[groups != "hub", i].mean()
        )
        stat_rows[-1]["hub_log2_enrichment"] = float(enrich[group_order.index("hub"), i])
    stats = pd.DataFrame(stat_rows)

    return {
        "feature_df": feature_df,
        "degree_df": degree_df,
        "categories": categories,
        "props": props,
        "counts": counts,
        "degree": degree,
        "groups": groups,
        "group_order": group_order,
        "group_labels": group_labels,
        "hub_cutoff": hub_cutoff,
        "enrich": enrich,
        "stats": stats,
        "comp_idx": comp_idx,
        "structural_idx": structural_idx,
    }


def _category_pca(props: np.ndarray):
    centered = props - props.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    var = (s**2) / np.sum(s**2)
    return coords, var


def plot(data: dict) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    categories = data["categories"]
    cat_colors = {c: CAT_COLORS[i % len(CAT_COLORS)] for i, c in enumerate(categories)}

    degree = data["degree"]
    props = data["props"]
    groups = data["groups"]
    hub_cutoff = data["hub_cutoff"]
    structural_idx = data["structural_idx"]

    fig = plt.figure(figsize=(7.2, 5.85))
    gs = fig.add_gridspec(2, 2, hspace=0.54, wspace=0.46)

    # a. Degree distribution as ECDF.
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    x = np.sort(degree)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x + 1, y * 100, color=P["blue"], lw=1.8)
    ax.fill_betweenx([0, 100], hub_cutoff + 1, x.max() + 1, color=P["rose_soft"], alpha=0.45)
    ax.axvline(hub_cutoff + 1, color=P["rose"], lw=0.9, ls="--")
    ax.text(
        hub_cutoff + 3,
        12,
        f"hub cutoff\n>= {hub_cutoff} partners",
        color=P["rose"],
        fontsize=5.8,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Full-graph degree + 1")
    ax.set_ylabel("Cumulative proteins (%)")
    ax.set_title("PRING Human degree distribution is strongly skewed", loc="left", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=P["grid"], lw=0.4, alpha=0.75)
    ax.text(
        0.04,
        0.90,
        f"n = {len(degree):,} proteins\nhub fraction = {(groups == 'hub').mean() * 100:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=P["ink"],
    )

    # b. Degree vs structural-motif SAE share.
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    structural = props[:, structural_idx] * 100
    nonhub = groups != "hub"
    hub = groups == "hub"
    ax.scatter(
        np.log10(degree[nonhub] + 1),
        structural[nonhub],
        s=7,
        color=P["blue_soft"],
        alpha=0.38,
        edgecolor="none",
        rasterized=True,
        label="non-hub",
    )
    ax.scatter(
        np.log10(degree[hub] + 1),
        structural[hub],
        s=9,
        color=P["rose"],
        alpha=0.45,
        edgecolor="none",
        rasterized=True,
        label="hub",
    )
    bins = np.quantile(np.log10(degree + 1), np.linspace(0, 1, 13))
    bins = np.unique(bins)
    xm, ym = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (np.log10(degree + 1) >= lo) & (np.log10(degree + 1) <= hi)
        if m.sum() < 10:
            continue
        xm.append(np.median(np.log10(degree[m] + 1)))
        ym.append(np.median(structural[m]))
    ax.plot(xm, ym, color=P["ink"], lw=1.2, marker="o", markersize=2.8, zorder=4)
    rho, pval = spearmanr(np.log1p(degree), structural)
    ax.text(
        0.04,
        0.95,
        f"Spearman rho = {rho:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=P["ink"],
        fontweight="bold",
    )
    ax.set_xlabel("log10(full-graph degree + 1)")
    ax.set_ylabel(f"Structural-motif share in top-{TOP_K} (%)")
    ax.set_title("Hubness modestly shifts structural-motif load", loc="left", fontweight="bold")
    ax.legend(loc="lower right", markerscale=1.4)
    ax.grid(color=P["grid"], lw=0.35, alpha=0.55)

    # c. Category-wise association with graph degree.
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    stats = data["stats"].copy()
    stats = stats[stats["overall_top256_share"] >= 0.005].copy()
    stats = stats.sort_values("spearman_log1p_degree")
    short = {
        "Post-translational modification": "PTM",
        "Membrane-associated": "Membrane assoc.",
        "Compositional bias": "Compos. bias",
    }
    y = np.arange(len(stats))
    ax.axvline(0, color=P["grid"], lw=0.8)
    for yi, (_, row) in enumerate(stats.iterrows()):
        rho = row["spearman_log1p_degree"]
        col = P["rose"] if rho > 0 else P["blue"]
        ax.plot([0, rho], [yi, yi], color=col, lw=1.25, alpha=0.65)
        ax.scatter(rho, yi, s=24, color=col, edgecolor="white", linewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([short.get(c, c) for c in stats["category"]])
    ax.set_xlabel("Spearman rho with log1p(degree)")
    ax.set_xlim(-0.10, 0.18)
    ax.set_title("Most SAE category shifts are weak but consistent", loc="left", fontweight="bold")
    ax.grid(axis="x", color=P["grid"], lw=0.4, alpha=0.7)
    ax.text(
        0.03,
        0.04,
        "categories with <0.5% top-256 share omitted",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=P["muted"],
    )

    # d. Category-composition landscape, colored by graph degree.
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    coords, var = _category_pca(props)
    cmap_degree = mpl.colors.LinearSegmentedColormap.from_list(
        "degree", [P["blue_soft"], "#FFFFFF", P["gold_soft"], P["rose"]]
    )
    color_vals = np.log10(degree + 1)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_vals,
        cmap=cmap_degree,
        s=5,
        alpha=0.62,
        linewidth=0,
        rasterized=True,
    )
    ax.scatter(
        coords[hub, 0],
        coords[hub, 1],
        s=7,
        facecolor="none",
        edgecolor=P["ink"],
        linewidth=0.25,
        alpha=0.18,
        rasterized=True,
    )
    ax.axhline(0, color=P["grid"], lw=0.5, zorder=0)
    ax.axvline(0, color=P["grid"], lw=0.5, zorder=0)
    ax.set_xlabel(f"category PC1 ({var[0] * 100:.0f}% var.)")
    ax.set_ylabel(f"category PC2 ({var[1] * 100:.0f}% var.)")
    ax.set_title("Hub proteins occupy a shifted SAE category landscape", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log10(degree + 1)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.2)
    ax.text(
        0.04,
        0.05,
        f"outlined points: hubs\n(top-{TOP_K} category composition)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=P["ink"],
    )

    fig.suptitle(
        "Human PPI hubness shows modest but structured shifts in ESM-C SAE category usage",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    for ext in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_FIG / f"human_sae_hub_category_association.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load]", FEATURE_TABLE, flush=True)
    print("[load]", HUMAN_CACHE, flush=True)
    print("[load]", DEGREE_TABLE, flush=True)
    data = prepare_data()
    stats = data["stats"]
    stats.to_csv(OUT_DIR / "human_sae_hub_category_stats.tsv", sep="\t", index=False)
    meta = {
        "feature_table": str(FEATURE_TABLE),
        "human_cache": str(HUMAN_CACHE),
        "degree_table": str(DEGREE_TABLE),
        "n_proteins_used": int(len(data["degree"])),
        "top_k": TOP_K,
        "hub_cutoff_degree": int(data["hub_cutoff"]),
        "group_counts": {g: int((data["groups"] == g).sum()) for g in data["group_order"]},
    }
    with (OUT_DIR / "human_sae_hub_category_stats.json").open("w") as f:
        json.dump({"meta": meta, "categories": stats.to_dict(orient="records")}, f, indent=2)
    plot(data)
    print("[done] wrote", OUT_FIG / "human_sae_hub_category_association.png", flush=True)
    print("[done] wrote", OUT_DIR / "human_sae_hub_category_stats.tsv", flush=True)


if __name__ == "__main__":
    main()
