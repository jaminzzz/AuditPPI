#!/usr/bin/env python3
"""Plot global ESM-C SAE feature-category usage in PRING Human proteins.

This maps pooled human protein SAE activations onto the public ESM-C SAE feature
annotation table:

  feature_id -> category

and summarizes both the SAE vocabulary composition and how often each category
appears in human proteins.

Run:
    /data/wmzhu/anaconda3/envs/genmol/bin/python scripts/plot_human_sae_feature_distribution.py
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

from conf.paths import AUDIT, FEATURE_TABLE, FIGURES

HUMAN_CACHE = AUDIT / "pring_participation" / "pring_human_esmc_sae_cache.pt"
OUT_FIG = FIGURES
OUT_DATA = AUDIT / "sae_feature_distribution"

DIM = 16384
EPS = 1e-12
TOP_KS = (64, 128, 256, 512)
FOCUS_TOP_K = 256

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

PALETTE = [
    P["blue"],
    P["teal"],
    P["gold"],
    P["violet"],
    P["rose"],
    P["green"],
    P["brown"],
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
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
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


def load_inputs() -> tuple[pd.DataFrame, dict]:
    feature_df = pd.read_parquet(
        FEATURE_TABLE,
        columns=["feature_id", "category", "uniref90_frequency", "threshold"],
    ).sort_values("feature_id")
    if feature_df.shape[0] != DIM:
        raise ValueError(f"expected {DIM} SAE features, got {feature_df.shape[0]}")
    if not np.array_equal(feature_df["feature_id"].to_numpy(), np.arange(DIM)):
        raise ValueError("feature_id must be the contiguous range 0..16383")
    feature_df["category"] = feature_df["category"].fillna("Unclassified")
    cache = torch.load(HUMAN_CACHE, map_location="cpu", weights_only=False)
    mat = cache["esmc_sae_max"]
    if tuple(mat.shape) != (len(cache["protein_ids"]), DIM):
        raise ValueError(f"unexpected esmc_sae_max shape: {tuple(mat.shape)}")
    return feature_df, cache


def compute_summary(feature_df: pd.DataFrame, cache: dict, chunk_size: int = 256):
    mat = cache["esmc_sae_max"].float()
    n_proteins = int(mat.shape[0])
    categories = (
        feature_df["category"]
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )
    cat_to_i = {c: i for i, c in enumerate(categories)}
    cat_idx = np.array([cat_to_i[c] for c in feature_df["category"]], dtype=np.int64)

    cat_t = torch.from_numpy(cat_idx)
    max_k = max(TOP_KS)
    topk_incidence = {k: torch.zeros(len(categories), dtype=torch.long) for k in TOP_KS}
    topk_mass = {k: torch.zeros(len(categories), dtype=torch.float64) for k in TOP_KS}
    focus_counts = []
    focus_dom = []
    focus_diversity = []

    for start in range(0, n_proteins, chunk_size):
        chunk = mat[start : start + chunk_size]
        vals, idx = torch.topk(chunk, k=max_k, dim=1)
        cats = cat_t[idx]
        for k in TOP_KS:
            ck = cats[:, :k].reshape(-1)
            vk = vals[:, :k].reshape(-1).double()
            topk_incidence[k] += torch.bincount(ck, minlength=len(categories))
            topk_mass[k] += torch.zeros(len(categories), dtype=torch.float64).scatter_add_(
                0, ck, vk
            )

        fcats = cats[:, :FOCUS_TOP_K]
        onehot_counts = torch.nn.functional.one_hot(
            fcats, num_classes=len(categories)
        ).sum(dim=1)
        focus_counts.append(onehot_counts.cpu().numpy())
        focus_dom.append(onehot_counts.argmax(dim=1).cpu().numpy())
        focus_diversity.append((onehot_counts > 0).sum(dim=1).cpu().numpy())

    focus_counts = np.concatenate(focus_counts, axis=0)
    focus_dom = np.concatenate(focus_dom)
    focus_diversity = np.concatenate(focus_diversity)

    rows = []
    total_uniref90_freq = float(feature_df["uniref90_frequency"].sum())
    feature_counts = feature_df["category"].value_counts().reindex(categories).to_numpy()
    feature_share = feature_counts / feature_counts.sum()
    topk_inc_np = {k: topk_incidence[k].numpy() for k in TOP_KS}
    topk_mass_np = {k: topk_mass[k].numpy() for k in TOP_KS}
    for cat in categories:
        mask = feature_df["category"].to_numpy() == cat
        ci = cat_to_i[cat]
        row = {
            "category": cat,
            "feature_count": int(mask.sum()),
            "feature_share": float(mask.mean()),
            "uniref90_frequency_sum": int(feature_df.loc[mask, "uniref90_frequency"].sum()),
            "uniref90_frequency_share": float(
                feature_df.loc[mask, "uniref90_frequency"].sum() / (total_uniref90_freq + EPS)
            ),
            "top256_features_per_protein_mean": float(focus_counts[:, ci].mean()),
            "top256_features_per_protein_median": float(np.median(focus_counts[:, ci])),
            "top256_proteins_with_category": int((focus_counts[:, ci] > 0).sum()),
            "top256_proteins_with_category_share": float((focus_counts[:, ci] > 0).mean()),
            "top256_dominant_protein_count": int((focus_dom == ci).sum()),
            "top256_dominant_protein_share": float((focus_dom == ci).mean()),
        }
        for k in TOP_KS:
            inc_total = float(topk_inc_np[k].sum())
            mass_total = float(topk_mass_np[k].sum())
            row[f"top{k}_incidence"] = int(topk_inc_np[k][ci])
            row[f"top{k}_incidence_share"] = float(topk_inc_np[k][ci] / (inc_total + EPS))
            row[f"top{k}_activation_mass"] = float(topk_mass_np[k][ci])
            row[f"top{k}_activation_mass_share"] = float(topk_mass_np[k][ci] / (mass_total + EPS))
            row[f"top{k}_log2_enrichment_vs_feature_share"] = float(
                np.log2((topk_inc_np[k][ci] / (inc_total + EPS) + EPS) / (feature_share[ci] + EPS))
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    overall = {
        "feature_table": str(FEATURE_TABLE),
        "human_cache": str(HUMAN_CACHE),
        "n_features": DIM,
        "n_categories": len(categories),
        "n_human_proteins": n_proteins,
        "pooled_feature_matrix": "esmc_sae_max",
        "top_ks": list(TOP_KS),
        "focus_top_k": FOCUS_TOP_K,
        "topk_definition": "for each protein, take the strongest pooled SAE features by esmc_sae_max",
        "top256_category_diversity_median": float(np.median(focus_diversity)),
        "top256_category_diversity_p10": float(np.percentile(focus_diversity, 10)),
        "top256_category_diversity_p90": float(np.percentile(focus_diversity, 90)),
    }
    arrays = {
        "categories": categories,
        "top256_category_counts": focus_counts,
        "top256_dominant_category": focus_dom,
        "top256_category_diversity": focus_diversity,
    }
    return summary, overall, arrays


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.11,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=P["ink"],
    )


def plot(summary: pd.DataFrame, overall: dict, arrays: dict) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    colors = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(summary["category"])}

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.43)

    # A. Feature dictionary composition as a compact donut.
    ax = fig.add_subplot(gs[0, 0])
    _panel_label(ax, "a")
    major = summary.sort_values("feature_count", ascending=False).head(8)
    minor = summary.loc[~summary["category"].isin(major["category"])]
    donut = major[["category", "feature_count", "feature_share"]].copy()
    if not minor.empty:
        donut = pd.concat(
            [
                donut,
                pd.DataFrame(
                    {
                        "category": ["Other minor"],
                        "feature_count": [int(minor["feature_count"].sum())],
                        "feature_share": [float(minor["feature_share"].sum())],
                    }
                ),
            ],
            ignore_index=True,
        )
    dcols = [colors.get(c, P["grey_soft"]) for c in donut["category"]]
    dcols[-1] = P["grey_soft"]
    wedges, _ = ax.pie(
        donut["feature_count"],
        startangle=92,
        counterclock=False,
        colors=dcols,
        wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 0.8},
    )
    ax.text(
        0,
        0.05,
        "16,384",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=P["ink"],
    )
    ax.text(0, -0.13, "SAE features", ha="center", va="center", fontsize=6.2, color=P["muted"])
    for w, (_, r) in zip(wedges, donut.iterrows()):
        if r["feature_share"] < 0.035 and r["category"] != "Other minor":
            continue
        ang = 0.5 * (w.theta1 + w.theta2)
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        ha = "left" if x > 0 else "right"
        ax.plot([0.80 * x, 1.05 * x], [0.80 * y, 1.05 * y], color=P["grid"], lw=0.6)
        ax.text(
            1.12 * x,
            1.12 * y,
            f"{r['category']}\n{r['feature_share'] * 100:.1f}%",
            ha=ha,
            va="center",
            fontsize=5.4,
            color=P["ink"],
        )
    ax.set_aspect("equal")
    ax.set_title("ESM-C SAE vocabulary by annotation category", loc="left", fontweight="bold")

    # B. Human top-K category mixture as a bubble matrix.
    ax = fig.add_subplot(gs[0, 1])
    _panel_label(ax, "b")
    top_cats = (
        summary.sort_values(f"top{FOCUS_TOP_K}_incidence_share", ascending=False)
        .head(8)["category"]
        .tolist()
    )
    x_labels = ["dict"] + [f"top-{k}" for k in TOP_KS]
    x = np.arange(len(x_labels))
    y = np.arange(len(top_cats))[::-1]
    for yi, cat in zip(y, top_cats):
        row = summary.loc[summary["category"] == cat].iloc[0]
        vals = [row["feature_share"] * 100]
        vals += [row[f"top{k}_incidence_share"] * 100 for k in TOP_KS]
        sizes = 18 + np.asarray(vals) * 17
        ax.scatter(
            x,
            np.full_like(x, yi, dtype=float),
            s=sizes,
            color=colors[cat],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.88,
            zorder=3,
        )
        for xx, v in zip(x, vals):
            if v >= 10:
                ax.text(
                    xx,
                    yi,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color="white" if v >= 16 else P["ink"],
                    fontweight="bold" if v >= 16 else "normal",
                    zorder=4,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y)
    short_labels = {
        "Compositional bias": "Compos. bias",
        "Structural motif": "Structural motif",
        "Post-translational modification": "PTM",
        "Ligand-binding site": "Ligand-binding",
        "Membrane-associated": "Membrane assoc.",
    }
    ax.set_yticklabels([short_labels.get(c, c) for c in top_cats])
    ax.set_xlim(-0.45, len(x_labels) - 0.55)
    ax.set_ylim(-0.65, len(top_cats) - 0.35)
    ax.set_title("Top pooled SAE features reshape category usage", loc="left", fontweight="bold")
    for xx in x:
        ax.axvline(xx, color=P["grid"], lw=0.35, alpha=0.55, zorder=0)
    for yy in y:
        ax.axhline(yy, color=P["grid"], lw=0.35, alpha=0.35, zorder=0)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.text(
        0.99,
        -0.14,
        "circle area = category share; numbers are %",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=P["muted"],
    )

    # C. Category enrichment among human top-256 features.
    ax = fig.add_subplot(gs[1, 0])
    _panel_label(ax, "c")
    enr_col = f"top{FOCUS_TOP_K}_log2_enrichment_vs_feature_share"
    sdf = summary.sort_values(enr_col, ascending=True)
    y = np.arange(len(sdf))
    ax.axvline(0, color=P["grid"], lw=0.8)
    for yi, (_, r) in enumerate(sdf.iterrows()):
        col = colors[r["category"]]
        ax.plot([0, r[enr_col]], [yi, yi], color=col, lw=1.2, alpha=0.7)
        ax.scatter(r[enr_col], yi, s=20, color=col, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sdf["category"])
    ax.set_xlabel(f"log2 enrichment in human top-{FOCUS_TOP_K} vs dictionary")
    ax.set_title("Human proteins emphasize a subset of SAE categories", loc="left", fontweight="bold")
    ax.grid(axis="x", color=P["grid"], lw=0.4, alpha=0.7)
    ax.text(0.02, 0.04, "left: under-represented\nright: over-represented",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color=P["muted"])

    # D. Protein-level composition manifold.
    ax = fig.add_subplot(gs[1, 1])
    _panel_label(ax, "d")
    counts = arrays["top256_category_counts"].astype(float)
    props = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    centered = props - props.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    var = (s**2) / np.sum(s**2)
    comp_col = props[:, arrays["categories"].index("Compositional bias")]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "comp_bias", [P["blue_soft"], "#FFFFFF", P["rose_soft"], P["rose"]]
    )
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=comp_col,
        cmap=cmap,
        s=5,
        alpha=0.62,
        linewidth=0,
        rasterized=True,
    )
    ax.axhline(0, color=P["grid"], lw=0.5, zorder=0)
    ax.axvline(0, color=P["grid"], lw=0.5, zorder=0)
    ax.set_xlabel(f"category PC1 ({var[0] * 100:.0f}% var.)")
    ax.set_ylabel(f"category PC2 ({var[1] * 100:.0f}% var.)")
    ax.set_title("Protein-level SAE category composition is heterogeneous", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Compositional-bias share", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.2)
    for q, lab in [(0.1, "low\nbias"), (0.9, "high\nbias")]:
        idx = np.argmin(np.abs(comp_col - np.quantile(comp_col, q)))
        ax.scatter(coords[idx, 0], coords[idx, 1], s=45, facecolor="none",
                   edgecolor=P["ink"], linewidth=0.7, zorder=4)
        ax.annotate(
            lab.replace("\n", " "),
            xy=(coords[idx, 0], coords[idx, 1]),
            xytext=(8, 9 if q < 0.5 else -12),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.2,
            color=P["ink"],
            arrowprops={"arrowstyle": "-", "lw": 0.45, "color": P["muted"]},
        )
    ax.text(
        0.98,
        0.06,
        f"n proteins = {overall['n_human_proteins']:,}\n"
        f"median categories / protein = {overall['top256_category_diversity_median']:.0f}\n"
        f"top-{FOCUS_TOP_K} pooled SAE features",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=P["ink"],
    )

    fig.suptitle(
        "Human proteins concentrate their strongest ESM-C SAE features into structured biological categories",
        x=0.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )

    for ext in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 400
        fig.savefig(OUT_FIG / f"human_sae_feature_distribution.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    setup_style()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    print("[load] feature table:", FEATURE_TABLE, flush=True)
    print("[load] human cache:", HUMAN_CACHE, flush=True)
    feature_df, cache = load_inputs()
    print(f"[compute] proteins={len(cache['protein_ids']):,} features={DIM:,}", flush=True)
    summary, overall, arrays = compute_summary(feature_df, cache)
    summary.to_csv(OUT_DATA / "human_pring_sae_category_summary.tsv", sep="\t", index=False)
    with (OUT_DATA / "human_pring_sae_category_summary.json").open("w") as f:
        json.dump({"overall": overall, "categories": summary.to_dict(orient="records")}, f, indent=2)
    plot(summary, overall, arrays)
    print("[done] wrote", OUT_FIG / "human_sae_feature_distribution.png", flush=True)
    print("[done] wrote", OUT_DATA / "human_pring_sae_category_summary.tsv", flush=True)


if __name__ == "__main__":
    main()
