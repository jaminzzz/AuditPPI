#!/usr/bin/env python3
"""Standalone supplementary figures for the AuditPPI audit.

This script intentionally does not modify plot_audit_figures.py. It turns the
already-computed audit outputs in AuditPPI and the upper SAE_PPI directory into
three small, logically separated supplementary figures:

S1: participation-bias controls
S2: SAE fingerprint anatomy and compactness
S3: transferable concepts and structural grounding

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/figures/plot_supplementary_audit_figures.py
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec, patches
from matplotlib.colors import LinearSegmentedColormap

from conf.paths import ROOT
from conf.paths import RESULTS_MISC, RESULTS_RESIDUE, FIGURES, SAE_SUPP_INPUTS

# precomputed results/propensity JSONs (was SAE_PPI/ppi_fingerprint/outputs/)
SAE_OUT = SAE_SUPP_INPUTS
OUT_FIG = FIGURES
OUT_DIR = RESULTS_MISC / "supplementary_audit_figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

CATEGORY_COLORS = {
    "Structural motif": P["blue"],
    "Disorder": P["rose"],
    "Post-translational modification": P["gold"],
    "Ligand-binding site": P["teal"],
    "Membrane-associated": P["violet"],
    "Domain": P["green"],
    "Compositional bias": P["brown"],
    "Interaction site": "#4A8FBA",
    "Catalytic function": "#A8679B",
    "Repeat": "#7A9E3D",
    "Sequence motif": "#B9803D",
    "Unannotated": P["muted"],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6.1,
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


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem} -> {OUT_FIG}")


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


def wrap_label(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def add_grid_x(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=P["grid"], lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def metric_rows_for_s1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prop = load_json(
        SAE_OUT / "esmc" / "propensity" / "participation_bias_c3_vs_bernett.json"
    )
    graph = load_json(SAE_OUT / "esmc" / "propensity" / "c3_graph_degree_prior.json")
    degree = load_json(SAE_OUT / "esmc" / "propensity" / "c3_degree_controlled.json")
    homology = load_json(SAE_OUT / "esmc" / "propensity" / "c3_homology_stratified.json")
    context = load_json(
        SAE_OUT / "esmc" / "propensity" / "c3_context_matched_negatives.json"
    )
    cross = load_json(
        SAE_OUT
        / "cross_species"
        / "esmc"
        / "propensity"
        / "participation_bias_cross_species.json"
    )
    sars = load_json(
        SAE_OUT
        / "sars_cov2"
        / "esmc"
        / "transfer"
        / "c3_xgb_full"
        / "metrics_c3_xgb_sym.json"
    )

    participation = pd.DataFrame(
        [
            {"dataset": "C3", **prop["c3_test"]},
            {"dataset": "Bernett", **prop["bernett_test"]},
        ]
    )

    hom_bins = homology["auroc_by_dense_cosine_min_q5"]
    hom_vals = [r["auroc"] for r in hom_bins]
    context_vals = []
    context_null = []
    for variant in context["variants"].values():
        for key, value in variant.items():
            if key.startswith("nbins") and isinstance(value, dict):
                context_vals.append(value["model_auroc"])
                context_null.append(value["sim_only_auroc_sanity"])

    controls = pd.DataFrame(
        [
            {
                "control": "Graph recurrence",
                "model_center": graph["best_graph_test_auroc"],
                "model_low": graph["best_graph_test_auroc"],
                "model_high": graph["best_graph_test_auroc"],
                "null_center": np.nan,
                "note": "graph-only",
            },
            {
                "control": "Degree matched",
                "model_center": degree["degree_matched_subset"]["model_auroc"],
                "model_low": degree["degree_matched_subset"]["model_auroc"],
                "model_high": degree["degree_matched_subset"]["model_auroc"],
                "null_center": degree["degree_matched_subset"]["degprod_auroc"],
                "note": "same degree product",
            },
            {
                "control": "Homology strata",
                "model_center": float(np.mean(hom_vals)),
                "model_low": float(np.min(hom_vals)),
                "model_high": float(np.max(hom_vals)),
                "null_center": np.nan,
                "note": "nearest-train bins",
            },
            {
                "control": "Context matched",
                "model_center": float(np.mean(context_vals)),
                "model_low": float(np.min(context_vals)),
                "model_high": float(np.max(context_vals)),
                "null_center": float(np.mean(context_null)),
                "note": "A-B similarity balanced",
            },
            {
                "control": "Full C3 model",
                "model_center": prop["c3_test"]["full_model_auroc"],
                "model_low": prop["c3_test"]["full_model_auroc"],
                "model_high": prop["c3_test"]["full_model_auroc"],
                "null_center": np.nan,
                "note": "reference",
            },
        ]
    )

    transfer = pd.DataFrame(
        [
            {
                "benchmark": "C3",
                "model_auroc": prop["c3_test"]["full_model_auroc"],
                "oracle_auroc": prop["c3_test"]["participation_oracle_min_auroc"],
            },
            {
                "benchmark": "Cross-species",
                "model_auroc": cross["mean_model"],
                "oracle_auroc": cross["mean_oracle"],
            },
            {
                "benchmark": "Bernett",
                "model_auroc": prop["bernett_test"]["full_model_auroc"],
                "oracle_auroc": prop["bernett_test"]["participation_oracle_min_auroc"],
            },
            {
                "benchmark": "C3 -> SARS-CoV-2",
                "model_auroc": sars["sars"]["auroc"],
                "oracle_auroc": np.nan,
            },
        ]
    )

    participation.to_csv(OUT_DIR / "s1_participation_summary.tsv", sep="\t", index=False)
    controls.to_csv(OUT_DIR / "s1_control_summary.tsv", sep="\t", index=False)
    transfer.to_csv(OUT_DIR / "s1_transfer_summary.tsv", sep="\t", index=False)
    return participation, controls, transfer


def make_s1_participation_controls() -> None:
    participation, controls, transfer = metric_rows_for_s1()
    fig = plt.figure(figsize=(7.15, 4.35))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.58, wspace=0.5)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    ax.axvspan(0, 0.2, color=P["rose_soft"], alpha=0.28, lw=0)
    ax.axvspan(0.8, 1.0, color=P["gold_soft"], alpha=0.28, lw=0)
    y_map = {"C3": 1, "Bernett": 0}
    color_map = {"C3": P["rose"], "Bernett": P["teal"]}
    soft_map = {"C3": P["rose_soft"], "Bernett": P["teal_soft"]}
    for _, row in participation.iterrows():
        y = y_map[row["dataset"]]
        mean = float(row["t_mean"])
        sd = float(row["t_std"])
        lo = max(0.0, mean - sd)
        hi = min(1.0, mean + sd)
        ax.plot([0, 1], [y, y], color=P["grid"], lw=1.2, zorder=1)
        ax.add_patch(
            patches.FancyBboxPatch(
                (lo, y - 0.12),
                hi - lo,
                0.24,
                boxstyle="round,pad=0.0,rounding_size=0.035",
                fc=soft_map[row["dataset"]],
                ec="none",
                alpha=0.95,
                zorder=2,
            )
        )
        ax.scatter(
            [mean],
            [y],
            s=54,
            color=color_map[row["dataset"]],
            edgecolor="white",
            lw=0.8,
            zorder=4,
        )
        ax.text(
            1.02,
            y,
            f"sd={sd:.2f}\nlow={row['frac_t_lt_0.2']:.0%}, high={row['frac_t_gt_0.8']:.0%}",
            ha="left",
            va="center",
            fontsize=5.9,
            color=P["muted"],
        )
    ax.text(0.1, 1.38, "mostly negative", ha="center", va="bottom", fontsize=5.8, color=P["rose"])
    ax.text(0.9, 1.38, "mostly positive", ha="center", va="bottom", fontsize=5.8, color=P["gold"])
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["C3", "Bernett"])
    ax.set_xlim(0, 1.22)
    ax.set_ylim(-0.48, 1.52)
    ax.set_xlabel("per-protein positive fraction, t(p)")
    ax.set_title("Participation imbalance", loc="left", fontweight="bold")
    add_grid_x(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    rows = participation[["dataset", "participation_oracle_min_auroc", "full_model_auroc"]]
    for i, (_, row) in enumerate(rows.iloc[::-1].iterrows()):
        y = i
        xs = [row["participation_oracle_min_auroc"], row["full_model_auroc"]]
        ax.plot(xs, [y, y], color=P["grid"], lw=2.4, solid_capstyle="round", zorder=1)
        ax.scatter(
            [xs[0]],
            [y],
            s=56,
            marker="o",
            color=P["gold"],
            edgecolor="white",
            lw=0.8,
            label="participation oracle" if i == 0 else None,
            zorder=3,
        )
        ax.scatter(
            [xs[1]],
            [y],
            s=56,
            marker="D",
            color=P["blue"],
            edgecolor="white",
            lw=0.8,
            label="full model" if i == 0 else None,
            zorder=3,
        )
        ax.text(max(xs) + 0.015, y, f"{row['dataset']}", ha="left", va="center", color=P["ink"])
    ax.axvline(0.5, color=P["grid"], lw=0.8, ls="--")
    ax.set_xlim(0.55, 0.98)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("test AUROC")
    ax.set_title("Oracle tracks the model", loc="left", fontweight="bold")
    ax.legend(loc="lower right", handletextpad=0.5)
    add_grid_x(ax)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    plot_df = controls.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    for yi, row in zip(y, plot_df.itertuples()):
        if row.model_low != row.model_high:
            ax.plot([row.model_low, row.model_high], [yi, yi], color=P["blue_soft"], lw=3.0)
        ax.scatter(
            [row.model_center],
            [yi],
            s=46,
            color=P["blue"] if row.control != "Graph recurrence" else P["muted"],
            edgecolor="white",
            lw=0.75,
            zorder=4,
        )
        if not math.isnan(row.null_center):
            ax.scatter(
                [row.null_center],
                [yi],
                s=32,
                color="white",
                edgecolor=P["muted"],
                lw=1.0,
                zorder=4,
            )
    ax.axvline(0.5, color=P["grid"], lw=0.85, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(v, 18) for v in plot_df["control"]])
    ax.set_xlim(0.46, 0.98)
    ax.set_xlabel("AUROC after control")
    ax.set_title("Candidate leakage controls", loc="left", fontweight="bold")
    ax.text(0.965, -0.45, "model", color=P["blue"], ha="right", va="center", fontsize=5.9)
    ax.text(0.535, -0.45, "null", color=P["muted"], ha="left", va="center", fontsize=5.9)
    add_grid_x(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    plot_df = transfer.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    for yi, row in zip(y, plot_df.itertuples()):
        if not math.isnan(row.oracle_auroc):
            ax.plot(
                [row.oracle_auroc, row.model_auroc],
                [yi, yi],
                color=P["grid"],
                lw=2.2,
                solid_capstyle="round",
                zorder=1,
            )
            ax.scatter([row.oracle_auroc], [yi], s=46, color=P["gold"], edgecolor="white", lw=0.75, zorder=3)
        ax.scatter([row.model_auroc], [yi], s=50, marker="D", color=P["blue"], edgecolor="white", lw=0.75, zorder=4)
    ax.axvline(0.5, color=P["grid"], lw=0.85, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(v, 18) for v in plot_df["benchmark"]])
    ax.set_xlim(0.40, 1.02)
    ax.set_xlabel("AUROC")
    ax.set_title("Construction predicts transfer behavior", loc="left", fontweight="bold")
    ax.scatter([], [], s=46, color=P["gold"], edgecolor="white", lw=0.75, label="participation oracle")
    ax.scatter([], [], s=50, marker="D", color=P["blue"], edgecolor="white", lw=0.75, label="model")
    ax.legend(loc="lower right", handletextpad=0.45)
    add_grid_x(ax)

    save_fig(fig, "supplementary_s1_participation_bias_controls")


def make_s2_fingerprint_interpretability() -> None:
    interp_dir = SAE_OUT / "esmc" / "interpretability"
    summary = load_json(interp_dir / "summary.json")
    categories = pd.read_csv(interp_dir / "1_category_attribution.csv")
    categories["category"] = categories["category"].fillna("Unannotated").replace({"nan": "Unannotated"})
    minimal = pd.read_csv(interp_dir / "2_minimal_fingerprint.csv")
    topk = pd.read_csv(SAE_OUT / "esmc" / "tabpfn_topk" / "summary_sae-id.csv")

    categories.to_csv(OUT_DIR / "s2_category_attribution.tsv", sep="\t", index=False)
    minimal.to_csv(OUT_DIR / "s2_minimal_fingerprint.tsv", sep="\t", index=False)
    topk.to_csv(OUT_DIR / "s2_tabpfn_topk.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.15, 4.45))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.52)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    top = categories.sort_values("pct_of_signal", ascending=False).head(9).iloc[::-1]
    y = np.arange(len(top))
    for yi, row in zip(y, top.itertuples()):
        color = CATEGORY_COLORS.get(row.category, P["muted"])
        ax.plot([0, row.pct_of_signal], [yi, yi], color=P["grid"], lw=1.6, zorder=1)
        ax.scatter([row.pct_of_signal], [yi], s=58, color=color, edgecolor="white", lw=0.75, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(c, 21) for c in top["category"]])
    ax.set_xlim(0, 22)
    ax.set_xlabel("share of total TreeSHAP signal (%)")
    ax.set_title("SAE concept categories", loc="left", fontweight="bold")
    add_grid_x(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    block = summary["block_share"]
    sizes = [block["AND(a*b)"], block["|a-b|"]]
    colors = [P["teal"], P["blue"]]
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.text(0, 0.06, "pair\nfeatures", ha="center", va="center", color=P["ink"], fontsize=7.1)
    ax.text(-1.18, -0.86, f"AND\n{sizes[0]:.1%}", ha="center", va="center", color=P["teal"], fontsize=6.3)
    ax.text(1.17, 0.86, f"|a-b|\n{sizes[1]:.1%}", ha="center", va="center", color=P["blue"], fontsize=6.3)
    ax.set_title("Interaction block anatomy", loc="left", fontweight="bold")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    plot = minimal.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    for yi, row in zip(y, plot.itertuples()):
        ax.plot([row.binary_auroc, row.sae_max_auroc], [yi, yi], color=P["grid"], lw=2.4, solid_capstyle="round")
        ax.scatter([row.binary_auroc], [yi], s=38, color=P["blue_soft"], edgecolor=P["blue"], lw=0.75, zorder=3)
        ax.scatter([row.sae_max_auroc], [yi], s=42, color=P["teal"], edgecolor="white", lw=0.75, zorder=4)
        if row.K == summary["minimal_fp"]["knee_within_1pt"]:
            ax.text(row.sae_max_auroc + 0.003, yi, "within 1 pt", ha="left", va="center", fontsize=5.8, color=P["teal"])
    ax.axvline(summary["minimal_fp"]["full_auroc"], color=P["muted"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"K={int(k)}" for k in plot["K"]])
    ax.set_xlim(0.825, 0.945)
    ax.set_xlabel("C3 test AUROC")
    ax.set_title("Compact fingerprint recovers C3 signal", loc="left", fontweight="bold")
    ax.text(0.917, 6.85, "binary", color=P["blue"], fontsize=5.9, ha="left", va="center")
    ax.text(0.932, 7.25, "max-pool", color=P["teal"], fontsize=5.9, ha="left", va="center")
    add_grid_x(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    arch_order = ["logreg", "xgb", "tabpfn"]
    arch_label = {"logreg": "LogReg", "xgb": "XGB", "tabpfn": "TabPFN"}
    k_order = sorted(topk["top_k"].unique())
    cmap = LinearSegmentedColormap.from_list("auc", [P["rose_soft"], "#FFFFFF", P["blue"]])
    vmin, vmax = 0.84, 0.94
    for _, row in topk.iterrows():
        x = arch_order.index(row["arch"])
        yv = k_order.index(row["top_k"])
        size = 28 + (row["input_dim"] / topk["input_dim"].max()) * 90
        ax.scatter(
            x,
            yv,
            s=size,
            c=[row["test_auroc"]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="white",
            lw=0.75,
            zorder=3,
        )
        ax.text(x, yv, f"{row['test_auroc']:.2f}", ha="center", va="center", fontsize=5.4, color=P["ink"])
    ax.set_xticks(range(len(arch_order)))
    ax.set_xticklabels([arch_label[a] for a in arch_order])
    ax.set_yticks(range(len(k_order)))
    ax.set_yticklabels([str(k) for k in k_order])
    ax.set_ylabel("top SAE features")
    ax.set_title("Top-K features across models", loc="left", fontweight="bold")
    ax.set_xlim(-0.55, len(arch_order) - 0.45)
    ax.set_ylim(-0.55, len(k_order) - 0.45)
    ax.invert_yaxis()
    ax.grid(color=P["grid"], lw=0.5, alpha=0.65)
    ax.set_axisbelow(True)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("test AUROC", labelpad=2)
    cb.outline.set_visible(False)

    save_fig(fig, "supplementary_s2_sae_fingerprint_interpretability")


def _enrichment_map(items: list[dict]) -> dict[str, float]:
    return {d["category"]: float(d["enrichment"]) for d in items}


def make_s3_transferable_grounding() -> None:
    transfer = load_json(
        SAE_OUT / "esmc" / "propensity" / "transferable_vs_bias_features.json"
    )
    rosetta = load_json(
        SAE_OUT
        / "rosetta_benchmark"
        / "sae_feature_compare"
        / "rosetta_vs_c3_shap_compare.json"
    )
    interface = pd.read_csv(
        RESULTS_RESIDUE
        / "interface_grounding"
        / "pdb_ppi_pos_sae_enrichment_all_noninterface"
        / "interface_grounded_category_summary.tsv",
        sep="\t",
    )

    with open(OUT_DIR / "s3_transferable_vs_bias_features.json", "w") as fh:
        json.dump(transfer, fh, indent=2)
    interface.to_csv(OUT_DIR / "s3_interface_grounding_categories.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(7.15, 4.55))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.56)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    part = _enrichment_map(transfer["category_enrichment"]["participation_set"])
    trans = _enrichment_map(transfer["category_enrichment"]["transferable_set"])
    focus = [
        "Disorder",
        "Post-translational modification",
        "Membrane-associated",
        "Interaction site",
        "Compositional bias",
        "Structural motif",
        "Domain",
        "Ligand-binding site",
    ]
    for cat in focus:
        if cat not in part or cat not in trans:
            continue
        y0, y1 = np.log2(part[cat]), np.log2(trans[cat])
        color = CATEGORY_COLORS.get(cat, P["muted"])
        ax.plot([0, 1], [y0, y1], color=color, lw=1.5, alpha=0.82)
        ax.scatter([0, 1], [y0, y1], s=34, color=color, edgecolor="white", lw=0.65, zorder=3)
        if cat in {"Disorder", "Membrane-associated", "Interaction site"}:
            y_offset = {
                "Membrane-associated": 0.12,
                "Interaction site": 0.00,
                "Disorder": -0.12,
            }[cat]
            ax.text(1.05, y1 + y_offset, wrap_label(cat, 18), ha="left", va="center", fontsize=5.8, color=color)
    ax.axhline(0, color=P["grid"], lw=0.8, ls="--")
    ax.set_xlim(-0.18, 1.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["C3-only\nbias", "Bernett\ntransfer"])
    ax.set_ylabel("log2 enrichment vs SAE vocabulary")
    ax.set_title("Feature themes diverge", loc="left", fontweight="bold")
    add_grid_x(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    ax.axis("off")
    ax.set_title("Top concepts barely overlap", loc="left", fontweight="bold")
    circle1 = patches.Circle((0.43, 0.52), 0.31, fc=P["rose_soft"], ec=P["rose"], lw=1.0, alpha=0.82)
    circle2 = patches.Circle((0.67, 0.52), 0.31, fc=P["teal_soft"], ec=P["teal"], lw=1.0, alpha=0.82)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(0.34, 0.53, "C3\nparticipation", ha="center", va="center", color=P["rose"], fontsize=6.5)
    ax.text(0.76, 0.53, "Bernett\ntransfer", ha="center", va="center", color=P["teal"], fontsize=6.5)
    ov = transfer["top_set_overlap"]
    ax.text(
        0.55,
        0.52,
        f"{ov['n_shared']}\nshared",
        ha="center",
        va="center",
        fontsize=7.0,
        color=P["ink"],
        fontweight="bold",
    )
    ax.text(
        0.55,
        0.16,
        f"top-{ov['topN']} SAE concepts: Jaccard={ov['jaccard']:.3f}\nall-feature Spearman={transfer['rank_correlation']['spearman_all_flat']:.3f}",
        ha="center",
        va="center",
        fontsize=6.2,
        color=P["muted"],
    )
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(0.05, 0.95)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    plot = interface.sort_values("enrichment", ascending=False).head(8).iloc[::-1]
    y = np.arange(len(plot))
    pvals = plot["fisher_greater_p"].clip(lower=1e-300)
    neglog = -np.log10(pvals)
    norm = mpl.colors.Normalize(vmin=float(neglog.min()), vmax=float(min(neglog.max(), 60)))
    cmap = LinearSegmentedColormap.from_list("pval", [P["grey_soft"], P["gold_soft"], P["rose"]])
    for yi, (_, row) in enumerate(plot.iterrows()):
        color = cmap(norm(min(-np.log10(max(row["fisher_greater_p"], 1e-300)), 60)))
        ax.plot([1, row["enrichment"]], [yi, yi], color=P["grid"], lw=1.6, zorder=1)
        ax.scatter(
            [row["enrichment"]],
            [yi],
            s=30 + row["n_interface_grounded"] * 0.16,
            color=color,
            edgecolor="white",
            lw=0.7,
            zorder=3,
        )
    ax.axvline(1, color=P["grid"], lw=0.85, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(c, 21) for c in plot["category"]])
    ax.set_xlim(0.55, max(2.8, plot["enrichment"].max() + 0.25))
    ax.set_xlabel("interface-grounding enrichment")
    ax.set_title("Residue-level grounding is category-biased", loc="left", fontweight="bold")
    add_grid_x(ax)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    theme = rosetta["theme_top30"]
    themes = [
        "signal/localization",
        "interface/scaffold/domain",
        "membrane/TM",
        "disorder/IDR",
        "RNA/ribosome/RNP",
        "other",
    ]
    theme_colors = {
        "signal/localization": P["gold"],
        "interface/scaffold/domain": P["blue"],
        "membrane/TM": P["violet"],
        "disorder/IDR": P["rose"],
        "RNA/ribosome/RNP": P["teal"],
        "other": P["muted"],
    }
    cols = ["c3", "rosetta"]
    for x, col in enumerate(cols):
        for yv, th in enumerate(themes):
            count = theme.get(col, {}).get(th, 0)
            if count == 0:
                ax.scatter(x, yv, s=14, color=P["grey_soft"], edgecolor="none")
                continue
            ax.scatter(
                x,
                yv,
                s=28 + count * 13,
                color=theme_colors[th],
                edgecolor="white",
                lw=0.75,
                zorder=3,
            )
            ax.text(x, yv, str(count), ha="center", va="center", fontsize=5.6, color="white" if count >= 7 else P["ink"])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["C3", "Rosetta"])
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels([wrap_label(t, 20) for t in themes])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, len(themes) - 0.45)
    ax.invert_yaxis()
    ax.grid(color=P["grid"], lw=0.5, alpha=0.65)
    ax.set_title("Structure benchmark shifts the theme mix", loc="left", fontweight="bold")

    save_fig(fig, "supplementary_s3_transferable_concept_grounding")


def main() -> None:
    setup_style()
    make_s1_participation_controls()
    make_s2_fingerprint_interpretability()
    make_s3_transferable_grounding()


if __name__ == "__main__":
    main()
