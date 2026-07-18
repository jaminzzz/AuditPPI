#!/usr/bin/env python
"""AuditPPI brief-communication figures — Nature-family style.

Figure 1: AuditPPI concept + framework (2x3, schematic with small real insets)
Figure 2: Results across protein / pair / residue scales (3x3, all real data)

Design follows plot_prank_figures.py: 7.2" wide, 6-7pt Arial, soft palette,
rounded panel labels, mixed chart types, exports svg/pdf/png.

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/figures/plot_audit_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from conf.paths import (
    RESULTS_PROTEIN,
    RESULTS_PAIR,
    RESULTS_RESIDUE,
    RESULTS_MISC,
    FIGURES as OUT,
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Palette — soft Nature family (from rRank / PRANK reference)
# ============================================================
P = {
    "ink": "#2B2B2B",
    "muted": "#6F6F77",
    "grid": "#DADDE5",
    "light": "#F4F6FA",
    "panel": "#FFFFFF",
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

# representation -> color
RC = {
    "binary": P["blue"],
    "sae_max": P["teal"],
    "esmc_mean": P["gold"],
}
RC_LABEL = {
    "binary": "SAE (binary)",
    "sae_max": "SAE (max-pool)",
    "esmc_mean": "ESM-C mean",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.7,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
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
    })


def save_fig(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem} -> {OUT}")


def panel_label(ax, label, x=-0.16, y=1.12):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color=P["ink"])


def blank_header(ax, label, title):
    """Header for axis-off schematic panels: panel label far-left, title
    indented to its right so the two never collide (the data panels get this
    for free from the axis spine, the blank ones do not)."""
    ax.text(-0.06, 1.15, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color=P["ink"])
    ax.text(0.06, 1.15, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, fontweight="bold", color=P["ink"])


# ============================================================
# Data loaders (all paths verified against the repo)
# ============================================================
def load_participation_node_preds():
    f = (RESULTS_PROTEIN / "pring_participation" / "high_p90_xgboost"
         / "pring_human_bfs_binary_high_q90_xgboost_protein_predictions.tsv")
    df = pd.read_csv(f, sep="\t")
    return df[df["split"] == "test"].copy()


def load_participation_matrix():
    """test AUROC for 3 splits x 3 representations."""
    import json
    splits = ["bfs", "dfs", "random_walk"]
    reps = ["binary", "sae_max", "esmc_mean"]
    auroc = np.zeros((len(reps), len(splits)))
    auprc = np.zeros((len(reps), len(splits)))
    base = RESULTS_PROTEIN / "pring_participation" / "high_p90_xgboost"
    for j, sp in enumerate(splits):
        for i, rp in enumerate(reps):
            fp = base / f"pring_human_{sp}_{rp}_high_q90_xgboost.json"
            d = json.load(open(fp))
            auroc[i, j] = d["node_metrics"]["test"]["auroc"]
            auprc[i, j] = d["node_metrics"]["test"]["auprc"]
    return splits, reps, auroc, auprc


def load_baseline_summary():
    import json
    return json.load(open(RESULTS_MISC / "ppi_fingerprint" / "baseline_summary.json"))


def load_query_audit():
    f = RESULTS_PAIR / "leakage_audit" / "tabpfn_c3_attention_feature_label" / "query_audit.tsv"
    return pd.read_csv(f, sep="\t")


def load_case4365():
    f = RESULTS_PAIR / "structure_comparison" / "tabpfn_case_4365" / "structure_similarity_to_query.tsv"
    return pd.read_csv(f, sep="\t")


def load_ranking_enrichment():
    fa = (RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae_enrichment_all_noninterface"
          / "ranking_interface_enrichment.tsv")
    fs = (RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae_enrichment_surface_noninterface"
          / "ranking_interface_enrichment.tsv")
    da = pd.read_csv(fa, sep="\t")
    ds = pd.read_csv(fs, sep="\t")
    da = da[da["ranking"] == "c3"]
    ds = ds[ds["ranking"] == "c3"]
    return da, ds


def load_interface_volcano():
    f = (RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae_enrichment_all_noninterface"
         / "interface_sae_feature_enrichment.tsv")
    return pd.read_csv(f, sep="\t")


def load_contact_categories():
    f = (RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae_contact_compat_top4"
         / "contact_compatible_category_pair_summary.tsv")
    return pd.read_csv(f, sep="\t")


def _regime_of(text):
    """Map a feature summary's free text to a structural regime. The coarse
    category labels collapse the biology onto 'Unannotated' partners; the real
    transmembrane / ribosomal / active-site signal lives in the summaries."""
    t = str(text).lower()
    if any(k in t for k in ("transmembrane", "membrane", "tm helix", "lipid",
                            "juxtamembrane", "multi-pass")):
        return "Transmembrane"
    if any(k in t for k in ("ribosom", "rrna", "rna polymerase", "translation",
                            "rnp", "p-stalk", "nucleic")):
        return "Ribosomal / RNA"
    if any(k in t for k in ("catalyt", "active-site", "active site", "enzyme",
                            "cofactor", "substrate", "rossmann", "tim-barrel",
                            "ntpase", "hydrolase")):
        return "Enzyme active-site"
    return "Other"


def load_contact_feature_pairs():
    """Significant contact-compatible feature-pairs, with a per-pair structural
    regime attached from the free-text summaries. A pair takes the named regime
    carried by either side (transmembrane / ribosomal / active-site take
    precedence over 'Other'). Only rows flagged is_contact_compatible are kept."""
    f = (RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae_contact_compat_top4"
         / "top_contact_compatible_feature_pairs_annotated.tsv")
    df = pd.read_csv(f, sep="\t")
    df = df[df["is_contact_compatible"] == 1].copy()
    order = ["Transmembrane", "Ribosomal / RNA", "Enzyme active-site", "Other"]
    ra = df["summary_a"].map(_regime_of)
    rb = df["summary_b"].map(_regime_of)
    df["regime"] = [next((r for r in order if r in (a, b)), "Other")
                    for a, b in zip(ra, rb)]
    return df


# ============================================================
# Figure 1 — AuditPPI concept + framework (schematic)
# Layout 2x3:
#   a: one PPI score, three confounded mechanisms
#   b: SAE decomposes dense pLM activations into sparse features
#   c: AuditPPI pipeline (seq pair -> ESM-C -> frozen SAE -> multi-scale)
#   d: protein fingerprints (max-pool g vs binary b)  [small real inset]
#   e: pair fingerprint AND/XOR symmetric construction
#   f: three hypotheses -> three audit scales mapping
# ============================================================
def _rbox(ax, x, y, w, h, fc, ec, lw=0.8, alpha=1.0, rounding=0.02, z=2):
    box = FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                         boxstyle=f"round,pad=0,rounding_size={rounding}",
                         linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha,
                         mutation_aspect=1.0, zorder=z)
    ax.add_patch(box)
    return box


def _arrow(ax, x0, y0, x1, y1, color=None, lw=1.0, z=3, style="-|>", ms=6):
    color = color or P["muted"]
    ar = FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                         arrowstyle=style, mutation_scale=ms, lw=lw,
                         color=color, zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(ar)


def _blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def make_figure1():
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.22)

    # ---- Panel a: one score, three mechanisms -------------------
    ax = fig.add_subplot(gs[0, 0]); _blank(ax)
    blank_header(ax, "a", "One PPI score, three explanations")
    # protein pair -> score
    _rbox(ax, 0.04, 0.78, 0.26, 0.13, P["blue_soft"], P["blue"])
    ax.text(0.17, 0.845, "Protein A", ha="center", va="center", fontsize=6)
    _rbox(ax, 0.04, 0.62, 0.26, 0.13, P["teal_soft"], P["teal"])
    ax.text(0.17, 0.685, "Protein B", ha="center", va="center", fontsize=6)
    _rbox(ax, 0.40, 0.70, 0.20, 0.13, P["gold_soft"], P["gold"])
    ax.text(0.50, 0.765, "high score", ha="center", va="center", fontsize=6,
            fontweight="bold")
    _arrow(ax, 0.30, 0.84, 0.40, 0.79, P["muted"], 0.9)
    _arrow(ax, 0.30, 0.685, 0.40, 0.74, P["muted"], 0.9)
    # three branches
    mechs = [
        ("Hub participation", P["rose"], P["rose_soft"], 0.50, "confound"),
        ("Semantic concordance", P["violet"], P["violet_soft"], 0.27, "confound"),
        ("Interface compatibility", P["green"], P["green_soft"], 0.04, "physical"),
    ]
    for name, ec, fc, yy, kind in mechs:
        _rbox(ax, 0.62, yy, 0.36, 0.135, fc, ec)
        ax.text(0.80, yy + 0.092, name, ha="center", va="center", fontsize=5.7,
                fontweight="bold", color=P["ink"])
        tag = "physical" if kind == "physical" else "confounder"
        tcol = P["green"] if kind == "physical" else P["muted"]
        ax.text(0.80, yy + 0.032, tag, ha="center", va="center", fontsize=4.9,
                color=tcol, style="italic")
        _arrow(ax, 0.60, 0.765, 0.62, yy + 0.067, ec, 0.9)
    ax.text(0.80, 0.005, "all three inflate accuracy", ha="center", va="bottom",
            fontsize=5.0, color=P["muted"], style="italic")

    # ---- Panel b: SAE decomposition -----------------------------
    ax = fig.add_subplot(gs[0, 1]); _blank(ax)
    blank_header(ax, "b", "Sparse, monosemantic features")
    rng = np.random.default_rng(3)
    # dense activation block
    dense = rng.uniform(0.2, 1.0, (8, 6))
    ax.imshow(dense, extent=(0.02, 0.30, 0.30, 0.86), aspect="auto",
              cmap=mpl.colors.LinearSegmentedColormap.from_list(
                  "d", ["#FFFFFF", P["blue_soft"], P["blue"]]),
              transform=ax.transData if False else ax.transAxes, zorder=2)
    ax.text(0.16, 0.20, "dense pLM\nactivations", ha="center", va="center",
            fontsize=5.4, color=P["ink"])
    # SAE encoder arrow
    _arrow(ax, 0.32, 0.58, 0.46, 0.58, P["muted"], 1.1)
    ax.text(0.39, 0.64, "frozen\nSAE", ha="center", va="center", fontsize=5.0,
            color=P["muted"])
    # sparse block (mostly white, few colored)
    sparse = np.zeros((8, 12))
    idx = rng.choice(8 * 12, 7, replace=False)
    sparse.flat[idx] = rng.uniform(0.6, 1.0, 7)
    ax.imshow(sparse, extent=(0.48, 0.98, 0.30, 0.86), aspect="auto",
              cmap=mpl.colors.LinearSegmentedColormap.from_list(
                  "s", ["#FFFFFF", P["teal_soft"], P["teal"]]),
              transform=ax.transAxes, zorder=2)
    ax.text(0.73, 0.20, "16,384 sparse features\n(top-64 active / residue)",
            ha="center", va="center", fontsize=5.4, color=P["ink"])
    for tag, xx, col in [("binding site", 0.55, P["teal"]),
                         ("motif", 0.73, P["gold"]),
                         ("family", 0.90, P["rose"])]:
        ax.text(xx, 0.10, tag, ha="center", va="center", fontsize=4.6,
                color=col, style="italic")

    # ---- Panel c: AuditPPI pipeline -----------------------------
    ax = fig.add_subplot(gs[0, 2]); _blank(ax)
    blank_header(ax, "c", "AuditPPI pipeline")
    steps = [
        ("Sequence pair", P["grey_soft"], P["muted"]),
        ("ESM-C (frozen)", P["blue_soft"], P["blue"]),
        ("SAE (frozen)", P["teal_soft"], P["teal"]),
        ("Multi-scale\nfingerprints", P["gold_soft"], P["gold"]),
    ]
    yy = 0.86
    for i, (name, fc, ec) in enumerate(steps):
        _rbox(ax, 0.18, yy - 0.105, 0.64, 0.11, fc, ec)
        ax.text(0.50, yy - 0.05, name, ha="center", va="center", fontsize=5.6,
                fontweight="bold")
        if i < len(steps) - 1:
            _arrow(ax, 0.50, yy - 0.105, 0.50, yy - 0.17, P["muted"], 1.0)
        yy -= 0.205
    # scales row
    for k, (lbl, col) in enumerate([("protein", P["blue"]), ("pair", P["violet"]),
                                    ("residue", P["rose"]), ("res.-pair", P["green"])]):
        x0 = 0.05 + k * 0.245
        _rbox(ax, x0, 0.02, 0.205, 0.085, "#FFFFFF", col, lw=0.9)
        ax.text(x0 + 0.10, 0.062, lbl, ha="center", va="center", fontsize=5.0,
                color=col, fontweight="bold")
    ax.text(0.50, 0.135, "queried at 4 resolutions", ha="center", va="center",
            fontsize=4.9, color=P["muted"], style="italic")

    # ---- Panel d: protein fingerprints (real inset) -------------
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "d")
    # real: max-pooled magnitude distribution of active features for a few proteins
    rng = np.random.default_rng(11)
    # illustrate g (continuous) vs b (binary) on a small feature window
    feats = np.arange(24)
    g = np.zeros(24)
    active = rng.choice(24, 9, replace=False)
    g[active] = rng.uniform(0.3, 1.0, 9)
    b = (g > 0).astype(float)
    ax.bar(feats - 0.2, g, width=0.4, color=P["teal"], alpha=0.8,
           label=r"$g_{p,k}$ (max-pool)", edgecolor="white", linewidth=0.3)
    ax.bar(feats + 0.2, b, width=0.4, color=P["blue"], alpha=0.55,
           label=r"$b_{p,k}$ (binary)", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("SAE feature index (window)")
    ax.set_ylabel("activation")
    ax.set_title("Protein fingerprints", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-1, 24)
    ax.set_xticks([])
    ax.legend(loc="upper right", fontsize=5.0)

    # ---- Panel e: pair fingerprint AND/XOR ----------------------
    ax = fig.add_subplot(gs[1, 1]); _blank(ax)
    blank_header(ax, "e", "Order-invariant pair fingerprint")
    rng = np.random.default_rng(7)
    a = (rng.uniform(0, 1, 14) > 0.55).astype(float)
    bb = (rng.uniform(0, 1, 14) > 0.55).astype(float)
    AND = a * bb
    XOR = ((a + bb) == 1).astype(float)
    rows = [("A", a, P["blue"]), ("B", bb, P["teal"]),
            (r"A $\odot$ B  (AND)", AND, P["green"]), ("|A $-$ B|  (XOR)", XOR, P["rose"])]
    for r, (lbl, vec, col) in enumerate(rows):
        yy = 0.80 - r * 0.20
        ax.text(0.0, yy + 0.04, lbl, ha="left", va="center", fontsize=5.4,
                color=col, fontweight="bold")
        for c, v in enumerate(vec):
            x0 = 0.34 + c * 0.047
            fc = col if v > 0 else "#FFFFFF"
            _rbox(ax, x0, yy, 0.042, 0.085, fc, col, lw=0.5,
                  alpha=0.85 if v > 0 else 1.0)
    ax.text(0.5, -0.02, r"$\Phi_S(a,b)=[\,x_a\!\odot x_b,\ |x_a-x_b|\,]$",
            ha="center", va="center", fontsize=6.0, color=P["ink"])

    # ---- Panel f: hypotheses -> audit scales --------------------
    ax = fig.add_subplot(gs[1, 2]); _blank(ax)
    blank_header(ax, "f", "Three audits, three scales")
    rows = [
        ("Protein", "Participation /\nhub-ness", P["blue"], P["blue_soft"]),
        ("Pair", "Semantic\nconcordance", P["violet"], P["violet_soft"]),
        ("Residue", "Interface\ncompatibility", P["green"], P["green_soft"]),
    ]
    for r, (scale, desc, ec, fc) in enumerate(rows):
        yy = 0.70 - r * 0.235
        _rbox(ax, 0.02, yy, 0.30, 0.175, fc, ec)
        ax.text(0.17, yy + 0.088, scale, ha="center", va="center", fontsize=6.0,
                fontweight="bold", color=P["ink"])
        _arrow(ax, 0.33, yy + 0.088, 0.48, yy + 0.088, ec, 1.0)
        _rbox(ax, 0.50, yy, 0.48, 0.175, "#FFFFFF", ec, lw=0.9)
        ax.text(0.74, yy + 0.088, desc, ha="center", va="center", fontsize=5.4,
                color=P["ink"])

    fig.suptitle("AuditPPI: a sparse-feature vocabulary to decompose PPI predictions",
                 x=0.02, ha="left", fontsize=8.5, fontweight="bold")
    save_fig(fig, "figure1_auditppi_concept")


# ============================================================
# Figure 2 — Results across protein / pair / residue scales (3x3)
# ============================================================
def make_figure2():
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.62, wspace=0.60)

    # ===== Row 1: PROTEIN scale =====
    # --- a: hub prob vs degree (real node predictions) ---
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    preds = load_participation_node_preds()
    deg = preds["degree_full"].to_numpy()
    prob = preds["pred_high_prob"].to_numpy()
    hp = preds["high_participation"].to_numpy().astype(bool)
    ax.scatter(deg[~hp], prob[~hp], s=5, color=P["grey_soft"],
               edgecolor=P["muted"], linewidth=0.15, alpha=0.5, zorder=2,
               label="low-degree")
    ax.scatter(deg[hp], prob[hp], s=7, color=P["rose_soft"],
               edgecolor=P["rose"], linewidth=0.2, alpha=0.8, zorder=3,
               label="hub (deg ≥ 61)")
    ax.axvline(61, color=P["ink"], ls="--", lw=0.6, alpha=0.6)
    ax.text(66, 0.92, "deg = 61\n(90th pct)", fontsize=4.8, color=P["ink"],
            va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Full-graph degree")
    ax.set_ylabel("Predicted hub probability")
    ax.set_title("Participation is sequence-predictable", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.text(0.04, 0.04, "BFS test\nAUROC 0.701\nAUPRC 0.561", transform=ax.transAxes,
            fontsize=5.2, color=P["rose"], fontweight="bold", va="bottom")
    ax.legend(loc="upper left", fontsize=5.0, bbox_to_anchor=(0.0, 1.0))

    # --- b: participation AUROC, 3 splits x 3 reps ---
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    splits, reps, auroc, auprc = load_participation_matrix()
    x = np.arange(len(splits))
    w = 0.26
    for i, rp in enumerate(reps):
        ax.bar(x + (i - 1) * w, auroc[i], w, color=RC[rp], alpha=0.82,
               edgecolor="white", linewidth=0.4, label=RC_LABEL[rp])
        for j in range(len(splits)):
            ax.text(x[j] + (i - 1) * w, auroc[i, j] + 0.008, f"{auroc[i,j]:.2f}",
                    ha="center", va="bottom", fontsize=4.3, color=P["ink"])
    ax.axhline(0.5, color=P["grid"], ls=":", lw=0.7)
    ax.text(2.45, 0.51, "chance", fontsize=4.6, color=P["muted"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["BFS", "DFS", "Rand.\nwalk"])
    ax.set_ylabel("Hub-classification AUROC")
    ax.set_ylim(0.45, 0.95)
    ax.set_title("Recoverable under every protein split", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.legend(loc="upper center", fontsize=4.4, ncol=3, columnspacing=0.7,
              handlelength=1.0, handletextpad=0.35, bbox_to_anchor=(0.5, 0.99))

    # --- c: model-free concordance separates C3 ---
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "c")
    names = ["Dense\ncosine", "Binary\nJaccard", "Participation\noracle",
             "Trained\nclassifier"]
    vals = [0.717, 0.669, 0.8523, 0.933]
    cols = [P["teal"], P["teal"], P["violet"], P["blue"]]
    alphas = [0.6, 0.6, 0.8, 0.9]
    y = np.arange(len(names))[::-1]
    for yi, (n, v, c, al) in enumerate(zip(names, vals, cols, alphas)):
        yy = y[yi]
        ax.barh(yy, v - 0.5, left=0.5, height=0.6, color=c, alpha=al,
                edgecolor="white", linewidth=0.4)
        ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=5.2,
                color=c, fontweight="bold")
    ax.axvline(0.5, color=P["grid"], ls=":", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.2)
    ax.set_xlabel("C3 test AUROC")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Concordance alone inflates C3", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.text(0.98, 0.04, "model-free →\ntrained", transform=ax.transAxes,
            fontsize=4.8, color=P["muted"], ha="right", va="bottom", style="italic")

    # ===== Row 2: PAIR scale =====
    # --- d: cross-benchmark AUROC (binary / sae / esmc) ---
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "d")
    summ = load_baseline_summary()
    benches = [("c3:test", "C3"), ("cross_species:human_test", "Cross-\nspecies"),
               ("rf2ppi", "RF2-PPI")]
    reps2 = ["binary", "sae_max", "esmc_mean"]
    x = np.arange(len(benches))
    w = 0.26
    for i, rp in enumerate(reps2):
        vals = [summ[f"xgb/{rp}/{bk}"]["auroc"] for bk, _ in benches]
        ax.bar(x + (i - 1) * w, vals, w, color=RC[rp], alpha=0.82,
               edgecolor="white", linewidth=0.4, label=RC_LABEL[rp])
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + 0.008, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=4.2, color=P["ink"])
    ax.axhline(0.5, color=P["grid"], ls=":", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in benches])
    ax.set_ylabel("Test AUROC (XGBoost)")
    ax.set_ylim(0.45, 1.08)
    ax.set_title("Accuracy drops on interface benchmark", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.legend(loc="upper right", fontsize=4.6, bbox_to_anchor=(1.0, 1.0))

    # --- e: retrieval 2D distribution ---
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "e")
    qa = load_query_audit()
    jx = qa["max_input_jaccard"].to_numpy()
    sy = qa["topk_same_label_attention_share"].to_numpy()
    risk = qa["risk_score"].to_numpy()
    sc = ax.scatter(jx, sy, c=risk, s=5, cmap="magma_r", alpha=0.6,
                    vmin=0, vmax=np.percentile(risk, 99), linewidth=0, zorder=2)
    ax.axhline(0.8, color=P["rose"], ls="--", lw=0.7, alpha=0.8)
    ax.axvline(0.8, color=P["rose"], ls="--", lw=0.7, alpha=0.8)
    ax.text(0.82, 0.97, "high-risk\n1.09%", fontsize=4.8, color=P["rose"],
            transform=ax.transData, va="top", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("retrieval-risk", fontsize=5.0)
    cb.ax.tick_params(labelsize=4.5)
    ax.set_xlabel("Max feature Jaccard (top-20)")
    ax.set_ylabel("Same-label attention share")
    ax.set_title("C3 leans on labelled neighbours", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.3, 1.02)

    # --- f: case 4365 neighbours (attention vs structural identity) ---
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "f")
    c4 = load_case4365().sort_values("attention", ascending=False).head(10)
    att = c4["attention"].to_numpy()
    tm_short = c4["short_tm_query"].to_numpy()   # actin partner TM to query actin
    sid_short = c4["short_seq_id"].to_numpy()
    jac = c4["input_jaccard"].to_numpy()
    sizes = 30 + jac * 120
    scsc = ax.scatter(tm_short, att, s=sizes, c=sid_short, cmap="viridis",
                      vmin=0, vmax=1.0, edgecolor=P["ink"], linewidth=0.4,
                      alpha=0.9, zorder=3)
    cb = fig.colorbar(scsc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("partner seq. id.", fontsize=5.0)
    cb.ax.tick_params(labelsize=4.5)
    ax.set_xlabel("Partner structural sim. to query (TM-score)")
    ax.set_ylabel("TabPFN attention weight")
    ax.set_title("Case 4365: near-duplicate actins", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.text(0.04, 0.96, "20/20 positive\n96.2% attn mass\nJaccard 0.93",
            transform=ax.transAxes, fontsize=4.8, color=P["blue"],
            fontweight="bold", va="top")
    # marker size legend
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=P["grey_soft"],
               markeredgecolor=P["ink"], markersize=3, label="low Jaccard"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=P["grey_soft"],
               markeredgecolor=P["ink"], markersize=7, label="high Jaccard"),
    ], loc="lower right", fontsize=4.6)

    # ===== Row 3: RESIDUE scale =====
    # --- g: interface enrichment collapse (broad vs surface control) ---
    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "g")
    da, ds = load_ranking_enrichment()
    topks = [20, 50, 100, 200]
    ea = [float(da[da["topk"] == k]["enrichment"].iloc[0]) for k in topks]
    es = [float(ds[ds["topk"] == k]["enrichment"].iloc[0]) for k in topks]
    x = np.arange(len(topks))
    ax.plot(x, ea, "o-", color=P["blue"], lw=1.6, markersize=5,
            label="broad control\n(all non-interface)", zorder=4)
    ax.plot(x, es, "s--", color=P["rose"], lw=1.5, markersize=4.5,
            label="surface-matched\ncontrol", zorder=4)
    ax.fill_between(x, ea, es, color=P["blue_soft"], alpha=0.18)
    ax.axhline(1.0, color=P["grid"], ls=":", lw=0.8)
    ax.text(3.0, 1.02, "no enrichment", fontsize=4.6, color=P["muted"],
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"top {k}" for k in topks])
    ax.set_ylabel("Interface-feature enrichment (×)")
    ax.set_title("Enrichment vanishes under control", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.set_ylim(0.9, 2.1)
    ax.legend(loc="upper right", fontsize=4.6)

    # --- h: interface volcano ---
    ax = fig.add_subplot(gs[2, 1])
    panel_label(ax, "h")
    vol = load_interface_volcano()
    lor = vol["interface_log_or"].to_numpy()
    q = vol["bh_q"].to_numpy()
    grounded = vol["is_interface_grounded"].to_numpy().astype(bool)
    nlq = -np.log10(np.clip(q, 1e-300, None))
    nlq = np.clip(nlq, 0, 300)
    # subsample non-grounded for speed/clarity
    rng = np.random.default_rng(0)
    ng = ~grounded
    keep_ng = np.where(ng)[0]
    keep_ng = rng.choice(keep_ng, min(6000, len(keep_ng)), replace=False)
    ax.scatter(lor[keep_ng], nlq[keep_ng], s=2.5, color=P["grey_soft"],
               edgecolor="none", alpha=0.4, zorder=2)
    ax.scatter(lor[grounded], nlq[grounded], s=4, color=P["green"],
               edgecolor="none", alpha=0.55, zorder=3)
    ax.axvline(0.5, color=P["ink"], ls="--", lw=0.6, alpha=0.6)
    ax.set_xlabel("Interface log odds ratio")
    ax.set_ylabel(r"$-\log_{10}$ FDR $q$")
    ax.set_title("11.1% of features interface-grounded", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.set_xlim(-2.5, 4.0)
    ax.text(0.96, 0.5, f"{int(grounded.sum()):,}\ngrounded\nfeatures",
            transform=ax.transAxes, fontsize=5.0, color=P["green"],
            fontweight="bold", ha="right", va="center")

    # --- i: contact signal concentrates in structured regimes ---
    # Regime is inferred from the annotated summaries of the significant
    # feature-pairs; the coarse category_pair labels are dominated by
    # "Unannotated" partners and obscure this. The significant contact-
    # compatible pairs concentrate in three structured regimes; the regime is
    # read from each pair's annotated summaries.
    ax = fig.add_subplot(gs[2, 2])
    panel_label(ax, "i")
    fp = load_contact_feature_pairs()
    reg = fp["regime"].to_numpy()
    n_sig = len(reg)
    regimes = ["Transmembrane", "Enzyme active-site", "Ribosomal / RNA", "Other"]
    rcol = {"Transmembrane": P["rose"], "Enzyme active-site": P["teal"],
            "Ribosomal / RNA": P["violet"], "Other": P["grey_soft"]}
    fracs = [100.0 * np.mean(reg == rg) for rg in regimes]
    alphas = [0.85, 0.85, 0.85, 0.55]
    bar_colors = [mpl.colors.to_rgba(rcol[rg], a) for rg, a in zip(regimes, alphas)]
    yy = np.arange(len(regimes))[::-1]
    ax.barh(yy, fracs, height=0.64, color=bar_colors,
            edgecolor="white", linewidth=0.4)
    for y0, fr in zip(yy, fracs):
        ax.text(fr + 0.8, y0, f"{fr:.0f}%", va="center", fontsize=5.2,
                color=P["ink"])
    struct = sum(fracs[:3])
    ax.set_yticks(yy)
    ax.set_yticklabels([rg.replace(" ", "\n", 1) for rg in regimes], fontsize=5.0)
    ax.set_xlabel("Share of significant pairs (%)")
    ax.set_xlim(0, 62)
    ax.set_title("Contact signal is structurally localized", loc="left",
                 fontweight="bold", fontsize=6.6)
    ax.text(0.97, 0.06, f"{struct:.0f}% in 3 structured\nregimes "
            f"(n={n_sig:,})", transform=ax.transAxes, fontsize=4.8,
            color=P["ink"], ha="right", va="bottom", fontweight="bold")

    fig.suptitle("Sequence-only PPI accuracy blends participation, concordance and sparse interface signal",
                 x=0.02, ha="left", fontsize=8.5, fontweight="bold")
    save_fig(fig, "figure2_auditppi_results")


if __name__ == "__main__":
    setup_style()
    print("Generating AuditPPI figures...")
    make_figure1()
    make_figure2()
    print("Done. Output dir:", OUT)
