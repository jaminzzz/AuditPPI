#!/usr/bin/env python3
"""Audit C3 TabPFN retrieval risk using attention, feature overlap, and labels.

This deliberately avoids structure comparisons. It asks whether C3 test rows
receive high TabPFN decoder attention from training rows that are both feature-
similar and label-concordant.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.model import DEFAULT_SEED
from conf.paths import (
    RESULTS_PAIR,
    PAIR_CACHES_BINARY,
    RAPPPID_C3_DIR,
    TABPFN_RANKING,
)
from src.runtime import setup_device
from src.eval.classification import probe_classification_metrics as metrics
from src.interpretability.pair_probe import (
    build_dense_sym_topk,
    predict_proba_chunked,
    read_feature_ranking as read_ranking,
    select_top_features,
)
from src.interpretability.tabpfn_retrieval import (
    decoder_attention_weights,
    embeddings_with_configs,
    input_overlap_stats,
    load_split_with_indices,
    read_split_csv,
    top_shared_features,
)
from src.models.estimators.tabpfn import fit_tabpfn
from src.experiments.results import dump_experiment


DEFAULT_EMBEDDING_DIR = PAIR_CACHES_BINARY
DEFAULT_RANKING = TABPFN_RANKING
DEFAULT_C3_DIR = RAPPPID_C3_DIR
DEFAULT_OUT = RESULTS_PAIR / "leakage_audit/tabpfn_c3_attention_feature_label"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    p.add_argument("--ranking-csv", type=Path, default=DEFAULT_RANKING)
    p.add_argument("--c3-dir", type=Path, default=DEFAULT_C3_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--query-split", choices=["val", "test"], default="test")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--tabpfn-device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--top-k-mode", choices=["sae-id", "flat"], default="sae-id")
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--max-queries", type=int, default=0, help="0 means all queries")
    p.add_argument("--query-batch-size", type=int, default=64)
    p.add_argument("--train-subsample", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--predict-batch-size", type=int, default=5000)
    p.add_argument("--tabpfn-n-estimators", type=int, default=4)
    p.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    p.add_argument("--ignore-pretraining-limits", action="store_true", default=True)
    p.add_argument("--jaccard-high", type=float, default=0.80)
    p.add_argument("--jaccard-moderate", type=float, default=0.60)
    p.add_argument("--write-shared-features", action="store_true", help="include shared feature ids in neighbor table")
    return p.parse_args()


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k == scores.shape[0]:
        idx = np.arange(scores.shape[0])
    else:
        idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def svg_text(x: float, y: float, s: str, size: int = 13, fill: str = "#1f2933", weight: str = "400") -> str:
    s = (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}">{s}</text>'


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 5) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def draw_hist(svg: list[str], values: np.ndarray, x: float, y: float, w: float, h: float, title: str, color: str) -> None:
    svg.append(svg_rect(x, y, w, h, "#ffffff", "#d9e2ec", 8))
    svg.append(svg_text(x + 18, y + 28, title, 15, "#111827", "700"))
    if len(values) == 0:
        svg.append(svg_text(x + 18, y + 60, "No values", 12, "#6b7280"))
        return
    hist, edges = np.histogram(values, bins=np.linspace(0.0, 1.0, 21))
    max_count = max(int(hist.max()), 1)
    plot_x = x + 42
    plot_y = y + 58
    plot_w = w - 70
    plot_h = h - 95
    bw = plot_w / len(hist)
    for i, count in enumerate(hist):
        bh = plot_h * float(count) / max_count
        svg.append(svg_rect(plot_x + i * bw + 1, plot_y + plot_h - bh, bw - 2, bh, color, "none", 1))
    svg.append(svg_text(plot_x, y + h - 18, "0", 11, "#6b7280"))
    svg.append(svg_text(plot_x + plot_w - 10, y + h - 18, "1", 11, "#6b7280"))
    svg.append(svg_text(x + w - 135, y + 28, f"median {np.median(values):.3f}", 12, "#4b5563"))


def draw_svg(out_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    if not rows:
        return
    arr = {key: np.array([float(r[key]) for r in rows], dtype=float) for key in [
        "max_input_jaccard",
        "attention_weighted_jaccard",
        "topk_same_label_rate",
        "topk_same_label_attention_share",
        "risk_score",
    ]}
    svg: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="980" viewBox="0 0 1500 980">',
        '<rect width="1500" height="980" fill="#fbfcfd"/>',
        '<style>text{font-family:Inter,Arial,Helvetica,sans-serif}.mono{font-family:Menlo,Consolas,monospace}</style>',
        svg_text(54, 58, "C3 TabPFN Attention/Feature/Label Audit", 28, "#111827", "700"),
        svg_text(54, 88, "Non-structural train-test near-duplicate risk scan over TabPFN decoder attention neighbors.", 15, "#4b5563"),
    ]

    cards = [
        ("queries", summary["n_queries"]),
        ("high risk", summary["high_risk_count"]),
        ("moderate risk", summary["moderate_risk_count"]),
        ("median max Jaccard", f"{summary['max_jaccard_median']:.3f}"),
        ("median same-label attn share", f"{summary['same_label_attention_share_median']:.3f}"),
    ]
    for i, (label, value) in enumerate(cards):
        x = 54 + i * 280
        svg.append(svg_rect(x, 122, 250, 92, "#ffffff", "#d9e2ec", 8))
        svg.append(svg_text(x + 18, 154, label, 13, "#6b7280", "700"))
        svg.append(svg_text(x + 18, 190, value, 25, "#111827", "700"))

    draw_hist(svg, arr["max_input_jaccard"], 54, 248, 430, 230, "Top-20 max feature Jaccard", "#2563eb")
    draw_hist(svg, arr["attention_weighted_jaccard"], 535, 248, 430, 230, "Attention-weighted Jaccard", "#16a34a")
    draw_hist(svg, arr["topk_same_label_attention_share"], 1016, 248, 430, 230, "Same-label attention share", "#d97706")
    draw_hist(svg, arr["risk_score"], 54, 520, 430, 230, "Risk score", "#b91c1c")
    draw_hist(svg, arr["topk_same_label_rate"], 535, 520, 430, 230, "Same-label neighbor rate", "#7c3aed")

    svg.append(svg_rect(1016, 520, 430, 360, "#ffffff", "#d9e2ec", 8))
    svg.append(svg_text(1034, 552, "Highest-risk examples", 15, "#111827", "700"))
    top_rows = sorted(rows, key=lambda r: float(r["risk_score"]), reverse=True)[:8]
    y = 586
    for r in top_rows:
        label = (
            f"idx {r['query_idx']} y={r['query_label']} p={float(r['tabpfn_proba']):.3f} "
            f"risk={float(r['risk_score']):.3f}"
        )
        detail = (
            f"Jmax {float(r['max_input_jaccard']):.3f}; "
            f"same-label attn {float(r['topk_same_label_attention_mass']):.3f}; "
            f"same-label rate {float(r['topk_same_label_rate']):.2f}"
        )
        svg.append(svg_text(1034, y, label, 12, "#111827", "700"))
        svg.append(svg_text(1034, y + 18, detail, 11, "#64748b"))
        y += 38

    svg.append(svg_text(54, 936, "Risk score = topK same-label attention mass x max input-feature Jaccard among topK attention neighbors.", 12, "#6b7280"))
    svg.append(svg_text(54, 956, "Attention is associative evidence over TabPFN row embeddings, not a causal proof of leakage.", 12, "#6b7280"))
    svg.append("</svg>")
    (out_dir / "attention_feature_label_audit.svg").write_text("\n".join(svg))


def make_report(out_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    top_rows = sorted(rows, key=lambda r: float(r["risk_score"]), reverse=True)[:20]
    lines = [
        "# C3 TabPFN Attention/Feature/Label Audit",
        "",
        "Scope: non-structural audit over TabPFN decoder-attention neighbors.",
        "",
        "## Summary",
        "",
        f"- Queries audited: {summary['n_queries']}",
        f"- Train rows: {summary['n_train']}",
        f"- Top attention neighbors per query: {summary['neighbors']}",
        f"- Test AUROC/AUPRC from this run: {summary['test_auroc']:.4f}/{summary['test_auprc']:.4f}",
        f"- High-risk queries: {summary['high_risk_count']} ({summary['high_risk_rate']:.2%})",
        f"- Moderate-risk queries: {summary['moderate_risk_count']} ({summary['moderate_risk_rate']:.2%})",
        f"- Median max Jaccard among topK: {summary['max_jaccard_median']:.4f}",
        f"- Median attention-weighted Jaccard: {summary['attention_weighted_jaccard_median']:.4f}",
        f"- Median same-label attention share inside topK: {summary['same_label_attention_share_median']:.4f}",
        "",
        "High risk is defined as `max_input_jaccard >= jaccard_high`, `topk_same_label_rate >= 0.8`, and `risk_score >= 0.10`.",
        "Moderate risk is defined as `max_input_jaccard >= jaccard_moderate`, `topk_same_label_rate >= 0.7`, and `risk_score >= 0.03`.",
        "",
        "## Top Risk Cases",
        "",
        "| rank | query_idx | label | proba | risk | max_jaccard | weighted_jaccard | same-label rate | same-label attn mass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(top_rows, 1):
        lines.append(
            f"| {i} | {r['query_idx']} | {r['query_label']} | {float(r['tabpfn_proba']):.4f} | "
            f"{float(r['risk_score']):.4f} | {float(r['max_input_jaccard']):.4f} | "
            f"{float(r['attention_weighted_jaccard']):.4f} | {float(r['topk_same_label_rate']):.3f} | "
            f"{float(r['topk_same_label_attention_mass']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `query_audit.tsv`: one row per query.",
            "- `neighbor_audit_topk.tsv`: top attention neighbors per query.",
            "- `summary.json`: machine-readable summary.",
            "- `attention_feature_label_audit.svg`: compact visual summary.",
            "",
            "Note: decoder attention and feature overlap support a retrieval-style leakage-risk argument. They do not by themselves prove exact duplicate leakage.",
        ]
    )
    (out_dir / "AUDIT_SUMMARY.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if args.tabpfn_device == "cpu":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        print("[device] using CPU; CUDA_VISIBLE_DEVICES is not changed", flush=True)
    else:
        setup_device(args.device_id)
    tabpfn_device = args.tabpfn_device
    if tabpfn_device == "auto":
        import torch

        tabpfn_device = "cuda" if torch.cuda.is_available() else "cpu"

    ranking = read_ranking(args.ranking_csv)
    flat_features, feature_meta = select_top_features(ranking, args.top_k, args.top_k_mode)
    (out_dir / f"selected_features_{args.top_k_mode}_k{args.top_k}.json").write_text(
        json.dumps(feature_meta, indent=2)
    )

    print(f"[load] train/{args.query_split} embeddings from {args.embedding_dir}", flush=True)
    atr, btr, ytr, train_original_idx = load_split_with_indices(
        args.embedding_dir / "train_embeddings.pt", args.train_subsample, args.seed
    )
    aq, bq, yq, query_original_idx = load_split_with_indices(
        args.embedding_dir / f"{args.query_split}_embeddings.pt", None, args.seed + 1
    )

    if args.max_queries and args.max_queries < len(yq):
        query_positions = np.arange(args.max_queries, dtype=np.int64)
    else:
        query_positions = np.arange(len(yq), dtype=np.int64)

    Xtr = build_dense_sym_topk(atr, btr, flat_features)
    Xq_all = build_dense_sym_topk(aq, bq, flat_features)
    del atr, btr, aq, bq
    gc.collect()
    print(f"[data] train={Xtr.shape} {args.query_split}={Xq_all.shape} audited_queries={len(query_positions)}", flush=True)

    print(f"[fit] TabPFN device={tabpfn_device}", flush=True)
    model = fit_tabpfn(
        Xtr,
        ytr,
        n_estimators=args.tabpfn_n_estimators,
        subsample_samples=args.tabpfn_subsample_samples,
        ignore_limits=args.ignore_pretraining_limits,
        seed=args.seed,
        device=tabpfn_device,
    )

    print("[predict] query probabilities", flush=True)
    proba_all = predict_proba_chunked(model, Xq_all, args.predict_batch_size)
    run_metrics = metrics(yq[query_positions], proba_all[query_positions])

    print("[embed] train rows", flush=True)
    Etr, train_configs = embeddings_with_configs(model, Xtr, "train")
    Etr_mean_norm = l2_normalize(Etr.mean(axis=0))

    train_df = read_split_csv(args.c3_dir, "train")
    query_df = read_split_csv(args.c3_dir, args.query_split)
    ytr = ytr.astype(np.int64, copy=False)
    yq = yq.astype(np.int64, copy=False)

    query_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    attention_available = True

    for batch_no, start in enumerate(range(0, len(query_positions), args.query_batch_size), 1):
        end = min(start + args.query_batch_size, len(query_positions))
        batch_pos = query_positions[start:end]
        Xq = Xq_all[batch_pos]
        print(f"[audit] query batch {batch_no}: rows {start}-{end} / {len(query_positions)}", flush=True)

        Eq, query_configs = embeddings_with_configs(model, Xq, "test")
        configs = query_configs if len(query_configs) == Eq.shape[0] else train_configs
        cosine = l2_normalize(Eq.mean(axis=0)) @ Etr_mean_norm.T
        attention = decoder_attention_weights(model, Etr, Eq, configs)
        if attention is None:
            attention_available = False
            scores_matrix = cosine
            score_type = "embedding_cosine"
        else:
            scores_matrix = attention
            score_type = "decoder_attention"

        for local_i, split_pos in enumerate(batch_pos):
            scores = scores_matrix[local_i]
            nn = top_indices(scores, args.neighbors)
            att = attention[local_i, nn] if attention is not None else np.full(len(nn), np.nan)
            cos = cosine[local_i, nn]
            inter, union, jac = input_overlap_stats(Xq[local_i], Xtr, nn)

            query_label = int(yq[split_pos])
            labels = ytr[nn]
            label_match = labels == query_label
            positive = labels == 1
            topk_attention_mass = float(np.nansum(att)) if attention is not None else float("nan")
            same_label_attn_mass = float(np.nansum(att[label_match])) if attention is not None else float("nan")
            positive_attn_mass = float(np.nansum(att[positive])) if attention is not None else float("nan")
            same_label_share = safe_div(same_label_attn_mass, topk_attention_mass) if attention is not None else float("nan")
            positive_share = safe_div(positive_attn_mass, topk_attention_mass) if attention is not None else float("nan")
            weighted_jaccard = safe_div(float(np.nansum(att * jac)), topk_attention_mass) if attention is not None else float(np.mean(jac))
            high_mask = jac >= args.jaccard_high
            moderate_mask = jac >= args.jaccard_moderate
            high_same_label_rate = safe_div(float(np.logical_and(high_mask, label_match).sum()), float(high_mask.sum()))
            moderate_same_label_rate = safe_div(float(np.logical_and(moderate_mask, label_match).sum()), float(moderate_mask.sum()))
            same_label_rate = float(label_match.mean()) if len(label_match) else float("nan")
            pos_rate = float(positive.mean()) if len(positive) else float("nan")
            max_jaccard = float(jac.max()) if len(jac) else float("nan")
            risk_score = same_label_attn_mass * max_jaccard if attention is not None else same_label_rate * max_jaccard
            all_same_label_attn_mass = (
                float(np.nansum(attention[local_i, ytr == query_label])) if attention is not None else float("nan")
            )
            all_positive_attn_mass = (
                float(np.nansum(attention[local_i, ytr == 1])) if attention is not None else float("nan")
            )

            query_csv_idx = int(query_original_idx[split_pos])
            qrow = query_df.iloc[query_csv_idx]
            query_rows.append(
                {
                    "query_split": args.query_split,
                    "query_pos": int(split_pos),
                    "query_idx": query_csv_idx,
                    "query_label": query_label,
                    "tabpfn_proba": float(proba_all[split_pos]),
                    "tabpfn_pred": int(proba_all[split_pos] >= 0.5),
                    "correct": int((proba_all[split_pos] >= 0.5) == bool(query_label)),
                    "neighbors": int(len(nn)),
                    "score_type": score_type,
                    "topk_attention_mass": topk_attention_mass,
                    "topk_same_label_attention_mass": same_label_attn_mass,
                    "topk_same_label_attention_share": same_label_share,
                    "all_train_same_label_attention_mass": all_same_label_attn_mass,
                    "topk_positive_attention_mass": positive_attn_mass,
                    "topk_positive_attention_share": positive_share,
                    "all_train_positive_attention_mass": all_positive_attn_mass,
                    "topk_same_label_count": int(label_match.sum()),
                    "topk_same_label_rate": same_label_rate,
                    "topk_positive_count": int(positive.sum()),
                    "topk_positive_rate": pos_rate,
                    "max_input_jaccard": max_jaccard,
                    "mean_input_jaccard": float(jac.mean()) if len(jac) else float("nan"),
                    "attention_weighted_jaccard": weighted_jaccard,
                    "high_jaccard_count": int(high_mask.sum()),
                    "high_jaccard_same_label_rate": high_same_label_rate,
                    "moderate_jaccard_count": int(moderate_mask.sum()),
                    "moderate_jaccard_same_label_rate": moderate_same_label_rate,
                    "risk_score": risk_score,
                    "query_seq_a_len": len(str(qrow["query"])),
                    "query_seq_b_len": len(str(qrow["text"])),
                }
            )

            for rank, ni in enumerate(nn, 1):
                train_csv_idx = int(train_original_idx[ni])
                trow = train_df.iloc[train_csv_idx]
                row = {
                    "query_split": args.query_split,
                    "query_idx": query_csv_idx,
                    "query_label": query_label,
                    "tabpfn_proba": float(proba_all[split_pos]),
                    "neighbor_rank": rank,
                    "train_idx": train_csv_idx,
                    "train_label": int(ytr[ni]),
                    "label_match": int(ytr[ni] == query_label),
                    "retrieval_score_type": score_type,
                    "retrieval_score": float(scores[ni]),
                    "decoder_attention": float(att[rank - 1]) if attention is not None else "",
                    "embedding_cosine": float(cos[rank - 1]),
                    "input_shared_active": int(inter[rank - 1]),
                    "input_union_active": int(union[rank - 1]),
                    "input_jaccard": float(jac[rank - 1]),
                    "high_jaccard": int(jac[rank - 1] >= args.jaccard_high),
                    "moderate_jaccard": int(jac[rank - 1] >= args.jaccard_moderate),
                    "train_seq_a_len": len(str(trow["query"])),
                    "train_seq_b_len": len(str(trow["text"])),
                }
                if args.write_shared_features:
                    row["shared_sae_features"] = top_shared_features(Xq[local_i], Xtr[ni], feature_meta)
                neighbor_rows.append(row)

        del Eq, cosine, attention, scores_matrix
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    query_fields = [
        "query_split",
        "query_pos",
        "query_idx",
        "query_label",
        "tabpfn_proba",
        "tabpfn_pred",
        "correct",
        "neighbors",
        "score_type",
        "topk_attention_mass",
        "topk_same_label_attention_mass",
        "topk_same_label_attention_share",
        "all_train_same_label_attention_mass",
        "topk_positive_attention_mass",
        "topk_positive_attention_share",
        "all_train_positive_attention_mass",
        "topk_same_label_count",
        "topk_same_label_rate",
        "topk_positive_count",
        "topk_positive_rate",
        "max_input_jaccard",
        "mean_input_jaccard",
        "attention_weighted_jaccard",
        "high_jaccard_count",
        "high_jaccard_same_label_rate",
        "moderate_jaccard_count",
        "moderate_jaccard_same_label_rate",
        "risk_score",
        "query_seq_a_len",
        "query_seq_b_len",
    ]
    neighbor_fields = [
        "query_split",
        "query_idx",
        "query_label",
        "tabpfn_proba",
        "neighbor_rank",
        "train_idx",
        "train_label",
        "label_match",
        "retrieval_score_type",
        "retrieval_score",
        "decoder_attention",
        "embedding_cosine",
        "input_shared_active",
        "input_union_active",
        "input_jaccard",
        "high_jaccard",
        "moderate_jaccard",
        "train_seq_a_len",
        "train_seq_b_len",
    ]
    if args.write_shared_features:
        neighbor_fields.append("shared_sae_features")

    write_tsv(out_dir / "query_audit.tsv", query_rows, query_fields)
    write_tsv(out_dir / "neighbor_audit_topk.tsv", neighbor_rows, neighbor_fields)

    max_j = np.array([float(r["max_input_jaccard"]) for r in query_rows], dtype=float)
    aw_j = np.array([float(r["attention_weighted_jaccard"]) for r in query_rows], dtype=float)
    same_share = np.array([float(r["topk_same_label_attention_share"]) for r in query_rows], dtype=float)
    same_rate = np.array([float(r["topk_same_label_rate"]) for r in query_rows], dtype=float)
    risk = np.array([float(r["risk_score"]) for r in query_rows], dtype=float)

    high_risk = (
        (max_j >= args.jaccard_high)
        & (same_rate >= 0.8)
        & (risk >= 0.10)
    )
    moderate_risk = (
        (max_j >= args.jaccard_moderate)
        & (same_rate >= 0.7)
        & (risk >= 0.03)
    )
    summary = {
        "dataset": "c3",
        "query_split": args.query_split,
        "n_train": int(len(ytr)),
        "n_queries_total": int(len(yq)),
        "n_queries": int(len(query_rows)),
        "neighbors": int(args.neighbors),
        "top_k_mode": args.top_k_mode,
        "top_k": int(args.top_k),
        "input_dim": int(len(flat_features)),
        "attention_available": bool(attention_available),
        "test_auroc": float(run_metrics["auroc"]),
        "test_auprc": float(run_metrics["auprc"]),
        "test_acc": float(run_metrics["acc"]),
        "test_f1": float(run_metrics["f1"]),
        "jaccard_high": float(args.jaccard_high),
        "jaccard_moderate": float(args.jaccard_moderate),
        "high_risk_count": int(high_risk.sum()),
        "high_risk_rate": float(high_risk.mean()),
        "moderate_risk_count": int(moderate_risk.sum()),
        "moderate_risk_rate": float(moderate_risk.mean()),
        "max_jaccard_mean": float(max_j.mean()),
        "max_jaccard_median": float(np.median(max_j)),
        "max_jaccard_p90": percentile(max_j, 90),
        "max_jaccard_p95": percentile(max_j, 95),
        "attention_weighted_jaccard_mean": float(aw_j.mean()),
        "attention_weighted_jaccard_median": float(np.median(aw_j)),
        "attention_weighted_jaccard_p90": percentile(aw_j, 90),
        "same_label_rate_mean": float(same_rate.mean()),
        "same_label_rate_median": float(np.median(same_rate)),
        "same_label_attention_share_mean": float(np.nanmean(same_share)),
        "same_label_attention_share_median": float(np.nanmedian(same_share)),
        "risk_score_mean": float(risk.mean()),
        "risk_score_median": float(np.median(risk)),
        "risk_score_p90": percentile(risk, 90),
        "risk_score_p95": percentile(risk, 95),
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    dump_experiment(
        out_dir / "summary.json",
        task="tabpfn_c3_attention_feature_label",
        dataset="c3",
        features=f"sae_top{args.top_k}_{args.top_k_mode}",
        split=args.query_split,
        model="tabpfn",
        seed=args.seed,
        payload=summary,
        metrics={"auroc": summary["test_auroc"], "auprc": summary["test_auprc"]},
    )
    make_report(out_dir, summary, query_rows)
    draw_svg(out_dir, query_rows, summary)

    print(f"[done] {out_dir / 'query_audit.tsv'}", flush=True)
    print(f"[done] {out_dir / 'neighbor_audit_topk.tsv'}", flush=True)
    print(f"[done] {out_dir / 'AUDIT_SUMMARY.md'}", flush=True)
    print("TABPFN_C3_ATTENTION_FEATURE_LABEL_AUDIT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
