#!/usr/bin/env python3
"""Explain TabPFN PPI predictions using retrieval and SAE-feature overlap.

Works for any RAPPPID leakage level (``--family c1|c2|c3``). Each family uses
its own pair-index caches, protein cache, and per-family binary/sym ranking
produced by ``run_clevel_tabpfn_topk.py``. Products land under a per-family
subdir so the three levels never overwrite each other::

    results/audit_pair/tabpfn/{family}/tabpfn_retrieval_explanations/

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/analysis/explain_tabpfn_retrieval.py --family c3
    $PY scripts/analysis/explain_tabpfn_retrieval.py --family c1
    $PY scripts/analysis/explain_tabpfn_retrieval.py --family c2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np

from conf.model import DEFAULT_BACKBONE, DEFAULT_SEED, resolve_backbone_layer
from conf.paths import (
    CLEVEL_PAIR_INDEX_CACHES,
    CLEVEL_SAE_CACHES,
    RAPPPID_C1_DIR,
    RAPPPID_C2_DIR,
    RAPPPID_C3_DIR,
    clevel_tabpfn_ranking,
    clevel_tabpfn_retrieval_dir,
)
from src.experiments.results import dump_experiment
from src.features.pairs import load_protein_feature_cache
from src.interp.pair_probe import (
    build_dense_sym_topk,
    fit_tabpfn_probe,
    materialize_pair_split,
    predict_proba_chunked,
    read_feature_ranking,
    sae_dim_for_backbone,
    select_top_features,
)
from src.interp.tabpfn_retrieval import (
    choose_queries,
    decoder_attention_weights,
    embeddings_with_configs,
    input_overlap_stats,
    l2_normalize,
    read_split_csv,
    short_sequence,
    top_shared_features,
)
from src.runtime import setup_device

FAMILIES = ("c1", "c2", "c3")
FAMILY_DIRS = {
    "c1": RAPPPID_C1_DIR,
    "c2": RAPPPID_C2_DIR,
    "c3": RAPPPID_C3_DIR,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--family",
        choices=FAMILIES,
        default="c3",
        help="RAPPPID leakage level. Ranking, caches, CSVs, and out-dir are "
        "all resolved from this (default: c3).",
    )
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=None,
        help="override ranking CSV; default is the family's own tabpfn_topk ranking.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="override product dir; default is tabpfn/{family}/tabpfn_retrieval_explanations.",
    )
    parser.add_argument(
        "--clevel-dir",
        type=Path,
        default=None,
        help="override RAPPPID CSV dir; default is data/raw/rapppid_{family}.",
    )
    parser.add_argument("--rep", choices=["binary", "sae_max"], default="binary")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="SAE layer; defaults to the backbone's default layer.",
    )
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-k-mode", choices=["sae-id", "flat"], default="sae-id")
    parser.add_argument("--train-subsample", type=int, default=None)
    parser.add_argument("--query-split", choices=["val", "test"], default="test")
    parser.add_argument("--query-indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--min-proba", type=float, default=0.8)
    parser.add_argument("--prefer-true-positive", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--predict-batch-size", type=int, default=5000)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=4)
    parser.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    parser.add_argument("--ignore-pretraining-limits", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_device(args.device_id)
    family = args.family
    ranking_csv = args.ranking_csv or clevel_tabpfn_ranking(family)
    out_dir = args.out_dir or clevel_tabpfn_retrieval_dir(family)
    clevel_dir = args.clevel_dir or FAMILY_DIRS[family]
    pair_index_caches = CLEVEL_PAIR_INDEX_CACHES[family]
    protein_cache_path = CLEVEL_SAE_CACHES[family]
    out_dir.mkdir(parents=True, exist_ok=True)

    layer = resolve_backbone_layer(args.backbone, args.layer)
    sae_dim = sae_dim_for_backbone(args.backbone)
    ranking = read_feature_ranking(ranking_csv)
    flat_features, feature_metadata = select_top_features(
        ranking, args.top_k, args.top_k_mode, sae_dim=sae_dim
    )
    (out_dir / f"selected_features_{args.top_k_mode}_k{args.top_k}.json").write_text(
        json.dumps(feature_metadata, indent=2)
    )

    # Cap train rows to TabPFN's context budget so Xtr/ytr/embeddings/attention
    # stay row-aligned (see audit_tabpfn_clevel_attention_feature_label.py).
    train_max_rows = args.train_subsample
    if train_max_rows is None and args.tabpfn_subsample_samples > 0:
        train_max_rows = args.tabpfn_subsample_samples

    print(
        f"[load] {family} train/query via pair-index caches "
        f"({args.rep} {args.backbone}L{layer}; train_max_rows={train_max_rows})",
        flush=True,
    )
    protein_cache = load_protein_feature_cache(protein_cache_path)
    train_a, train_b, train_y, train_original_indices = materialize_pair_split(
        pair_index_caches["train"],
        protein_cache,
        rep=args.rep,
        backbone=args.backbone,
        layer=layer,
        max_rows=train_max_rows,
        seed=args.seed,
        return_indices=True,
    )
    query_a, query_b, query_y, query_original_indices = materialize_pair_split(
        pair_index_caches[args.query_split],
        protein_cache,
        rep=args.rep,
        backbone=args.backbone,
        layer=layer,
        max_rows=None,
        seed=args.seed + 1,
        return_indices=True,
    )
    train_x = build_dense_sym_topk(train_a, train_b, flat_features, sae_dim=sae_dim)
    query_x_all = build_dense_sym_topk(query_a, query_b, flat_features, sae_dim=sae_dim)
    del train_a, train_b, query_a, query_b
    gc.collect()

    print("[fit] TabPFN", flush=True)
    model = fit_tabpfn_probe(
        train_x,
        train_y,
        n_estimators=args.tabpfn_n_estimators,
        subsample_samples=args.tabpfn_subsample_samples,
        ignore_limits=args.ignore_pretraining_limits,
        seed=args.seed,
    )
    probabilities = predict_proba_chunked(model, query_x_all, args.predict_batch_size)
    query_positions = choose_queries(
        probabilities,
        query_y,
        query_indices=args.query_indices,
        min_probability=args.min_proba,
        prefer_true_positive=args.prefer_true_positive,
        num_queries=args.num_queries,
    )
    query_x = query_x_all[query_positions]
    print(f"[query] selected={len(query_positions)}", flush=True)

    train_embeddings, train_configs = embeddings_with_configs(model, train_x, "train")
    query_embeddings, query_configs = embeddings_with_configs(model, query_x, "test")
    configs = query_configs if len(query_configs) == query_embeddings.shape[0] else train_configs
    cosine = l2_normalize(query_embeddings.mean(axis=0)) @ l2_normalize(
        train_embeddings.mean(axis=0)
    ).T
    attention = decoder_attention_weights(model, train_embeddings, query_embeddings, configs)
    if attention is None:
        print("[warn] decoder attention unavailable; using embedding cosine", flush=True)
        retrieval_scores, score_name = cosine, "embedding_cosine"
    else:
        retrieval_scores, score_name = attention, "decoder_attention"

    train_frame = read_split_csv(clevel_dir, "train", family=family)
    query_frame = read_split_csv(clevel_dir, args.query_split, family=family)
    detail_rows = []
    summary_rows = []
    for query_rank, split_position in enumerate(query_positions, 1):
        scores = retrieval_scores[query_rank - 1]
        neighbors = np.argsort(scores)[::-1][: args.neighbors]
        neighbor_attention = (
            attention[query_rank - 1, neighbors]
            if attention is not None
            else np.full(len(neighbors), np.nan)
        )
        neighbor_cosine = cosine[query_rank - 1, neighbors]
        intersection, union, jaccard = input_overlap_stats(
            query_x[query_rank - 1], train_x, neighbors
        )
        positive_mass = (
            float(np.nansum(neighbor_attention[train_y[neighbors] == 1]))
            if attention is not None
            else float("nan")
        )
        positive_count = int((train_y[neighbors] == 1).sum())
        query_csv_index = int(query_original_indices[split_position])
        query_row = query_frame.iloc[query_csv_index]
        summary_rows.append(
            {
                "family": family,
                "query_rank": query_rank,
                "query_split": args.query_split,
                "query_idx": query_csv_index,
                "query_label": int(query_y[split_position]),
                "tabpfn_proba": float(probabilities[split_position]),
                "top_neighbor_positive_count": positive_count,
                "top_neighbor_count": int(len(neighbors)),
                "top_neighbor_positive_rate": float(positive_count / max(len(neighbors), 1)),
                "top_neighbor_positive_attention_mass": positive_mass,
                "query_seq_a_len": len(str(query_row["query"])),
                "query_seq_b_len": len(str(query_row["text"])),
                "query_seq_a_short": short_sequence(query_row["query"]),
                "query_seq_b_short": short_sequence(query_row["text"]),
            }
        )
        for neighbor_rank, values in enumerate(
            zip(
                neighbors,
                neighbor_attention,
                neighbor_cosine,
                intersection,
                union,
                jaccard,
                strict=False,
            ),
            1,
        ):
            neighbor, attn, cos, inter, total, jac = values
            train_csv_index = int(train_original_indices[neighbor])
            train_row = train_frame.iloc[train_csv_index]
            detail_rows.append(
                {
                    "family": family,
                    "query_rank": query_rank,
                    "query_split": args.query_split,
                    "query_idx": query_csv_index,
                    "query_label": int(query_y[split_position]),
                    "tabpfn_proba": float(probabilities[split_position]),
                    "neighbor_rank": neighbor_rank,
                    "train_idx": train_csv_index,
                    "train_label": int(train_y[neighbor]),
                    "retrieval_score_type": score_name,
                    "retrieval_score": float(scores[neighbor]),
                    "decoder_attention": float(attn) if not np.isnan(attn) else "",
                    "embedding_cosine": float(cos),
                    "input_shared_active": int(inter),
                    "input_union_active": int(total),
                    "input_jaccard": float(jac),
                    "shared_sae_features": top_shared_features(
                        query_x[query_rank - 1], train_x[neighbor], feature_metadata
                    ),
                    "train_seq_a_len": len(str(train_row["query"])),
                    "train_seq_b_len": len(str(train_row["text"])),
                    "train_seq_a_short": short_sequence(train_row["query"]),
                    "train_seq_b_short": short_sequence(train_row["text"]),
                }
            )

    summary_path = out_dir / "query_summary.csv"
    detail_path = out_dir / "neighbor_details.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    config = {
        "dataset": family,
        "family": family,
        "protein_cache": str(protein_cache_path),
        "rep": args.rep,
        "backbone": args.backbone,
        "layer": layer,
        "sae_dim": sae_dim,
        "ranking_csv": str(ranking_csv),
        "top_k_mode": args.top_k_mode,
        "top_k": args.top_k,
        "input_dim": len(flat_features),
        "train_n": int(len(train_y)),
        "query_split": args.query_split,
        "query_n_total": int(len(query_y)),
        "query_n_explained": int(len(query_positions)),
        "neighbor_k": args.neighbors,
        "retrieval_score_type": score_name,
        "decoder_attention_available": attention is not None,
    }
    dump_experiment(
        out_dir / "config.json",
        task="interpretability.tabpfn_retrieval",
        dataset=family,
        features=f"binary_sym_{args.top_k_mode}_k{args.top_k}",
        split=args.query_split,
        model="tabpfn",
        seed=args.seed,
        payload=config,
        metrics={
            "query_n_explained": config["query_n_explained"],
            "input_dim": config["input_dim"],
            "train_n": config["train_n"],
        },
        hyperparameters={
            "family": family,
            "top_k_mode": args.top_k_mode,
            "top_k": args.top_k,
            "neighbor_k": args.neighbors,
            "retrieval_score_type": score_name,
        },
    )
    report = [
        f"# TabPFN Retrieval Explanations ({family.upper()})",
        "",
        f"Dataset: {family.upper()} `{args.query_split}`. Top-K SAE ids: {args.top_k}; "
        f"input dim: {len(flat_features)}.",
        f"Train context: {len(train_y)} pairs. Explained queries: {len(query_positions)}.",
        f"Primary retrieval score: `{score_name}`.",
        "",
        "| query | idx | label | proba | top positive rate | positive attention mass |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        mass = row["top_neighbor_positive_attention_mass"]
        mass_text = "NA" if np.isnan(mass) else f"{mass:.6f}"
        report.append(
            f"| {row['query_rank']} | {row['query_idx']} | {row['query_label']} | "
            f"{row['tabpfn_proba']:.4f} | {row['top_neighbor_positive_rate']:.3f} | "
            f"{mass_text} |"
        )
    report.extend(
        [
            "",
            "Files:",
            "- `query_summary.csv`",
            "- `neighbor_details.csv`",
            "- `config.json`",
            "",
            "Decoder attention is a retrieval-style association over TabPFN row "
            "embeddings, not a causal explanation.",
        ]
    )
    (out_dir / "RETRIEVAL_EXPLANATIONS.md").write_text("\n".join(report))
    print(f"[summary] {summary_path}", flush=True)
    print(f"[details] {detail_path}", flush=True)
    print("TABPFN_RETRIEVAL_EXPLANATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
