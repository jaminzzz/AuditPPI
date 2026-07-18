#!/usr/bin/env python
"""Compute interface enrichment for residue-level SAE features.

Inputs are the geometry-derived side masks from
``build_pdb_ppi_interface_masks.py`` and a residue-level SAE LMDB cache
readable by ``src.data.sae_cache.SaeCacheReader``.

The feature definition is independent of feature text/SwissProt annotations:
for every SAE feature, test whether it is active more often on interface
residues than on same-chain non-interface residues.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from conf.audit import FDR_ALPHA
from conf.model import ESMC_SAE_DIM
from conf.paths import RESULTS_RESIDUE
from src.data.sae_cache import SaeCacheReader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side-masks", type=Path, required=True)
    p.add_argument("--sae-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae")
    p.add_argument("--positive-column", default="interface_indices")
    p.add_argument(
        "--control-column",
        default=None,
        help="optional residue-index column for controls; default is all same-chain non-positive residues",
    )
    p.add_argument("--activation-threshold", type=float, default=0.0)
    p.add_argument("--max-sides", type=int, default=0)
    p.add_argument("--dim", type=int, default=ESMC_SAE_DIM)
    p.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA)
    p.add_argument("--min-log-or", type=float, default=0.5)
    p.add_argument("--min-interface-active", type=int, default=25)
    p.add_argument("--min-side-support", type=int, default=5)
    p.add_argument(
        "--ranking",
        action="append",
        default=[],
        help="optional ranking for comparison, formatted name=/path/to/file.csv-or-tsv",
    )
    p.add_argument("--topks", default="20,50,100,200,500")
    return p.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def parse_indices(text: str) -> np.ndarray:
    if not text:
        return np.asarray([], dtype=np.int64)
    return np.fromiter((int(x) for x in text.split(",") if x), dtype=np.int64)


def bincount_active(idx: np.ndarray, val: np.ndarray, threshold: float, dim: int) -> np.ndarray:
    if idx.size == 0:
        return np.zeros(dim, dtype=np.int64)
    active = idx[val > threshold]
    if active.size == 0:
        return np.zeros(dim, dtype=np.int64)
    return np.bincount(active.astype(np.int64, copy=False), minlength=dim)[:dim]


def active_unique(idx: np.ndarray, val: np.ndarray, threshold: float) -> np.ndarray:
    if idx.size == 0:
        return np.asarray([], dtype=np.int64)
    active = idx[val > threshold]
    if active.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.unique(active.astype(np.int64, copy=False))


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * n / rank)
        q[i] = val
        prev = val
    return np.clip(q, 0.0, 1.0)


def detect_delimiter(path: Path) -> str:
    return "\t" if path.suffix in {".tsv", ".gz"} and ".tsv" in path.name else ","


def read_ranking_features(path: Path) -> list[int]:
    feature_cols = ("sae_feature", "feature_id", "feature_index", "feature", "idx")
    with open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter=detect_delimiter(path))
        cols = reader.fieldnames or []
        feature_col = next((c for c in feature_cols if c in cols), None)
        if feature_col is None:
            raise ValueError(f"cannot find feature column in {path}; columns={cols}")
        feats = []
        seen = set()
        for row in reader:
            raw = row.get(feature_col, "")
            if raw == "":
                continue
            try:
                feat = int(float(raw))
            except ValueError:
                continue
            if feat not in seen:
                feats.append(feat)
                seen.add(feat)
        return feats


def compute_ranking_overlap(args, grounded: np.ndarray, log_or: np.ndarray) -> Path | None:
    if not args.ranking:
        return None
    topks = [int(x) for x in args.topks.split(",") if x]
    global_frac = float(grounded.mean()) if grounded.size else 0.0
    out_path = args.out_dir / "ranking_interface_enrichment.tsv"
    with out_path.open("w", newline="") as fh:
        fields = [
            "ranking",
            "path",
            "topk",
            "n_ranked",
            "n_in_dim",
            "n_grounded",
            "grounded_fraction",
            "global_grounded_fraction",
            "enrichment",
            "mean_interface_log_or",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in args.ranking:
            if "=" not in item:
                raise ValueError(f"--ranking must be name=path, got {item}")
            name, raw_path = item.split("=", 1)
            path = Path(raw_path)
            feats = read_ranking_features(path)
            for k in topks:
                top = [f for f in feats[:k] if 0 <= f < len(grounded)]
                if top:
                    n_grounded = int(grounded[top].sum())
                    frac = n_grounded / len(top)
                    mean_log_or = float(np.mean(log_or[top]))
                else:
                    n_grounded = 0
                    frac = 0.0
                    mean_log_or = math.nan
                writer.writerow(
                    {
                        "ranking": name,
                        "path": str(path),
                        "topk": k,
                        "n_ranked": min(k, len(feats)),
                        "n_in_dim": len(top),
                        "n_grounded": n_grounded,
                        "grounded_fraction": frac,
                        "global_grounded_fraction": global_frac,
                        "enrichment": frac / global_frac if global_frac > 0 else math.nan,
                        "mean_interface_log_or": mean_log_or,
                    }
                )
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reader = SaeCacheReader(str(args.sae_cache_dir))
    dim = int(reader.meta.get("dim", args.dim))

    int_active = np.zeros(dim, dtype=np.int64)
    ctrl_active = np.zeros(dim, dtype=np.int64)
    side_support = np.zeros(dim, dtype=np.int64)
    total_interface = 0
    total_control = 0
    n_sides = n_cached = n_len_mismatch = n_no_interface = 0
    missing_keys = {}

    with open_text(args.side_masks) as fh:
        side_reader = csv.DictReader(fh, delimiter="\t")
        for n_sides, row in enumerate(side_reader, 1):
            if args.max_sides and n_sides > args.max_sides:
                n_sides -= 1
                break
            cache_key = row["cache_key"]
            rec = reader.get(cache_key)
            if rec is None:
                missing_keys[cache_key] = missing_keys.get(cache_key, 0) + 1
                continue
            idx, val = rec
            n_cached += 1
            length = int(row["length"])
            if idx.shape[0] != length:
                n_len_mismatch += 1
            L = min(idx.shape[0], length)
            iface = parse_indices(row[args.positive_column])
            iface = iface[(iface >= 0) & (iface < L)]
            if iface.size == 0:
                n_no_interface += 1
                continue
            mask = np.zeros(L, dtype=bool)
            mask[iface] = True
            if args.control_column:
                ctrl = parse_indices(row.get(args.control_column, ""))
                ctrl = ctrl[(ctrl >= 0) & (ctrl < L)]
                ctrl = ctrl[~mask[ctrl]]
            else:
                ctrl = np.flatnonzero(~mask)
            if ctrl.size == 0:
                continue

            int_idx = idx[:L][iface]
            int_val = val[:L][iface]
            ctrl_idx = idx[:L][ctrl]
            ctrl_val = val[:L][ctrl]
            int_active += bincount_active(int_idx, int_val, args.activation_threshold, dim)
            ctrl_active += bincount_active(ctrl_idx, ctrl_val, args.activation_threshold, dim)
            side_support[active_unique(int_idx, int_val, args.activation_threshold)] += 1
            total_interface += int(iface.size)
            total_control += int(ctrl.size)
            if n_sides % 5000 == 0:
                print(f"processed sides={n_sides} cached={n_cached}", flush=True)

    int_inactive = total_interface - int_active
    ctrl_inactive = total_control - ctrl_active
    log_or = np.log((int_active + 0.5) * (ctrl_inactive + 0.5)) - np.log(
        (int_inactive + 0.5) * (ctrl_active + 0.5)
    )
    int_rate = int_active / max(total_interface, 1)
    ctrl_rate = ctrl_active / max(total_control, 1)
    pvals = np.ones(dim, dtype=np.float64)
    for f in range(dim):
        _, p = fisher_exact(
            [
                [int(int_active[f]), int(int_inactive[f])],
                [int(ctrl_active[f]), int(ctrl_inactive[f])],
            ],
            alternative="greater",
        )
        pvals[f] = p
    qvals = bh_qvalues(pvals)
    grounded = (
        (qvals <= args.fdr_alpha)
        & (log_or >= args.min_log_or)
        & (int_active >= args.min_interface_active)
        & (side_support >= args.min_side_support)
    )

    out_path = args.out_dir / "interface_sae_feature_enrichment.tsv"
    order = np.lexsort((-int_active, qvals, -log_or))
    with out_path.open("w", newline="") as fh:
        fields = [
            "feature_id",
            "is_interface_grounded",
            "interface_log_or",
            "interface_rate",
            "control_rate",
            "fisher_p",
            "bh_q",
            "interface_active_residues",
            "control_active_residues",
            "interface_total_residues",
            "control_total_residues",
            "interface_side_support",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for f in order:
            writer.writerow(
                {
                    "feature_id": int(f),
                    "is_interface_grounded": int(grounded[f]),
                    "interface_log_or": float(log_or[f]),
                    "interface_rate": float(int_rate[f]),
                    "control_rate": float(ctrl_rate[f]),
                    "fisher_p": float(pvals[f]),
                    "bh_q": float(qvals[f]),
                    "interface_active_residues": int(int_active[f]),
                    "control_active_residues": int(ctrl_active[f]),
                    "interface_total_residues": int(total_interface),
                    "control_total_residues": int(total_control),
                    "interface_side_support": int(side_support[f]),
                }
            )

    ranking_path = compute_ranking_overlap(args, grounded, log_or)
    summary = {
        "side_masks": str(args.side_masks),
        "sae_cache_dir": str(args.sae_cache_dir),
        "positive_column": args.positive_column,
        "control_column": args.control_column,
        "activation_threshold": args.activation_threshold,
        "n_sides_read": n_sides,
        "n_sides_cached": n_cached,
        "n_sides_missing_cache": n_sides - n_cached,
        "n_unique_missing_cache_keys": len(missing_keys),
        "n_length_mismatch_sides": n_len_mismatch,
        "n_no_interface_sides": n_no_interface,
        "total_interface_residues": int(total_interface),
        "total_control_residues": int(total_control),
        "fdr_alpha": args.fdr_alpha,
        "min_log_or": args.min_log_or,
        "min_interface_active": args.min_interface_active,
        "min_side_support": args.min_side_support,
        "n_interface_grounded_features": int(grounded.sum()),
        "feature_enrichment_tsv": str(out_path),
        "ranking_interface_enrichment_tsv": str(ranking_path) if ranking_path else None,
        "missing_cache_key_examples": sorted(missing_keys)[:20],
    }
    with (args.out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
