#!/usr/bin/env python
"""Compute SAE feature-pair compatibility at structural PPI contacts.

This audit asks a stricter question than single-feature interface enrichment:
are pairs of SAE residue features enriched specifically on cross-chain contact
residue pairs, compared with same-complex surface non-contact residue pairs?

Definitions:
  * positive residue pair: Cbeta-Cbeta <= contact_threshold Angstrom.
  * control residue pair: both residues are surface non-interface residues and
    Cbeta-Cbeta >= control_min_distance Angstrom, sampled within the same PDB
    chain pair.
  * active feature-pair: an unordered pair formed from the top-m positive SAE
    features on the two residues.

The script reuses the existing PDB_PPI structures and residue-level SAE LMDB
cache; it does not run ESM-C or SAE inference.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import hypergeom

from conf.paths import RESULTS_RESIDUE, FEATURE_TABLE, PDB_PPI_SAE_CACHE, PPI_DATA
from src.data.sae_cache import SaeCacheReader

DEFAULT_DATA_ROOT = PPI_DATA
DEFAULT_SURFACE_SIDES = (
    RESULTS_RESIDUE / "interface_grounding/pdb_ppi_pos_surface/posi_nohomo.side_surface_controls.tsv.gz"
)
DEFAULT_SAE_CACHE = PDB_PPI_SAE_CACHE
DEFAULT_FEATURE_TABLE = FEATURE_TABLE

THREE2ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side-surface-controls", type=Path, default=DEFAULT_SURFACE_SIDES)
    p.add_argument("--sae-cache-dir", type=Path, default=DEFAULT_SAE_CACHE)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_RESIDUE / "interface_grounding/pdb_ppi_pos_sae_contact_compat_top4",
    )
    p.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    p.add_argument("--contact-threshold", type=float, default=8.0)
    p.add_argument("--control-min-distance", type=float, default=12.0)
    p.add_argument("--control-per-contact", type=float, default=1.0)
    p.add_argument("--top-m", type=int, default=4)
    p.add_argument("--activation-threshold", type=float, default=0.0)
    p.add_argument("--dim", type=int, default=16384)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--max-pairs", type=int, default=0, help="cap PDB chain pairs for smoke runs")
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--write-top-n", type=int, default=50000)
    p.add_argument(
        "--write-all-tsv",
        action="store_true",
        help="write every observed feature-pair as annotated TSV; can be very large on full data",
    )
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--min-log-or", type=float, default=0.5)
    p.add_argument("--min-contact-count", type=int, default=100)
    p.add_argument("--min-pair-support", type=int, default=10)
    p.add_argument(
        "--exhaustive-control-max",
        type=int,
        default=200000,
        help="exhaustively enumerate control candidates when |surface_a|*|surface_b| is at most this",
    )
    p.add_argument("--sample-batch-size", type=int, default=8192)
    return p.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def parse_indices(text: str) -> np.ndarray:
    if not text:
        return np.asarray([], dtype=np.int64)
    return np.fromiter((int(x) for x in text.split(",") if x), dtype=np.int64)


def format_threshold(t: float) -> str:
    return str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")


def pdb_chain_to_path(root: Path, chain1: str, chain2: str, which: str) -> Path:
    pdbid = chain1.split("_")[0]
    sub = pdbid[1:3]
    base = root / "PDB_PPI" / "posi_pdbs_extracted" / sub
    path = base / f"{chain1}-{chain2}__{which}.pdb"
    if path.is_file():
        return path
    hits = list((root / "PDB_PPI" / "posi_pdbs_extracted").glob(f"*/{chain1}-{chain2}__{which}.pdb"))
    return hits[0] if hits else path


def parse_pdb(path: Path):
    """Return (sequence, Cbeta-like coords) for one extracted chain."""
    if not path.is_file():
        return None, None
    residues = OrderedDict()
    try:
        with path.open() as fh:
            for line in fh:
                if not line.startswith("ATOM") or len(line) < 54:
                    continue
                atom_name = line[12:16].strip()
                resn = line[17:20].strip()
                key = (line[21], line[22:26].strip(), line[26])
                if key not in residues:
                    residues[key] = [THREE2ONE.get(resn, "X"), {}]
                if atom_name not in ("N", "CA", "C", "CB"):
                    continue
                if atom_name in residues[key][1]:
                    continue
                try:
                    residues[key][1][atom_name] = (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                except ValueError:
                    continue
    except OSError:
        return None, None
    if not residues:
        return None, None
    seq = "".join(v[0] for v in residues.values())
    cb = [_reconstruct_cb(atoms) for _, atoms in residues.values()]
    return seq, cb


def _reconstruct_cb(atoms: dict[str, tuple[float, float, float]]):
    if "N" in atoms and "CA" in atoms and "C" in atoms:
        n = np.asarray(atoms["N"], dtype=np.float64)
        ca = np.asarray(atoms["CA"], dtype=np.float64)
        c = np.asarray(atoms["C"], dtype=np.float64)
        b, cc = ca - n, c - ca
        a = np.cross(b, cc)
        cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca
        return cb.astype(np.float32)
    if "CB" in atoms:
        return np.asarray(atoms["CB"], dtype=np.float32)
    if "CA" in atoms:
        return np.asarray(atoms["CA"], dtype=np.float32)
    return None


def valid_coord_arrays(cb_a, cb_b, len_a: int, len_b: int):
    idx_a = np.asarray([i for i, c in enumerate(cb_a[:len_a]) if c is not None], dtype=np.int64)
    idx_b = np.asarray([j for j, c in enumerate(cb_b[:len_b]) if c is not None], dtype=np.int64)
    if idx_a.size == 0 or idx_b.size == 0:
        return idx_a, idx_b, None, None
    arr_a = np.stack([cb_a[int(i)] for i in idx_a]).astype(np.float32, copy=False)
    arr_b = np.stack([cb_b[int(j)] for j in idx_b]).astype(np.float32, copy=False)
    return idx_a, idx_b, arr_a, arr_b


def contact_pairs(idx_a, idx_b, arr_a, arr_b, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if arr_a is None or arr_b is None:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    tree_b = cKDTree(arr_b)
    neigh = tree_b.query_ball_point(arr_a, r=threshold)
    out_i = []
    out_j = []
    for ka, blist in enumerate(neigh):
        if not blist:
            continue
        i = int(idx_a[ka])
        for kb in blist:
            out_i.append(i)
            out_j.append(int(idx_b[kb]))
    return np.asarray(out_i, dtype=np.int64), np.asarray(out_j, dtype=np.int64)


def coords_for_indices(indices: np.ndarray, cb: list, max_len: int):
    keep = indices[(indices >= 0) & (indices < max_len)]
    good = [int(i) for i in keep if cb[int(i)] is not None]
    if not good:
        return np.asarray([], dtype=np.int64), None
    return np.asarray(good, dtype=np.int64), np.stack([cb[i] for i in good]).astype(np.float32, copy=False)


def stable_pair_seed(seed: int, pair_id: str) -> int:
    h = hashlib.blake2b(pair_id.encode(), digest_size=8).digest()
    return (int.from_bytes(h, "little") ^ int(seed)) % (2**63 - 1)


def sample_control_pairs(
    surf_a: np.ndarray,
    surf_b: np.ndarray,
    cb_a: list,
    cb_b: list,
    len_a: int,
    len_b: int,
    n_target: int,
    min_distance: float,
    exhaustive_max: int,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_target <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    si, ca = coords_for_indices(surf_a, cb_a, len_a)
    sj, cb = coords_for_indices(surf_b, cb_b, len_b)
    if ca is None or cb is None:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    n_possible = int(si.size * sj.size)
    if n_possible == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    if n_possible <= exhaustive_max:
        dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
        ii, jj = np.nonzero(dist >= min_distance)
        n_valid = int(ii.size)
        if n_valid == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
        take = min(n_target, n_valid)
        chosen = rng.choice(n_valid, size=take, replace=False)
        return si[ii[chosen]], sj[jj[chosen]]

    out_i = []
    out_j = []
    seen = set()
    attempts = 0
    max_attempts = 100
    while len(out_i) < n_target and attempts < max_attempts:
        attempts += 1
        need = n_target - len(out_i)
        batch = max(batch_size, need * 2)
        ia = rng.integers(0, si.size, size=batch)
        jb = rng.integers(0, sj.size, size=batch)
        d = np.linalg.norm(ca[ia] - cb[jb], axis=1)
        good = np.flatnonzero(d >= min_distance)
        for g in good:
            i = int(si[ia[g]])
            j = int(sj[jb[g]])
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            out_i.append(i)
            out_j.append(j)
            if len(out_i) >= n_target:
                break
    return np.asarray(out_i, dtype=np.int64), np.asarray(out_j, dtype=np.int64)


def top_features(idx: np.ndarray, val: np.ndarray, top_m: int, threshold: float) -> np.ndarray:
    """Return (L, top_m) int32 feature ids, padded with -1 for inactive slots."""
    if idx.ndim != 2 or idx.shape != val.shape:
        raise ValueError(f"bad SAE shapes: idx={idx.shape} val={val.shape}")
    L, k = idx.shape
    top_m = min(top_m, k)
    values = val.astype(np.float32, copy=False)
    if top_m == k:
        order = np.argsort(values, axis=1)[:, ::-1]
    else:
        part = np.argpartition(values, kth=k - top_m, axis=1)[:, -top_m:]
        part_vals = np.take_along_axis(values, part, axis=1)
        order_in_part = np.argsort(part_vals, axis=1)[:, ::-1]
        order = np.take_along_axis(part, order_in_part, axis=1)
    feats = np.take_along_axis(idx.astype(np.int32, copy=False), order, axis=1)
    vals = np.take_along_axis(values, order, axis=1)
    feats = feats.copy()
    feats[vals <= threshold] = -1
    return feats


def add_feature_pair_counts(
    counter: Counter,
    support_set: set[int] | None,
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    inds_a: np.ndarray,
    inds_b: np.ndarray,
    dim: int,
    chunk_size: int = 50000,
) -> int:
    """Count unordered feature-pairs over residue-pair indices."""
    n = int(min(inds_a.size, inds_b.size))
    if n == 0:
        return 0
    counted_residue_pairs = 0
    m = feats_a.shape[1]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        fa = feats_a[inds_a[start:end]]
        fb = feats_b[inds_b[start:end]]
        lo = np.minimum(fa[:, :, None], fb[:, None, :])
        hi = np.maximum(fa[:, :, None], fb[:, None, :])
        valid = (lo >= 0) & (hi >= 0)
        keys = (lo.astype(np.int64) * dim + hi.astype(np.int64)).reshape(end - start, m * m)
        keys[~valid.reshape(end - start, m * m)] = -1

        # One feature pair should count once per residue-residue pair even if it
        # can be generated by both (f_i, f_j) and (f_j, f_i).
        keys.sort(axis=1)
        keep = keys >= 0
        keep[:, 1:] &= keys[:, 1:] != keys[:, :-1]
        flat = keys[keep]
        if flat.size == 0:
            continue
        uniq, cnt = np.unique(flat, return_counts=True)
        for key, c in zip(uniq, cnt):
            k_int = int(key)
            counter[k_int] += int(c)
            if support_set is not None:
                support_set.add(k_int)
        counted_residue_pairs += end - start
    return counted_residue_pairs


def paired_side_rows(path: Path):
    pending: dict[str, dict[str, dict]] = {}
    with open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pair_id = row["pair_id"]
            pair = pending.setdefault(pair_id, {})
            pair[row["side"]] = row
            if "a" in pair and "b" in pair:
                yield pair_id, pair["a"], pair["b"]
                del pending[pair_id]
    if pending:
        print(f"warning: {len(pending)} unpaired side rows left in {path}", file=sys.stderr)


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


def load_feature_annotations(path: Path) -> dict[int, dict]:
    if not path or not path.is_file():
        return {}
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional path
        print(f"warning: pandas unavailable, skipping feature annotations: {exc}", file=sys.stderr)
        return {}
    df = pd.read_parquet(path, columns=["feature_id", "category", "summary"])
    out = {}
    for row in df.itertuples(index=False):
        out[int(row.feature_id)] = {
            "category": "" if row.category is None else str(row.category),
            "summary": "" if row.summary is None else str(row.summary),
        }
    return out


def category_pair(cat_a: str, cat_b: str) -> str:
    a = cat_a or "Unannotated"
    b = cat_b or "Unannotated"
    return "|".join(sorted((a, b)))


def write_results(
    args: argparse.Namespace,
    pos_counts: Counter,
    ctrl_counts: Counter,
    pair_support: Counter,
    total_pos_pairs: int,
    total_ctrl_pairs: int,
    annotations: dict[int, dict],
    dim: int,
) -> tuple[Path, Path | None, Path, Path]:
    keys = np.asarray(sorted(set(pos_counts) | set(ctrl_counts)), dtype=np.int64)
    pos = np.asarray([pos_counts[int(k)] for k in keys], dtype=np.int64)
    ctrl = np.asarray([ctrl_counts[int(k)] for k in keys], dtype=np.int64)
    support = np.asarray([pair_support[int(k)] for k in keys], dtype=np.int64)

    total = int(total_pos_pairs + total_ctrl_pairs)
    pair_totals = pos + ctrl
    # Fisher's exact one-sided p-value can be computed as a hypergeometric
    # survival function under fixed margins.
    pvals = hypergeom.sf(pos - 1, total, pair_totals, int(total_pos_pairs))
    qvals = bh_qvalues(pvals)
    pos_inactive = total_pos_pairs - pos
    ctrl_inactive = total_ctrl_pairs - ctrl
    log_or = np.log((pos + 0.5) * (ctrl_inactive + 0.5)) - np.log(
        (pos_inactive + 0.5) * (ctrl + 0.5)
    )
    fold = (pos / max(total_pos_pairs, 1)) / ((ctrl + 0.5) / max(total_ctrl_pairs, 1))
    enriched = (
        (qvals <= args.fdr_alpha)
        & (log_or >= args.min_log_or)
        & (pos >= args.min_contact_count)
        & (support >= args.min_pair_support)
    )
    order = np.lexsort((-pos, qvals, -log_or))

    npz_path = args.out_dir / "feature_pair_contact_compatibility_arrays.npz"
    np.savez_compressed(
        npz_path,
        keys=keys,
        contact_active_pairs=pos,
        control_active_pairs=ctrl,
        pdb_pair_support=support,
        hypergeom_p=pvals,
        bh_q=qvals,
        contact_log_or=log_or,
        contact_fold_change=fold,
        is_contact_compatible=enriched.astype(np.int8),
        dim=np.asarray([dim], dtype=np.int64),
        contact_total_pairs=np.asarray([total_pos_pairs], dtype=np.int64),
        control_total_pairs=np.asarray([total_ctrl_pairs], dtype=np.int64),
    )

    out_path = args.out_dir / "feature_pair_contact_compatibility.tsv"
    top_path = args.out_dir / "top_contact_compatible_feature_pairs_annotated.tsv"
    fields = [
        "feature_a",
        "feature_b",
        "is_contact_compatible",
        "contact_log_or",
        "contact_fold_change",
        "contact_rate",
        "control_rate",
        "hypergeom_p",
        "bh_q",
        "contact_active_pairs",
        "control_active_pairs",
        "contact_total_pairs",
        "control_total_pairs",
        "pdb_pair_support",
        "category_a",
        "category_b",
        "category_pair",
        "summary_a",
        "summary_b",
    ]
    out_fh = out_path.open("w", newline="") if args.write_all_tsv else None
    with top_path.open("w", newline="") as top_fh:
        writer = (
            csv.DictWriter(out_fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
            if out_fh is not None
            else None
        )
        top_writer = csv.DictWriter(top_fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if writer is not None:
            writer.writeheader()
        top_writer.writeheader()
        n_top = 0
        for idx_i in order:
            key = int(keys[idx_i])
            fa = key // dim
            fb = key % dim
            ann_a = annotations.get(fa, {})
            ann_b = annotations.get(fb, {})
            cat_a = ann_a.get("category", "")
            cat_b = ann_b.get("category", "")
            row = {
                "feature_a": fa,
                "feature_b": fb,
                "is_contact_compatible": int(enriched[idx_i]),
                "contact_log_or": float(log_or[idx_i]),
                "contact_fold_change": float(fold[idx_i]),
                "contact_rate": float(pos[idx_i] / max(total_pos_pairs, 1)),
                "control_rate": float(ctrl[idx_i] / max(total_ctrl_pairs, 1)),
                "hypergeom_p": float(pvals[idx_i]),
                "bh_q": float(qvals[idx_i]),
                "contact_active_pairs": int(pos[idx_i]),
                "control_active_pairs": int(ctrl[idx_i]),
                "contact_total_pairs": int(total_pos_pairs),
                "control_total_pairs": int(total_ctrl_pairs),
                "pdb_pair_support": int(support[idx_i]),
                "category_a": cat_a,
                "category_b": cat_b,
                "category_pair": category_pair(cat_a, cat_b),
                "summary_a": ann_a.get("summary", ""),
                "summary_b": ann_b.get("summary", ""),
            }
            if writer is not None:
                writer.writerow(row)
            if n_top < args.write_top_n:
                top_writer.writerow(row)
                n_top += 1
    if out_fh is not None:
        out_fh.close()
    else:
        out_path = None

    cat_counts = Counter()
    cat_bg_counts = Counter()
    for idx_i, key in enumerate(keys):
        fa = int(key // dim)
        fb = int(key % dim)
        cp = category_pair(
            annotations.get(fa, {}).get("category", ""),
            annotations.get(fb, {}).get("category", ""),
        )
        cat_bg_counts[cp] += 1
        if enriched[idx_i]:
            cat_counts[cp] += 1
    cat_path = args.out_dir / "contact_compatible_category_pair_summary.tsv"
    with cat_path.open("w", newline="") as fh:
        fields2 = [
            "category_pair",
            "n_contact_compatible_feature_pairs",
            "n_observed_feature_pairs",
            "compatible_fraction",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields2, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for cp, n_obs in sorted(cat_bg_counts.items(), key=lambda x: (-cat_counts[x[0]], x[0])):
            n_hit = cat_counts[cp]
            writer.writerow(
                {
                    "category_pair": cp,
                    "n_contact_compatible_feature_pairs": int(n_hit),
                    "n_observed_feature_pairs": int(n_obs),
                    "compatible_fraction": float(n_hit / n_obs) if n_obs else 0.0,
                }
            )
    return npz_path, out_path, top_path, cat_path


def main() -> None:
    args = parse_args()
    if args.top_m <= 0:
        raise ValueError("--top-m must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reader = SaeCacheReader(str(args.sae_cache_dir))
    dim = int(reader.meta.get("dim", args.dim))

    @lru_cache(maxsize=4096)
    def cached_top_features(cache_key: str):
        rec = reader.get(cache_key)
        if rec is None:
            return None
        idx, val = rec
        return top_features(idx, val, args.top_m, args.activation_threshold)

    pos_counts: Counter = Counter()
    ctrl_counts: Counter = Counter()
    pair_support: Counter = Counter()
    stats = {
        "side_surface_controls": str(args.side_surface_controls),
        "sae_cache_dir": str(args.sae_cache_dir),
        "data_root": str(args.data_root),
        "contact_threshold": args.contact_threshold,
        "control_min_distance": args.control_min_distance,
        "control_per_contact": args.control_per_contact,
        "top_m": args.top_m,
        "activation_threshold": args.activation_threshold,
        "seed": args.seed,
        "n_pairs_seen": 0,
        "n_pairs_processed": 0,
        "n_pairs_missing_pdb": 0,
        "n_pairs_missing_cache": 0,
        "n_pairs_no_contacts": 0,
        "n_pairs_no_controls": 0,
        "n_contact_residue_pairs": 0,
        "n_control_residue_pairs": 0,
        "n_contact_residue_pairs_counted": 0,
        "n_control_residue_pairs_counted": 0,
        "missing_examples": [],
    }

    for pair_id, row_a, row_b in paired_side_rows(args.side_surface_controls):
        if args.max_pairs and stats["n_pairs_seen"] >= args.max_pairs:
            break
        stats["n_pairs_seen"] += 1

        chain_a = row_a["chain_id"]
        chain_b = row_b["chain_id"]
        path_a = pdb_chain_to_path(args.data_root, chain_a, chain_b, chain_a)
        path_b = pdb_chain_to_path(args.data_root, chain_a, chain_b, chain_b)
        seq_a, cb_a = parse_pdb(path_a)
        seq_b, cb_b = parse_pdb(path_b)
        if seq_a is None or seq_b is None:
            stats["n_pairs_missing_pdb"] += 1
            if len(stats["missing_examples"]) < 20:
                stats["missing_examples"].append({"pair_id": pair_id, "path_a": str(path_a), "path_b": str(path_b)})
            continue

        feats_a = cached_top_features(row_a["cache_key"])
        feats_b = cached_top_features(row_b["cache_key"])
        if feats_a is None or feats_b is None:
            stats["n_pairs_missing_cache"] += 1
            continue

        len_a = min(int(row_a["length"]), len(seq_a), feats_a.shape[0])
        len_b = min(int(row_b["length"]), len(seq_b), feats_b.shape[0])
        idx_a, idx_b, arr_a, arr_b = valid_coord_arrays(cb_a, cb_b, len_a, len_b)
        con_i, con_j = contact_pairs(idx_a, idx_b, arr_a, arr_b, args.contact_threshold)
        if con_i.size == 0:
            stats["n_pairs_no_contacts"] += 1
            continue

        surf_a = parse_indices(row_a.get("surface_noninterface_indices", ""))
        surf_b = parse_indices(row_b.get("surface_noninterface_indices", ""))
        n_controls_target = int(math.ceil(con_i.size * args.control_per_contact))
        pair_rng = np.random.default_rng(stable_pair_seed(args.seed, pair_id))
        ctrl_i, ctrl_j = sample_control_pairs(
            surf_a,
            surf_b,
            cb_a,
            cb_b,
            len_a,
            len_b,
            n_controls_target,
            args.control_min_distance,
            args.exhaustive_control_max,
            pair_rng,
            args.sample_batch_size,
        )
        if ctrl_i.size == 0:
            stats["n_pairs_no_controls"] += 1
            continue

        stats["n_pairs_processed"] += 1
        stats["n_contact_residue_pairs"] += int(con_i.size)
        stats["n_control_residue_pairs"] += int(ctrl_i.size)

        pair_keys: set[int] = set()
        stats["n_contact_residue_pairs_counted"] += add_feature_pair_counts(
            pos_counts, pair_keys, feats_a, feats_b, con_i, con_j, dim
        )
        stats["n_control_residue_pairs_counted"] += add_feature_pair_counts(
            ctrl_counts, None, feats_a, feats_b, ctrl_i, ctrl_j, dim
        )
        for key in pair_keys:
            pair_support[key] += 1

        if args.progress_every and stats["n_pairs_seen"] % args.progress_every == 0:
            print(
                "processed "
                f"seen={stats['n_pairs_seen']} ok={stats['n_pairs_processed']} "
                f"contacts={stats['n_contact_residue_pairs']} controls={stats['n_control_residue_pairs']} "
                f"pos_feature_pairs={len(pos_counts)}",
                flush=True,
            )

    annotations = load_feature_annotations(args.feature_table)
    arrays_path, result_path, top_path, cat_path = write_results(
        args,
        pos_counts,
        ctrl_counts,
        pair_support,
        stats["n_contact_residue_pairs_counted"],
        stats["n_control_residue_pairs_counted"],
        annotations,
        dim,
    )
    stats.update(
        {
            "dim": dim,
            "n_observed_contact_feature_pairs": len(pos_counts),
            "n_observed_control_feature_pairs": len(ctrl_counts),
            "feature_pair_contact_compatibility_arrays_npz": str(arrays_path),
            "feature_pair_contact_compatibility_tsv": str(result_path) if result_path else None,
            "top_contact_compatible_feature_pairs_annotated_tsv": str(top_path),
            "contact_compatible_category_pair_summary_tsv": str(cat_path),
            "fdr_alpha": args.fdr_alpha,
            "min_log_or": args.min_log_or,
            "min_contact_count": args.min_contact_count,
            "min_pair_support": args.min_pair_support,
        }
    )
    with (args.out_dir / "summary.json").open("w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
