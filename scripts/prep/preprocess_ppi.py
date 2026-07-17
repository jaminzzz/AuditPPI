#!/usr/bin/env python
"""Preprocess raw PDB_PPI / AFDB_DDI into consolidated train/val JSONL for AuditPPI.

This is the heavy, one-time extraction job (see the increment-2 plan). It reads the raw
metadata CSVs + per-chain extracted PDB coordinate files under ``PPI_data/`` and writes one
JSON record per pair containing the two chain sequences and the inter-chain contact list, so
that training (``src/data/ppi_datamodule.py``) never has to touch a PDB file.

Contacts are stored as inter-chain **Cβ–Cβ** residue pairs with distance < ``--dmax`` (20 Å),
each as ``[i, j, dist]`` (Cβ reconstructed from backbone N/Cα/C exactly as RoseTTAFold2 does).
This single sparse representation supports the downstream binary contact target:
contact = Cβ–Cβ < 12 Å.
Negatives are cross-structure → empty pair list (all "no-contact"). Full pairs are stored, so the
threshold / binning is a training-time choice and never needs re-preprocessing.

Splits are the homology-aware ones already shipped with the data:
  * PDB_PPI : ``{posi,nega}_val.list`` hold the CLUSTER ids reserved for validation.
  * AFDB_DDI: ``pretrain_split_50id80cov/{train,val}_ids.strict50id80cov.txt`` (lines ``sp|ACC|NAME``).

Usage (genmol env)::

    PY=/data/wmzhu/anaconda3/envs/genmol/bin/python
    $PY scripts/preprocess_ppi.py --limit 200          # tiny smoke slice
    $PY scripts/preprocess_ppi.py                       # full datasets (the long job)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

from conf.paths import PPI_DATA

# --------------------------------------------------------------------------- constants
DEFAULT_DATA_ROOT = str(PPI_DATA)

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine -> Met (very common in crystal structures)
}

# CSV columns (verified against the data)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# --------------------------------------------------------------------------- PDB parsing
def parse_pdb(path):
    """Parse a single-chain extracted PDB file.

    Returns ``(seq, cb)`` where ``seq`` is the 1-letter sequence in file order and ``cb`` is a
    list (one per residue) of the Cβ coordinate as a float32 ``(3,)`` array, or ``None`` if the
    residue has no usable backbone. Cβ is **reconstructed from N/Cα/C** exactly as RF2 does
    (``util.get_Cb``), so glycine and missing-Cβ residues are handled uniformly. ``(None, None)``
    if the file is missing/unreadable/empty.
    """
    if not os.path.isfile(path):
        return None, None
    # (resseq, icode) -> [aa1, {atom_name: xyz}]
    residues = OrderedDict()
    try:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                resn = line[17:20].strip()
                key = (line[22:26].strip(), line[26])  # (resSeq, iCode)
                if key not in residues:
                    residues[key] = [THREE2ONE.get(resn, "X"), {}]
                try:
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                # keep only the backbone atoms needed to reconstruct Cβ (+ the real CB as fallback)
                if atom_name in ("N", "CA", "C", "CB") and atom_name not in residues[key][1]:
                    residues[key][1][atom_name] = xyz
    except OSError:
        return None, None
    if not residues:
        return None, None
    seq = "".join(v[0] for v in residues.values())
    cb = [_reconstruct_cb(atoms) for _, atoms in residues.values()]
    return seq, cb


def _reconstruct_cb(atoms):
    """Cβ from N/Cα/C (RF2 `util.get_Cb`); fall back to the real CB, then Cα; else None."""
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


def compute_cb_pairs(cb_a, cb_b, dmax):
    """Inter-chain residue pairs with Cβ–Cβ distance < dmax (Å).

    Returns a sorted list of ``[i, j, dist]`` (0-based residue indices + rounded distance).
    Stored up to ``dmax`` (=20 Å); the binary Cβ-contact label thresholds ``dist`` at training time.
    """
    idx_a = [i for i, c in enumerate(cb_a) if c is not None]
    idx_b = [j for j, c in enumerate(cb_b) if c is not None]
    if not idx_a or not idx_b:
        return []
    A = np.stack([cb_a[i] for i in idx_a])
    B = np.stack([cb_b[j] for j in idx_b])
    treeB = cKDTree(B)
    neighbors = treeB.query_ball_point(A, r=dmax)  # B-residues within dmax of each A-residue
    pairs = []
    for ka, blist in enumerate(neighbors):
        if not blist:
            continue
        i = idx_a[ka]
        for kb in blist:
            d = float(np.linalg.norm(A[ka] - B[kb]))
            pairs.append([i, idx_b[kb], round(d, 3)])
    pairs.sort()
    return pairs


# --------------------------------------------------------------------------- worker
def _process(job, dmax):
    """Worker: parse both chains, compute Cβ–Cβ pairs (<dmax) for positives. Returns a status tuple."""
    seq_a, cb_a = parse_pdb(job["pdb_a"])
    seq_b, cb_b = parse_pdb(job["pdb_b"])
    if seq_a is None or seq_b is None:
        missing = job["pdb_a"] if seq_a is None else job["pdb_b"]
        return ("missing", job["source"], job["pair_id"], missing)

    cb_pairs = compute_cb_pairs(cb_a, cb_b, dmax) if job["label"] == 1 else []

    # length-consistency check vs metadata (kept, not dropped — coords are truth)
    mismatch = False
    if job.get("len_a_meta") is not None:
        mismatch = (len(seq_a) != job["len_a_meta"]) or (len(seq_b) != job["len_b_meta"])

    record = {
        "source": job["source"],
        "pair_id": job["pair_id"],
        "label": job["label"],
        "is_pdb": job["source"] == "ppi",
        "seq_a": seq_a,
        "seq_b": seq_b,
        "len_a": len(seq_a),
        "len_b": len(seq_b),
        # inter-chain Cβ–Cβ pairs within dmax: [i, j, dist] (sorted). Empty for negatives.
        "cb_pairs": cb_pairs,
        "n_pairs": len(cb_pairs),
        "split": job["split"],
    }
    return ("ok", record, mismatch, len(cb_pairs))


# --------------------------------------------------------------------------- job building
def _read_csv(path):
    with open(path, newline="") as fh:
        yield from csv.DictReader(fh)


def _pdb_chain_to_path(root, polarity, chain1, chain2, which):
    """PDB_PPI: <root>/PDB_PPI/<pol>_pdbs_extracted/<c2c3>/<CHAIN1>-<CHAIN2>__<which>.pdb"""
    pdbid = chain1.split("_")[0]
    sub = pdbid[1:3]
    base = os.path.join(root, "PDB_PPI", f"{polarity}_pdbs_extracted", sub)
    fname = f"{chain1}-{chain2}__{which}.pdb"
    path = os.path.join(base, fname)
    if os.path.isfile(path):
        return path
    # fallback: the directory key occasionally differs — glob for the pair
    hits = glob.glob(os.path.join(root, "PDB_PPI", f"{polarity}_pdbs_extracted", "*",
                                  f"{chain1}-{chain2}__{which}.pdb"))
    return hits[0] if hits else path  # return nominal path (worker flags missing)


def _ddi_dom_to_path(root, dom):
    """AFDB_DDI: <root>/AFDB_DDI/dompdbs_extracted/<acc[-2:]>/<acc>_D<n>.pdb"""
    acc = dom.rsplit("_", 1)[0]
    sub = acc[-2:]
    return os.path.join(root, "AFDB_DDI", "dompdbs_extracted", sub, f"{dom}.pdb")


def build_pdb_jobs(root, polarity, label, val_clusters, limit):
    csv_path = os.path.join(root, "PDB_PPI", f"{polarity}_nohomo.csv")
    jobs = []
    for n, row in enumerate(_read_csv(csv_path)):
        if limit is not None and n >= limit:
            break
        chain1, chain2 = row["CHAIN1:CHAIN2"].split(":")
        len1, len2 = (int(x) for x in row["LEN1:LEN2"].split(":"))
        split = "val" if row["CLUSTER"] in val_clusters else "train"
        jobs.append({
            "source": "ppi", "pair_id": row["CHAIN1:CHAIN2"], "label": label, "split": split,
            "pdb_a": _pdb_chain_to_path(root, polarity, chain1, chain2, chain1),
            "pdb_b": _pdb_chain_to_path(root, polarity, chain1, chain2, chain2),
            "len_a_meta": len1, "len_b_meta": len2,
        })
    return jobs


def build_ddi_jobs(root, polarity, label, val_every, limit):
    """AFDB_DDI ships no val split, so hold out a deterministic fraction of its OWN CLUSTER
    column (``int(cluster) % val_every == 0`` -> val). Homology-aware (clusters never straddle
    train/val), pos & neg share AFDB_DDI's cluster space so the rule is consistent across them.
    AFDB_DDI clusters are a SEPARATE id system from PDB_PPI's — never mixed."""
    csv_path = os.path.join(root, "AFDB_DDI", f"{polarity}_nohomo.csv")
    jobs = []
    for n, row in enumerate(_read_csv(csv_path)):
        if limit is not None and n >= limit:
            break
        dom_a, dom_b = row["PAIRID"].split(":")
        len_a, len_b = (int(x) for x in row["LEN"].split(":"))
        try:
            is_val = (int(row["CLUSTER"]) % val_every == 0)
        except ValueError:
            is_val = False
        jobs.append({
            "source": "ddi", "pair_id": row["PAIRID"], "label": label,
            "split": "val" if is_val else "train",
            "pdb_a": _ddi_dom_to_path(root, dom_a),
            "pdb_b": _ddi_dom_to_path(root, dom_b),
            "len_a_meta": len_a, "len_b_meta": len_b,
        })
    return jobs


def load_pdb_val_clusters(root, polarity):
    path = os.path.join(root, "PDB_PPI", f"{polarity}_val.list")
    clusters = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and line != "CLUSTER":
                clusters.add(line)
    return clusters


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"))
    ap.add_argument("--datasets", default="pdb,ddi", help="comma list: pdb,ddi")
    ap.add_argument("--dmax", type=float, default=20.0,
                    help="store inter-chain Cβ–Cβ pairs closer than this (Å)")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="AFDB_DDI held-out cluster fraction (PDB_PPI uses its shipped val.list)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunksize", type=int, default=64)
    # caps (None => all). --limit caps every csv (handy for smoke tests).
    ap.add_argument("--limit", type=int, default=None, help="cap EACH csv to N rows (smoke test)")
    ap.add_argument("--max-pdb-pos", type=int, default=None)
    ap.add_argument("--max-pdb-neg", type=int, default=None)
    ap.add_argument("--max-ddi-pos", type=int, default=None)
    ap.add_argument("--max-ddi-neg", type=int, default=None)
    args = ap.parse_args()

    root = args.data_root
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    val_every = max(2, round(1.0 / args.val_frac)) if args.val_frac > 0 else 10**9

    def cap(specific):
        if args.limit is not None:
            return args.limit if specific is None else min(args.limit, specific)
        return specific

    # ---- build job list (cheap; main process) ----
    jobs = []
    if "pdb" in datasets:
        pos_clusters = load_pdb_val_clusters(root, "posi")
        neg_clusters = load_pdb_val_clusters(root, "nega")
        jobs += build_pdb_jobs(root, "posi", 1, pos_clusters, cap(args.max_pdb_pos))
        jobs += build_pdb_jobs(root, "nega", 0, neg_clusters, cap(args.max_pdb_neg))
    if "ddi" in datasets:
        jobs += build_ddi_jobs(root, "posi", 1, val_every, cap(args.max_ddi_pos))
        jobs += build_ddi_jobs(root, "nega", 0, val_every, cap(args.max_ddi_neg))

    print(f"[preprocess] {len(jobs)} pairs to process; "
          f"datasets={datasets} ddi_val_every={val_every}; workers={args.workers}", flush=True)

    # ---- process (parallel) and stream to split files ----
    out = {s: open(os.path.join(args.out_dir, f"{s}.jsonl"), "w") for s in ("train", "val")}
    missing_fh = open(os.path.join(args.out_dir, "missing.tsv"), "w")
    missing_fh.write("source\tpair_id\tmissing_path\n")

    summary = {
        "data_root": root, "datasets": datasets, "dmax": args.dmax,
        "contact_def": "cb_cb", "ddi_val_every": val_every,
        "n_jobs": len(jobs),
        "written": {"train": 0, "val": 0}, "missing": 0, "mismatch": 0,
        "by_source": {}, "cb_pairs_total": 0, "rows_with_no_pairs": 0,
    }

    def bump(rec, n_pairs, mismatch):
        s = summary["by_source"].setdefault(
            rec["source"], {"train": 0, "val": 0, "pos": 0, "neg": 0,
                            "cb_pairs_total": 0, "rows_with_no_pairs": 0})
        s[rec["split"]] += 1
        s["pos" if rec["label"] == 1 else "neg"] += 1
        s["cb_pairs_total"] += n_pairs
        summary["cb_pairs_total"] += n_pairs
        if rec["label"] == 1 and n_pairs == 0:
            s["rows_with_no_pairs"] += 1
            summary["rows_with_no_pairs"] += 1
        summary["written"][rec["split"]] += 1
        if mismatch:
            summary["mismatch"] += 1

    worker = partial(_process, dmax=args.dmax)
    done = 0
    if args.workers and args.workers > 1:
        pool = Pool(args.workers)
        results = pool.imap_unordered(worker, jobs, chunksize=args.chunksize)
    else:
        pool = None
        results = (worker(j) for j in jobs)

    for res in results:
        done += 1
        if res[0] == "missing":
            _, source, pair_id, path = res
            missing_fh.write(f"{source}\t{pair_id}\t{path}\n")
            summary["missing"] += 1
        else:
            _, rec, mismatch, n_pairs = res
            out[rec["split"]].write(json.dumps(rec) + "\n")
            bump(rec, n_pairs, mismatch)
        if done % 5000 == 0:
            print(f"[preprocess] {done}/{len(jobs)} "
                  f"(train={summary['written']['train']} val={summary['written']['val']} "
                  f"missing={summary['missing']})", flush=True)

    if pool is not None:
        pool.close()
        pool.join()
    for fh in out.values():
        fh.close()
    missing_fh.close()

    with open(os.path.join(args.out_dir, "prepare_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[preprocess] done. train={summary['written']['train']} "
          f"val={summary['written']['val']} missing={summary['missing']} "
          f"mismatch={summary['mismatch']} cb_pairs={summary['cb_pairs_total']}", flush=True)
    print(f"[preprocess] summary -> {os.path.join(args.out_dir, 'prepare_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
