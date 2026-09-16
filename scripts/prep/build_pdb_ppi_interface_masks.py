#!/usr/bin/env python
"""Build geometry-defined interface masks from PDB_PPI positive complexes.

This script intentionally uses only structural geometry from PDB_PPI positive
pairs. It does not use SAE annotations or feature text, so the resulting masks
can be used as an independent reference for interface-grounding audits.

Outputs:
  * pair summary TSV.GZ: one row per positive pair.
  * side mask TSV.GZ: one row per pair side, with residue indices at the interface.
  * unique chain TSV.GZ: sequence/path manifest keyed like the SAE cache (``ppi:<chain>``).
  * JSON summary: aggregate statistics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import glob
import hashlib
import json
import os
import sys
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from conf.audit import CONTACT_DISTANCE_ANGSTROM
from conf.paths import PPI_DATA, RESULTS_RESIDUE

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_DATA_ROOT = PPI_DATA
DEFAULT_OUT_DIR = RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos"

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
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--csv-name", default="posi_nohomo.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--contact-threshold", type=float, default=CONTACT_DISTANCE_ANGSTROM)
    p.add_argument(
        "--report-thresholds",
        default="8,10,12",
        help="comma-separated Cbeta distance thresholds to summarize",
    )
    p.add_argument("--limit", type=int, default=0, help="cap positive rows for a smoke run")
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    p.add_argument("--chunksize", type=int, default=64)
    return p.parse_args()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def encode_indices(indices: list[int]) -> str:
    return ",".join(str(i) for i in indices)


def read_val_clusters(root: Path) -> set[str]:
    path = root / "PDB_PPI" / "posi_val.list"
    if not path.is_file():
        return set()
    with path.open() as fh:
        return {line.strip() for line in fh if line.strip()}


def pdb_chain_to_path(root: Path, chain1: str, chain2: str, which: str) -> Path:
    pdbid = chain1.split("_")[0]
    sub = pdbid[1:3]
    base = root / "PDB_PPI" / "posi_pdbs_extracted" / sub
    path = base / f"{chain1}-{chain2}__{which}.pdb"
    if path.is_file():
        return path
    hits = glob.glob(str(root / "PDB_PPI" / "posi_pdbs_extracted" / "*" / f"{chain1}-{chain2}__{which}.pdb"))
    return Path(hits[0]) if hits else path


def parse_pdb(path: Path):
    """Return (sequence, cb_coords) for a single-chain extracted PDB."""
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


def cb_pairs_under_thresholds(cb_a, cb_b, thresholds: list[float]):
    idx_a = [i for i, c in enumerate(cb_a) if c is not None]
    idx_b = [j for j, c in enumerate(cb_b) if c is not None]
    if not idx_a or not idx_b:
        return {t: 0 for t in thresholds}, [], [], None, len(idx_a), len(idx_b)

    a = np.stack([cb_a[i] for i in idx_a])
    b = np.stack([cb_b[j] for j in idx_b])
    max_t = max(thresholds)
    tree_b = cKDTree(b)
    neighbors = tree_b.query_ball_point(a, r=max_t)

    counts = {t: 0 for t in thresholds}
    iface_a = set()
    iface_b = set()
    min_dist = None
    contact_t = thresholds[0]
    for ka, blist in enumerate(neighbors):
        if not blist:
            continue
        i = idx_a[ka]
        for kb in blist:
            d = float(np.linalg.norm(a[ka] - b[kb]))
            if min_dist is None or d < min_dist:
                min_dist = d
            for t in thresholds:
                if d <= t:
                    counts[t] += 1
            if d <= contact_t:
                iface_a.add(i)
                iface_b.add(idx_b[kb])
    return counts, sorted(iface_a), sorted(iface_b), min_dist, len(idx_a), len(idx_b)


def build_jobs(args: argparse.Namespace, val_clusters: set[str]):
    csv_path = args.data_root / "PDB_PPI" / args.csv_name
    with csv_path.open(newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh)):
            if args.limit and n >= args.limit:
                break
            chain1, chain2 = row["CHAIN1:CHAIN2"].split(":")
            len1, len2 = (int(x) for x in row["LEN1:LEN2"].split(":"))
            cluster = row["CLUSTER"]
            yield {
                "pair_id": row["CHAIN1:CHAIN2"],
                "chain_a": chain1,
                "chain_b": chain2,
                "cluster": cluster,
                "split": "val" if cluster in val_clusters else "train",
                "seqid_pair": row["SEQID1:SEQID2"],
                "taxid": row.get("TAXID", ""),
                "len_a_meta": len1,
                "len_b_meta": len2,
                "pdb_a": str(pdb_chain_to_path(args.data_root, chain1, chain2, chain1)),
                "pdb_b": str(pdb_chain_to_path(args.data_root, chain1, chain2, chain2)),
            }


def process_job(payload):
    job, thresholds = payload
    seq_a, cb_a = parse_pdb(Path(job["pdb_a"]))
    seq_b, cb_b = parse_pdb(Path(job["pdb_b"]))
    if seq_a is None or seq_b is None:
        return {
            "status": "missing",
            "pair": job,
            "missing_path": job["pdb_a"] if seq_a is None else job["pdb_b"],
        }

    counts, iface_a, iface_b, min_dist, valid_a, valid_b = cb_pairs_under_thresholds(cb_a, cb_b, thresholds)
    len_a = len(seq_a)
    len_b = len(seq_b)
    pair = {
        **job,
        "status": "ok",
        "len_a": len_a,
        "len_b": len_b,
        "valid_cb_a": valid_a,
        "valid_cb_b": valid_b,
        "n_iface_a": len(iface_a),
        "n_iface_b": len(iface_b),
        "frac_iface_a": len(iface_a) / len_a if len_a else 0.0,
        "frac_iface_b": len(iface_b) / len_b if len_b else 0.0,
        "min_cb_dist": min_dist,
        "length_mismatch": int(len_a != job["len_a_meta"] or len_b != job["len_b_meta"]),
    }
    for t in thresholds:
        pair[f"n_cb_pairs_le_{format_threshold(t)}"] = counts[t]

    sides = [
        {
            "pair_id": job["pair_id"],
            "cluster": job["cluster"],
            "split": job["split"],
            "taxid": job["taxid"],
            "side": "a",
            "chain_id": job["chain_a"],
            "partner_chain_id": job["chain_b"],
            "cache_key": f"ppi:{job['chain_a']}",
            "length": len_a,
            "n_interface": len(iface_a),
            "interface_fraction": len(iface_a) / len_a if len_a else 0.0,
            "interface_indices": encode_indices(iface_a),
        },
        {
            "pair_id": job["pair_id"],
            "cluster": job["cluster"],
            "split": job["split"],
            "taxid": job["taxid"],
            "side": "b",
            "chain_id": job["chain_b"],
            "partner_chain_id": job["chain_a"],
            "cache_key": f"ppi:{job['chain_b']}",
            "length": len_b,
            "n_interface": len(iface_b),
            "interface_fraction": len(iface_b) / len_b if len_b else 0.0,
            "interface_indices": encode_indices(iface_b),
        },
    ]
    chains = [
        {
            "chain_id": job["chain_a"],
            "cache_key": f"ppi:{job['chain_a']}",
            "length": len_a,
            "sequence_sha1": sha1(seq_a),
            "sequence": seq_a,
            "pdb_path": job["pdb_a"],
        },
        {
            "chain_id": job["chain_b"],
            "cache_key": f"ppi:{job['chain_b']}",
            "length": len_b,
            "sequence_sha1": sha1(seq_b),
            "sequence": seq_b,
            "pdb_path": job["pdb_b"],
        },
    ]
    return {"status": "ok", "pair": pair, "sides": sides, "chains": chains}


def format_threshold(t: float) -> str:
    if float(t).is_integer():
        return str(int(t))
    return str(t).replace(".", "p")


def write_tsv_row(writer: csv.DictWriter, row: dict, fields: list[str]):
    writer.writerow({k: row.get(k, "") for k in fields})


def percentile(values: list[float], q: float):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({args.contact_threshold, *[float(x) for x in args.report_thresholds.split(",") if x]})
    thresholds.remove(args.contact_threshold)
    thresholds.insert(0, args.contact_threshold)

    val_clusters = read_val_clusters(args.data_root)
    jobs = list(build_jobs(args, val_clusters))

    stem = "posi_nohomo"
    if args.limit:
        stem += f".limit{args.limit}"
    pair_path = args.out_dir / f"{stem}.pair_summary.tsv.gz"
    side_path = args.out_dir / f"{stem}.side_interface_masks.tsv.gz"
    chain_path = args.out_dir / f"{stem}.unique_chains.tsv.gz"
    summary_path = args.out_dir / f"{stem}.summary.json"

    pair_fields = [
        "pair_id",
        "cluster",
        "split",
        "seqid_pair",
        "taxid",
        "chain_a",
        "chain_b",
        "len_a_meta",
        "len_b_meta",
        "len_a",
        "len_b",
        "valid_cb_a",
        "valid_cb_b",
        "n_iface_a",
        "n_iface_b",
        "frac_iface_a",
        "frac_iface_b",
        "min_cb_dist",
        "length_mismatch",
    ] + [f"n_cb_pairs_le_{format_threshold(t)}" for t in thresholds]
    side_fields = [
        "pair_id",
        "cluster",
        "split",
        "taxid",
        "side",
        "chain_id",
        "partner_chain_id",
        "cache_key",
        "length",
        "n_interface",
        "interface_fraction",
        "interface_indices",
    ]
    chain_fields = ["chain_id", "cache_key", "length", "sequence_sha1", "sequence", "pdb_path"]

    stats = {
        "input_csv": str(args.data_root / "PDB_PPI" / args.csv_name),
        "n_input_pairs": len(jobs),
        "contact_threshold": args.contact_threshold,
        "report_thresholds": thresholds,
        "n_ok_pairs": 0,
        "n_missing_pairs": 0,
        "n_length_mismatch_pairs": 0,
        "n_pairs_with_interface": 0,
        "missing_examples": [],
    }
    frac_values = []
    min_dist_values = []
    n_iface_values = []
    chain_seen: dict[str, dict] = {}
    chain_conflicts = []

    with gzip.open(pair_path, "wt", newline="") as pair_fh, gzip.open(side_path, "wt", newline="") as side_fh:
        pair_writer = csv.DictWriter(pair_fh, fieldnames=pair_fields, delimiter="\t", lineterminator="\n")
        side_writer = csv.DictWriter(side_fh, fieldnames=side_fields, delimiter="\t", lineterminator="\n")
        pair_writer.writeheader()
        side_writer.writeheader()

        iterator = ((job, thresholds) for job in jobs)
        if args.workers == 1:
            results = map(process_job, iterator)
        else:
            pool = Pool(args.workers)
            results = pool.imap_unordered(process_job, iterator, chunksize=args.chunksize)

        try:
            for i, res in enumerate(results, 1):
                if i % 5000 == 0:
                    print(f"processed {i}/{len(jobs)} pairs", flush=True)
                if res["status"] != "ok":
                    stats["n_missing_pairs"] += 1
                    if len(stats["missing_examples"]) < 20:
                        stats["missing_examples"].append(res)
                    continue

                pair = res["pair"]
                write_tsv_row(pair_writer, pair, pair_fields)
                for side in res["sides"]:
                    write_tsv_row(side_writer, side, side_fields)
                    frac_values.append(float(side["interface_fraction"]))
                    n_iface_values.append(int(side["n_interface"]))
                for chain in res["chains"]:
                    old = chain_seen.get(chain["cache_key"])
                    if old is None:
                        chain_seen[chain["cache_key"]] = chain
                    elif old["sequence_sha1"] != chain["sequence_sha1"]:
                        chain_conflicts.append(
                            {
                                "cache_key": chain["cache_key"],
                                "old_sha1": old["sequence_sha1"],
                                "new_sha1": chain["sequence_sha1"],
                                "old_path": old["pdb_path"],
                                "new_path": chain["pdb_path"],
                            }
                        )

                stats["n_ok_pairs"] += 1
                stats["n_length_mismatch_pairs"] += int(pair["length_mismatch"])
                if int(pair["n_iface_a"]) > 0 or int(pair["n_iface_b"]) > 0:
                    stats["n_pairs_with_interface"] += 1
                if pair["min_cb_dist"] is not None:
                    min_dist_values.append(float(pair["min_cb_dist"]))
        finally:
            if args.workers != 1:
                pool.close()
                pool.join()

    with gzip.open(chain_path, "wt", newline="") as chain_fh:
        chain_writer = csv.DictWriter(chain_fh, fieldnames=chain_fields, delimiter="\t", lineterminator="\n")
        chain_writer.writeheader()
        for chain in sorted(chain_seen.values(), key=lambda x: x["cache_key"]):
            write_tsv_row(chain_writer, chain, chain_fields)

    stats.update(
        {
            "n_unique_chains": len(chain_seen),
            "n_chain_sequence_conflicts": len(chain_conflicts),
            "chain_conflict_examples": chain_conflicts[:20],
            "pair_summary_tsv_gz": str(pair_path),
            "side_interface_masks_tsv_gz": str(side_path),
            "unique_chains_tsv_gz": str(chain_path),
            "side_interface_fraction": {
                "mean": float(np.mean(frac_values)) if frac_values else None,
                "p50": percentile(frac_values, 50),
                "p75": percentile(frac_values, 75),
                "p90": percentile(frac_values, 90),
                "p95": percentile(frac_values, 95),
            },
            "side_n_interface_residues": {
                "mean": float(np.mean(n_iface_values)) if n_iface_values else None,
                "p50": percentile(n_iface_values, 50),
                "p75": percentile(n_iface_values, 75),
                "p90": percentile(n_iface_values, 90),
                "p95": percentile(n_iface_values, 95),
            },
            "pair_min_cb_distance": {
                "mean": float(np.mean(min_dist_values)) if min_dist_values else None,
                "p50": percentile(min_dist_values, 50),
                "p90": percentile(min_dist_values, 90),
                "p95": percentile(min_dist_values, 95),
            },
        }
    )
    with summary_path.open("w") as fh:
        json.dump(stats, fh, indent=2)
        fh.write("\n")
    print(json.dumps(stats, indent=2)[:4000])


if __name__ == "__main__":
    main()
