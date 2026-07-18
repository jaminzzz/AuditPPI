#!/usr/bin/env python
"""Add surface non-interface control residues to PDB_PPI side masks.

Surface residues are called from single-chain SASA using MDTraj Shrake-Rupley.
The output keeps all original side-mask columns and appends:

  * surface_indices
  * surface_noninterface_indices
  * n_surface
  * n_surface_noninterface

This provides a cleaner control set for interface-grounded SAE enrichment:
interface residues are compared against residues that are solvent-exposed in
the same chain but not part of the observed partner interface.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from conf.audit import SURFACE_RSASA_THRESHOLD
from conf.paths import RESULTS_RESIDUE

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

# Maximum solvent accessibility in A^2, used for relative SASA.
# Values are standard extended-state residue maxima; the exact table has little
# effect because we threshold coarsely at rSASA >= 0.20.
MAX_ASA = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "Q": 214.0,
    "E": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0,
    "X": 180.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side-masks", type=Path, required=True)
    p.add_argument("--unique-chains", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_surface")
    p.add_argument("--rsasa-threshold", type=float, default=SURFACE_RSASA_THRESHOLD)
    p.add_argument("--n-sphere-points", type=int, default=960)
    p.add_argument("--no-sanitize-pdb", action="store_true", help="load raw PDBs directly into MDTraj")
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    p.add_argument("--limit-chains", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=16)
    return p.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def open_write_text(path: Path):
    return gzip.open(path, "wt", newline="") if path.name.endswith(".gz") else path.open("w", newline="")


def encode_indices(indices) -> str:
    return ",".join(str(int(i)) for i in indices)


def parse_indices(text: str) -> np.ndarray:
    if not text:
        return np.asarray([], dtype=np.int64)
    return np.fromiter((int(x) for x in text.split(",") if x), dtype=np.int64)


def read_chains(path: Path, limit: int):
    rows = []
    with open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def compute_surface(job):
    import mdtraj as md

    cache_key = job["cache_key"]
    pdb_path = job["pdb_path"]
    expected_len = int(job["length"])
    load_path = pdb_path
    tmp_path = None
    try:
        if job.get("sanitize_pdb", True):
            tmp_path = sanitize_pdb_for_sasa(pdb_path)
            load_path = tmp_path
        traj = md.load(load_path)
        sasa_nm2 = md.shrake_rupley(
            traj,
            mode="residue",
            n_sphere_points=job["n_sphere_points"],
        )[0]
    except Exception as exc:
        return {
            "cache_key": cache_key,
            "status": "error",
            "error": repr(exc),
            "surface_indices": "",
            "n_residues": 0,
            "expected_len": expected_len,
        }
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    aas = []
    for res in traj.topology.residues:
        aas.append(THREE2ONE.get(res.name.strip().upper(), "X"))
    L = min(len(aas), len(sasa_nm2), expected_len)
    max_asa = np.asarray([MAX_ASA.get(aa, MAX_ASA["X"]) for aa in aas[:L]], dtype=np.float64)
    rsasa = (np.asarray(sasa_nm2[:L], dtype=np.float64) * 100.0) / max_asa
    surface = np.flatnonzero(rsasa >= job["rsasa_threshold"])
    return {
        "cache_key": cache_key,
        "status": "ok",
        "surface_indices": encode_indices(surface),
        "n_surface": int(surface.size),
        "surface_fraction": float(surface.size / L) if L else 0.0,
        "n_residues": int(L),
        "expected_len": expected_len,
        "mdtraj_n_residues": int(traj.n_residues),
        "length_mismatch": int(L != expected_len or traj.n_residues != expected_len),
    }


def sanitize_pdb_for_sasa(pdb_path: str) -> str:
    """Write a temporary PDB with duplicate/alternate atoms removed.

    Some extracted PDB chains contain alternate locations or duplicate atom
    records with identical coordinates. MDTraj's Shrake-Rupley implementation
    can abort on exactly overlapping atoms, so keep one atom per
    (chain, residue, atom name) and one atom per rounded coordinate.
    """
    seen_atoms = set()
    seen_xyz = set()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", prefix="sasa_", delete=False)
    with open(pdb_path) as fh, tmp:
        serial = 1
        for line in fh:
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            altloc = line[16]
            if altloc not in (" ", "A", "1"):
                continue
            atom_name = line[12:16].strip()
            if not atom_name or atom_name.upper().startswith("H"):
                continue
            try:
                xyz = (
                    round(float(line[30:38]), 3),
                    round(float(line[38:46]), 3),
                    round(float(line[46:54]), 3),
                )
            except ValueError:
                continue
            atom_key = (line[21], line[22:26].strip(), line[26], atom_name)
            if atom_key in seen_atoms or xyz in seen_xyz:
                continue
            seen_atoms.add(atom_key)
            seen_xyz.add(xyz)
            clean = f"{line[:6]}{serial:5d}{line[11:16]} {line[17:]}"
            tmp.write(clean.rstrip("\n") + "\n")
            serial += 1
        tmp.write("END\n")
    return tmp.name


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "posi_nohomo"
    if args.limit_chains:
        stem += f".limit{args.limit_chains}"
    surface_manifest = args.out_dir / f"{stem}.chain_surface.tsv.gz"
    side_out = args.out_dir / f"{stem}.side_surface_controls.tsv.gz"
    summary_out = args.out_dir / f"{stem}.summary.json"

    chain_rows = read_chains(args.unique_chains, args.limit_chains)
    jobs = [
        {
            "cache_key": row["cache_key"],
            "pdb_path": row["pdb_path"],
            "length": row["length"],
            "rsasa_threshold": args.rsasa_threshold,
            "n_sphere_points": args.n_sphere_points,
            "sanitize_pdb": not args.no_sanitize_pdb,
        }
        for row in chain_rows
    ]

    surface_by_key = {}
    errors = []
    if args.workers == 1:
        results = map(compute_surface, jobs)
        pool = None
    else:
        pool = Pool(args.workers)
        results = pool.imap_unordered(compute_surface, jobs, chunksize=args.chunksize)
    try:
        for i, res in enumerate(results, 1):
            if i % 5000 == 0:
                print(f"computed SASA {i}/{len(jobs)} chains", flush=True)
            surface_by_key[res["cache_key"]] = res
            if res["status"] != "ok" and len(errors) < 50:
                errors.append(res)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    surface_fields = [
        "cache_key",
        "status",
        "n_residues",
        "expected_len",
        "mdtraj_n_residues",
        "length_mismatch",
        "n_surface",
        "surface_fraction",
        "surface_indices",
        "error",
    ]
    with open_write_text(surface_manifest) as fh:
        writer = csv.DictWriter(fh, fieldnames=surface_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for key in sorted(surface_by_key):
            row = surface_by_key[key]
            writer.writerow({k: row.get(k, "") for k in surface_fields})

    n_side_rows = n_side_with_surface_control = 0
    n_missing_surface = 0
    with open_text(args.side_masks) as in_fh, open_write_text(side_out) as out_fh:
        reader = csv.DictReader(in_fh, delimiter="\t")
        side_fields = list(reader.fieldnames or []) + [
            "surface_indices",
            "surface_noninterface_indices",
            "n_surface",
            "n_surface_noninterface",
            "surface_fraction",
            "surface_noninterface_fraction",
        ]
        writer = csv.DictWriter(out_fh, fieldnames=side_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            n_side_rows += 1
            surf_row = surface_by_key.get(row["cache_key"])
            if surf_row is None or surf_row.get("status") != "ok":
                n_missing_surface += 1
                surface = np.asarray([], dtype=np.int64)
            else:
                surface = parse_indices(surf_row.get("surface_indices", ""))
            iface = parse_indices(row.get("interface_indices", ""))
            if surface.size and iface.size:
                iface_set = set(int(x) for x in iface)
                control = np.asarray([int(x) for x in surface if int(x) not in iface_set], dtype=np.int64)
            else:
                control = surface
            length = int(row["length"])
            row["surface_indices"] = encode_indices(surface)
            row["surface_noninterface_indices"] = encode_indices(control)
            row["n_surface"] = int(surface.size)
            row["n_surface_noninterface"] = int(control.size)
            row["surface_fraction"] = float(surface.size / length) if length else 0.0
            row["surface_noninterface_fraction"] = float(control.size / length) if length else 0.0
            n_side_with_surface_control += int(control.size > 0)
            writer.writerow({k: row.get(k, "") for k in side_fields})

    ok = [r for r in surface_by_key.values() if r.get("status") == "ok"]
    summary = {
        "side_masks": str(args.side_masks),
        "unique_chains": str(args.unique_chains),
        "rsasa_threshold": args.rsasa_threshold,
        "n_sphere_points": args.n_sphere_points,
        "sanitize_pdb": not args.no_sanitize_pdb,
        "n_chains_input": len(chain_rows),
        "n_chains_ok": len(ok),
        "n_chains_error": len(chain_rows) - len(ok),
        "n_chain_length_mismatch": sum(int(r.get("length_mismatch", 0)) for r in ok),
        "n_side_rows": n_side_rows,
        "n_side_missing_surface": n_missing_surface,
        "n_side_with_surface_control": n_side_with_surface_control,
        "chain_surface_tsv_gz": str(surface_manifest),
        "side_surface_controls_tsv_gz": str(side_out),
        "error_examples": errors,
    }
    with summary_out.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
