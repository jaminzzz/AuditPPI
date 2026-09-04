#!/usr/bin/env python
"""Quantify the impact of the Cbeta definition on PDB_PPI interface masks.

Compares two conventions on the same complexes:
  * current   -- reconstruct virtual Cbeta whenever N/CA/C present, else deposited CB, else CA
                 (scripts/prep/build_pdb_ppi_interface_masks.py:_reconstruct_cb)
  * standard  -- deposited CB for non-Gly, virtual Cbeta for Gly (and for non-Gly with no CB),
                 else CA

Reports per-residue coordinate deviation and the resulting interface-mask
symmetric difference at the 8A Cbeta threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conf.audit import CONTACT_DISTANCE_ANGSTROM
from conf.paths import PPI_DATA
from scripts.prep.build_pdb_ppi_interface_masks import THREE2ONE, pdb_chain_to_path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def virtual_cb(n, ca, c):
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def parse_pdb_both(path: Path):
    """Return (seq, cb_current, cb_standard, provenance counters)."""
    if not path.is_file():
        return None
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
        return None
    if not residues:
        return None

    seq = "".join(v[0] for v in residues.values())
    cb_cur, cb_std, devs, prov = [], [], [], Counter()

    for aa, atoms in residues.values():
        has_bb = "N" in atoms and "CA" in atoms and "C" in atoms
        vcb = None
        if has_bb:
            vcb = virtual_cb(
                np.asarray(atoms["N"], dtype=np.float64),
                np.asarray(atoms["CA"], dtype=np.float64),
                np.asarray(atoms["C"], dtype=np.float64),
            ).astype(np.float32)
        dcb = np.asarray(atoms["CB"], dtype=np.float32) if "CB" in atoms else None
        ca = np.asarray(atoms["CA"], dtype=np.float32) if "CA" in atoms else None

        # current: virtual first, deposited fallback, CA last
        cur = vcb if vcb is not None else (dcb if dcb is not None else ca)
        # standard: deposited for non-Gly, virtual for Gly / missing-CB, CA last
        if aa == "G":
            std = vcb if vcb is not None else ca
        else:
            std = dcb if dcb is not None else (vcb if vcb is not None else ca)

        if vcb is None and dcb is not None:
            prov["fallback_deposited_no_backbone"] += 1
        if aa != "G" and dcb is None:
            prov["nongly_missing_cb"] += 1
        if aa == "G":
            prov["gly"] += 1
        if vcb is not None and dcb is not None and aa != "G":
            devs.append(float(np.linalg.norm(vcb - dcb)))
            prov["nongly_both_available"] += 1

        cb_cur.append(cur)
        cb_std.append(std)

    return seq, cb_cur, cb_std, devs, prov


def interface_sets(cb_a, cb_b, thr: float):
    idx_a = [i for i, c in enumerate(cb_a) if c is not None]
    idx_b = [j for j, c in enumerate(cb_b) if c is not None]
    if not idx_a or not idx_b:
        return set(), set()
    a = np.stack([cb_a[i] for i in idx_a])
    b = np.stack([cb_b[j] for j in idx_b])
    pairs = cKDTree(a).query_ball_tree(cKDTree(b), r=thr)
    iface_a, iface_b = set(), set()
    for ka, blist in enumerate(pairs):
        if blist:
            iface_a.add(idx_a[ka])
            iface_b.update(idx_b[kb] for kb in blist)
    return iface_a, iface_b


def process(payload):
    job, thr = payload
    ra = parse_pdb_both(Path(job["pdb_a"]))
    rb = parse_pdb_both(Path(job["pdb_b"]))
    if ra is None or rb is None:
        return None
    seq_a, cur_a, std_a, dev_a, prov_a = ra
    seq_b, cur_b, std_b, dev_b, prov_b = rb

    cur_ia, cur_ib = interface_sets(cur_a, cur_b, thr)
    std_ia, std_ib = interface_sets(std_a, std_b, thr)

    out = {"devs": dev_a + dev_b, "prov": prov_a + prov_b, "sides": []}
    for cur_set, std_set, seq in ((cur_ia, std_ia, seq_a), (cur_ib, std_ib, seq_b)):
        union = cur_set | std_set
        inter = cur_set & std_set
        added = std_set - cur_set  # residues the standard definition gains
        removed = cur_set - std_set  # residues the standard definition loses
        out["sides"].append(
            {
                "n_res": len(seq),
                "n_cur": len(cur_set),
                "n_std": len(std_set),
                "n_added": len(added),
                "n_removed": len(removed),
                "jaccard": len(inter) / len(union) if union else 1.0,
                "identical": int(cur_set == std_set),
                "flip_aa": Counter(seq[i] for i in (added | removed) if i < len(seq)),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=PPI_DATA)
    ap.add_argument("--csv-name", default="posi_nohomo.csv")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=0, help="take every Nth row instead of the first N")
    ap.add_argument("--threshold", type=float, default=CONTACT_DISTANCE_ANGSTROM)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    csv_path = args.data_root / "PDB_PPI" / args.csv_name
    jobs = []
    with csv_path.open(newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh)):
            if args.stride and n % args.stride:
                continue
            chain1, chain2 = row["CHAIN1:CHAIN2"].split(":")
            jobs.append(
                {
                    "pdb_a": str(pdb_chain_to_path(args.data_root, chain1, chain2, chain1)),
                    "pdb_b": str(pdb_chain_to_path(args.data_root, chain1, chain2, chain2)),
                }
            )
            if args.limit and len(jobs) >= args.limit:
                break

    print(f"[cb-diff] {len(jobs)} pairs, threshold {args.threshold}A", flush=True)
    with Pool(args.workers) as pool:
        results = [r for r in pool.imap_unordered(process, ((j, args.threshold) for j in jobs), 32) if r]

    devs = np.asarray([d for r in results for d in r["devs"]], dtype=np.float64)
    prov = Counter()
    for r in results:
        prov += r["prov"]
    sides = [s for r in results for s in r["sides"]]

    n_sides = len(sides)
    identical = sum(s["identical"] for s in sides)
    jac = np.asarray([s["jaccard"] for s in sides])
    n_cur = np.asarray([s["n_cur"] for s in sides], dtype=np.float64)
    n_std = np.asarray([s["n_std"] for s in sides], dtype=np.float64)
    added = np.asarray([s["n_added"] for s in sides], dtype=np.float64)
    removed = np.asarray([s["n_removed"] for s in sides], dtype=np.float64)
    flip_aa = Counter()
    for s in sides:
        flip_aa += s["flip_aa"]

    summary = {
        "n_pairs": len(results),
        "n_sides": n_sides,
        "threshold": args.threshold,
        "residue_provenance": dict(prov),
        "virtual_vs_deposited_cb_deviation_A": {
            "n": int(devs.size),
            "mean": float(devs.mean()) if devs.size else None,
            "p50": float(np.percentile(devs, 50)) if devs.size else None,
            "p90": float(np.percentile(devs, 90)) if devs.size else None,
            "p99": float(np.percentile(devs, 99)) if devs.size else None,
            "max": float(devs.max()) if devs.size else None,
        },
        "interface_mask": {
            "frac_sides_identical": identical / n_sides if n_sides else None,
            "jaccard_mean": float(jac.mean()),
            "jaccard_p05": float(np.percentile(jac, 5)),
            "jaccard_min": float(jac.min()),
            "mean_n_iface_current": float(n_cur.mean()),
            "mean_n_iface_standard": float(n_std.mean()),
            "mean_added_per_side": float(added.mean()),
            "mean_removed_per_side": float(removed.mean()),
            "total_iface_residues_current": int(n_cur.sum()),
            "total_iface_residues_standard": int(n_std.sum()),
            "frac_residues_flipped": float((added.sum() + removed.sum()) / max(1.0, n_cur.sum())),
        },
        "flipped_residue_aa_composition": dict(flip_aa.most_common()),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[cb-diff] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
