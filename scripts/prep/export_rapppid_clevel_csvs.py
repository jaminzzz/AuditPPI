#!/usr/bin/env python3
"""Export RAPPPID C1/C2/C3 splits from the h5 store into (query,text,label) CSVs.

The RAPPPID h5 (``RAPPPID_H5``) holds all three Park & Marcotte leakage levels under
``interactions/{c1,c2,c3}/{level}_{split}`` as (protein_id1, protein_id2, label)
rows, plus a shared ``sequences`` table (name=UniProt accession, sequence). The
audit pipeline consumes flat ``(query,text,label)`` CSVs where query/text are the
raw SEQUENCES (this is the schema of the existing ``data/raw/rapppid_c3/c3.*.csv``).

This script reproduces that schema for any requested level, so C1/C2 become
first-class benchmarks alongside the pre-existing C3 set. Output layout mirrors
the C3 directory:

    data/raw/rapppid_c1/c1.{train,val,test}.csv
    data/raw/rapppid_c2/c2.{train,val,test}.csv
    data/raw/rapppid_c3/c3.{train,val,test}.csv   (regenerated only if --levels c3)

Each id is resolved to its sequence via the h5 ``sequences`` table. Sequences in
the store are fixed-width ``S3000``; proteins longer than 3000 aa are truncated
AT SOURCE (a pre-existing property inherited by the C3 CSVs, not introduced here).

Run in E1 (includes h5py + hdf5plugin):
    /data/wmzhu/anaconda3/envs/E1/bin/python \
        scripts/prep/export_rapppid_clevel_csvs.py --levels c1 c2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from conf.paths import RAPPPID_H5, RAPPPID_CLEVEL_CSVS


def _load_id2seq(h5_path: Path) -> dict[str, str]:
    """Build UniProt id -> sequence map from the h5 ``sequences`` table."""
    import hdf5plugin  # noqa: F401  (registers the blosc filter)

    os.environ.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGINS_PATH)
    import h5py

    id2seq: dict[str, str] = {}
    with h5py.File(h5_path, "r") as h:
        for r in h["sequences"]:
            id2seq[r["name"].decode()] = r["sequence"].decode()
    return id2seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=RAPPPID_H5)
    ap.add_argument("--levels", nargs="+", default=["c1", "c2"],
                    choices=["c1", "c2", "c3"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    import hdf5plugin  # noqa: F401
    os.environ.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGINS_PATH)
    import h5py
    import pandas as pd

    id2seq = _load_id2seq(args.h5)
    print(f"[h5] sequences table: {len(id2seq)} proteins")

    with h5py.File(args.h5, "r") as h:
        for level in args.levels:
            csv_map = RAPPPID_CLEVEL_CSVS[level]
            out_dir = csv_map["train"].parent
            out_dir.mkdir(parents=True, exist_ok=True)
            for split in args.splits:
                ds = h[f"interactions/{level}/{level}_{split}"]
                ids_a = [r["protein_id1"].decode() for r in ds]
                ids_b = [r["protein_id2"].decode() for r in ds]
                labels = [int(r["label"]) for r in ds]

                missing = {i for i in (*ids_a, *ids_b) if i not in id2seq}
                if missing:
                    raise KeyError(
                        f"[{level}/{split}] {len(missing)} ids absent from h5 sequences table"
                    )

                df = pd.DataFrame(
                    {
                        "query": [id2seq[i] for i in ids_a],
                        "text": [id2seq[i] for i in ids_b],
                        "label": labels,
                    }
                )
                dst = csv_map[split]
                df.to_csv(dst, index=False)
                pos = int(df["label"].sum())
                print(
                    f"[{level}/{split}] n={len(df)} pos={pos} neg={len(df) - pos} "
                    f"pos_rate={pos / len(df):.4f} -> {dst}"
                )


if __name__ == "__main__":
    main()
