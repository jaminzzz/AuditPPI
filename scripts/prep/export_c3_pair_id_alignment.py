#!/usr/bin/env python3
"""Export a row-aligned (uniprot_a, uniprot_b, label) table for each C3 split.

This is the shared foundation for the negative-sampling audit. The SAE reps
(``{split}_embeddings.pt``, keys ``emb_a/emb_b/label``) are built row-for-row
from ``rapppid-c3/c3.{split}.csv`` (columns ``query``/``text``/``label`` = raw
SEQUENCES), while the RAPPPID h5 carries the UniProt protein ids. We bridge the
two via the h5 ``sequences`` table (name=UniProt accession, sequence), which is a
clean sequence->id bijection (verified: 0 collisions). Each CSV row's
(query_seq, text_seq) therefore maps to a unique (id_a, id_b) pair, giving every
reps row a stable protein identity WITHOUT relying on the h5 pair-row order.

The emitted parquet has one row per reps row, in reps/CSV order:
    row, id_a, id_b, label
so downstream audits can attach per-protein annotations (degree, localization)
and slice the cached SAE fingerprints by the same row index.

Run (needs hdf5plugin -> genmol env):
    /data/wmzhu/anaconda3/envs/genmol/bin/python scripts/export_c3_pair_id_alignment.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from conf.paths import RAPPPID_C3_DIR as CSV_DIR, C3_H5, AUDIT  # noqa: E402

OUT_DIR = AUDIT / "negative_sampling_audit"


def _load_seq2id(h5_path: Path) -> dict[str, str]:
    """Build sequence -> UniProt id map from the h5 ``sequences`` table.

    Raises if any sequence maps to more than one id (would break the bridge)."""
    import hdf5plugin  # noqa: F401  (registers the blosc filter)

    os.environ.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGINS_PATH)
    import h5py

    seq2id: dict[str, str] = {}
    collisions = 0
    with h5py.File(h5_path, "r") as h:
        for r in h["sequences"]:
            name = r["name"].decode()
            seq = r["sequence"].decode()
            if seq in seq2id and seq2id[seq] != name:
                collisions += 1
            seq2id[seq] = name
    if collisions:
        raise RuntimeError(f"{collisions} sequences map to >1 UniProt id; bridge is not a bijection")
    return seq2id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    ap.add_argument("--csv-pattern", default="c3.{split}.csv")
    ap.add_argument("--h5", type=Path, default=C3_H5)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--col-a", default="query")
    ap.add_argument("--col-b", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    import pandas as pd

    seq2id = _load_seq2id(args.h5)
    print(f"[h5] sequences table: {len(seq2id)} unique seqs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        csv = args.csv_dir / args.csv_pattern.format(split=split)
        df = pd.read_csv(csv, usecols=[args.col_a, args.col_b, args.label_col])
        sa = df[args.col_a].astype(str)
        sb = df[args.col_b].astype(str)
        missing = set(sa).union(sb).difference(seq2id)
        if missing:
            raise KeyError(f"[{split}] {len(missing)} sequences absent from h5 sequences table")
        out = pd.DataFrame(
            {
                "row": range(len(df)),
                "id_a": [seq2id[s] for s in sa],
                "id_b": [seq2id[s] for s in sb],
                "label": df[args.label_col].astype(int).to_numpy(),
            }
        )
        dst = args.out_dir / f"c3_{split}_pair_ids.parquet"
        out.to_parquet(dst, index=False)
        pos = int(out["label"].sum())
        print(
            f"[{split}] n={len(out)} pos={pos} neg={len(out) - pos} "
            f"pos_rate={pos / len(out):.4f} n_proteins={len(set(out['id_a']).union(out['id_b']))} -> {dst}"
        )


if __name__ == "__main__":
    main()
