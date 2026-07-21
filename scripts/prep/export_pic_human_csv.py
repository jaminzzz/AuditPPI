#!/usr/bin/env python3
"""Export the PIC human essentiality set from its pickle into a flat CSV.

PIC (Protein Importance Calculator) ships one legacy-numpy pickle per dataset
with columns ``ID`` (Ensembl ENSP accession), ``sequence``, and a per-dataset
binary label column (``human`` for the human set). The v1 protein-cache slicer
(``scripts/prep/slice_dataset_protein_cache.py``) reads FASTA/CSV/TSV only, so
this step lowers the pickle into the flat schema the slicer consumes:

    data/raw/pic/pic_human.csv    columns: ID, sequence, label

The read reuses ``src.data.proteins.load_pic`` so the ID/sequence/label contract
lives in exactly one place (and the numpy-core shim for the old pickle stays
there). Downstream, slice the v1 cache with the real ENSP ids:

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/prep/slice_dataset_protein_cache.py \
        --input data/raw/pic/pic_human.csv --sequence-cols sequence --id-cols ID \
        --esmc-pooled data/sae/seq_caches/pooled_esmc_l60_l80_max1022_features.pt \
        --esm2-pooled data/sae/seq_caches/pooled_esm2_l33_max1022_features.pt \
        --output data/sae/protein_caches/pic_human_protein_features_max1022.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from conf.paths import PIC_HUMAN_CSV
from src.data.proteins import load_pic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=PIC_HUMAN_CSV)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite to replace it")

    import pandas as pd

    dataset = load_pic("human")
    rows = {
        "ID": dataset.ids,
        "sequence": [dataset.seqs[identifier] for identifier in dataset.ids],
        "label": dataset.labels.tolist(),
    }
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    n_pos = int(frame["label"].sum())
    print(
        f"[pic_human] n={len(frame)} pos={n_pos} neg={len(frame) - n_pos} "
        f"pos_rate={n_pos / len(frame):.4f} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
