#!/usr/bin/env python3
"""Collect unique protein sequences across all AuditPPI datasets.

Builds a sequence-keyed pooled manifest for the shared feature pool:

    sequence -> sorted list of source datasets that contain it

Output (default under data/sae/seq_caches/):
  - pooled_sequence_manifest.parquet   one row per unique normalized sequence
  - pooled_sequence_manifest.tsv.gz    same content, easy to inspect
  - pooled_sequence_manifest.summary.json

Columns
-------
sequence
    Normalized protein sequence (uppercase, whitespace stripped).
seq_id
    Stable hash id (``seq_`` + sha1 prefix).
length
    Sequence length after normalization.
sources
    Semicolon-separated sorted dataset tags (a sequence may belong to many).
n_sources
    Number of distinct source tags.
example_ids
    Up to N protein/endpoint ids observed for this sequence (semicolon-separated).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from conf.paths import (
    BERNETT_DIR,
    BERNETT_SPLIT_CSVS,
    CROSS_SPECIES_CSVS,
    CROSS_SPECIES_DIR,
    PIC_DATASET_PKL,
    PRING_ROOT,
    RAPPPID_CLEVEL_CSVS,
    RF2PPI_FASTA,
    SEQ_CACHES,
)
from src.data.pairs import _bernett_clean_seq
from src.data.proteins import _read_legacy_dataframe
from src.data.sequences import iter_fasta, normalize_sequence, sequence_id


# ---------------------------------------------------------------------------
# Source collectors: each yields (source_tag, protein_id | None, sequence)
# ---------------------------------------------------------------------------

def _iter_pair_csv(
    path: Path,
    *,
    source: str,
    seq_cols: tuple[str, str],
    id_cols: tuple[str, str] | None = None,
) -> list[tuple[str, str | None, str]]:
    if not path.is_file():
        print(f"[skip] missing {path}", flush=True)
        return []
    rows: list[tuple[str, str | None, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        for col in seq_cols:
            if col not in fields:
                raise ValueError(f"{path}: missing column {col!r}; have {sorted(fields)}")
        for row in reader:
            for i, col in enumerate(seq_cols):
                seq = row.get(col, "")
                pid = row.get(id_cols[i]) if id_cols is not None else None
                rows.append((source, pid, seq))
    return rows


def collect_rapppid() -> list[tuple[str, str | None, str]]:
    out: list[tuple[str, str | None, str]] = []
    for level, splits in RAPPPID_CLEVEL_CSVS.items():
        source = f"rapppid_{level}"
        for split, path in splits.items():
            # CSVs store raw sequences in query/text (no separate protein ids).
            out.extend(
                _iter_pair_csv(
                    Path(path),
                    source=source,
                    seq_cols=("query", "text"),
                )
            )
            print(f"[read] {source}/{split}: {path}", flush=True)
    return out


def collect_cross_species() -> list[tuple[str, str | None, str]]:
    out: list[tuple[str, str | None, str]] = []
    for name in CROSS_SPECIES_CSVS:
        path = CROSS_SPECIES_DIR / name
        # Keep one family tag so multi-source joins stay readable; file name is
        # recoverable from provenance in the summary.
        out.extend(
            _iter_pair_csv(
                path,
                source="cross_species",
                seq_cols=("query", "text"),
            )
        )
        print(f"[read] cross_species/{name}", flush=True)
    return out


def collect_bernett() -> list[tuple[str, str | None, str]]:
    out: list[tuple[str, str | None, str]] = []
    for split, filename in BERNETT_SPLIT_CSVS.items():
        path = BERNETT_DIR / filename
        if not path.is_file():
            print(f"[skip] missing {path}", flush=True)
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for col in ("seq1", "seq2"):
                    raw = row.get(col, "")
                    # Match the Bernett loader cleaning before normalization.
                    cleaned = _bernett_clean_seq(raw)
                    out.append(("bernett", None, cleaned))
        print(f"[read] bernett/{split}: {path}", flush=True)
    return out


def collect_rf2ppi() -> list[tuple[str, str | None, str]]:
    """RF2-PPI formal benchmark proteins only (not the raw sequence lookup FASTA)."""
    out: list[tuple[str, str | None, str]] = []
    path = Path(RF2PPI_FASTA)
    if not path.is_file():
        print(f"[skip] missing {path}", flush=True)
        return out
    n = 0
    for pid, seq in iter_fasta(path):
        out.append(("rf2ppi", pid, seq))
        n += 1
    print(f"[read] rf2ppi: {path} ({n} records)", flush=True)
    return out


def collect_pring() -> list[tuple[str, str | None, str]]:
    out: list[tuple[str, str | None, str]] = []
    for species in ("human", "yeast", "ecoli", "arath"):
        source = f"pring_{species}"
        # Prefer the simple FASTA used by the SAE cache builders.
        fasta = PRING_ROOT / species / f"{species}_simple.fasta"
        csv_path = PRING_ROOT / species / f"{species}_protein_id.csv"
        if fasta.is_file():
            n = 0
            for pid, seq in iter_fasta(fasta):
                out.append((source, pid, seq))
                n += 1
            print(f"[read] {source} fasta: {fasta} ({n})", flush=True)
        elif csv_path.is_file():
            n = 0
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                # Common PRING columns: uniprot_id, sequence (tolerate aliases).
                fields = set(reader.fieldnames or [])
                id_col = next(
                    (c for c in ("uniprot_id", "protein_id", "id", "ID") if c in fields),
                    None,
                )
                seq_col = next(
                    (c for c in ("sequence", "seq", "Sequence") if c in fields),
                    None,
                )
                if seq_col is None:
                    raise ValueError(f"{csv_path}: no sequence column in {sorted(fields)}")
                for row in reader:
                    out.append((source, row.get(id_col) if id_col else None, row.get(seq_col, "")))
                    n += 1
            print(f"[read] {source} csv: {csv_path} ({n})", flush=True)
        else:
            print(f"[skip] missing PRING sequences for {species}", flush=True)
    return out


def collect_pic() -> list[tuple[str, str | None, str]]:
    out: list[tuple[str, str | None, str]] = []
    for dataset, path in PIC_DATASET_PKL.items():
        path = Path(path)
        if not path.is_file():
            print(f"[skip] missing {path}", flush=True)
            continue
        frame = _read_legacy_dataframe(path)
        source = f"pic_{dataset}"
        n = 0
        for pid, seq in zip(frame["ID"].tolist(), frame["sequence"].tolist()):
            out.append((source, str(pid), str(seq)))
            n += 1
        print(f"[read] {source}: {path} ({n})", flush=True)
    return out


COLLECTORS = (
    ("rapppid", collect_rapppid),
    ("cross_species", collect_cross_species),
    ("bernett", collect_bernett),
    ("rf2ppi", collect_rf2ppi),
    ("pring", collect_pring),
    ("pic", collect_pic),
)


def build_manifest(
    records: list[tuple[str, str | None, str]],
    *,
    max_example_ids: int = 5,
) -> list[dict]:
    """Collapse (source, id, seq) rows into unique-sequence records."""
    sources_by_seq: dict[str, set[str]] = defaultdict(set)
    ids_by_seq: dict[str, list[str]] = defaultdict(list)
    id_seen: dict[str, set[str]] = defaultdict(set)
    n_empty = 0
    n_raw = 0

    for source, pid, raw_seq in records:
        n_raw += 1
        seq = normalize_sequence(raw_seq)
        if not seq:
            n_empty += 1
            continue
        sources_by_seq[seq].add(source)
        if pid is not None:
            token = str(pid).strip()
            if token and token not in id_seen[seq]:
                if len(ids_by_seq[seq]) < max_example_ids:
                    ids_by_seq[seq].append(token)
                id_seen[seq].add(token)

    rows: list[dict] = []
    for seq in sorted(sources_by_seq.keys(), key=lambda s: (len(s), s)):
        sources = sorted(sources_by_seq[seq])
        rows.append(
            {
                "seq_id": sequence_id(seq),
                "sequence": seq,
                "length": len(seq),
                "sources": ";".join(sources),
                "n_sources": len(sources),
                "example_ids": ";".join(ids_by_seq.get(seq, [])),
            }
        )
    return rows, {"n_raw_endpoint_records": n_raw, "n_empty_after_normalize": n_empty}


def write_outputs(rows: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "pooled_sequence_manifest.parquet"
    tsv_path = out_dir / "pooled_sequence_manifest.tsv.gz"
    summary_path = out_dir / "pooled_sequence_manifest.summary.json"

    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(tsv_path, sep="\t", index=False, compression="gzip")
        print(f"[saved] {parquet_path}  rows={len(frame)}", flush=True)
        print(f"[saved] {tsv_path}", flush=True)
    except Exception as exc:  # pragma: no cover - fallback without pyarrow/pandas quirks
        print(f"[warn] parquet/tsv via pandas failed ({exc}); writing plain TSV", flush=True)
        tsv_plain = out_dir / "pooled_sequence_manifest.tsv"
        with tsv_plain.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["seq_id", "sequence", "length", "sources", "n_sources", "example_ids"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {tsv_plain}", flush=True)

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[saved] {summary_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SEQ_CACHES,
        help="directory for pooled_sequence_manifest.* (default: data/sae/seq_caches)",
    )
    parser.add_argument(
        "--max-example-ids",
        type=int,
        default=5,
        help="cap on example protein ids retained per sequence",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[name for name, _ in COLLECTORS],
        help="optional subset of collector families to include",
    )
    args = parser.parse_args(argv)

    selected = set(args.only) if args.only else {name for name, _ in COLLECTORS}
    records: list[tuple[str, str | None, str]] = []
    per_family_raw: dict[str, int] = {}
    for name, fn in COLLECTORS:
        if name not in selected:
            continue
        print(f"\n=== collect {name} ===", flush=True)
        family_records = fn()
        per_family_raw[name] = len(family_records)
        records.extend(family_records)

    rows, build_stats = build_manifest(records, max_example_ids=args.max_example_ids)

    # Per-source unique sequence counts and multi-source histogram.
    source_counts: dict[str, int] = defaultdict(int)
    multi_hist: dict[int, int] = defaultdict(int)
    for row in rows:
        multi_hist[row["n_sources"]] += 1
        for source in row["sources"].split(";"):
            source_counts[source] += 1

    summary = {
        "n_unique_sequences": len(rows),
        "n_raw_endpoint_records": build_stats["n_raw_endpoint_records"],
        "n_empty_after_normalize": build_stats["n_empty_after_normalize"],
        "per_family_raw_records": per_family_raw,
        "per_source_unique_sequences": dict(sorted(source_counts.items())),
        "n_sources_histogram": {str(k): multi_hist[k] for k in sorted(multi_hist)},
        "sources": sorted(source_counts),
        "out_dir": str(args.out_dir.resolve()),
    }

    print("\n=== summary ===", flush=True)
    print(f"unique sequences: {summary['n_unique_sequences']}", flush=True)
    print(f"raw endpoint records: {summary['n_raw_endpoint_records']}", flush=True)
    print("per-source unique sequences:", flush=True)
    for source, n in summary["per_source_unique_sequences"].items():
        print(f"  {source:28s} {n:7d}", flush=True)
    print("n_sources histogram:", summary["n_sources_histogram"], flush=True)

    write_outputs(rows, summary, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
