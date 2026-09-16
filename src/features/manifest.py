"""Dataset-neutral protein manifest loading.

Inputs may be FASTA files or CSV/TSV files containing one or more sequence
columns. Identical sequences are encoded once while every supplied protein ID
is retained as an alias to the shared feature row.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.data.sequences import iter_fasta, normalize_sequence, sequence_id


@dataclass
class ProteinManifest:
    """Unique sequences plus all ID aliases that refer to their feature rows."""

    protein_ids: list[str]
    sequences: list[str]
    id2idx: dict[str, int]
    seq2idx: dict[str, int]
    sources: list[str]

    def __len__(self) -> int:
        return len(self.sequences)


class _ManifestBuilder:
    def __init__(self) -> None:
        self.protein_ids: list[str] = []
        self.sequences: list[str] = []
        self.id2idx: dict[str, int] = {}
        self.seq2idx: dict[str, int] = {}

    def add(self, protein_id: str | None, sequence: str) -> None:
        seq = normalize_sequence(sequence)
        if not seq:
            return
        pid = str(protein_id).strip() if protein_id is not None else sequence_id(seq)
        if not pid:
            pid = sequence_id(seq)

        existing_id = self.id2idx.get(pid)
        if existing_id is not None and self.sequences[existing_id] != seq:
            raise ValueError(f"protein ID {pid!r} maps to more than one sequence")

        idx = self.seq2idx.get(seq)
        if idx is None:
            idx = len(self.sequences)
            self.seq2idx[seq] = idx
            self.sequences.append(seq)
            self.protein_ids.append(pid)
        self.id2idx[pid] = idx


def _delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def _read_table(
    path: Path,
    *,
    sequence_cols: Sequence[str],
    id_cols: Sequence[str] | None,
) -> Iterable[tuple[str | None, str]]:
    csv.field_size_limit(min(2**31 - 1, 10_000_000))
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=_delimiter(path))
        fields = set(reader.fieldnames or [])
        missing = set(sequence_cols) - fields
        if missing:
            raise ValueError(f"{path}: missing sequence columns {sorted(missing)}; columns={sorted(fields)}")
        if id_cols is not None:
            missing_ids = set(id_cols) - fields
            if missing_ids:
                raise ValueError(f"{path}: missing ID columns {sorted(missing_ids)}")
        for row in reader:
            for i, seq_col in enumerate(sequence_cols):
                id_col = id_cols[i] if id_cols is not None else None
                yield (row.get(id_col) if id_col else None), row.get(seq_col, "")


def _is_fasta(path: Path, input_format: str) -> bool:
    if input_format == "fasta":
        return True
    if input_format == "table":
        return False
    return path.suffix.lower() in {".fa", ".faa", ".fasta", ".fas"}


def load_protein_manifest(
    paths: Sequence[Path],
    *,
    input_format: str = "auto",
    sequence_cols: Sequence[str] = ("sequence",),
    id_cols: Sequence[str] | None = None,
) -> ProteinManifest:
    """Load and deduplicate proteins from one or more FASTA/CSV/TSV inputs."""
    if input_format not in {"auto", "fasta", "table"}:
        raise ValueError("input_format must be auto, fasta, or table")
    if id_cols is not None and len(id_cols) != len(sequence_cols):
        raise ValueError("--id-cols must have the same number of entries as --sequence-cols")

    builder = _ManifestBuilder()
    sources: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(str(path.resolve()))
        rows = iter_fasta(path) if _is_fasta(path, input_format) else _read_table(
            path, sequence_cols=sequence_cols, id_cols=id_cols
        )
        for protein_id, sequence in rows:
            builder.add(protein_id, sequence)

    return ProteinManifest(
        protein_ids=builder.protein_ids,
        sequences=builder.sequences,
        id2idx=builder.id2idx,
        seq2idx=builder.seq2idx,
        sources=sources,
    )


def manifest_from_protein_cache(path: Path) -> ProteinManifest:
    """Rebuild a :class:`ProteinManifest` from a v1 protein feature cache.

    Reads only the manifest fields (``protein_ids``/``sequences``/``id2idx``/
    ``seq2idx``) and never materializes the (multi-GB) feature tensors, so a
    per-protein baseline can re-derive the exact unique-sequence set and row
    order of an existing ``auditppi_protein_features_v1`` cache. This keeps the
    baseline's rows aligned to the pair-side protein caches by construction.
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    fmt = payload.get("format")
    if fmt != "auditppi_protein_features_v1":
        raise ValueError(f"{path}: expected auditppi_protein_features_v1, got {fmt!r}")
    meta = payload.get("meta", {})
    return ProteinManifest(
        protein_ids=list(payload["protein_ids"]),
        sequences=list(payload["sequences"]),
        id2idx=dict(payload["id2idx"]),
        seq2idx=dict(payload["seq2idx"]),
        sources=list(meta.get("sources", []) or [str(path)]),
    )
