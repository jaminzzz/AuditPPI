"""Shared sequence normalization and FASTA helpers.

These helpers are intentionally independent of feature extraction and
benchmark loading. They provide one canonical spelling for protein sequences
and a stable identifier for inputs that do not carry protein IDs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator


def normalize_sequence(sequence: str) -> str:
    """Remove whitespace and normalize a protein sequence to uppercase."""
    return "".join(str(sequence).split()).upper()


def sequence_id(sequence: str, *, prefix: str = "seq_", digest_chars: int = 20) -> str:
    """Return a deterministic ID for a normalized sequence."""
    if digest_chars <= 0:
        raise ValueError("digest_chars must be positive")
    normalized = normalize_sequence(sequence)
    return prefix + hashlib.sha1(normalized.encode()).hexdigest()[:digest_chars]


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(first_header_token, normalized_sequence)`` records from FASTA."""
    protein_id: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_id is not None:
                    yield protein_id, normalize_sequence("".join(chunks))
                protein_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if protein_id is not None:
        yield protein_id, normalize_sequence("".join(chunks))


def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA file into an ID-to-sequence mapping."""
    return dict(iter_fasta(path))


__all__ = ["iter_fasta", "normalize_sequence", "read_fasta", "sequence_id"]
