"""Benchmark data objects and loaders used across AuditPPI.

All loaders return the same :class:`Benchmark` object so evaluation,
participation diagnostics, and fingerprint baselines can share one data
contract. Optional heavy dependencies (``h5py``, ``hdf5plugin``, and pandas)
are imported only by the loader that needs them.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from conf.paths import (
    BENCHMARK_TSV as RF2PPI_TSV,
    C3_H5,
    CROSS_SPECIES_DIR,
    RF2PPI_FASTA,
)
from src.data.sequences import read_fasta

Pair = Tuple[Hashable, Hashable]


@dataclass
class Benchmark:
    """A named pair-label dataset with optional endpoint sequences."""

    name: str
    pairs: List[Pair]
    labels: np.ndarray
    seqs: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pairs = [tuple(pair) for pair in self.pairs]
        if any(len(pair) != 2 for pair in self.pairs):
            raise ValueError("every benchmark pair must contain exactly two endpoints")
        self.labels = np.asarray(self.labels, dtype=int).reshape(-1)
        if len(self.pairs) != self.labels.size:
            raise ValueError(
                f"benchmark {self.name!r} has {len(self.pairs)} pairs but "
                f"{self.labels.size} labels"
            )
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError(f"benchmark {self.name!r} labels must be binary 0/1")
        self.seqs = dict(self.seqs)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def protein_ids(self) -> frozenset[Hashable]:
        """Unique endpoint identifiers appearing in the benchmark pairs."""
        return frozenset(endpoint for pair in self.pairs for endpoint in pair)

    @property
    def positive_rate(self) -> float | None:
        """Fraction of positive pairs, or ``None`` for an empty benchmark."""
        return float(self.labels.mean()) if self.labels.size else None


CROSS_SPECIES_CSV = {
    "human_train": "human.ppi.qrels.seq.train.csv",
    "human_test": "human.ppi.qrels.seq.test.csv",
    "ecoli": "ecoli.ppi.qrels.seq.test.csv",
    "fly": "fly.ppi.qrels.seq.test.csv",
    "mouse": "mouse.ppi.qrels.seq.test.csv",
    "worm": "worm.ppi.qrels.seq.test.csv",
    "yeast": "yeast.ppi.qrels.seq.test.csv",
}


def _sequence_pair_id(sequence: str) -> str:
    """Preserve the historical cross-species ``cs_`` ID contract."""
    return "cs_" + hashlib.sha1(sequence.encode()).hexdigest()[:16]


def load_rf2ppi(
    tsv: Path = RF2PPI_TSV,
    fasta: Optional[Path] = RF2PPI_FASTA,
    attach_seqs: bool = True,
) -> Benchmark:
    """Load ``positives_and_negatives.tsv`` and optional benchmark FASTA."""
    pairs: list[Pair] = []
    labels: list[int] = []
    with Path(tsv).open() as handle:
        try:
            header = next(handle)
        except StopIteration as exc:
            raise ValueError(f"empty RF2-PPI benchmark: {tsv}") from exc
        if "pair" not in header.lower():
            raise ValueError(f"unexpected RF2-PPI header: {header!r}")
        for line_number, line in enumerate(handle, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(f"{tsv}:{line_number}: expected pair and label")
            pair_str, category = fields
            parts = pair_str.split("_")
            if len(parts) != 2:
                raise ValueError(f"{tsv}:{line_number}: invalid pair token {pair_str!r}")
            pairs.append((parts[0], parts[1]))
            labels.append(1 if category.strip().lower().startswith("pos") else 0)
    seqs = read_fasta(Path(fasta)) if attach_seqs and fasta and Path(fasta).exists() else {}
    return Benchmark("rf2ppi", pairs, np.asarray(labels, dtype=int), seqs)


def load_clevel(
    level: str = "c3",
    split: str = "test",
    h5_path: Path = C3_H5,
    attach_seqs: bool = True,
) -> Benchmark:
    """Load one RAPPPID C1/C2/C3 split from the compressed HDF5 store."""
    level = level.lower()
    if level not in {"c1", "c2", "c3"}:
        raise ValueError("level must be c1, c2, or c3")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")

    import hdf5plugin  # noqa: F401  (registers the blosc filter)

    os.environ.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGINS_PATH)
    import h5py

    with h5py.File(h5_path, "r") as handle:
        dataset = handle[f"interactions/{level}/{level}_{split}"]
        pairs = [
            (row["protein_id1"].decode(), row["protein_id2"].decode())
            for row in dataset
        ]
        labels = np.asarray([int(row["label"]) for row in dataset], dtype=int)
        seqs: dict[str, str] = {}
        if attach_seqs:
            needed = {endpoint for pair in pairs for endpoint in pair}
            for row in handle["sequences"]:
                name = row["name"].decode()
                if name in needed:
                    seqs[name] = row["sequence"].decode()
    return Benchmark(f"{level}_{split}", pairs, labels, seqs)


def load_c3(
    split: str = "test",
    h5_path: Path = C3_H5,
    attach_seqs: bool = True,
) -> Benchmark:
    """Backward-compatible convenience wrapper for RAPPPID C3."""
    return load_clevel("c3", split=split, h5_path=h5_path, attach_seqs=attach_seqs)


def load_cross_species(
    species: str = "human_test",
    attach_seqs: bool = True,
) -> Benchmark:
    """Load one sequence-based cross-species qrels split."""
    import pandas as pd

    if species not in CROSS_SPECIES_CSV:
        raise ValueError(f"unknown species {species!r}; choose from {list(CROSS_SPECIES_CSV)}")
    frame = pd.read_csv(
        CROSS_SPECIES_DIR / CROSS_SPECIES_CSV[species],
        usecols=["query", "text", "label"],
    )
    pairs: list[Pair] = []
    seqs: dict[str, str] = {}
    for sequence_a, sequence_b in zip(frame["query"].astype(str), frame["text"].astype(str)):
        id_a, id_b = _sequence_pair_id(sequence_a), _sequence_pair_id(sequence_b)
        pairs.append((id_a, id_b))
        if attach_seqs:
            seqs[id_a] = sequence_a
            seqs[id_b] = sequence_b
    labels = frame["label"].astype(int).to_numpy()
    return Benchmark(f"cross_species_{species}", pairs, labels, seqs)


def list_cross_species() -> list[str]:
    return list(CROSS_SPECIES_CSV)


def load_benchmark(name: str, *, attach_seqs: bool = True) -> Benchmark:
    """Dispatch ``rf2ppi``, ``c1[:split]``, ``c2[:split]``, ``c3[:split]``, or ``cross_species:x``."""
    key, _, argument = name.partition(":")
    if key == "rf2ppi":
        return load_rf2ppi(attach_seqs=attach_seqs)
    if key in {"c1", "c2", "c3"}:
        return load_clevel(key, split=argument or "test", attach_seqs=attach_seqs)
    if key == "cross_species":
        return load_cross_species(species=argument or "human_test", attach_seqs=attach_seqs)
    raise ValueError(
        f"unknown benchmark {name!r} (rf2ppi | c1[:split] | c2[:split] | c3[:split] | "
        "cross_species:species)"
    )


__all__ = [
    "Benchmark",
    "CROSS_SPECIES_CSV",
    "load_benchmark",
    "load_c3",
    "load_clevel",
    "load_cross_species",
    "load_rf2ppi",
    "list_cross_species",
]
