"""Pair-level benchmark data objects and loaders used across AuditPPI.

All pair loaders return the same :class:`Benchmark` object so evaluation,
participation diagnostics, and fingerprint baselines can share one data
contract. Protein-level datasets live in :mod:`src.data.proteins`. Optional
heavy dependencies (``h5py``, ``hdf5plugin``, and pandas) are imported only by
the loader that needs them.
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
    BERNETT_DIR,
    BERNETT_SPLIT_CSVS,
    CROSS_SPECIES_DIR,
    PRING_CROSS_SPECIES,
    PRING_ROOT,
    RAPPPID_H5,
    RF2PPI_FASTA,
)
from src.data.pring_graph import METHODS as PRING_METHODS
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
    h5_path: Path = RAPPPID_H5,
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
    h5_path: Path = RAPPPID_H5,
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


def _bernett_clean_seq(sequence: str) -> str:
    """Mirror the MINT Bernett cache cleaning (``*``/``f`` dropped, ``J``→``L``)."""
    return str(sequence).replace("*", "").replace("f", "").replace("J", "L")


def _bernett_seq_id(sequence: str) -> str:
    """Stable per-sequence endpoint id (Bernett CSVs carry no protein ids)."""
    return "bn_" + hashlib.sha1(sequence.encode()).hexdigest()[:16]


def load_bernett(split: str = "test", attach_seqs: bool = True) -> Benchmark:
    """Load one Bernett gold-standard split (MINT GeneralPPI ``Intra{1,0,2}``).

    The ``{split}_seqs.csv`` files carry only ``seq1``/``seq2``/``labels`` (no
    protein ids), so endpoints are keyed by a stable ``bn_`` sequence hash after
    applying the same cleaning as the caching script.
    """
    import pandas as pd

    if split not in BERNETT_SPLIT_CSVS:
        raise ValueError(f"unknown Bernett split {split!r}; choose from {list(BERNETT_SPLIT_CSVS)}")
    frame = pd.read_csv(BERNETT_DIR / BERNETT_SPLIT_CSVS[split], usecols=["seq1", "seq2", "labels"])
    pairs: list[Pair] = []
    seqs: dict[str, str] = {}
    for sequence_a, sequence_b in zip(frame["seq1"].astype(str), frame["seq2"].astype(str)):
        clean_a, clean_b = _bernett_clean_seq(sequence_a), _bernett_clean_seq(sequence_b)
        id_a, id_b = _bernett_seq_id(clean_a), _bernett_seq_id(clean_b)
        pairs.append((id_a, id_b))
        if attach_seqs:
            seqs[id_a] = clean_a
            seqs[id_b] = clean_b
    labels = frame["labels"].astype(int).to_numpy()
    return Benchmark(f"bernett_{split}", pairs, labels, seqs)


def _read_pring_pair_file(
    path: Path, *, drop_self_pairs: bool = False
) -> tuple[list[Pair], np.ndarray]:
    """Parse a labelled PRING pair file (``id_a  id_b  label``), matching the audit.

    ``drop_self_pairs`` filters ``a == b`` rows; used by participation workflows that
    score only cross-protein pairs. The default ``False`` preserves the historical
    benchmark-loader contract (and the unit tests that pin it).
    """
    pairs: list[Pair] = []
    labels: list[int] = []
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            endpoint_a, endpoint_b = parts[0], parts[1]
            if drop_self_pairs and endpoint_a == endpoint_b:
                continue
            pairs.append((endpoint_a, endpoint_b))
            labels.append(int(parts[2]))
    return pairs, np.asarray(labels, dtype=int)


def _load_pring_pair_sequences(species: str, root: Path) -> dict[str, str]:
    """Read ``{species}_protein_id.csv`` (uniprot_id → sequence)."""
    import pandas as pd

    frame = pd.read_csv(root / species / f"{species}_protein_id.csv", usecols=["uniprot_id", "sequence"])
    return {
        str(identifier): str(sequence)
        for identifier, sequence in zip(frame["uniprot_id"], frame["sequence"])
    }


def load_pring_pairs(
    species: str = "human",
    split: str = "test",
    method: str = "BFS",
    *,
    root: Path = PRING_ROOT,
    attach_seqs: bool = True,
) -> Benchmark:
    """Load a PRING pair benchmark (UniProt-keyed).

    Human has ``train``/``val``/``test`` splits per sampling ``method``
    (``human/{METHOD}/human_{split}_ppi.txt``). Cross-species graphs
    (yeast/ecoli/arath) are test-only (``{sp}_test_ppi.txt``); they accept only
    ``split="test"`` and ignore ``method``. Endpoint sequences come from
    ``{species}_protein_id.csv`` when ``attach_seqs`` is set.
    """
    species = species.lower()
    if species == "human":
        method = method.upper()
        if method not in PRING_METHODS:
            raise ValueError(f"method must be one of {PRING_METHODS}")
        if split not in {"train", "val", "test"}:
            raise ValueError("human split must be train, val, or test")
        path = root / "human" / method / f"human_{split}_ppi.txt"
        name = f"pring_human_{method}_{split}"
    elif species in PRING_CROSS_SPECIES:
        if split != "test":
            raise ValueError(f"{species!r} PRING graph is test-only; use split='test'")
        path = root / species / f"{species}_test_ppi.txt"
        name = f"pring_{species}_test"
    else:
        raise ValueError(
            f"unknown PRING species {species!r} (human | {' | '.join(PRING_CROSS_SPECIES)})"
        )

    pairs, labels = _read_pring_pair_file(path)
    seqs: dict[str, str] = {}
    if attach_seqs:
        all_seqs = _load_pring_pair_sequences(species, root)
        needed = {endpoint for pair in pairs for endpoint in pair}
        seqs = {identifier: all_seqs[identifier] for identifier in needed if identifier in all_seqs}
    return Benchmark(name, pairs, labels, seqs)


def load_benchmark(name: str, *, attach_seqs: bool = True) -> Benchmark:
    """Dispatch pair benchmarks by string key.

    Supports ``rf2ppi``, ``c1[:split]``, ``c2[:split]``, ``c3[:split]``,
    ``cross_species:species``, ``bernett[:split]``, and
    ``pring:species[:split[:method]]``.
    """
    key, _, argument = name.partition(":")
    if key == "rf2ppi":
        return load_rf2ppi(attach_seqs=attach_seqs)
    if key in {"c1", "c2", "c3"}:
        return load_clevel(key, split=argument or "test", attach_seqs=attach_seqs)
    if key == "cross_species":
        return load_cross_species(species=argument or "human_test", attach_seqs=attach_seqs)
    if key == "bernett":
        return load_bernett(split=argument or "test", attach_seqs=attach_seqs)
    if key == "pring":
        species, _, rest = argument.partition(":")
        split, _, method = rest.partition(":")
        return load_pring_pairs(
            species=species or "human",
            split=split or "test",
            method=method or "BFS",
            attach_seqs=attach_seqs,
        )
    raise ValueError(
        f"unknown benchmark {name!r} (rf2ppi | c1[:split] | c2[:split] | c3[:split] | "
        "cross_species:species | bernett[:split] | pring:species[:split[:method]])"
    )


__all__ = [
    "Benchmark",
    "CROSS_SPECIES_CSV",
    "PRING_CROSS_SPECIES",
    "PRING_METHODS",
    "load_benchmark",
    "load_bernett",
    "load_c3",
    "load_clevel",
    "load_cross_species",
    "load_pring_pairs",
    "load_rf2ppi",
    "list_cross_species",
]
