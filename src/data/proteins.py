"""Protein-level (single-sequence) datasets and loaders.

Parallel to :mod:`src.data.pairs`, but for tasks whose unit is one protein
rather than a pair: PIC essentiality (binary) and PRING graph participation
(continuous degree/participation). Pair benchmarks cannot host these — the
:class:`~src.data.pairs.Benchmark` contract requires exactly-two endpoints per
row — so they get their own :class:`ProteinDataset` contract here.

Loaders are thin wrappers over the logic that already produced the manuscript
numbers (``load_pic`` mirrors ``scripts/cache/cache_pic_human_esmc_sae.py``;
``load_pring_participation`` reuses ``src.data.pring_graph``), so on-disk
results are unaffected. Optional heavy dependencies (pandas) are imported only
by the loader that needs them.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from conf.paths import PIC_DATASET_PKL, PRING_ROOT
from src.data.pring_graph import full_graph_participation_labels, load_pring_human_split

LABEL_KINDS = ("binary", "continuous")


def _read_legacy_dataframe(path: str | Path):
    """Unpickle a DataFrame written with an older numpy, robustly.

    The vendored PIC pickle stores numpy arrays that reference the historical
    ``numpy.core`` module path. Current numpy keeps that path only as a noisy
    deprecated shim, and a future numpy will drop it entirely -- at which point a
    naive :func:`pickle.load` would fail with ``ModuleNotFoundError``. Alias the
    old path onto the current ``numpy._core`` so the load survives that removal,
    and silence the (understood) deprecation while reading.
    """
    core = getattr(np, "_core", None)
    if core is not None:
        sys.modules.setdefault("numpy.core", core)
        for sub in ("multiarray", "numeric", "umath", "_multiarray_umath"):
            submodule = getattr(core, sub, None)
            if submodule is not None:
                sys.modules.setdefault(f"numpy.core.{sub}", submodule)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with open(path, "rb") as handle:
            return pickle.load(handle)


@dataclass
class ProteinDataset:
    """A named per-protein label set with optional sequences.

    ``label_kind`` selects the label contract: ``"binary"`` coerces labels to
    ``int`` and requires 0/1 values; ``"continuous"`` keeps them as ``float``.
    """

    name: str
    ids: List[str]
    labels: np.ndarray
    seqs: Dict[str, str] = field(default_factory=dict)
    label_kind: str = "binary"

    def __post_init__(self) -> None:
        if self.label_kind not in LABEL_KINDS:
            raise ValueError(f"label_kind must be one of {LABEL_KINDS}")
        self.ids = [str(identifier) for identifier in self.ids]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"protein dataset {self.name!r} has duplicate ids")
        if self.label_kind == "binary":
            self.labels = np.asarray(self.labels, dtype=int).reshape(-1)
            if not np.isin(self.labels, (0, 1)).all():
                raise ValueError(f"protein dataset {self.name!r} labels must be binary 0/1")
        else:
            self.labels = np.asarray(self.labels, dtype=float).reshape(-1)
        if len(self.ids) != self.labels.size:
            raise ValueError(
                f"protein dataset {self.name!r} has {len(self.ids)} ids but "
                f"{self.labels.size} labels"
            )
        self.seqs = dict(self.seqs)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def positive_rate(self) -> float | None:
        """Fraction of positive proteins (binary only), or ``None`` if empty."""
        if self.label_kind != "binary" or not self.labels.size:
            return None
        return float(self.labels.mean())


def load_pic(dataset: str = "human", *, label_col: Optional[str] = None) -> ProteinDataset:
    """Load a PIC essentiality dataset as a binary :class:`ProteinDataset`.

    Mirrors the ``load_pic`` reader in the caching script: ``ID`` /
    ``sequence`` (upper-cased) / ``label_col``. ``label_col`` defaults to the
    dataset name for ``human``/``mouse``; the ``cell`` pickle carries one
    column per cell line, so an explicit ``label_col`` is required there.
    """
    if dataset not in PIC_DATASET_PKL:
        raise ValueError(f"unknown PIC dataset {dataset!r}; choose from {list(PIC_DATASET_PKL)}")
    if label_col is None:
        if dataset == "cell":
            raise ValueError("PIC 'cell' dataset has per-cell-line columns; pass an explicit label_col")
        label_col = dataset

    frame = _read_legacy_dataframe(PIC_DATASET_PKL[dataset])
    if label_col not in frame.columns:
        raise ValueError(f"label_col {label_col!r} not in PIC {dataset!r} columns")

    ids = [str(identifier) for identifier in frame["ID"].tolist()]
    seqs = {identifier: str(seq).upper() for identifier, seq in zip(ids, frame["sequence"].tolist())}
    labels = np.asarray([int(value) for value in frame[label_col].tolist()], dtype=int)
    return ProteinDataset(f"pic_{dataset}", ids, labels, seqs, label_kind="binary")


def _load_pring_sequences(species: str, *, root: Path) -> dict[str, str]:
    """Read ``{species}_protein_id.csv`` (uniprot_id → sequence).

    Delegates to the shared reader in :mod:`src.data.pairs` so the CSV contract
    lives in exactly one place.
    """
    from src.data.pairs import _load_pring_pair_sequences

    return _load_pring_pair_sequences(species, root)


def load_pring_participation(
    species: str = "human",
    *,
    self_loop_mode: str = "drop",
    split: Optional[str] = None,
    root: Path = PRING_ROOT,
    attach_seqs: bool = True,
) -> ProteinDataset:
    """Load PRING graph participation as a continuous :class:`ProteinDataset`.

    Labels are the normalized participation ``t`` from
    :func:`src.data.pring_graph.full_graph_participation_labels` (the same
    values the participation audits consume). ``split`` (``"train"``/``"test"``)
    filters to the human method split via ``load_pring_human_split`` and is only
    valid for human; cross-species graphs are test-only and take ``split=None``.
    """
    species = species.lower()
    labels_obj = full_graph_participation_labels(
        root=root, species=species, self_loop_mode=self_loop_mode
    )
    keep: Optional[set[str]] = None
    name = f"pring_participation_{species}"
    if split is not None:
        if species != "human":
            raise ValueError("split filtering is only defined for the human PRING graph")
        # split may be "METHOD:train"/"METHOD:test" or a bare "train"/"test" (default BFS).
        method, _, side = split.partition(":")
        if not side:
            method, side = "BFS", method
        train_ids, test_ids = load_pring_human_split(method, root=root)
        if side == "train":
            keep = train_ids
        elif side == "test":
            keep = test_ids
        else:
            raise ValueError("split side must be 'train' or 'test'")
        name = f"pring_participation_human_{method}_{side}"

    ids = sorted(labels_obj.t)
    if keep is not None:
        ids = [identifier for identifier in ids if identifier in keep]
    labels = np.asarray([labels_obj.t[identifier] for identifier in ids], dtype=float)

    seqs: dict[str, str] = {}
    if attach_seqs:
        all_seqs = _load_pring_sequences(species, root=root)
        seqs = {identifier: all_seqs[identifier] for identifier in ids if identifier in all_seqs}

    return ProteinDataset(name, ids, labels, seqs, label_kind="continuous")


__all__ = [
    "LABEL_KINDS",
    "ProteinDataset",
    "load_pic",
    "load_pring_participation",
]
