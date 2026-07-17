"""Small I/O helpers shared by the independent baseline extraction scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from src.data.sequences import normalize_sequence


def count_pair_rows(path: Path, limit: int = 0) -> int:
    with path.open(newline="") as handle:
        count = max(0, sum(1 for _ in handle) - 1)
    return min(count, limit) if limit else count


def iter_pair_rows(
    path: Path,
    *,
    a_col: str,
    b_col: str,
    label_col: str,
    limit: int = 0,
) -> Iterator[tuple[int, str, str, float]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    csv.field_size_limit(min(2**31 - 1, 10_000_000))
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = set(reader.fieldnames or [])
        missing = {a_col, b_col, label_col} - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row_index, row in enumerate(reader):
            if limit and row_index >= limit:
                break
            seq_a = normalize_sequence(row[a_col])
            seq_b = normalize_sequence(row[b_col])
            if not seq_a or not seq_b:
                raise ValueError(f"{path}: empty sequence at row {row_index}")
            yield row_index, seq_a, seq_b, float(row[label_col])


def truncate_pair_balanced(seq_a: str, seq_b: str, max_total_tokens: int) -> tuple[str, str]:
    """Deterministically fit two sequences plus four special tokens into a budget."""
    residue_budget = max_total_tokens - 4
    if residue_budget < 2:
        raise ValueError("max_total_tokens must leave room for two non-empty chains")
    if len(seq_a) + len(seq_b) <= residue_budget:
        return seq_a, seq_b

    half = residue_budget // 2
    keep_a = min(len(seq_a), half)
    keep_b = min(len(seq_b), half)
    remaining = residue_budget - keep_a - keep_b
    if remaining:
        add_a = min(len(seq_a) - keep_a, remaining)
        keep_a += add_a
        remaining -= add_a
    if remaining:
        keep_b += min(len(seq_b) - keep_b, remaining)
    return seq_a[:keep_a], seq_b[:keep_b]


def save_baseline_pair_cache(
    path: Path,
    *,
    baseline: str,
    labels: torch.Tensor,
    features: dict[str, torch.Tensor],
    meta: dict[str, Any],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "auditppi_baseline_pair_features_v1",
        "baseline": baseline,
        "labels": labels,
        "features": features,
        "meta": {
            **meta,
            "feature_names": list(features),
            "feature_shapes": {name: list(value.shape) for name, value in features.items()},
        },
    }
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(payload["meta"], indent=2, sort_keys=True)
    )
