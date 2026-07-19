"""Protein-pair feature construction and AB/BA concat protocol."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.data.sequences import normalize_sequence

PAIR_MODES = ("sym", "product", "absdiff", "concat")


def pair_features(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """Construct one of the agreed pair representations.

    ``sym`` is the primary order-invariant representation
    ``[A * B, abs(A - B)]``. ``product`` and ``absdiff`` are its ablations.
    ``concat`` is ordered and must use the train/eval AB/BA protocol below.
    """
    if mode not in PAIR_MODES:
        raise ValueError(f"mode must be one of {PAIR_MODES}")
    if a.shape != b.shape:
        raise ValueError(f"A/B shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if not (a.is_floating_point() and b.is_floating_point()):
        a = a.float()
        b = b.float()
    else:
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype)
        b = b.to(dtype)
    if mode == "product":
        return a * b
    if mode == "absdiff":
        return (a - b).abs()
    if mode == "sym":
        return torch.cat([a * b, (a - b).abs()], dim=-1)
    return torch.cat([a, b], dim=-1)


def sym_features(A, B, cols: Optional[np.ndarray] = None) -> np.ndarray:
    """``sym = [A⊙B, |A−B|]`` as float32 numpy (n × 2·rep_dim); optional column subset.

    Numpy adapter over the canonical torch :func:`pair_features` (``mode="sym"``)
    so the ``[A*B, abs(A-B)]`` definition stays in exactly one place. A/B arrive as
    float32 torch tensors, so this is bit-for-bit ``concatenate([(A*B), (A-B).abs()])``;
    the numpy cast + column select is the fingerprint-baseline specific bit (TabPFN's
    feature cap needs a fixed column subset).
    """
    x = pair_features(A, B, "sym").numpy().astype(np.float32, copy=False)
    return x[:, cols] if cols is not None else x


def concat_training_examples(
    a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AB/BA training augmentation with an index back to the biological pair."""
    n = int(labels.shape[0])
    x = torch.cat([pair_features(a, b, "concat"), pair_features(b, a, "concat")], dim=0)
    y = torch.cat([labels, labels], dim=0)
    original_index = torch.arange(n, dtype=torch.long).repeat(2)
    direction = torch.cat(
        [torch.zeros(n, dtype=torch.int8), torch.ones(n, dtype=torch.int8)], dim=0
    )
    return x, y, original_index, direction


def concat_evaluation_examples(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate AB/BA matrices; callers average their predictions per pair."""
    return pair_features(a, b, "concat"), pair_features(b, a, "concat")


def load_protein_feature_cache(path: Path) -> dict[str, Any]:
    """Load an ``auditppi_protein_features_v1`` protein feature cache.

    Distinct from :func:`src.features.pooled_assembly.load_pooled_payload` (pooled
    fingerprint caches with optional id maps) and
    :func:`src.features.protein_cache.load_pooled_cache` (filtered pooled keys).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "auditppi_protein_features_v1":
        raise ValueError(f"{path} is not an auditppi_protein_features_v1 cache")
    return payload


def _resolve_row(cache: dict[str, Any], value: str, pair_key: str) -> int | None:
    if pair_key in {"auto", "id"}:
        idx = cache["id2idx"].get(str(value).strip())
        if idx is not None:
            return int(idx)
        if pair_key == "id":
            return None
    sequence = normalize_sequence(value)
    idx = cache["seq2idx"].get(sequence)
    return int(idx) if idx is not None else None


def load_pair_indices(
    path: Path,
    *,
    cache: dict[str, Any],
    a_col: str,
    b_col: str,
    label_col: str,
    pair_key: str = "auto",
    skip_missing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    rows_a: list[int] = []
    rows_b: list[int] = []
    labels: list[int] = []
    kept: list[int] = []
    total = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = set(reader.fieldnames or [])
        missing_cols = {a_col, b_col, label_col} - fields
        if missing_cols:
            raise ValueError(f"{path}: missing columns {sorted(missing_cols)}")
        for row_no, row in enumerate(reader):
            total += 1
            ia = _resolve_row(cache, row[a_col], pair_key)
            ib = _resolve_row(cache, row[b_col], pair_key)
            if ia is None or ib is None:
                if skip_missing:
                    continue
                raise KeyError(f"pair row {row_no}: endpoint missing from feature cache")
            rows_a.append(ia)
            rows_b.append(ib)
            labels.append(int(float(row[label_col])))
            kept.append(row_no)
    return (
        torch.as_tensor(rows_a, dtype=torch.long),
        torch.as_tensor(rows_b, dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.long),
        kept,
        total,
    )


def build_pair_payload(
    *,
    cache: dict[str, Any],
    feature_name: str,
    rows_a: torch.Tensor,
    rows_b: torch.Tensor,
    labels: torch.Tensor,
    kept_pair_indices: list[int],
    n_total: int,
    mode: str,
    concat_protocol: str,
    output_dtype: torch.dtype,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    if feature_name not in cache["features"]:
        raise KeyError(f"unknown feature {feature_name!r}; choose from {list(cache['features'])}")
    matrix = cache["features"][feature_name]
    n = int(labels.shape[0])
    dim = int(matrix.shape[1])
    chunk_size = max(1, int(chunk_size))
    payload: dict[str, Any] = {
        "format": "auditppi_pair_features_v1",
        "feature_name": feature_name,
        "mode": mode,
        "labels": labels,
        "kept_pair_indices": kept_pair_indices,
        "n_total": n_total,
        "n_kept": len(kept_pair_indices),
    }
    if mode != "concat":
        out_dim = dim * 2 if mode == "sym" else dim
        x = torch.empty((n, out_dim), dtype=output_dtype)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            a = matrix.index_select(0, rows_a[start:end]).to(output_dtype)
            b = matrix.index_select(0, rows_b[start:end]).to(output_dtype)
            x[start:end].copy_(pair_features(a, b, mode))
        payload["X"] = x
        return payload
    if concat_protocol == "train":
        x = torch.empty((n * 2, dim * 2), dtype=output_dtype)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            a = matrix.index_select(0, rows_a[start:end]).to(output_dtype)
            b = matrix.index_select(0, rows_b[start:end]).to(output_dtype)
            x[start:end].copy_(pair_features(a, b, "concat"))
            x[n + start : n + end].copy_(pair_features(b, a, "concat"))
        y = torch.cat([labels, labels], dim=0)
        original_index = torch.arange(n, dtype=torch.long).repeat(2)
        direction = torch.cat(
            [torch.zeros(n, dtype=torch.int8), torch.ones(n, dtype=torch.int8)], dim=0
        )
        payload.update(
            X=x,
            labels=y,
            original_pair_index=original_index,
            direction=direction,
            concat_protocol="train_ab_ba_augmentation",
        )
        return payload
    if concat_protocol == "eval":
        x_ab = torch.empty((n, dim * 2), dtype=output_dtype)
        x_ba = torch.empty_like(x_ab)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            a = matrix.index_select(0, rows_a[start:end]).to(output_dtype)
            b = matrix.index_select(0, rows_b[start:end]).to(output_dtype)
            x_ab[start:end].copy_(pair_features(a, b, "concat"))
            x_ba[start:end].copy_(pair_features(b, a, "concat"))
        payload.update(
            X_ab=x_ab,
            X_ba=x_ba,
            concat_protocol="eval_average_ab_ba_predictions",
        )
        return payload
    raise ValueError("concat_protocol must be train or eval")
