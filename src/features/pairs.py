"""Protein-pair feature construction and AB/BA concat protocol."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from conf.model import DEFAULT_BACKBONE
from src.data.sequences import normalize_sequence

# Multiplier = output_dim / endpoint_dim. Shared by every pair consumer
# (mlp_pair, tabm_pair, pair-feature caches, fingerprint baselines).
PAIR_MODE_MULTIPLIER = {
    "sym": 2,       # [A⊙B, |A−B|]
    "concat": 2,    # [A‖B]  -- the only order-sensitive mode (AB/BA protocol)
    "rich": 4,      # [A, B, A⊙B, |A−B|]  -- no AB/BA (single forward)
    "product": 1,   # A⊙B
    "absdiff": 1,   # |A−B|
    "sum": 1,       # A+B
}
PAIR_MODES = tuple(PAIR_MODE_MULTIPLIER)
# Only ``concat`` needs train-time AB/BA doubling + eval-time AB/BA averaging.
# ``rich`` contains ordered [A,B] but is used as a single-shot feature (user decision).
ORDER_SENSITIVE_MODES = frozenset({"concat"})


def pair_mode_dim(feat_dim: int, mode: str) -> int:
    """Output feature width for a pair mode given per-endpoint dim."""
    if mode not in PAIR_MODE_MULTIPLIER:
        raise ValueError(f"mode must be one of {PAIR_MODES}")
    return int(feat_dim) * PAIR_MODE_MULTIPLIER[mode]


def needs_abba(mode: str) -> bool:
    """Whether ``mode`` requires the AB/BA train/eval protocol."""
    if mode not in PAIR_MODE_MULTIPLIER:
        raise ValueError(f"mode must be one of {PAIR_MODES}")
    return mode in ORDER_SENSITIVE_MODES


def pair_features(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """Construct one of the agreed pair representations.

    Primary modes:
      ``sym``    = [A⊙B, |A−B|]           order-invariant (default tabular)
      ``concat`` = [A‖B]                  order-sensitive → AB/BA protocol
      ``rich``   = [A, B, A⊙B, |A−B|]     single-shot (no AB/BA)

    Ablations (order-invariant, single-shot):
      ``product`` / ``absdiff`` / ``sum``.
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
    if mode == "sum":
        return a + b
    if mode == "sym":
        return torch.cat([a * b, (a - b).abs()], dim=-1)
    if mode == "concat":
        return torch.cat([a, b], dim=-1)
    if mode == "rich":
        return torch.cat([a, b, a * b, (a - b).abs()], dim=-1)
    raise ValueError(f"mode must be one of {PAIR_MODES}")


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
    fingerprint caches with optional id maps).
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
    # Same field-size bump as manifest/baseline_io: pair CSVs can carry full
    # sequences in query/text columns that exceed the default 128 KiB limit.
    csv.field_size_limit(min(2**31 - 1, 10_000_000))
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


PAIR_INDEX_FORMAT = "auditppi_pair_index_v1"


def save_pair_index_cache(
    path: Path,
    *,
    rows_a: torch.Tensor,
    rows_b: torch.Tensor,
    labels: torch.Tensor,
    kept_pair_indices: list[int],
    n_total: int,
    source_cache: str,
    source_pairs: str,
    pair_key: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a lightweight pair-index cache (row indices + labels, no features).

    The payload stores only the per-pair endpoint *row indices* into a protein
    feature cache plus the labels and provenance -- never any materialized
    feature vectors. One index cache therefore serves every ``(backbone, layer,
    rep)`` channel and every pair mode: a consumer loads it, then gathers
    ``matrix.index_select(0, rows_a/rows_b)`` from whichever channel it wants.
    ``kept_pair_indices`` preserves the original CSV row order so downstream
    row-aligned tables (e.g. the C3 pair-id alignment parquet) line up.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
    payload: dict[str, Any] = {
        "format": PAIR_INDEX_FORMAT,
        "rows_a": rows_a.to(torch.long),
        "rows_b": rows_b.to(torch.long),
        "labels": labels.to(torch.long),
        "kept_pair_indices": list(kept_pair_indices),
        "n_total": int(n_total),
        "n_kept": int(len(kept_pair_indices)),
        "source_cache": str(source_cache),
        "source_pairs": str(source_pairs),
        "pair_key": str(pair_key),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_pair_index_cache(path: Path) -> dict[str, Any]:
    """Load an ``auditppi_pair_index_v1`` pair-index cache."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != PAIR_INDEX_FORMAT:
        raise ValueError(f"{path} is not an {PAIR_INDEX_FORMAT} cache")
    return payload


def materialize_pair_endpoints(
    index_cache: dict[str, Any],
    protein_cache: dict[str, Any],
    *,
    rep: str,
    backbone: str = DEFAULT_BACKBONE,
    layer: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather per-pair endpoint feature rows from a pair-index cache.

    Reads the endpoint row indices from an ``auditppi_pair_index_v1`` cache and
    ``index_select``-s the requested representation channel out of an
    ``auditppi_protein_features_v1`` protein cache, returning
    ``(emb_a, emb_b, labels)`` row-aligned to the index cache's
    ``kept_pair_indices``. This is the pair-scale analogue of
    :func:`src.features.protein_cache.representation_matrix`: one lightweight
    index cache serves every ``(backbone, layer, rep)`` channel and every pair
    mode, replacing the old per-rep ``{split}_embeddings.pt`` dumps (which
    duplicated every shared endpoint's vector across each pair it appeared in).
    ``binary`` arrives as a bool matrix (``sae_max > 0``).
    """
    from src.features.protein_cache import representation_matrix

    matrix = representation_matrix(protein_cache, rep, layer, backbone)
    rows_a = index_cache["rows_a"].to(torch.long)
    rows_b = index_cache["rows_b"].to(torch.long)
    emb_a = matrix.index_select(0, rows_a)
    emb_b = matrix.index_select(0, rows_b)
    return emb_a, emb_b, index_cache["labels"]


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
        out_dim = pair_mode_dim(dim, mode)
        x = torch.empty((n, out_dim), dtype=output_dtype)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            a = matrix.index_select(0, rows_a[start:end]).to(output_dtype)
            b = matrix.index_select(0, rows_b[start:end]).to(output_dtype)
            x[start:end].copy_(pair_features(a, b, mode))
        payload["X"] = x
        return payload
    if concat_protocol == "train":
        x = torch.empty((n * 2, pair_mode_dim(dim, "concat")), dtype=output_dtype)
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
