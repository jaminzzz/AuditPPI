"""Reusable analysis utilities for compact SAE pair-feature probes."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np

from conf.model import ESMC_SAE_DIM
from src.models.estimators.tabpfn import fit_tabpfn as _fit_tabpfn

SAE_DIM = ESMC_SAE_DIM  # SAE codebook size; kept as a module alias for back-compat
BLOCK_PRODUCT = "AND(a*b)"
BLOCK_ABSDIFF = "|a-b|"


def stratified_indices(
    labels: np.ndarray,
    max_rows: int | None,
    seed: int,
) -> np.ndarray | None:
    """Return class-stratified row indices, or ``None`` to keep every row."""
    if max_rows is None or max_rows >= len(labels):
        return None
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    chosen = []
    remaining = max_rows
    for index, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        if index == len(classes) - 1:
            n_take = remaining
        else:
            n_take = int(round(max_rows * len(candidates) / len(labels)))
            n_take = min(n_take, len(candidates), remaining)
        chosen.append(rng.choice(candidates, size=n_take, replace=False))
        remaining -= n_take
    output = np.concatenate(chosen)
    rng.shuffle(output)
    return output


def load_pair_embedding_split(path: Path, max_rows: int | None, seed: int):
    """Load endpoint tensors and labels from an AuditPPI embedding split."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    labels_full = payload["label"].numpy().astype(np.int64, copy=False)
    indices = stratified_indices(labels_full, max_rows, seed)
    if indices is None:
        return payload["emb_a"].contiguous(), payload["emb_b"].contiguous(), labels_full
    tensor_indices = torch.as_tensor(indices, dtype=torch.long)
    return (
        payload["emb_a"].index_select(0, tensor_indices).contiguous(),
        payload["emb_b"].index_select(0, tensor_indices).contiguous(),
        labels_full[indices],
    )


def read_feature_ranking(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            row["rank"] = int(row["rank"])
            row["flat_feature"] = int(row["flat_feature"])
            row["sae_feature"] = int(row["sae_feature"])
            row["rank_score"] = float(row["rank_score"])
            rows.append(row)
    rows.sort(key=lambda row: row["rank"])
    return rows


def select_top_features(
    ranking: list[dict],
    top_k: int,
    mode: str,
    *,
    sae_dim: int = SAE_DIM,
) -> tuple[list[int], list[dict]]:
    """Select ranked flat columns or paired product/absdiff SAE columns."""
    if mode not in {"flat", "sae-id"}:
        raise ValueError("mode must be flat or sae-id")
    if mode == "flat":
        flat_features = [int(row["flat_feature"]) for row in ranking[:top_k]]
        metadata = []
        for flat_feature in flat_features:
            block = BLOCK_PRODUCT if flat_feature < sae_dim else BLOCK_ABSDIFF
            metadata.append(
                {
                    "flat_feature": flat_feature,
                    "sae_feature": flat_feature % sae_dim,
                    "block": block,
                }
            )
        return flat_features, metadata

    sae_features = []
    seen = set()
    for row in ranking:
        feature_id = int(row["sae_feature"])
        if feature_id in seen:
            continue
        seen.add(feature_id)
        sae_features.append(feature_id)
        if len(sae_features) == top_k:
            break
    flat_features = sae_features + [feature_id + sae_dim for feature_id in sae_features]
    metadata = [
        {"flat_feature": feature_id, "sae_feature": feature_id, "block": BLOCK_PRODUCT}
        for feature_id in sae_features
    ] + [
        {
            "flat_feature": feature_id + sae_dim,
            "sae_feature": feature_id,
            "block": BLOCK_ABSDIFF,
        }
        for feature_id in sae_features
    ]
    return flat_features, metadata


def build_dense_sym_topk(
    endpoint_a,
    endpoint_b,
    flat_features: list[int],
    *,
    sae_dim: int = SAE_DIM,
) -> np.ndarray:
    """Materialize selected columns from ``[A*B, abs(A-B)]``."""
    product_ids = [index for index in flat_features if index < sae_dim]
    absdiff_ids = [index - sae_dim for index in flat_features if index >= sae_dim]
    blocks = []
    if product_ids:
        a = endpoint_a[:, product_ids].numpy().astype(np.float32, copy=False)
        b = endpoint_b[:, product_ids].numpy().astype(np.float32, copy=False)
        blocks.append(a * b)
    if absdiff_ids:
        a = endpoint_a[:, absdiff_ids].numpy().astype(np.float32, copy=False)
        b = endpoint_b[:, absdiff_ids].numpy().astype(np.float32, copy=False)
        blocks.append(np.abs(a - b))
    if not blocks:
        return np.empty((endpoint_a.shape[0], 0), dtype=np.float32)
    grouped = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    grouped_order = product_ids + [feature_id + sae_dim for feature_id in absdiff_ids]
    if grouped_order == flat_features:
        return grouped
    positions = {feature_id: index for index, feature_id in enumerate(grouped_order)}
    return grouped[:, [positions[feature_id] for feature_id in flat_features]]


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return error if len(labels) else float("nan")


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": float(expected_calibration_error(labels, probabilities)),
        "f1": float(f1_score(labels, predictions)),
        "acc": float(accuracy_score(labels, predictions)),
    }


def predict_proba_chunked(model, matrix: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size <= 0 or matrix.shape[0] <= batch_size:
        return model.predict_proba(matrix)[:, 1]
    chunks = []
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        chunks.append(model.predict_proba(matrix[start:end])[:, 1])
        print(f"    predict rows {end}/{matrix.shape[0]}", flush=True)
    return np.concatenate(chunks)


def fit_tabpfn_probe(
    train_x,
    train_y,
    *,
    n_estimators: int,
    subsample_samples: int,
    ignore_limits: bool,
    seed: int,
    device: str = "cuda",
):
    return _fit_tabpfn(
        train_x,
        train_y,
        n_estimators=n_estimators,
        subsample_samples=subsample_samples,
        ignore_limits=ignore_limits,
        seed=seed,
        device=device,
    )


def fit_xgb_probe(
    train_x,
    train_y,
    val_x,
    val_y,
    *,
    trees: int,
    depth: int,
    learning_rate: float,
    seed: int,
    cpu: bool,
):
    import torch
    import xgboost as xgb

    use_gpu = (not cpu) and torch.cuda.is_available()
    model = xgb.XGBClassifier(
        n_estimators=trees,
        max_depth=depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=seed,
        device="cuda" if use_gpu else "cpu",
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=50)
    if use_gpu:
        model.get_booster().set_param({"device": "cpu"})
    return model


def fit_logistic_probe(train_x, train_y, *, seed: int):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=seed)
    model.fit(train_x, train_y)
    return model


def evaluate_species(
    model,
    species: str,
    embedding_dir: Path,
    flat_features: list[int],
    *,
    test_subsample: int | None,
    seed: int,
    predict_batch_size: int,
) -> dict:
    endpoint_a, endpoint_b, labels = load_pair_embedding_split(
        embedding_dir / f"{species}_embeddings.pt", test_subsample, seed + 17
    )
    matrix = build_dense_sym_topk(endpoint_a, endpoint_b, flat_features)
    del endpoint_a, endpoint_b
    gc.collect()
    probabilities = predict_proba_chunked(model, matrix, predict_batch_size)
    output = {
        "n": int(len(labels)),
        "pos_rate": float(labels.mean()),
        **classification_metrics(labels, probabilities),
    }
    del matrix, labels, probabilities
    gc.collect()
    return output


# Historical names retained inside the new package for concise analysis imports.
load_split = load_pair_embedding_split
read_ranking = read_feature_ranking
metrics = classification_metrics
ece_binary = expected_calibration_error

__all__ = [
    "BLOCK_ABSDIFF",
    "BLOCK_PRODUCT",
    "SAE_DIM",
    "build_dense_sym_topk",
    "classification_metrics",
    "evaluate_species",
    "expected_calibration_error",
    "fit_logistic_probe",
    "fit_tabpfn_probe",
    "fit_xgb_probe",
    "load_pair_embedding_split",
    "predict_proba_chunked",
    "read_feature_ranking",
    "select_top_features",
    "stratified_indices",
]
