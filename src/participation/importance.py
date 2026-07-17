"""Feature selection and importance tables for participation predictors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.participation.models import fit_xgb_logdegree


def extract_xgb_importance(
    model,
    full_dim: int,
    selected_columns: Optional[np.ndarray] = None,
) -> Optional[Dict[str, np.ndarray]]:
    if not hasattr(model, "get_booster"):
        return None
    booster = model.get_booster()
    output = {
        "gain": np.zeros(full_dim, dtype=float),
        "weight": np.zeros(full_dim, dtype=float),
        "cover": np.zeros(full_dim, dtype=float),
        "total_gain": np.zeros(full_dim, dtype=float),
        "total_cover": np.zeros(full_dim, dtype=float),
    }
    for importance_type, values in output.items():
        for key, value in booster.get_score(importance_type=importance_type).items():
            if not key.startswith("f"):
                continue
            local_index = int(key[1:])
            if selected_columns is not None:
                if local_index >= len(selected_columns):
                    continue
                feature_index = int(selected_columns[local_index])
            else:
                feature_index = local_index
            if feature_index < full_dim:
                values[feature_index] = float(value)
    return output


def select_topk_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    top_k: int,
    seed: int,
    max_depth: int,
    learning_rate: float,
    device: str,
    importance_estimators: int,
    early_stopping_rounds: int,
) -> tuple[Optional[np.ndarray], Optional[object]]:
    if top_k <= 0 or top_k >= train_x.shape[1]:
        return None, None
    quick_model = fit_xgb_logdegree(
        train_x,
        train_y,
        val_x,
        val_y,
        seed=seed,
        n_estimators=importance_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        device=device,
        early_stopping_rounds=early_stopping_rounds,
    )
    importance = extract_xgb_importance(quick_model, train_x.shape[1])
    if importance is None:
        return None, quick_model
    score = importance["total_gain"]
    if not np.any(score):
        score = importance["gain"]
    if not np.any(score) and hasattr(quick_model, "feature_importances_"):
        score = np.asarray(quick_model.feature_importances_, dtype=float)
    columns = np.argsort(score)[::-1][: min(top_k, train_x.shape[1])]
    return np.sort(columns.astype(int)), quick_model


def build_importance_rows(
    *,
    feature_names: Sequence[str],
    arrays: Dict[str, np.ndarray],
    selected_columns: Optional[np.ndarray],
) -> list[dict]:
    score = arrays["total_gain"]
    if not np.any(score):
        score = arrays["gain"]
    if not np.any(score):
        score = arrays["weight"]
    order = np.argsort(score)[::-1]
    selected = None if selected_columns is None else {int(index) for index in selected_columns}
    rows = []
    for rank, feature_index in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "feature_index": int(feature_index),
                "feature_name": feature_names[feature_index],
                "selected_for_model": selected is None or int(feature_index) in selected,
                "gain": float(arrays["gain"][feature_index]),
                "weight": float(arrays["weight"][feature_index]),
                "cover": float(arrays["cover"][feature_index]),
                "total_gain": float(arrays["total_gain"][feature_index]),
                "total_cover": float(arrays["total_cover"][feature_index]),
            }
        )
    return rows


def write_feature_importance(path: Path, rows: Sequence[dict]) -> None:
    columns = [
        "rank",
        "feature_index",
        "feature_name",
        "selected_for_model",
        "gain",
        "weight",
        "cover",
        "total_gain",
        "total_cover",
    ]
    with path.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")


__all__ = [
    "build_importance_rows",
    "extract_xgb_importance",
    "select_topk_features",
    "write_feature_importance",
]
