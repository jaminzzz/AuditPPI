"""SAE feature-table annotation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conf.paths import FEATURE_TABLE

ANNOTATION_COLUMNS = (
    "feature_id",
    "category",
    "summary",
    "activation_pattern",
    "uniref90_frequency",
)


def add_sae_annotations(
    frame: pd.DataFrame,
    representation: str,
    *,
    feature_table: Path = FEATURE_TABLE,
) -> pd.DataFrame:
    """Left-join standard SAE annotations when the representation supports them."""
    if representation not in {"sae_max", "binary"} or not feature_table.exists():
        return frame
    annotations = pd.read_parquet(feature_table, columns=list(ANNOTATION_COLUMNS))
    annotations["category"] = annotations["category"].fillna("Unclassified")
    return frame.merge(annotations, on="feature_id", how="left")


__all__ = ["ANNOTATION_COLUMNS", "add_sae_annotations"]
