"""Full TreeSHAP explanation objects for the frozen sym pair boosters.

The existing pipeline (:func:`src.interp.pair_probe.rank_booster_by_shap`) collapses
TreeSHAP to two scalars per column (mean|shap|, mean signed shap) and throws the
matrix away -- enough to rank features, not enough to *plot* them. This module keeps
the matrix, wraps it in a :class:`shap.Explanation` so the official ``shap.plots.*``
API works directly, and caches the (already sparse) result to disk.

Why a cache: TreeSHAP over a 32768-column booster costs minutes per family and the
dense matrix is up to ~8 GB (bernett). Only ~4k of the 32768 columns are ever used
by a booster, so we persist the **active columns only** as float32 plus the column
index -- typically <100 MB, and enough to rebuild any plot.

Feature naming: each flat sym column ``j`` maps to SAE feature ``j % sae_dim`` in
block ``AND(a*b)`` (``j < sae_dim``) or ``|a-b|`` (``j >= sae_dim``). Labels join the
SAE feature table for category/summary, so plots carry biology rather than ``f23646``.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.interp.annotations import add_sae_annotations
from src.interp.pair_probe import BLOCK_ABSDIFF, BLOCK_PRODUCT

CACHE_VERSION = 2


@dataclass
class ShapBundle:
    """Active-column TreeSHAP values plus the feature values that produced them.

    ``shap_values``/``data`` are ``(n_rows, n_active)`` float32 arrays restricted to
    the columns the booster actually splits on; ``columns`` holds the original flat
    column ids so everything can be mapped back to SAE features.
    """

    family: str
    shap_values: np.ndarray      # (n, n_active) float32
    data: np.ndarray             # (n, n_active) float32 -- raw sym feature values
    columns: np.ndarray          # (n_active,) int32 -- original flat column ids
    base_value: float
    labels: np.ndarray           # (n,) int8 -- val labels
    sae_dim: int
    n_total_columns: int

    # ---- derived views -------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return int(self.shap_values.shape[0])

    def mean_abs(self) -> np.ndarray:
        """mean|SHAP| per active column."""
        return np.abs(self.shap_values).mean(axis=0)

    def feature_frame(self) -> pd.DataFrame:
        """Per-active-column table: flat id, SAE id, block, mean|shap|, mean signed."""
        flat = self.columns.astype(np.int64)
        frame = pd.DataFrame({
            "flat_feature": flat,
            "feature_id": flat % self.sae_dim,
            "block": np.where(flat < self.sae_dim, BLOCK_PRODUCT, BLOCK_ABSDIFF),
            "mean_abs_shap": self.mean_abs(),
            "mean_signed_shap": self.shap_values.mean(axis=0),
        })
        return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def annotated_frame(self) -> pd.DataFrame:
        """:meth:`feature_frame` left-joined with SAE category/summary annotations."""
        frame = add_sae_annotations(self.feature_frame(), "sae_max")
        if "category" in frame.columns:
            frame["category"] = frame["category"].fillna("Unclassified")
        return frame

    def top_columns(self, n: int) -> np.ndarray:
        """Positions (into the active axis) of the ``n`` highest mean|SHAP| columns."""
        return np.argsort(self.mean_abs())[::-1][:n]

    def explanation(self, positions: np.ndarray | None = None, *, labels: list[str] | None = None):
        """Build a :class:`shap.Explanation` for the given active-column positions."""
        import shap

        pos = np.arange(self.shap_values.shape[1]) if positions is None else np.asarray(positions)
        names = labels if labels is not None else [f"f{int(c)}" for c in self.columns[pos]]
        return shap.Explanation(
            values=self.shap_values[:, pos].astype(np.float64),
            base_values=np.full(self.n_rows, self.base_value, dtype=np.float64),
            data=self.data[:, pos].astype(np.float64),
            feature_names=list(names),
        )

    # ---- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            shap_values=self.shap_values,
            data=self.data,
            columns=self.columns,
            labels=self.labels,
            meta=np.array(json.dumps({
                "cache_version": CACHE_VERSION,
                "family": self.family,
                "base_value": float(self.base_value),
                "sae_dim": int(self.sae_dim),
                "n_total_columns": int(self.n_total_columns),
                "n_rows": self.n_rows,
                "n_active": int(self.shap_values.shape[1]),
            })),
        )

    @classmethod
    def load(cls, path: Path) -> "ShapBundle":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("cache_version") != CACHE_VERSION:
                raise ValueError(
                    f"{path} has cache_version={meta.get('cache_version')}, "
                    f"expected {CACHE_VERSION}; delete it and recompute"
                )
            return cls(
                family=meta["family"],
                shap_values=z["shap_values"],
                data=z["data"],
                columns=z["columns"],
                base_value=float(meta["base_value"]),
                labels=z["labels"],
                sae_dim=int(meta["sae_dim"]),
                n_total_columns=int(meta["n_total_columns"]),
            )


def compute_shap_bundle(
    booster,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    family: str,
    sae_dim: int,
    row_cap: int | None = None,
    seed: int = 0,
    chunk_rows: int = 4096,
) -> ShapBundle:
    """TreeSHAP over ``val_x``, kept as a matrix and trimmed to active columns.

    ``row_cap`` subsamples rows (stratified by label) before explaining -- beeswarm
    plots saturate well below 10k points and bernett's 59k x 32768 matrix is 8 GB.
    Rows are explained in chunks so peak memory stays near one chunk, not the whole
    matrix.
    """
    import xgboost as xgb

    n_total_columns = int(val_x.shape[1])
    rows = np.arange(val_x.shape[0])
    if row_cap is not None and len(rows) > row_cap:
        rng = np.random.default_rng(seed)
        pos = rows[val_y == 1]
        neg = rows[val_y == 0]
        take_pos = min(len(pos), row_cap // 2)
        take_neg = min(len(neg), row_cap - take_pos)
        rows = np.sort(np.concatenate([
            rng.choice(pos, take_pos, replace=False),
            rng.choice(neg, take_neg, replace=False),
        ]))

    # Columns the booster never splits on contribute exactly zero SHAP, so we can
    # find the active set from the model rather than from the (huge) matrix.
    active = np.zeros(n_total_columns, dtype=bool)
    for key in booster.get_score(importance_type="gain"):
        if key.startswith("f"):
            idx = int(key[1:])
            if 0 <= idx < n_total_columns:
                active[idx] = True
    columns = np.flatnonzero(active).astype(np.int32)

    shap_parts: list[np.ndarray] = []
    base_value = 0.0
    for start in range(0, len(rows), chunk_rows):
        block = rows[start : start + chunk_rows]
        contribs = booster.predict(xgb.DMatrix(val_x[block]), pred_contribs=True)
        shap_parts.append(contribs[:, columns].astype(np.float32))
        base_value = float(contribs[0, -1])
        del contribs
        gc.collect()

    return ShapBundle(
        family=family,
        shap_values=np.concatenate(shap_parts, axis=0),
        data=val_x[np.ix_(rows, columns)].astype(np.float32),
        columns=columns,
        base_value=base_value,
        labels=val_y[rows].astype(np.int8),
        sae_dim=int(sae_dim),
        n_total_columns=n_total_columns,
    )


def short_label(row: pd.Series, *, width: int = 46) -> str:
    """Compact axis label: ``SAE 6675 COACT · Ligand-binding site``.

    ``COACT``/``DIFF`` are the display names for the two pair blocks; the on-disk
    ``block`` values stay ``AND(a*b)``/``|a-b|`` so cached bundles and already
    exported source data keep reading back unchanged.
    """
    block = "COACT" if row["block"] == BLOCK_PRODUCT else "DIFF"
    category = str(row.get("category") or "Unclassified")
    if category == "None":
        category = "Unclassified"
    label = f"SAE {int(row['feature_id'])} {block} · {category}"
    return label if len(label) <= width else label[: width - 1] + "…"


__all__ = ["ShapBundle", "compute_shap_bundle", "short_label", "CACHE_VERSION"]
