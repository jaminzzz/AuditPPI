"""Regressor selection for the PRING sequence-to-participation workflows.

A small dispatcher over the four log-degree regressors the participation
workflows offer (XGBoost / TabPFN / random forest / ridge). The XGBoost and
TabPFN fitters live in :mod:`src.models.estimators.xgboost` /
:mod:`src.models.estimators.tabpfn`; the two sklearn one-liners are inlined here.
"""

from __future__ import annotations

import numpy as np

from src.models.estimators.tabpfn import fit_tabpfn_regressor
from src.models.estimators.xgboost import fit_xgb_logdegree

MODEL_KINDS = ("xgboost", "tabpfn", "random_forest", "ridge")


def fit_participation_model(
    model_kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
    early_stopping_rounds: int,
    tabpfn_estimators: int,
    tabpfn_subsample_samples: int,
):
    model_kind = model_kind.lower()
    if model_kind == "xgboost":
        return fit_xgb_logdegree(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            device=device,
            early_stopping_rounds=early_stopping_rounds,
        )
    if model_kind == "tabpfn":
        return fit_tabpfn_regressor(
            train_x,
            train_y,
            seed=seed,
            n_estimators=tabpfn_estimators,
            device=device,
            subsample_samples=tabpfn_subsample_samples,
        )
    if model_kind == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        depth = None if max_depth <= 0 else max_depth
        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=depth,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
        regressor.fit(train_x, train_y)
        return regressor
    if model_kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        regressor.fit(train_x, train_y)
        return regressor
    raise ValueError(f"model_kind must be one of {MODEL_KINDS}")


__all__ = ["MODEL_KINDS", "fit_participation_model"]
