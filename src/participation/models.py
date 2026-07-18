"""Regressors used by protein participation analyses."""

from __future__ import annotations

import inspect

import numpy as np

from conf.paths import TABPFN_SRC
from src.runtime import ensure_on_sys_path

MODEL_KINDS = ("xgboost", "tabpfn", "random_forest", "ridge")


def fit_xgb_logdegree(
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
):
    import xgboost as xgb

    def fit(selected_device: str):
        regressor = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="hist",
            random_state=seed,
            device=selected_device,
        )
        regressor.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        if selected_device == "cuda":
            regressor.get_booster().set_param({"device": "cpu"})
        return regressor

    if device == "cuda":
        try:
            return fit("cuda")
        except Exception as exc:  # noqa: BLE001
            print(
                f"    [fit_xgb_logdegree] CUDA failed ({str(exc)[:80]}...); "
                "retrying on CPU",
                flush=True,
            )
    return fit("cpu")


def fit_tabpfn_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    device: str,
    subsample_samples: int,
):
    try:
        from tabpfn import TabPFNRegressor
    except (ImportError, ModuleNotFoundError):
        # Fall back to the vendored TabPFN source tree when the package is absent.
        ensure_on_sys_path(TABPFN_SRC)
        from tabpfn import TabPFNRegressor

    kwargs = {
        "device": device,
        "n_estimators": n_estimators,
        "ignore_pretraining_limits": True,
        "random_state": seed,
    }
    if subsample_samples > 0:
        kwargs["inference_config"] = {"SUBSAMPLE_SAMPLES": subsample_samples}
    signature = inspect.signature(TabPFNRegressor)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    regressor = TabPFNRegressor(**kwargs)
    regressor.fit(train_x, train_y)
    return regressor


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


__all__ = [
    "MODEL_KINDS",
    "fit_participation_model",
    "fit_tabpfn_regressor",
    "fit_xgb_logdegree",
]
