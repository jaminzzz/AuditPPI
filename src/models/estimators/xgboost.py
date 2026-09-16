"""XGBoost estimator factories used by the fingerprint baselines."""

from __future__ import annotations

from typing import Callable, TypeVar

from conf.model import DEFAULT_SEED

T = TypeVar("T")


def fit_with_cpu_fallback(
    fit_fn: Callable[[str], T],
    *,
    try_cuda: bool,
    log_tag: str,
) -> T:
    """Try ``fit_fn("cuda")`` when requested, falling back to CPU on any error.

    ``fit_fn`` owns estimator construction / training / post-fit re-homing; this
    helper only owns the shared try-CUDA / print / retry-CPU scaffold used by
    every XGB factory in this module and by callers that need the same policy
    (e.g. :func:`src.features.feature_selection.xgb_topk_columns`).
    """
    if try_cuda:
        try:
            return fit_fn("cuda")
        except Exception as exc:  # noqa: BLE001 - XGBoost uses a generic CUDA error
            print(
                f"    [{log_tag}] GPU failed ({str(exc)[:60]}...); retrying on CPU",
                flush=True,
            )
    return fit_fn("cpu")


def fit_xgb(
    Xtr,
    ytr,
    Xva,
    yva,
    *,
    trees: int = 1000,
    depth: int = 4,
    lr: float = 0.05,
    seed: int = DEFAULT_SEED,
    cpu: bool = False,
    verbose: bool | int = False,
):
    """Fit the canonical pair classifier, retrying on CPU after a GPU error.

    ``verbose`` is forwarded to ``XGBClassifier.fit`` (default ``False`` keeps the
    fingerprint baseline silent; analysis probes pass ``50`` for progress).
    """
    import torch
    import xgboost as xgb

    def _fit(device: str):
        clf = xgb.XGBClassifier(
            n_estimators=trees,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            early_stopping_rounds=50,
            tree_method="hist",
            random_state=seed,
            device=device,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=verbose)
        if device == "cuda":
            clf.get_booster().set_param({"device": "cpu"})
        return clf

    return fit_with_cpu_fallback(
        _fit,
        try_cuda=(not cpu) and torch.cuda.is_available(),
        log_tag="fit_xgb",
    )


def fit_xgb_regressor(
    Xtr,
    ttr,
    Xva,
    tva,
    *,
    objective: str = "reg:logistic",
    eval_metric: str = "logloss",
    sample_weight=None,
    sample_weight_eval=None,
    trees: int = 600,
    depth: int = 4,
    lr: float = 0.05,
    early_stopping_rounds: int = 50,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    _log_tag: str = "fit_xgb_regressor",
):
    """Fit a participation/degree regressor, retrying on CPU after a GPU error.

    Unified over the two participation regressors that previously lived in
    separate places:

      * ``t(p)`` sequence oracle  -> ``reg:logistic`` / ``logloss`` (degree-weighted)
      * ``log1p(degree)``          -> ``reg:squarederror`` / ``rmse`` (see
        :func:`fit_xgb_logdegree`)

    ``device`` selects the training device: ``"auto"`` uses CUDA when available
    else CPU (the old ``cpu=False`` behaviour); ``"cuda"`` tries CUDA then falls
    back to CPU; ``"cpu"`` trains on CPU only. A successful CUDA fit is re-homed to
    CPU (``set_param device=cpu``) for a portable ``predict``. Passing
    ``sample_weight=None`` is equivalent to omitting it, so both call sites stay
    bit-identical to their pre-merge implementations.
    """
    import torch
    import xgboost as xgb

    def _fit(selected_device: str):
        reg = xgb.XGBRegressor(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=trees,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="hist",
            random_state=seed,
            device=selected_device,
        )
        reg.fit(
            Xtr,
            ttr,
            sample_weight=sample_weight,
            eval_set=[(Xva, tva)],
            sample_weight_eval_set=(
                [sample_weight_eval] if sample_weight_eval is not None else None
            ),
            verbose=False,
        )
        if selected_device == "cuda":
            reg.get_booster().set_param({"device": "cpu"})
        return reg

    try_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    return fit_with_cpu_fallback(_fit, try_cuda=try_cuda, log_tag=_log_tag)


def fit_xgb_logdegree(
    train_x,
    train_y,
    val_x,
    val_y,
    *,
    seed: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
    early_stopping_rounds: int,
):
    """Fit the ``log1p(degree)`` regressor (squared-error / RMSE).

    Thin wrapper over :func:`fit_xgb_regressor` so the PRING participation
    workflows keep their existing call shape while the fit logic lives in exactly
    one place. ``device`` is a bare ``"cpu"``/``"cuda"`` string here (its historic
    contract), mapped straight onto the unified fitter.
    """
    return fit_xgb_regressor(
        train_x,
        train_y,
        val_x,
        val_y,
        objective="reg:squarederror",
        eval_metric="rmse",
        trees=n_estimators,
        depth=max_depth,
        lr=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        device=device,
        _log_tag="fit_xgb_logdegree",
    )


def fit_xgb_classifier(
    Xtr,
    ytr,
    Xva,
    yva,
    *,
    seed: int = DEFAULT_SEED,
    n_estimators: int = 3000,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    device: str = "cpu",
    early_stopping_rounds: int = 200,
    eval_metric: str = "aucpr",
):
    """Fit a class-balanced binary XGBoost classifier with CUDA fallback."""
    import numpy as np
    import xgboost as xgb

    ytr = np.asarray(ytr)
    positives = max(1, int(ytr.sum()))
    negatives = max(1, int(ytr.size - ytr.sum()))
    scale_pos_weight = negatives / positives

    def _fit(selected_device: str):
        classifier = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=eval_metric,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="hist",
            random_state=seed,
            device=selected_device,
        )
        classifier.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        if selected_device == "cuda":
            classifier.get_booster().set_param({"device": "cpu"})
        return classifier

    return fit_with_cpu_fallback(
        _fit,
        try_cuda=(device == "cuda"),
        log_tag="fit_xgb_classifier",
    )


__all__ = [
    "fit_with_cpu_fallback",
    "fit_xgb",
    "fit_xgb_classifier",
    "fit_xgb_logdegree",
    "fit_xgb_regressor",
]
