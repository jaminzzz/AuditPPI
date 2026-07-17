"""XGBoost estimator factories used by the fingerprint baselines."""

from __future__ import annotations


def fit_xgb(
    Xtr,
    ytr,
    Xva,
    yva,
    *,
    trees: int = 1000,
    depth: int = 4,
    lr: float = 0.05,
    seed: int = 42,
    cpu: bool = False,
):
    """Fit the canonical pair classifier, retrying on CPU after a GPU error."""
    import torch
    import xgboost as xgb

    def _fit(device):
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
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return clf

    if (not cpu) and torch.cuda.is_available():
        try:
            clf = _fit("cuda")
            clf.get_booster().set_param({"device": "cpu"})
            return clf
        except Exception as exc:  # noqa: BLE001 - XGBoost uses a generic CUDA error
            print(
                f"    [fit_xgb] GPU failed ({str(exc)[:60]}...); retrying on CPU",
                flush=True,
            )
    return _fit("cpu")


def fit_xgb_regressor(
    Xtr,
    ttr,
    Xva,
    tva,
    *,
    sample_weight=None,
    sample_weight_eval=None,
    trees: int = 600,
    depth: int = 4,
    lr: float = 0.05,
    seed: int = 42,
    cpu: bool = False,
):
    """Fit the degree-weighted sequence-to-participation regressor."""
    import torch
    import xgboost as xgb

    def _fit(device):
        reg = xgb.XGBRegressor(
            objective="reg:logistic",
            eval_metric="logloss",
            n_estimators=trees,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            tree_method="hist",
            random_state=seed,
            device=device,
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
        return reg

    if (not cpu) and torch.cuda.is_available():
        try:
            reg = _fit("cuda")
            reg.get_booster().set_param({"device": "cpu"})
            return reg
        except Exception as exc:  # noqa: BLE001 - XGBoost uses a generic CUDA error
            print(
                f"    [fit_xgb_regressor] GPU failed ({str(exc)[:60]}...); retrying on CPU",
                flush=True,
            )
    return _fit("cpu")


def fit_xgb_classifier(
    Xtr,
    ytr,
    Xva,
    yva,
    *,
    seed: int = 42,
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

    if device == "cuda":
        try:
            return _fit("cuda")
        except Exception as exc:  # noqa: BLE001 - backend reports generic CUDA errors
            print(
                f"    [fit_xgb_classifier] CUDA failed "
                f"({str(exc)[:100]}...); retrying on CPU",
                flush=True,
            )
    return _fit("cpu")


__all__ = ["fit_xgb", "fit_xgb_classifier", "fit_xgb_regressor"]
