"""Degree calibration functions for participation predictors."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression

CALIBRATIONS = ("log_linear", "scale", "none")


def fit_degree_calibration(
    kind: str,
    predicted_log_degree: np.ndarray,
    validation_degree: np.ndarray,
) -> tuple[dict, Callable[[np.ndarray], np.ndarray]]:
    if kind not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {CALIBRATIONS}")
    if kind == "none":
        return {"kind": "none"}, lambda values: np.maximum(np.expm1(values), 0.0)

    raw_degree = np.maximum(np.expm1(predicted_log_degree), 0.0)
    if kind == "scale":
        denominator = float(raw_degree.sum())
        scale = float(validation_degree.sum() / denominator) if denominator > 0 else 1.0
        return (
            {"kind": "scale", "scale": scale},
            lambda values: np.maximum(np.expm1(values), 0.0) * scale,
        )

    regression = LinearRegression()
    regression.fit(predicted_log_degree.reshape(-1, 1), np.log1p(validation_degree))
    coefficient = float(regression.coef_[0])
    intercept = float(regression.intercept_)

    def calibrate(values: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(coefficient * values + intercept), 0.0)

    return (
        {"kind": "log_linear", "coef": coefficient, "intercept": intercept},
        calibrate,
    )


__all__ = ["CALIBRATIONS", "fit_degree_calibration"]
