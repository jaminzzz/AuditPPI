"""No-interaction endpoint EBM helpers for additive protein-pair scoring."""

from __future__ import annotations

import numpy as np

from conf.model import DEFAULT_SEED


def make_endpoint_ebm(
    *,
    feature_names=None,
    validation_size: float = 0.15,
    outer_bags: int = 8,
    learning_rate: float = 0.01,
    max_rounds: int = 5000,
    early_stopping_rounds: int = 100,
    min_samples_leaf: int = 4,
    max_bins: int = 256,
    n_jobs: int = -2,
    # Uses the project-wide seed. This line previously pinned 7 to reproduce the
    # manuscript's originally cached EBM/endpoint-additive results; that carve-out
    # was retired in favor of a single project seed, so re-runs now differ from
    # those first cached fits (bagging + validation split reseed on 42).
    random_state: int = DEFAULT_SEED,
):
    """Construct the endpoint EBM with pair interactions disabled."""
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(
        feature_names=feature_names,
        interactions=0,
        validation_size=validation_size,
        outer_bags=outer_bags,
        learning_rate=learning_rate,
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
        min_samples_leaf=min_samples_leaf,
        max_bins=max_bins,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def fit_endpoint_ebm(x, y, **kwargs):
    """Construct and fit a no-interaction endpoint EBM."""
    model = make_endpoint_ebm(**kwargs)
    model.fit(x, y)
    return model


def endpoint_alpha(ebm, x: np.ndarray) -> np.ndarray:
    """Return the endpoint contribution after removing the EBM intercept."""
    intercept = float(np.ravel(ebm.intercept_)[0])
    return np.asarray(ebm.decision_function(x)) - intercept


def endpoint_pair_predictions(
    ebm, xa: np.ndarray, xb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pair probability and the two no-intercept endpoint scores."""
    alpha_a = endpoint_alpha(ebm, xa)
    alpha_b = endpoint_alpha(ebm, xb)
    logits = alpha_a + alpha_b
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return probabilities, alpha_a, alpha_b


__all__ = [
    "endpoint_alpha",
    "endpoint_pair_predictions",
    "fit_endpoint_ebm",
    "make_endpoint_ebm",
]
