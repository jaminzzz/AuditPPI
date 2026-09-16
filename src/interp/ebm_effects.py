"""Feature contribution and shape-effect tables for endpoint EBMs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.interp.annotations import add_sae_annotations


def endpoint_matrix(
    split: dict[str, np.ndarray],
    feature_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate(
        [split["a"][:, feature_ids], split["b"][:, feature_ids]], axis=0
    ).astype(np.float32, copy=False)
    labels = np.concatenate([split["y"], split["y"]]).astype(np.int8, copy=False)
    return features, labels


def ebm_feature_tables(
    ebm,
    split: dict[str, np.ndarray],
    feature_ids: np.ndarray,
    selected_features: pd.DataFrame,
    representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-feature importance and per-bin shape-effect tables."""
    endpoint_features, _ = endpoint_matrix(split, feature_ids)
    contributions = ebm.eval_terms(endpoint_features)
    if contributions.ndim != 2:
        raise RuntimeError(f"unexpected eval_terms shape: {contributions.shape}")
    summary = pd.DataFrame(
        {
            "feature_id": feature_ids.astype(int),
            "importance_mean_abs_test_contribution": np.mean(
                np.abs(contributions), axis=0
            ),
            "mean_signed_test_contribution": np.mean(contributions, axis=0),
            "effect_range": [float(np.max(scores) - np.min(scores)) for scores in ebm.term_scores_],
            "max_abs_effect": [float(np.max(np.abs(scores))) for scores in ebm.term_scores_],
            "term_name": ebm.term_names_,
        }
    )
    summary = summary.merge(selected_features, on="feature_id", how="left")
    summary = add_sae_annotations(summary, representation).sort_values(
        "importance_mean_abs_test_contribution", ascending=False
    )

    effect_rows = []
    for term_index, feature_id in enumerate(feature_ids):
        for bin_index, effect in enumerate(np.asarray(ebm.term_scores_[term_index])):
            effect_rows.append(
                {
                    "feature_id": int(feature_id),
                    "term_index": int(term_index),
                    "bin_index": int(bin_index),
                    "effect": float(effect),
                    "term_name": ebm.term_names_[term_index],
                }
            )
    effects = add_sae_annotations(pd.DataFrame(effect_rows), representation)
    return summary, effects


__all__ = ["ebm_feature_tables", "endpoint_matrix"]
