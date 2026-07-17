"""Protein participation labels, features, predictors, and experiment pipelines.

Exports are resolved lazily so lightweight label/feature users do not eagerly
import model backends such as XGBoost, TabPFN, or PyTorch.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "METHODS": ("src.participation.labels", "METHODS"),
    "ParticipationLabels": ("src.participation.labels", "ParticipationLabels"),
    "full_graph_participation_labels": ("src.participation.labels", "full_graph_participation_labels"),
    "load_pring_human_split": ("src.participation.labels", "load_pring_human_split"),
    "FEATURE_KINDS": ("src.participation.features", "FEATURE_KINDS"),
    "FORMAL_FEATURE_KINDS": ("src.participation.features", "FORMAL_FEATURE_KINDS"),
    "sequence_features": ("src.participation.features", "sequence_features"),
    "MODEL_KINDS": ("src.participation.models", "MODEL_KINDS"),
    "OUT_DIR": ("src.participation.config", "OUT_DIR"),
    "PRING_ROOT": ("conf.paths", "PRING_ROOT"),
    "DEFAULT_PRING_CACHE": ("src.participation.config", "DEFAULT_PRING_CACHE"),
    "CROSS_SPECIES": ("src.participation.config", "CROSS_SPECIES"),
    "PAIR_EVALS": ("src.participation.config", "PAIR_EVALS"),
    "prepare_features": ("src.participation.cache", "prepare_features"),
    "species_cache_path": ("src.participation.cache", "species_cache_path"),
    "run_pring_participation_oracle": ("src.participation.pipeline", "run_pring_participation_oracle"),
    "run_pring_cross_species_generalization": (
        "src.participation.cross_species",
        "run_pring_cross_species_generalization",
    ),
    "run_participation_oracle": ("src.participation.predictor", "run_participation_oracle"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(module_name), attribute)


__all__ = list(_EXPORTS)
