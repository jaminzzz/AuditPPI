"""Lazy name registry for model architectures and estimator integrations."""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType

_ARCHITECTURE_PATHS = {
    "dual_tower": "src.models.architectures.dual_tower:DualTowerNet",
    "endpoint_mlp": "src.models.architectures.endpoint_mlp:EndpointMLP",
    "tabm_pair": "src.models.architectures.tabm_pair:TabMPair",
}

_ESTIMATOR_PATHS = {
    "xgboost": "src.models.estimators.xgboost:fit_xgb",
    "xgboost_classifier": "src.models.estimators.xgboost:fit_xgb_classifier",
    "xgboost_regressor": "src.models.estimators.xgboost:fit_xgb_regressor",
    "tabpfn": "src.models.estimators.tabpfn:fit_tabpfn",
    "endpoint_ebm": "src.models.estimators.ebm:make_endpoint_ebm",
}

ARCHITECTURES = MappingProxyType(_ARCHITECTURE_PATHS)
ESTIMATORS = MappingProxyType(_ESTIMATOR_PATHS)


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _resolve(path: str):
    module_name, attribute = path.split(":", 1)
    return getattr(import_module(module_name), attribute)


def get_architecture_class(name: str):
    """Resolve an architecture class without eagerly importing optional packages."""
    key = _normalize(name)
    try:
        return _resolve(_ARCHITECTURE_PATHS[key])
    except KeyError as exc:
        raise KeyError(
            f"unknown architecture {name!r}; choose from {tuple(ARCHITECTURES)}"
        ) from exc


def build_architecture(name: str, **kwargs):
    return get_architecture_class(name)(**kwargs)


def get_estimator_factory(name: str):
    """Resolve an estimator constructor/fitter by its stable registry name."""
    key = _normalize(name)
    try:
        return _resolve(_ESTIMATOR_PATHS[key])
    except KeyError as exc:
        raise KeyError(f"unknown estimator {name!r}; choose from {tuple(ESTIMATORS)}") from exc


__all__ = [
    "ARCHITECTURES",
    "ESTIMATORS",
    "build_architecture",
    "get_architecture_class",
    "get_estimator_factory",
]
