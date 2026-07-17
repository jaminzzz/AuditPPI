"""External estimators exposed through AuditPPI's stable interfaces."""

from .ebm import (
    endpoint_alpha,
    endpoint_pair_predictions,
    fit_endpoint_ebm,
    make_endpoint_ebm,
)
from .tabpfn import fit_tabpfn, predict_proba_chunked
from .xgboost import fit_xgb, fit_xgb_classifier, fit_xgb_regressor

__all__ = [
    "endpoint_alpha",
    "endpoint_pair_predictions",
    "fit_endpoint_ebm",
    "fit_tabpfn",
    "fit_xgb",
    "fit_xgb_classifier",
    "fit_xgb_regressor",
    "make_endpoint_ebm",
    "predict_proba_chunked",
]
