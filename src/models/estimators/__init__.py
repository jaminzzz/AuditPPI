"""External estimators exposed through AuditPPI's stable interfaces."""

from .ebm import (
    endpoint_alpha,
    endpoint_pair_predictions,
    make_endpoint_ebm,
)
from .tabpfn import fit_tabpfn, fit_tabpfn_regressor, predict_proba_chunked
from .xgboost import (
    fit_with_cpu_fallback,
    fit_xgb,
    fit_xgb_classifier,
    fit_xgb_logdegree,
    fit_xgb_regressor,
)

# Participation (log-degree) regressors offered by the PRING workflows.
PARTICIPATION_MODEL_KINDS = ("xgboost", "tabpfn")

__all__ = [
    "PARTICIPATION_MODEL_KINDS",
    "endpoint_alpha",
    "endpoint_pair_predictions",
    "fit_tabpfn",
    "fit_tabpfn_regressor",
    "fit_with_cpu_fallback",
    "fit_xgb",
    "fit_xgb_classifier",
    "fit_xgb_logdegree",
    "fit_xgb_regressor",
    "make_endpoint_ebm",
    "predict_proba_chunked",
]
