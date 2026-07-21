"""Model architectures and estimator integrations used by AuditPPI.

Consumers import concrete fitters/classes directly from their submodules
(e.g. ``from src.models.estimators.xgboost import fit_xgb_classifier`` or
``from src.models.architectures.endpoint_mlp import EndpointMLP``); there is no
name-indirection layer.
"""
