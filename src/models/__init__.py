"""Model architectures and estimator integrations used by AuditPPI.

Consumers import concrete fitters/classes directly from their submodules
(e.g. ``from src.models.estimators.xgboost import fit_xgb_classifier`` or
``from src.models.architectures.mlp_endpoint import MLPEndpoint``); there is no
name-indirection layer.
"""
