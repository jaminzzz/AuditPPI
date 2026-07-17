# Participation package

`src.participation` owns protein-level participation and hubness workflows.
It is separate from generic `src.analysis` because it defines domain-specific
labels, feature contracts, model fitting, calibration, and experiment protocol.

- `labels.py`: full-graph degree/participation labels and PRING splits.
- `features.py`: sequence features and participation feature-name contracts.
- `config.py`: shared output, cache, species, and pair-evaluation constants.
- `cache.py`: pooled-cache lookup, protein alignment, and feature split assembly.
- `models.py`: participation regressors and estimator dispatch.
- `calibration.py`: conversion from predicted log-degree to degree.
- `importance.py`: XGBoost feature selection and importance tables.
- `evaluation.py`: participation node/pair metrics and prediction-table output.
- `pipeline.py`: PRING within-human workflow and stable compatibility exports.
- `cross_species.py`: human-to-species zero-shot participation workflow.
- `predictor.py`: C3/cross-species pooled-fingerprint participation oracle.

Executable entry points remain in `scripts/audit_protein/`.
