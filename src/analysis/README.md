# Analysis package

`src.analysis` contains generic reusable statistical analyses over cached
features, labels, and predictions. Command-line orchestration belongs under
`scripts/analysis/`.

Current modules:

- `cross_species_probe.py`: stratified sampling, ranked SAE feature selection,
  compact symmetric pair matrices, calibration/classification statistics, and
  cross-species probe evaluation.
Protein participation is a distinct project domain and lives in
`src.participation`.

Code that inspects model internals, attention, gradients, or feature effects
belongs in `src.interpretability` instead.
