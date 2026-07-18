# Interpretability package

`src.interpretability` contains reusable methods for explaining model outputs
and SAE-feature behavior. Command-line entry points belong under
`scripts/interpretability/`.

Current module:

- `tabpfn_retrieval.py`: TabPFN ensemble embeddings, decoder attention,
  retrieval-neighbor scoring, and active SAE-feature overlap.
- `attribution.py`: endpoint-level gradient × input aggregation.
- `ebm_effects.py`: EBM term contributions and shape-effect tables.
- `annotations.py`: shared SAE feature-table annotation joins.

Pure statistics over saved predictions or feature matrices belong in
`src.features`.
