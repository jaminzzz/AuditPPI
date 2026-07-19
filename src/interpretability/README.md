# Interpretability package

`src.interpretability` contains reusable methods for explaining model outputs
and SAE-feature behavior. Command-line entry points belong under
`scripts/interpretability/`.

Current modules:

- `pair_probe.py`: compact SAE pair-feature probes (ranking I/O, sym top-k
  materialization, embedding-split load, thin fit wrappers). Used by analysis
  and TabPFN retrieval scripts; metrics live in `src.eval.classification`.
- `tabpfn_retrieval.py`: TabPFN ensemble embeddings, decoder attention,
  retrieval-neighbor scoring, and active SAE-feature overlap.
- `attribution.py`: endpoint-level gradient × input aggregation.
- `ebm_effects.py`: EBM term contributions and shape-effect tables.
- `annotations.py`: shared SAE feature-table annotation joins.

Feature extraction / pooled-cache assembly belongs in `src.features`.
