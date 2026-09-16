# Interpretability package

`src.interp` contains reusable methods for explaining model outputs
and SAE-feature behavior. Command-line entry points live under
`scripts/audit_pair/` (e.g. `explain_tabpfn_retrieval_cases.py`,
`run_cross_species_tabpfn_topk.py`).

Current modules:

- `pair_probe.py`: compact SAE pair-feature probes (ranking I/O, sym top-k
  materialization, embedding-split load, thin fit wrappers). Product/absdiff
  column offsets use the backbone codebook width via `sae_dim=` /
  `sae_dim_for_backbone` (ESM-C 16384, ESM-2 10240). Used by analysis and
  TabPFN retrieval scripts; metrics live in `src.eval.classification`.
- `tabpfn_retrieval.py`: TabPFN ensemble embeddings, decoder attention,
  retrieval-neighbor scoring, and active SAE-feature overlap.
- `attribution.py`: endpoint-level gradient × input aggregation.
- `ebm_effects.py`: EBM term contributions and shape-effect tables.
- `annotations.py`: shared SAE feature-table annotation joins.

Feature extraction / pooled-cache assembly belongs in `src.features`.
