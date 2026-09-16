# PPI fingerprint package

`src.ppi_fingerprint` implements the pooled protein-fingerprint method as a
complete downstream protocol. Primary outputs land under
`results/main/ppi_fingerprint/{family}/{model}/seed_{S}/`.

```text
ppi_fingerprint/
├── config.py     Representations, caches, native train, result path helpers
└── baseline.py   Train-once / multi-eval orchestration
```

The package owns the method-specific *protocol*:

- `binary`, `sae_max`, and `esmc_mean` representations;
- native training data for each benchmark family;
- train → (shared fit) → multi-eval → score runs;
- multi-seed matrix (`FINGERPRINT_SEEDS = 42, 43, 44`); every run writes under `seed_{S}/`;
- result layout under `results/main/ppi_fingerprint/`.

The reusable building blocks it composes live in the shared layers: the
pooled-cache / pair-row assembly and symmetric pair features (`[A*B, abs(A-B)]`)
in `src.features` (`protein_cache`, `pairs`), XGB Top-K column selection in
`src.features.feature_selection`, class-stratified subsampling in
`src.features.sampling`, benchmark loading in `src.data`, models in `src.models`,
and the participation workflows in `scripts/audit_protein/`.
