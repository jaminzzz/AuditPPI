# PPI fingerprint package

`src.ppi_fingerprint` implements the pooled protein-fingerprint method as a
complete downstream protocol.

```text
ppi_fingerprint/
├── config.py     Supported representations, cache mapping, native train sets
└── baseline.py   Train/validation/evaluation orchestration
```

The package owns the method-specific *protocol*:

- `binary`, `sae_max`, and `esmc_mean` representations;
- native training data for C3, cross-species, and RF2-PPI evaluation;
- how those pieces are composed into the train → eval → score run.

The reusable building blocks it composes live in the shared layers: the
pooled-cache / pair-row assembly and symmetric pair features (`[A*B, abs(A-B)]`)
in `src.features` (`protein_cache`, `pairs`), XGB Top-K column selection in
`src.features.feature_selection`, class-stratified subsampling in
`src.features.sampling`, benchmark loading in `src.data`, models in `src.models`,
and the participation workflows in `scripts/audit_protein/`.
