# PPI fingerprint package

`src.ppi_fingerprint` implements the pooled protein-fingerprint method as a
complete downstream protocol.

```text
ppi_fingerprint/
├── config.py     Supported representations, cache mapping, native train sets
├── features.py   Pooled-cache loading and protein-pair feature assembly
└── baseline.py   Train/validation/evaluation orchestration
```

The package owns method-specific decisions such as:

- `binary`, `sae_max`, and `esmc_mean` representations;
- native training data for C3, cross-species, and RF2-PPI evaluation;
- symmetric pair features `[A*B, abs(A-B)]`;
- XGBoost Top-K selection for the TabPFN path.

Generic benchmark loading belongs in `src.data`, reusable model definitions in
`src.models`, and participation workflows in `src.participation`.
