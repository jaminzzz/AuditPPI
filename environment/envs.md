# Conda environments

AuditPPI scripts run under one of three conda environments. Each script's header
comment names the environment it was validated with; the table below is the
authoritative mapping. There is no `python` on the bare `PATH` — always call the
interpreter by absolute path.

| env | interpreter | purpose |
|-----|-------------|---------|
| `genmol`   | `/data/wmzhu/anaconda3/envs/genmol/bin/python`   | default: baselines, participation oracles, most figures, GAM/MLP additive variants, smoke tests |
| `primenet` | `/data/wmzhu/anaconda3/envs/primenet/bin/python` | anything needing `interpret` (EBM additive variant) or the local ESM-C transformers fork (PIC caching, UniProt localization fetch) |
| `E1`       | `/data/wmzhu/anaconda3/envs/E1/bin/python`       | Biohub ESM-C / SAE model loading for feature caching (`scripts/cache/cache_pdb_afdb_sae.py`, `scripts/cache/cache_pring_human_esmc_sae.py`) |

Most scripts add the project root to `sys.path` themselves via the `.project-root`
anchor, so `PYTHONPATH` is usually unnecessary. A few (the ESM-C caching scripts)
document `PYTHONPATH=.` in their header — keep that when copying their command.

## Key package versions

Captured from the live environments (2026-07). Only versions that matter for
reproducibility are listed.

| package | genmol | primenet | E1 |
|---------|--------|----------|-----|
| python        | 3.10.0        | 3.10.19      | 3.12.12 |
| torch         | 2.6.0+cu126   | 2.2.0+cu118  | 2.7.0+cu126 |
| numpy         | 1.26.4        | 1.26.4       | 2.4.2 |
| pandas        | 2.1.0         | 2.1.0        | 2.3.3 |
| scikit-learn  | 1.2.2         | 1.7.2        | 1.9.0 |
| scipy         | 1.15.3        | 1.15.3       | 1.16.3 |
| transformers  | 4.52.4        | 4.52.4       | 4.57.6 |
| matplotlib    | 3.10.7        | 3.10.8       | 3.10.9 |
| pyarrow       | 22.0.0        | 24.0.0       | 17.0.0 |
| xgboost       | 3.2.0         | –            | – |
| tabpfn        | 8.0.6         | 8.0.6        | – |
| interpret     | –             | 0.7.8        | – |
| mdtraj        | 1.10.3        | –            | 1.11.1 |
| h5py          | 3.16.0        | –            | – |
| hdf5plugin    | 6.0.0         | –            | – |

Notes:
- The unified formal extractor under `scripts/features/` uses `E1` for
  ESM-C-6B layers 60/80 and `genmol` for ESM-2-650M + InterPLM SAE. Pair
  feature construction is model-free and runs in `genmol`.
- Baseline feature entry points live under `scripts/baseline/features/`.
  DeepNano uses `E1` for ESM-C and `genmol` for ESM-2; the local PPLM and MINT
  extractors use `genmol`; FlashPPI uses the local transformers implementation
  available in `E1`.
- `tabpfn` is cloned under `external/TabPFN/src` and put on `sys.path`
  by `conf.paths.TABPFN_SRC`; the pip `tabpfn==8.0.6` above is the runtime dep set.
- `scripts/analysis/run_cross_species_tabpfn_topk.py` and
  `scripts/interpretability/explain_tabpfn_retrieval.py` use `genmol`.
- `interpret` is only in `primenet`, so `run_c3_sae_endpoint_additive_ebm.py`
  (Explainable Boosting Machine) must run there. The `_binned_gam` and `_mlp`
  additive variants use only sklearn/torch and run under `genmol`.
- `h5py` + `hdf5plugin` (blosc filter) are only needed to read the RAPPPID C3
  HDF5 sequence store in `export_c3_pair_id_alignment.py` — `genmol` only.
- ESM-C / SAE model loading (`E1`) is only needed to *regenerate* SAE feature
  caches. The audit and figure pipeline reads precomputed caches and never
  touches the models.
