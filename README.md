# AuditPPI: Auditing Protein-Protein Interaction Signals with Sparse Autoencoder Fingerprints

A post-hoc interpretability framework that audits sequence-based protein–protein
interaction (PPI) predictors. Rather than treating benchmark accuracy as a single
biological quantity, AuditPPI uses the sparse-autoencoder (SAE) features of a
**frozen** protein language model (ESM-C / ESMC-6B) as an interpretable vocabulary
to decompose *why* a predictor scores a pair as interacting, across three scales:

| Scale | Question | Confounded signal it isolates |
|-------|----------|-------------------------------|
| **Protein** (Layer 1) | Is node degree / "hub-ness" recoverable from sequence alone? | network-topology shortcut |
| **Pair** (Layer 2) | Does pairwise SAE-feature concordance inflate a leakage-controlled benchmark, and is it a retrieval shortcut? | semantic concordance / negative-sampling bias |
| **Residue** (Layer 3) | Are benchmark-important features actually interface-specific, or just surface propensity? | genuine interface grounding |

The framework is **diagnostic, not a new PPI scorer**: it re-examines existing
benchmark splits, cached representations, and trained-model artifacts (feature
rankings, TabPFN attention neighbourhoods). See
`manuscripts/ncs_brief_communication_draft_v2.md` for the full write-up.

---

## Layout

```
AuditPPI/
├── conf/paths.py          Single source of truth for every path (edit here to relocate).
├── .project-root          Anchor marker; scripts find ROOT by walking up to this file.
├── data/
│   ├── audit/             Audit outputs + source data for figures.
│   ├── sae/               SAE artifacts: per-protein reps, feature table, seq caches.
│   └── tabpfn/            TabPFN feature ranking + retrieval explanations.
├── external/              Symlinks to shared read-only data lakes (see below).
├── src/
│   ├── data/              Benchmark loaders, sequence I/O, and sparse SAE cache codec.
│   ├── eval/              Scoring metrics + participation-oracle diagnostics.
│   ├── features/          Unified ESM-C/ESM-2 protein features + pair assembly.
│   ├── analysis/          Reusable statistical analyses over cached results.
│   ├── participation/     Protein participation labels, predictors, and protocols.
│   ├── interpretability/  SAE/model attribution, retrieval, and feature explanations.
│   ├── models/            Reusable architectures + external estimator integrations.
│   ├── runtime/           Device selection and executable runtime setup.
│   └── ppi_fingerprint/   PPI fingerprint features, protocol config, and orchestration.
├── scripts/               Pipeline, grouped by audit stage (see below).
├── tests/                 CPU unit tests for data, features, models, and analyses.
├── manuscripts/figures/   Final figures (SVG/PDF/PNG) written by scripts/figures/.
└── environment/envs.md    Conda environments + which env each script needs.
```

### Path configuration

Every script resolves paths through `conf/paths.py`. There are **no hardcoded
absolute paths** in `scripts/` or `src/`. To relocate the project or repoint a
data source, edit `conf/paths.py` or the entries under `external/` — nothing else.

Scripts bootstrap the import root by walking up to the `.project-root` anchor, so
they work regardless of which `scripts/<group>/` subfolder they live in.

Main downstream models live under `src/models/`: project-owned PyTorch modules
are in `architectures/`, while third-party `fit/predict` integrations are in
`estimators/`. The `src/ppi_fingerprint/` package directly composes these models
with its pooled fingerprint features and benchmark-specific training protocol.
Published baseline implementations such as DeepNano, PPLM, FlashPPI, and MINT
remain isolated under `baselines/` and their feature scripts under
`scripts/baseline/`.

### External data (`external/`)

`external/` mixes two kinds of entries. A few large assets that this project now
**owns outright** live here as *real directories* (moved out of the retired
`SAE_PPI` repo). The rest are *symlinks* into shared read-only data lakes that
other projects also use. If a shared source moves, re-point the symlink (or the
corresponding constant in `conf/paths.py`) — nothing else needs to change.

| Entry | Kind | Purpose |
|-------|------|---------|
| `ESMC/ESMC-6B`, `ESMC/ESMC-6B-sae-k64-codebook16384` | real dir (owned) | ESM-C-6B weights + layer-60 SAE (only needed to *re-cache* features) |
| `InterPLM` | real dir (owned) | vendored InterPLM SAE reference code + checkpoints (historical / provenance) |
| `TabPFN` | upstream Git clone | Official PriorLabs TabPFN source used when the package is not installed |
| `PPI_data` | symlink | PDB_PPI complexes + benchmark TSVs |

Two benchmark CSV sets are now in-project (only the CSVs the audit reads were
copied in; the tens of GB of upstream embeddings that shared each source
directory were left behind):

| In-project dir | Constant | Size | Contents |
|----------------|----------|------|----------|
| `data/cross_species/` | `CROSS_SPECIES_DIR` | 509M | 7 cross-species qrels CSVs |
| `data/rapppid_c3/` | `RAPPPID_C3_DIR` (+ `C3_{TRAIN,VAL,TEST}_CSV`) | 55M | 3 RAPPPID C3 split CSVs |
| `data/rapppid_c1/` | `RAPPPID_C1_DIR` (+ `C1_{TRAIN,VAL,TEST}_CSV`) | 119M | 3 RAPPPID C1 split CSVs |
| `data/rapppid_c2/` | `RAPPPID_C2_DIR` (+ `C2_{TRAIN,VAL,TEST}_CSV`) | 86M | 3 RAPPPID C2 split CSVs |

The C1/C2 CSVs are extracted from the same `C3_H5` store (all three Park &
Marcotte leakage levels live under `interactions/{c1,c2,c3}/`) by
`scripts/prep/export_rapppid_clevel_csvs.py`, in the same `(query,text,label)`
schema as the C3 set. `conf.paths.RAPPPID_CLEVEL_CSVS` maps level → split → CSV,
and `datasets.load_clevel(level, split)` / `load_benchmark("c1"|"c2"|"c3")`
load any of them. C3 is the strictest (both proteins unseen); C1 both seen,
C2 one seen.

### Baseline repos (`baselines/`)

The upstream baseline repositories (DeepNano, FlashPPI, mint, PIC, PPLM, PRING,
RoseTTAFold2-PPI, SWING, pllm-ppi-data-leakage) live in-project under `baselines/`.
Three of them supply audit inputs, wired through `conf/paths.py`:

| Constant | Path under `baselines/` | Used by |
|----------|-------------------------|---------|
| `PIC_DATA` | `PIC/data/human_data.pkl` | PIC essentiality classifier |
| `PRING_ROOT` | `PRING/data_process/pring_dataset` | PRING participation / endpoint-additive audits |
| `C3_H5` | `pllm-ppi-data-leakage/.../rapppid_[...].h5` | C1/C2/C3 pair + sequence store (datasets loader; C1/C2 CSV extraction) |

---

## Scripts by stage

Run with the conda env noted in `environment/envs.md`; most use `genmol`.

**`features/`** — dataset-neutral formal feature extraction. It produces
ESM-C layers 60/80 dense mean/max and SAE mean/max/binary features, or ESM-2
final-layer dense mean/max and InterPLM-SAE mean/max/binary features. The same
directory builds the primary symmetric pair representation
`[A*B, abs(A-B)]`, its product/absolute-difference ablations, and the AB/BA
concat protocol. See `scripts/features/README.md`.

**`prep/`** — one-time data preparation (re-caches SAE features, builds interface
masks, collects benchmark sequences). GPU + model weights required for the
`cache_*` scripts. You do not need to re-run these; their outputs are already in `data/`.

**`baseline/`** — pooled-SAE fingerprint baseline + participation-oracle diagnostics.
Baseline-specific feature entry points for DeepNano, PPLM, FlashPPI and MINT
live under `scripts/baseline/features/`; see its README for the different
per-protein versus pair-conditioned cache contracts.

**`analysis/`** — executable statistical analyses over cached features and
predictions. Reusable implementations live in `src/analysis/`.

**`audit_protein/` / `src/participation/`** — executable and reusable
sequence-to-participation, hubness, and PRING cross-species workflows.

**`interpretability/`** — executable model/SAE explanation workflows such as
TabPFN retrieval attention and active-feature overlap. Reusable algorithms live
in `src/interpretability/`.

**`audit_protein/`** (Layer 1) — sequence→participation oracles and
high-participation classifiers on PRING / C3 / PIC.

**`audit_pair/`** (Layer 2) — C3 endpoint-additive models (linear / GAM / EBM / MLP),
TabPFN retrieval-attention audit, negative-sampling & localization confound analyses.

**`audit_residue/`** (Layer 3) — interface SAE enrichment (with surface-matched control)
and contact-pair compatibility on PDB_PPI structures.

**`figures/`** — regenerate all manuscript figures into `manuscripts/figures/`.

**`smoke/`** — fast end-to-end sanity checks (no GPU).

---

## Quick start / reproduction

```bash
# 0. pick a python (see environment/envs.md); genmol covers most scripts
PY=/data/wmzhu/anaconda3/envs/genmol/bin/python

# 1. sanity check: participation-oracle math + real RF2-PPI load (no GPU)
$PY scripts/smoke/smoke_eval.py

# 2. reproduce a baseline (pooled SAE fingerprint + classifier)
PYTHONPATH=. $PY scripts/baseline/run_ppi_fingerprint_baseline.py --help

# 3. regenerate the main figures from cached source data
PYTHONPATH=. $PY scripts/figures/plot_audit_figures.py
```

Scripts insert `ROOT` onto `sys.path` themselves, but running from the project
root with `PYTHONPATH=.` is the safe default.

### Figure ↔ source-data map

All figure scripts import `AUDIT` (source data) and `FIGURES` (output) from
`conf/paths.py`:

| Figure | Script | Reads from |
|--------|--------|-----------|
| `figure1_auditppi_concept`, `figure2_auditppi_results` | `plot_audit_figures.py` | `data/audit/` |
| `human_sae_degree_pca` | `plot_human_sae_degree_pca.py` | `data/audit/pring_participation` |
| `human_sae_feature_distribution` | `plot_human_sae_feature_distribution.py` | `data/audit/pring_participation` + feature table |
| `human_sae_hub_category_association` | `plot_human_sae_hub_category_association.py` | `data/audit/pring_participation` + feature table |
| `supplementary_s1/s2/s3` | `plot_supplementary_audit_figures.py` | `data/audit/` + `data/sae/supplementary_inputs` |
| `supplementary_s5_external_baseline_comparison` | `plot_supplementary_baseline_comparison.py` | `data/audit/ppi_fingerprint` + `supplementary_inputs` |
| `tabpfn_case_4365_retrieval` | `plot_tabpfn_case_4365.py` | `data/tabpfn/tabpfn_retrieval_explanations` |

---

## Provenance
Large shared data lakes remain in place and are reached via `external/` symlinks; project-exclusive data was moved in.
