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
`manuscripts/ncs_brief_communication_draft.md` for the full write-up.

---

## Layout

```
AuditPPI/
├── conf/paths.py          Single source of truth for every path (edit here to relocate).
├── .project-root          Anchor marker; scripts find ROOT by walking up to this file.
├── data/
│   ├── raw/               Raw benchmark CSVs + RF2-PPI FASTAs (pair-level inputs).
│   └── sae/               SAE artifacts (see the cache pipeline below):
│       ├── seq_caches/     Stage 1: per-sequence pooled ESM-C/SAE fingerprints.
│       ├── protein_caches/ Stage 2: per-dataset protein caches (UniProt-ID indexed).
│       ├── pair_caches/    Stage 3: per-pair engineered feature matrices (was reps/).
│       ├── residue_caches/ Residue-level SAE cache for the interface-grounding audit.
│       └── feature_table/  ESMC-SAE feature reference table.
├── results/               Audit PRODUCTS (sibling of data/, not inside it):
│   ├── audit_protein/     Layer 1 -- participation / hubness / essentiality.
│   ├── audit_pair/        Layer 2 -- endpoint-additive, TabPFN, neg-sampling.
│   ├── audit_residue/     Layer 3 -- interface grounding.
│   ├── analysis/          Cross-layer analyses (e.g. C3↔PRING feature overlap).
│   └── misc/              Baseline fingerprint, cross-layer figure data, spare caches.
├── external/              Symlinks to shared read-only data lakes (see below).
├── src/
│   ├── data/              Benchmark loaders, PRING graph labels, sequence I/O, and sparse SAE cache codec.
│   ├── eval/              Scoring metrics + participation-oracle diagnostics.
│   ├── features/          Unified ESM-C/ESM-2 protein features + pair assembly.
│   ├── interpretability/  SAE/model attribution, pair probes, TabPFN retrieval, feature explanations.
│   ├── models/            Reusable architectures + external estimator integrations.
│   ├── runtime/           Device selection, seeding, and vendored-import setup.
│   ├── experiments/       Result envelope, experiment registry, run provenance.
│   └── ppi_fingerprint/   PPI fingerprint features, protocol config, and orchestration.
├── scripts/               Pipeline, grouped by audit stage (see below).
│   └── run_experiments.py Matrix runner over src/experiments/registry.py.
├── tests/                 CPU unit tests for data, features, models, and analyses.
└── manuscripts/           Draft, figures/ (SVG/PDF/PNG), and scripts/ that render them.
```

### Path configuration

Every script resolves paths through `conf/paths.py`. There are **no hardcoded
absolute paths** in `scripts/` or `src/`. To relocate the project or repoint a
data source, edit `conf/paths.py` or the entries under `external/` — nothing else.

After a one-time `pip install -e . --no-deps`, `conf` and `src` are importable
from any directory, so project-owned modules need no `sys.path` bootstrap and
work regardless of which `scripts/<group>/` subfolder they live in. The only
remaining path surgery is `src.runtime.ensure_on_sys_path`, used solely to
import *vendored upstream* trees under `external/` / `baselines/` (InterPLM,
PPLM, MINT, optional TabPFN source fallback) that are deliberately not packaged.

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
| `data/raw/cross_species/` | `CROSS_SPECIES_DIR` | 509M | 7 cross-species qrels CSVs |
| `data/raw/rapppid_c3/` | `RAPPPID_C3_DIR` (+ `C3_{TRAIN,VAL,TEST}_CSV`) | 55M | 3 RAPPPID C3 split CSVs |
| `data/raw/rapppid_c1/` | `RAPPPID_C1_DIR` (+ `C1_{TRAIN,VAL,TEST}_CSV`) | 119M | 3 RAPPPID C1 split CSVs |
| `data/raw/rapppid_c2/` | `RAPPPID_C2_DIR` (+ `C2_{TRAIN,VAL,TEST}_CSV`) | 86M | 3 RAPPPID C2 split CSVs |

The C1/C2 CSVs are extracted from the same `RAPPPID_H5` store (all three Park &
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
| `RAPPPID_H5` | `pllm-ppi-data-leakage/.../rapppid_[...].h5` | C1/C2/C3 pair + sequence store (datasets loader; C1/C2 CSV extraction) |

---

## Scripts by stage

Run all project-owned workflows with the conda env that owns the dependencies
(see `pyproject.toml` for the required packages).

**`features/`** — dataset-neutral formal feature extraction. It produces
ESM-C layers 60/80 dense mean/max and SAE mean/max/binary features, or ESM-2
final-layer dense mean/max and InterPLM-SAE mean/max/binary features. The same
directory builds the primary symmetric pair representation
`[A*B, abs(A-B)]`, its product/absolute-difference ablations, and the AB/BA
concat protocol. See `scripts/features/README.md`.

**`prep/`** — one-time data preparation (re-caches SAE features, builds interface
masks, collects benchmark sequences). GPU + model weights required for the
`cache_*` scripts. You do not need to re-run these; their outputs are already in `data/`.

**`baseline/`** — external published methods used as controls (DeepNano, PPLM,
FlashPPI, MINT, …). Feature entry points live under `scripts/baseline/features/`;
see its README for the different per-protein versus pair-conditioned cache
contracts. Train/eval runners that score those features will land here later.

**`analysis/`** — executable statistical analyses over cached features,
predictions, and benchmarks (e.g. cross-species TabPFN top-k probes, model-free
participation diagnostics). Shared probe helpers live in
`src/interpretability/pair_probe.py`.

**`interpretability/`** — executable model/SAE explanation workflows such as
TabPFN retrieval attention and active-feature overlap. Reusable algorithms live
in `src/interpretability/` (including `pair_probe`, `tabpfn_retrieval`,
attribution, and EBM effects).

**`audit_protein/`** (Layer 1) — sequence→participation oracles and
high-participation classifiers on PRING / C3 / PIC.

**`audit_pair/`** (Layer 2) — C3 endpoint-additive models (EBM / MLP),
PPI fingerprinting, and prediction on C1/C2/C3 and cross-species benchmarks, PRING and Bernett (xgboost),
TabPFN retrieval-attention audit, negative-sampling & localization confound analyses.

**`audit_residue/`** (Layer 3) — interface SAE enrichment (with surface-matched control)
and contact-pair compatibility on PDB_PPI structures.

**`smoke/`** — CPU plumbing check for the experiment matrix / runner / envelope
(`scripts/smoke/smoke_runner.py`).

**`run_experiments.py`** — the matrix runner. Declares every audit cell in
`src/experiments/registry.py`, probes readiness from declared inputs, runs
ready cells, and records a provenance sidecar + `results/runs.jsonl` history
line for each invocation. Result payloads themselves stay frozen manuscript
numbers, wrapped in a common envelope via `src.experiments.results.dump_experiment`.

---

## Quick start / reproduction

```bash
# 0. use the conda env that owns the dependencies
PY=/data/wmzhu/anaconda3/envs/E1/bin/python

# 1. one-time: install the project as an editable package so `conf` and `src`
#    import from anywhere (deps stay owned by the conda env, hence --no-deps).
$PY -m pip install -e . --no-deps

# 2. plumbing smoke: registry coherence + runner dry-run + envelope (no GPU)
$PY scripts/smoke/smoke_runner.py

# 3. inspect / dry-run the experiment matrix (readiness-probed, no execution)
$PY scripts/run_experiments.py --all --dry-run
$PY scripts/run_experiments.py --layer pair --dry-run

# 4. re-run one cell, a whole layer, or the auto matrix
$PY scripts/run_experiments.py --experiment pair.c3_endpoint_additive_mlp.sae_max
$PY scripts/run_experiments.py --layer protein --skip-existing
$PY scripts/run_experiments.py --all --skip-existing

# 5. expensive GPU cache builders are auto=False; run them explicitly if needed
$PY scripts/run_experiments.py --experiment prep.seq_cache.c3
```

After `pip install -e .`, `conf` and `src` resolve from any working directory —
no `PYTHONPATH=.` and no per-script project bootstrap. `conf.paths.ROOT`
still anchors on the `.project-root` marker, so relocating the project needs no
code change (just re-run the editable install).

Every audit result JSON is written as a common spine around the script's own
payload (`schema_version`, `task`, `dataset`, `features`, `split`, `model`,
`seed`, optional headline `metrics`/`hyperparameters`, and the original dict
under `payload`). Run provenance (git rev, env, input fingerprints) lives next
to the product as a `.prov.json` sidecar and in `results/runs.jsonl` — never
inside the payload.

---

## Provenance
Large shared data lakes remain in place and are reached via `external/` symlinks; project-exclusive data was moved in.


# E1 environment

AuditPPI uses one Python environment for all project-owned workflows:

```bash
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
```

This includes:

- ESM-C layers 60/80 with dense and SAE pooling;
- ESM-2 final-layer features with InterPLM SAE;
- DeepNano, FlashPPI, PPLM, and MINT feature wrappers;
- XGBoost, TabPFN, EBM, MLP, and other downstream models;
- participation, pair, residue, interpretability, and figure workflows;
- HDF5 preparation scripts and the CPU test suite.

## Project installation

AuditPPI and the official TabPFN clone are installed in editable mode. Changes
under `src/`, `conf/`, or `external/TabPFN/` take effect without reinstalling.

```bash
cd /data/wmzhu/PPI/AuditPPI

$PY -m pip install -e . --no-deps
$PY -m pip install -e external/TabPFN --no-deps
```

Re-run the corresponding editable install only after moving the repository or
changing package metadata/build configuration.

## Key versions

| package | E1 version |
|---|---:|
| Python | 3.12.12 |
| PyTorch | 2.7.0+cu126 |
| Transformers | 4.57.6 |
| NumPy | 2.4.2 |
| pandas | 2.3.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.2.0 |
| TabPFN | 8.0.6 |
| interpret | 0.7.8 |
| LightGBM | 4.6.0 |
| h5py | 3.16.0 |
| hdf5plugin | 6.0.0 |
| matplotlib | 3.10.9 |
| pyarrow | 17.0.0 |
| mdtraj | 1.11.1 |
| pytest | 9.1.1 |

TabPFN uses the official editable source under `external/TabPFN` and the model
checkpoint already cached under `~/.cache/tabpfn/`.

## ESM implementations
```text
ESM-C -> transformers.models.esmc + local ESM-C TopK SAE
ESM-2 -> transformers.models.esm + local InterPLM ReLUSAE
```

The installed `esm==3.3.0` distribution is not used by these extractors. Do not
install `fair-esm` into E1 because both distributions expose a top-level `esm`
module. ESM-C and ESM-2 should be run as separate commands so their large model
weights are not resident in memory simultaneously.

ESM-C CPU extraction is supported through the PyTorch SDPA fallback. A
single-protein layer-80 dense+SAE smoke used about 16.7 GiB peak RAM.

## Upstream baseline requirements

Do not install the complete historical requirement files from InterPLM,
DeepNano, PPLM, or MINT into E1. They pin mutually incompatible old versions of
Python, PyTorch, Transformers, or NumPy. AuditPPI's baseline wrappers import
only the required local model implementations and have been validated in E1.
