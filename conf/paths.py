"""Centralized path configuration for AuditPPI.

Every script imports paths from here instead of hard-coding absolute paths.
This is the single source of truth: to relocate the project or repoint an
external data source, edit only this file (or the symlinks under ``external/``).

Root discovery is anchored on the ``.project-root`` marker file, so scripts work
regardless of which ``scripts/<subdir>/`` they live in.

Layout
------
    AuditPPI/
    ├── conf/paths.py          <- this file
    ├── data/
    │   ├── audit/             <- audit outputs / source data (was AuditPPI/data)
    │   ├── sae/
    │   │   ├── reps/esmc/     <- {sae_max, binary_thr0, esmc_mean}
    │   │   ├── feature_table/ESMC-SAE-Features/
    │   │   └── seq_caches/    <- {rosetta,cross_species,bernett}_esmc_seq_cache.pt
    │   └── tabpfn/            <- {tabpfn_topk, tabpfn_retrieval_explanations}
    ├── external/              <- external repositories, model assets, and data links
    └── manuscripts/figures/
"""
from __future__ import annotations

from pathlib import Path


def _find_root(anchor: str = ".project-root") -> Path:
    """Walk up from this file until the anchor marker is found."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / anchor).exists():
            return parent
    # Fallback: conf/ is directly under the project root.
    return here.parent.parent


# --- Roots -----------------------------------------------------------------
ROOT = _find_root()
DATA = ROOT / "data"
CONF = ROOT / "conf"
MANUSCRIPTS = ROOT / "manuscripts"
FIGURES = MANUSCRIPTS / "figures"
EXTERNAL = ROOT / "external"
# Baseline repos (DeepNano/FlashPPI/mint/PIC/PPLM/PRING/RoseTTAFold2-PPI/SWING/
# pllm-ppi-data-leakage) pulled into the project. Was `external/` symlinks for PIC,
# pring_dataset and pllm-ppi-data-leakage.
BASELINES = ROOT / "baselines"

# --- Audit outputs / source data (was AuditPPI/data/*) --------------------
# Scripts that used `ROOT / "data" / X` now use `AUDIT / X`.
AUDIT = DATA / "audit"

# --- SAE artifacts (moved from SAE_PPI) ------------------------------------
SAE = DATA / "sae"
SAE_REPS = SAE / "reps" / "esmc"                    # contains sae_max/, binary_thr0/, esmc_mean/
SAE_REPS_SAE_MAX = SAE_REPS / "sae_max"
SAE_REPS_BINARY = SAE_REPS / "binary_thr0"
FEATURE_TABLE_DIR = SAE / "feature_table" / "ESMC-SAE-Features"
FEATURE_TABLE = FEATURE_TABLE_DIR / "uniref90_feature_table.parquet"

SEQ_CACHES = SAE / "seq_caches"
ROSETTA_SEQ_CACHE = SEQ_CACHES / "rosetta_esmc_seq_cache.pt"
CROSS_SPECIES_SEQ_CACHE = SEQ_CACHES / "cross_species_esmc_seq_cache.pt"
BERNETT_SEQ_CACHE = SEQ_CACHES / "bernett_esmc_seq_cache.pt"
ESMC_DEFAULT_SEQ_CACHE = SEQ_CACHES / "esmc_default_seq_cache.pt"
# ESM-2 (650M) + InterPLM-SAE pooled cache (legacy ESM-2 fingerprint line).
ESM2_SEQ_CACHE = SEQ_CACHES / "esm2_650m_seq_cache.pt"

# --- Precomputed inputs for supplementary figures (results/propensity JSONs) ---
# Structure mirrors the old SAE_PPI outputs/ layout: esmc/{results,propensity}/, baselines/results/.
SAE_SUPP_INPUTS = SAE / "supplementary_inputs"

# --- TabPFN artifacts (moved from SAE_PPI) ---------------------------------
TABPFN = DATA / "tabpfn"
TABPFN_TOPK = TABPFN / "tabpfn_topk"
TABPFN_RANKING = TABPFN_TOPK / "feature_ranking_binary_sym.csv"
TABPFN_RETRIEVAL = TABPFN / "tabpfn_retrieval_explanations"

# --- External upstream repositories ----------------------------------------
# Unmodified clone of https://github.com/PriorLabs/tabpfn. Keep the repository
# (including its own .git metadata) outside src/; only its Python source path is
# added at runtime when TabPFN is not installed in the active environment.
TABPFN_REPO = EXTERNAL / "TabPFN"
TABPFN_SRC = TABPFN_REPO / "src"

# --- External read-only data sources (symlinks under external/) ------------
PPI_DATA = EXTERNAL / "PPI_data"
BENCHMARK_TSV = PPI_DATA / "benchmarks" / "positives_and_negatives.tsv"
# RF2-PPI benchmark protein FASTA (collected into audit data).
RF2PPI_BENCHMARK_DIR = AUDIT / "rf2ppi_benchmark"
RF2PPI_FASTA = RF2PPI_BENCHMARK_DIR / "benchmark_proteins.fasta"
# Verified 0-missing source FASTA for regenerating the benchmark protein set.
RF2PPI_SEQ_SOURCE = RF2PPI_BENCHMARK_DIR / "all_sequences_100pct.fasta"

RAPPPID_C3_DIR = DATA / "rapppid_c3"
C3_TRAIN_CSV = RAPPPID_C3_DIR / "c3.train.csv"
C3_VAL_CSV = RAPPPID_C3_DIR / "c3.val.csv"
C3_TEST_CSV = RAPPPID_C3_DIR / "c3.test.csv"

# RAPPPID C1 / C2 splits (Park & Marcotte leakage levels; extracted from the same
# C3_H5 store into (query,text,label) CSVs, identical schema to the C3 set).
RAPPPID_C1_DIR = DATA / "rapppid_c1"
C1_TRAIN_CSV = RAPPPID_C1_DIR / "c1.train.csv"
C1_VAL_CSV = RAPPPID_C1_DIR / "c1.val.csv"
C1_TEST_CSV = RAPPPID_C1_DIR / "c1.test.csv"

RAPPPID_C2_DIR = DATA / "rapppid_c2"
C2_TRAIN_CSV = RAPPPID_C2_DIR / "c2.train.csv"
C2_VAL_CSV = RAPPPID_C2_DIR / "c2.val.csv"
C2_TEST_CSV = RAPPPID_C2_DIR / "c2.test.csv"

# Per-level CSV split maps, keyed by level then split (used by loaders/exporters).
RAPPPID_CLEVEL_CSVS = {
    "c1": {"train": C1_TRAIN_CSV, "val": C1_VAL_CSV, "test": C1_TEST_CSV},
    "c2": {"train": C2_TRAIN_CSV, "val": C2_VAL_CSV, "test": C2_TEST_CSV},
    "c3": {"train": C3_TRAIN_CSV, "val": C3_VAL_CSV, "test": C3_TEST_CSV},
}

# RAPPPID C1/C2/C3 HDF5 sequence store (in-project baseline repo). Holds all three
# leakage levels under interactions/{c1,c2,c3}/ plus the shared sequences table.
C3_H5 = (
    BASELINES / "pllm-ppi-data-leakage" / "data" / "data" / "ppi"
    / "rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]"
      "_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"
)

# Cross-species PPI benchmark CSVs (human/ecoli/fly/mouse/worm/yeast); in-project.
CROSS_SPECIES_DIR = DATA / "cross_species"

# Cross-species CSV filenames (human train/test + 5 held-out species tests).
CROSS_SPECIES_CSVS = [
    "human.ppi.qrels.seq.train.csv",
    "human.ppi.qrels.seq.test.csv",
    "ecoli.ppi.qrels.seq.test.csv",
    "fly.ppi.qrels.seq.test.csv",
    "mouse.ppi.qrels.seq.test.csv",
    "worm.ppi.qrels.seq.test.csv",
    "yeast.ppi.qrels.seq.test.csv",
]

# Bernett gold-standard PPI (MINT GeneralPPI split): Intra1/0/2 = train/val/test.
BERNETT_DIR = BASELINES / "mint" / "downstream" / "GeneralPPI" / "ppi"
BERNETT_SPLIT_CSVS = {"train": "Intra1_seqs.csv", "val": "Intra0_seqs.csv", "test": "Intra2_seqs.csv"}

PRING_ROOT = BASELINES / "PRING" / "data_process" / "pring_dataset"
PIC_DATA = BASELINES / "PIC" / "data" / "human_data.pkl"

# PRING cross-species generalization: train the participation oracle on human,
# zero-shot test on the held-out species (each has its own full {sp}_graph.pkl
# but NO train/val split -- they are test-only graphs). Pooled ESM-C/SAE caches
# live alongside the human PRING cache under AUDIT/pring_participation/.
PRING_PARTICIPATION_DIR = AUDIT / "pring_participation"
PRING_TRAIN_SPECIES = "human"
PRING_CROSS_SPECIES = ("yeast", "ecoli", "arath")
PRING_HUMAN_SAE_CACHE = PRING_PARTICIPATION_DIR / "pring_human_esmc_sae_cache.pt"
# {species: pooled ESM-C/SAE cache}; human reuses the pre-existing human cache.
PRING_SPECIES_SAE_CACHES = {
    "human": PRING_HUMAN_SAE_CACHE,
    **{sp: PRING_PARTICIPATION_DIR / f"pring_{sp}_esmc_sae_cache.pt" for sp in PRING_CROSS_SPECIES},
}

# AlphaFold structures for the case-4365 analysis are the 14 PDBs already cached
# under AUDIT/structure_comparison/tabpfn_case_4365/structures/ (the script's
# STRUCT_DIR); anything missing is fetched from the AlphaFold EBI API at runtime.

# --- ESM-C model + SAE weights (symlinks; only needed to re-cache features) -
ESMC_MODEL = EXTERNAL / "ESMC" / "ESMC-6B"
ESMC_SAE = EXTERNAL / "ESMC" / "ESMC-6B-sae-k64-codebook16384"

# --- ESM-2 / InterPLM SAE line (legacy fingerprint provenance) -------------
# ESM-2-650M is loaded from HuggingFace by name (must be cached locally when
# HF_HUB_OFFLINE=1). The InterPLM ReLU-SAE for layer 33 lives in external/.
ESM2_650M_MODEL = "facebook/esm2_t33_650M_UR50D"
INTERPLM_ROOT = EXTERNAL / "InterPLM"
ESM2_SAE_CKPT = INTERPLM_ROOT / "sae_ckpts" / "esm2-650m" / "layer_33" / "ae_normalized.pt"

# --- DeepNano-style frozen-embedding baseline (mean/min/max pooled cache) ---
# Products are not consumed by the audit yet; kept for reproducibility.
DEEPNANO_DIR = SAE / "deepnano"

# --- External tools --------------------------------------------------------
USALIGN = Path("/data/wmzhu/tools/usalign/USalign")


__all__ = [
    "ROOT", "DATA", "CONF", "MANUSCRIPTS", "FIGURES", "EXTERNAL", "BASELINES",
    "AUDIT",
    "SAE", "SAE_REPS", "SAE_REPS_SAE_MAX", "SAE_REPS_BINARY",
    "FEATURE_TABLE_DIR", "FEATURE_TABLE",
    "SEQ_CACHES", "ROSETTA_SEQ_CACHE", "CROSS_SPECIES_SEQ_CACHE", "BERNETT_SEQ_CACHE",
    "ESMC_DEFAULT_SEQ_CACHE", "ESM2_SEQ_CACHE", "SAE_SUPP_INPUTS",
    "TABPFN", "TABPFN_TOPK", "TABPFN_RANKING", "TABPFN_RETRIEVAL",
    "TABPFN_REPO", "TABPFN_SRC",
    "PPI_DATA", "BENCHMARK_TSV",
    "RF2PPI_BENCHMARK_DIR", "RF2PPI_FASTA", "RF2PPI_SEQ_SOURCE",
    "RAPPPID_C3_DIR", "C3_TRAIN_CSV", "C3_VAL_CSV", "C3_TEST_CSV", "C3_H5",
    "RAPPPID_C1_DIR", "C1_TRAIN_CSV", "C1_VAL_CSV", "C1_TEST_CSV",
    "RAPPPID_C2_DIR", "C2_TRAIN_CSV", "C2_VAL_CSV", "C2_TEST_CSV",
    "RAPPPID_CLEVEL_CSVS",
    "CROSS_SPECIES_DIR",
    "BERNETT_DIR", "BERNETT_SPLIT_CSVS",
    "PRING_ROOT", "PIC_DATA",
    "PRING_PARTICIPATION_DIR", "PRING_TRAIN_SPECIES", "PRING_CROSS_SPECIES",
    "PRING_HUMAN_SAE_CACHE", "PRING_SPECIES_SAE_CACHES",
    "ESMC_MODEL", "ESMC_SAE",
    "ESM2_650M_MODEL", "INTERPLM_ROOT", "ESM2_SAE_CKPT",
    "DEEPNANO_DIR",
    "USALIGN",
]
