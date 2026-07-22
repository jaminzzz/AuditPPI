"""Centralized path configuration for AuditPPI.

Every script imports paths from here instead of hard-coding absolute paths.
This is the single source of truth: to relocate the project or repoint an
external data source, edit only this file (or the symlinks under ``external/``).

Root discovery is anchored on the ``.project-root`` marker file, so scripts work
regardless of which ``scripts/<subdir>/`` they live in.

Sections
--------
    Roots                 project anchors (ROOT/DATA/CONF/EXTERNAL/BASELINES/RESULTS)
    Results               audit PRODUCTS, layered by scale (protein/pair/residue/
                          analysis/main). Was data/audit + data/tabpfn.
    SAE artifacts         cache pipeline (seq/protein/pair/residue), feature
                          table, supp inputs
    TabPFN artifacts      topk / retrieval products under results/audit_pair/tabpfn
                          + upstream repo source
    Datasets              raw inputs to the data layer (under data/raw/), split to
                          mirror it:
                            - Pair-level   -> loaded by ``src.data.pairs``
                            - Protein-level -> loaded by ``src.data.proteins``
    Models & SAE weights   ESM-C / ESM-2 + InterPLM SAE checkpoints
    Misc                  frozen-embedding baseline, external tools

Layout
------
    AuditPPI/
    ├── conf/paths.py          <- this file
    ├── data/
    │   ├── raw/               <- benchmark CSVs + RF2-PPI FASTAs (inputs)
    │   └── sae/               <- SAE cache pipeline (coarsening granularity):
    │       ├── seq_caches/    <-   per SEQUENCE  {rosetta,cross_species,bernett,...}_esmc_seq_cache.pt
    │       ├── protein_caches/ <-  per PROTEIN   {pring_*,pic_*}_esmc_sae_cache.pt (UniProt-ID indexed)
    │       ├── pair_caches/esmc/ <- per PAIR     {sae_max, binary_thr0, esmc_mean} feature matrices
    │       ├── residue_caches/ <-  per RESIDUE   pdb_ppi_pos_sae_cache_* (SaeCacheReader dir)
    │       └── feature_table/ESMC-SAE-Features/
    ├── results/               <- audit PRODUCTS (sibling of data/, not inside it)
    │   ├── audit_protein/     <- Ladder 1 participation / hubness / essentiality
    │   ├── audit_pair/        <- Ladder 2 endpoint-additive / TabPFN / neg-sampling
    │   ├── audit_residue/     <- Ladder 3 interface grounding
    │   ├── analysis/          <- cross-ladder analyses
    │   └── main/              <- primary PPI outputs (fingerprint baselines, …)
    ├── baselines/             <- in-project baseline repos (PIC/PRING/mint/...)
    ├── external/              <- external repositories, model assets, and data links
    └── manuscripts/figures/
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_root(anchor: str = ".project-root") -> Path:
    """Walk up from this file until the anchor marker is found."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / anchor).exists():
            return parent
    # Fallback: conf/ is directly under the project root.
    return here.parent.parent


# === Roots =================================================================
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


# === Audit outputs (was data/audit + data/tabpfn) =========================
# All audit PRODUCTS live under the top-level results/ tree (sibling of data/,
# not inside it: these are outputs, not inputs). results/ mirrors the three
# audit scales plus two cross-cutting buckets:
#   audit_protein/  Ladder 1 -- sequence->participation / hubness / essentiality
#   audit_pair/     Ladder 2 -- endpoint-additive models, TabPFN, neg-sampling
#   audit_residue/  Ladder 3 -- interface grounding
#   analysis/       cross-ladder analyses (e.g. C3<->PRING feature overlap)
#   main/           primary PPI outputs (fingerprint baselines)
# Source FASTA that used to sit under data/audit/rf2ppi_benchmark/ moved to
# data/raw/ (it is a dataset input, not a result) -- see RF2PPI_* below.
RESULTS = ROOT / "results"
RESULTS_PROTEIN = RESULTS / "audit_protein"
RESULTS_PAIR = RESULTS / "audit_pair"
RESULTS_RESIDUE = RESULTS / "audit_residue"
RESULTS_ANALYSIS = RESULTS / "analysis"
RESULTS_MAIN = RESULTS / "main"
# Back-compat alias -- prefer RESULTS_MAIN in new code.
RESULTS_MISC = RESULTS_MAIN


# === SAE artifacts (moved from SAE_PPI) ====================================
SAE = DATA / "sae"
FEATURE_TABLE_DIR = SAE / "feature_table" / "ESMC-SAE-Features"
FEATURE_TABLE = FEATURE_TABLE_DIR / "uniref90_feature_table.parquet"

# --- SAE cache pipeline (three stages, coarsening granularity) -------------
# The cheap-to-reuse products of the expensive ESM-C(+SAE) forward pass. All
# three live side-by-side under SAE/ and share the ``*_caches/`` naming. They
# form one pipeline; reuse decreases left to right:
#
#   seq_caches/     per unique SEQUENCE -> pooled fingerprint {seq2idx, esmc_*}
#                   keyed by sequence string. Most expensive (ESM-C forward),
#                   broadest reuse: any dataset with the same sequence hits it.
#   protein_caches/ per DATASET's proteins -> same fingerprints + UniProt-ID
#                   index {protein_ids, uniprotid2idx}. Seeded from seq_caches,
#                   only missing proteins are recomputed. For ID-keyed protein-
#                   level experiments (PRING/PIC).
#   pair_caches/    per protein PAIR -> engineered product/absdiff/sym/concat
#                   feature matrix {X, y, X_ab, X_ba}. Built by
#                   build_pair_features.py from a seq_cache + a pair list.
#                   Cheapest to recompute, narrowest reuse (bound to a specific
#                   split + mode); consumed by the TabM/XGBoost endpoint models.

# Stage 1: per-sequence pooled fingerprints (was SEQ_CACHES).
#
# Naming convention (NOT subdirectories) isolates model family and layer, so a
# single flat seq_caches/ dir holds everything and future variants slot in by
# name alone:
#   * Model family  -> filename infix. All current caches are ESM-C 6B and carry
#                      ``_esmc_`` (rosetta_esmc_seq_cache.pt); the ESM-2 line uses
#                      ``_esm2_`` (esm2_650m_seq_cache.pt). Never introduce an
#                      esmc/ vs esm2/ dir split -- the infix already disambiguates.
#   * Layer         -> in-file key prefix, NOT the filename. Every cache on disk
#                      today is single-layer ESM-C L60, stored under flat keys
#                      ``esmc_mean`` / ``esmc_sae_max`` with the
#                      layer recorded in meta{'layer': 60}. When a cache carries
#                      multiple layers (the extractor default is layers=(60, 80)),
#                      keys gain a layer prefix ``esmc_l60_*`` / ``esmc_l80_*`` to
#                      coexist in one file -- again no new directory.
SEQ_CACHES = SAE / "seq_caches"
ROSETTA_SEQ_CACHE = SEQ_CACHES / "rosetta_esmc_seq_cache.pt"
CROSS_SPECIES_SEQ_CACHE = SEQ_CACHES / "cross_species_esmc_seq_cache.pt"
BERNETT_SEQ_CACHE = SEQ_CACHES / "bernett_esmc_seq_cache.pt"
ESMC_DEFAULT_SEQ_CACHE = SEQ_CACHES / "esmc_default_seq_cache.pt"
# ESM-2 (650M) + InterPLM-SAE pooled cache (legacy ESM-2 fingerprint line).
ESM2_SEQ_CACHE = SEQ_CACHES / "esm2_650m_seq_cache.pt"

# Pooled unique-sequence feature caches (auditppi_protein_features_v1) holding
# every benchmark sequence's channels. The per-dataset protein caches are sliced
# out of these on CPU (see scripts/prep/slice_dataset_protein_cache.py). The
# max1022 suffix pins the cross-backbone-comparable residue budget.
POOLED_ESMC_SEQ_CACHE = SEQ_CACHES / "pooled_esmc_l60_l80_max1022_features.pt"
POOLED_ESM2_SEQ_CACHE = SEQ_CACHES / "pooled_esm2_l33_max1022_features.pt"

# Benchmark -> pooled per-sequence cache. Which seq-cache each benchmark's pooled
# fingerprints live in; shared by the ppi_fingerprint baseline and the C3 /
# cross-species sequence-participation oracle. (Was ppi_fingerprint.config.CACHE.)
POOLED_SEQ_CACHES = {
    "c3": ESMC_DEFAULT_SEQ_CACHE,
    "cross_species": CROSS_SPECIES_SEQ_CACHE,
    "rf2ppi": ROSETTA_SEQ_CACHE,
}

# Ordered fallback pooled caches probed for a diagnostic hint when a PRING-
# specific cache is absent. (Was participation.config.FALLBACK_CACHE_CANDIDATES.)
PRING_FALLBACK_CACHES = (
    ROSETTA_SEQ_CACHE,
    CROSS_SPECIES_SEQ_CACHE,
    BERNETT_SEQ_CACHE,
    ESMC_DEFAULT_SEQ_CACHE,
)

# Stage 2: per-dataset protein-level pooled caches (UniProt-ID indexed). Moved
# here from results/audit_protein/{pring_participation,pic_essentiality}/ so the reusable cache
# no longer sits beside the disposable xgboost outputs it feeds.
PROTEIN_SAE_CACHES = SAE / "protein_caches"
# v1 per-dataset caches (auditppi_protein_features_v1): every ESM-C L60/L80 and
# ESM-2 L33 channel in one payload, sliced from the pooled seq caches. The
# max1022 suffix pins the cross-backbone-comparable residue budget.
PRING_HUMAN_SAE_CACHE = PROTEIN_SAE_CACHES / "pring_human_protein_features_max1022.pt"
PIC_HUMAN_SAE_CACHE = PROTEIN_SAE_CACHES / "pic_human_protein_features_max1022.pt"
# RAPPPID C1/C2/C3 endpoint caches. The CSVs carry raw query/text sequences
# (no protein ids), so these caches are keyed by sequence via seq2idx, not id2idx.
C1_SAE_CACHE = PROTEIN_SAE_CACHES / "c1_protein_features_max1022.pt"
C2_SAE_CACHE = PROTEIN_SAE_CACHES / "c2_protein_features_max1022.pt"
C3_SAE_CACHE = PROTEIN_SAE_CACHES / "c3_protein_features_max1022.pt"
# Cross-species (human train + held-out species tests) and Bernett gold-standard
# endpoint caches. Both are sequence-keyed (no protein ids in the source CSVs).
CROSS_SPECIES_SAE_CACHE = PROTEIN_SAE_CACHES / "cross_species_protein_features_max1022.pt"
BERNETT_SAE_CACHE = PROTEIN_SAE_CACHES / "bernett_protein_features_max1022.pt"

# Ladder-2 PPI prediction: benchmark family -> its v1 protein feature cache. One
# cache per family holds every endpoint sequence across that family's splits, so
# a single cache serves the family's native-train + eval protocol. PRING is
# per-species (human train graph + yeast/ecoli/arath test graphs), so it maps to
# the per-species caches keyed by species rather than one family cache.
PPI_PREDICTION_CACHES = {
    "c1": C1_SAE_CACHE,
    "c2": C2_SAE_CACHE,
    "c3": C3_SAE_CACHE,
    "cross_species": CROSS_SPECIES_SAE_CACHE,
    "bernett": BERNETT_SAE_CACHE,
    "pring": PRING_HUMAN_SAE_CACHE,
}

# Stage 3: per-pair engineered feature matrices (was reps/esmc/). Renamed to
# pair_caches to name its true granularity (protein PAIRS, not "representations").
PAIR_CACHES = SAE / "pair_caches" / "esmc"          # contains sae_max/, binary_thr0/, esmc_mean/
PAIR_CACHES_SAE_MAX = PAIR_CACHES / "sae_max"
PAIR_CACHES_BINARY = PAIR_CACHES / "binary_thr0"

# Lightweight pair-index caches (auditppi_pair_index_v1): per-pair endpoint row
# indices into a v1 protein cache + labels, no materialized features. One file
# per (dataset, split) serves every backbone/layer/rep channel and pair mode;
# built by scripts/prep/build_pair_index_cache.py. Replaces the old per-rep
# {split}_embeddings.pt dumps. Row order matches the pair CSV (and thus the C3
# pair-id alignment parquet) via kept_pair_indices.
PAIR_INDEX_CACHES = SAE / "pair_caches"
C3_PAIR_INDEX_CACHES = {
    split: PAIR_INDEX_CACHES / "c3" / f"{split}_pairs.pt"
    for split in ("train", "val", "test")
}
# Cross-species pair-index caches. No val CSV exists on disk: the human_train
# graph is the only train source and val is carved from it in-memory (stratified,
# matching the ppi_fingerprint baseline convention). So we build one cache for
# human_train plus one per held-out species test graph; the topk consumer splits
# human_train into train/val itself.
CROSS_SPECIES_PAIR_SPLITS = (
    "human_train", "ecoli", "fly", "mouse", "worm", "yeast",
)
CROSS_SPECIES_PAIR_INDEX_CACHES = {
    split: PAIR_INDEX_CACHES / "cross_species" / f"{split}_pairs.pt"
    for split in CROSS_SPECIES_PAIR_SPLITS
}

# Residue-level SAE cache (SaeCacheReader LMDB-style dir format) for the
# interface-grounding audit. Residue SAE cache lives here (~8.8G);
# enrichment/compat OUTPUTS live under results/audit_residue/interface_grounding/.
RESIDUE_SAE_CACHES = SAE / "residue_caches"
PDB_PPI_SAE_CACHE = RESIDUE_SAE_CACHES / "pdb_ppi_pos_sae_cache_gpu0"
PDB_PPI_SAE_CACHE_META = RESIDUE_SAE_CACHES / "pdb_ppi_pos_sae_cache_meta"

# Precomputed inputs for supplementary figures (results/propensity JSONs).
# Structure mirrors the old SAE_PPI outputs/ layout: esmc/{results,propensity}/, baselines/results/.
SAE_SUPP_INPUTS = SAE / "supplementary_inputs"


# === TabPFN artifacts =====================================================
# TabPFN products are a Ladder-2 (pair-scale) result, so they live under
# results/audit_pair/tabpfn/. feature_ranking_binary_sym.csv (TABPFN_RANKING)
# is a product that later scripts also consume as a downstream INPUT.
TABPFN = RESULTS_PAIR / "tabpfn"
TABPFN_TOPK = TABPFN / "tabpfn_topk"
TABPFN_RANKING = TABPFN_TOPK / "feature_ranking_binary_sym.csv"
TABPFN_RETRIEVAL = TABPFN / "tabpfn_retrieval_explanations"

# Unmodified clone of https://github.com/PriorLabs/tabpfn. Keep the repository
# (including its own .git metadata) outside src/; only its Python source path is
# added at runtime when TabPFN is not installed in the active environment.
TABPFN_REPO = EXTERNAL / "TabPFN"
TABPFN_SRC = TABPFN_REPO / "src"


# === Datasets ==============================================================
# Raw inputs to the data layer. Split to mirror it: pair-level sources feed
# `src.data.pairs` (the `Benchmark` contract), protein-level sources feed
# `src.data.proteins` (the `ProteinDataset` contract). PRING spans both — its
# edge lists are pair-level, its participation counts are protein-level.

# --- Pair-level  (loaded by src.data.pairs) --------------------------------

# RF2-PPI benchmark: positives/negatives TSV + protein FASTA.
PPI_DATA = EXTERNAL / "PPI_data"
BENCHMARK_TSV = PPI_DATA / "benchmarks" / "positives_and_negatives.tsv"
RF2PPI_BENCHMARK_DIR = DATA / "raw" / "rf2ppi_benchmark"
RF2PPI_FASTA = RF2PPI_BENCHMARK_DIR / "benchmark_proteins.fasta"
# Verified 0-missing source FASTA for regenerating the benchmark protein set.
RF2PPI_SEQ_SOURCE = RF2PPI_BENCHMARK_DIR / "all_sequences_100pct.fasta"

# RAPPPID C1/C2/C3 leakage-level splits (Park & Marcotte). All three share one
# (query,text,label) CSV schema and one HDF5 sequence store.
RAPPPID_C1_DIR = DATA / "raw" / "rapppid_c1"
C1_TRAIN_CSV = RAPPPID_C1_DIR / "c1.train.csv"
C1_VAL_CSV = RAPPPID_C1_DIR / "c1.val.csv"
C1_TEST_CSV = RAPPPID_C1_DIR / "c1.test.csv"

RAPPPID_C2_DIR = DATA / "raw" / "rapppid_c2"
C2_TRAIN_CSV = RAPPPID_C2_DIR / "c2.train.csv"
C2_VAL_CSV = RAPPPID_C2_DIR / "c2.val.csv"
C2_TEST_CSV = RAPPPID_C2_DIR / "c2.test.csv"

RAPPPID_C3_DIR = DATA / "raw" / "rapppid_c3"
C3_TRAIN_CSV = RAPPPID_C3_DIR / "c3.train.csv"
C3_VAL_CSV = RAPPPID_C3_DIR / "c3.val.csv"
C3_TEST_CSV = RAPPPID_C3_DIR / "c3.test.csv"

# Per-level CSV split maps, keyed by level then split (used by loaders/exporters).
RAPPPID_CLEVEL_CSVS = {
    "c1": {"train": C1_TRAIN_CSV, "val": C1_VAL_CSV, "test": C1_TEST_CSV},
    "c2": {"train": C2_TRAIN_CSV, "val": C2_VAL_CSV, "test": C2_TEST_CSV},
    "c3": {"train": C3_TRAIN_CSV, "val": C3_VAL_CSV, "test": C3_TEST_CSV},
}

# RAPPPID C1/C2/C3 HDF5 sequence store (in-project baseline repo). Holds all three
# leakage levels under interactions/{c1,c2,c3}/ plus the shared sequences table,
# so the name is level-agnostic (was C3_H5, which misleadingly implied C3-only).
RAPPPID_H5 = (
    BASELINES / "pllm-ppi-data-leakage" / "data" / "data" / "ppi"
    / "rapppid_[common_string_9606.protein.links.detailed.v12.0_upkb.csv]"
      "_Mz70T9t-4Y-i6jWD9sEtcjOr0X8=.h5"
)

# Cross-species PPI benchmark CSVs (human train/test + 5 held-out species tests).
CROSS_SPECIES_DIR = DATA / "raw" / "cross_species"
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

# PRING edge lists (pair-level). Human has BFS/DFS/RANDOM_WALK train/val/test
# splits; held-out species are test-only graphs. See PRING_* below for the
# protein-level participation view of the same dataset.
PRING_ROOT = BASELINES / "PRING" / "data_process" / "pring_dataset"

# --- Protein-level  (loaded by src.data.proteins) --------------------------

# PIC (Protein Importance Calculator) essentiality data: one pickle per dataset
# with columns ID / sequence / {label columns}. PIC_DATA kept for back-compat
# (the human_data.pkl the cache script defaults to).
PIC_DATA_DIR = BASELINES / "PIC" / "data"
PIC_DATA = PIC_DATA_DIR / "human_data.pkl"
PIC_DATASET_PKL = {
    "human": PIC_DATA_DIR / "human_data.pkl",
    "mouse": PIC_DATA_DIR / "mouse_data.pkl",
    "cell": PIC_DATA_DIR / "cell_data.pkl",
}
# Flat CSV export of the PIC human essentiality set (ID,sequence,label), written
# by scripts/prep/export_pic_human_csv.py from the legacy-numpy pickle. This is
# the format-neutral input the v1 protein-cache slicer consumes (--id-cols ID),
# isolating the PIC-specific pickle read in the prep step.
PIC_HUMAN_DIR = DATA / "raw" / "pic"
PIC_HUMAN_CSV = PIC_HUMAN_DIR / "pic_human.csv"

# PRING cross-species generalization (participation oracle): train on human,
# zero-shot test on the held-out species (each has its own full {sp}_graph.pkl
# but NO train/val split -- they are test-only graphs). PRING_PARTICIPATION_DIR
# is the experiment OUTPUT dir; the reusable pooled ESM-C/SAE caches now live
# under PROTEIN_SAE_CACHES (see SAE section). PRING_HUMAN_SAE_CACHE is defined there.
PRING_PARTICIPATION_DIR = RESULTS_PROTEIN / "pring_participation"
PRING_TRAIN_SPECIES = "human"
PRING_CROSS_SPECIES = ("yeast", "ecoli", "arath")
# {species: pooled ESM-C/SAE cache}; human reuses the pre-existing human cache.
PRING_SPECIES_SAE_CACHES = {
    "human": PRING_HUMAN_SAE_CACHE,
    **{
        sp: PROTEIN_SAE_CACHES / f"pring_{sp}_protein_features_max1022.pt"
        for sp in PRING_CROSS_SPECIES
    },
}

# AlphaFold structures for the case-4365 analysis are the 14 PDBs already cached
# under results/audit_pair/structure_comparison/tabpfn_case_4365/structures/ (the script's
# STRUCT_DIR); anything missing is fetched from the AlphaFold EBI API at runtime.


# === Models & SAE weights ==================================================
# ESM-C model + SAE weights (symlinks; only needed to re-cache features).
ESMC_MODEL = EXTERNAL / "ESMC" / "ESMC-6B"
ESMC_SAE = EXTERNAL / "ESMC" / "ESMC-6B-sae-k64-codebook16384"

# ESM-2 / InterPLM SAE line (legacy fingerprint provenance). ESM-2-650M is loaded
# from HuggingFace by name (must be cached locally when HF_HUB_OFFLINE=1). The
# InterPLM ReLU-SAE for layer 33 lives in external/.
ESM2_650M_MODEL = "facebook/esm2_t33_650M_UR50D"
INTERPLM_ROOT = EXTERNAL / "InterPLM"
ESM2_SAE_CKPT = INTERPLM_ROOT / "sae_ckpts" / "esm2-650m" / "layer_33" / "ae_normalized.pt"


# === Misc ==================================================================
# DeepNano-style frozen-embedding baseline (mean/min/max pooled cache). Products
# are not consumed by the audit yet; kept for reproducibility.
DEEPNANO_DIR = SAE / "deepnano"

# External tools. The ONLY machine-dependent path in this file: US-align lives
# outside the project tree, so it can't hang off ROOT/EXTERNAL. Override with the
# ``AUDITPPI_USALIGN`` env var when the binary lives elsewhere.
USALIGN = Path(os.environ.get("AUDITPPI_USALIGN", "/data/wmzhu/tools/usalign/USalign"))


# Export every module-level constant (UPPER_CASE names); regenerated on import so
# it never goes stale as paths are added. Excludes helpers (_find_root) and imports.
__all__ = sorted(name for name in dict(globals()) if name.isupper())
