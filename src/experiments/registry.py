"""The experiment matrix: every runnable audit cell, declared once.

This module is the single source of truth for *what experiments exist*. The
runner (``scripts/run_experiments.py``) consumes it; nothing here executes a
script or touches the filesystem beyond read-only readiness probes.

Design
------
Each :class:`Experiment` is one invocation of one script with one set of args.
The matrix axes (dataset x rep x model x method x species) are expanded by the
per-layer builder functions below, so adding a rep or a species is a one-line
edit, not a copy-paste of a cell.

Readiness is *probed at runtime*, never stored as a static ``enabled`` flag.
Every cell lists the input paths it needs (caches, or upstream products for the
consumer layers). ``Experiment.missing_inputs()`` returns the subset that does
not exist on disk right now. A cell whose inputs are all present is *ready*;
one with missing inputs is declared but skipped. This is exactly the behaviour
the user asked for: the full dataset matrix is *configured* even before its
cache exists, and dropping a cache file in place auto-lights the cell on the
next run -- no code change.

Because the consumer layers (figures / analysis / interpretability) list their
*upstream products* as inputs, the same probe doubles as dependency
resolution: a figure whose upstream JSON has not been produced yet is simply
not ready. Combined with the layer ordering (see ``LAYER_ORDER``) this gives a
topological execution order without a hand-maintained DAG.

Out-dir policy
--------------
The registry does **not** redirect ``--out-dir``. Every script writes to its
own established default location, because the figure/analysis consumers read
their upstream products from hard-coded paths (e.g.
``RESULTS_PROTEIN/pring_participation/high_p90_xgboost``). Per-run traceability
is provided by the sidecar + history written *next to* each product by the
runner (see ``src.experiments.history``), which does not depend on the directory
layout. Moving products into a faceted tree would sever the consumer chain for
zero traceability gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from conf.paths import (
    BERNETT_DIR,
    BERNETT_SEQ_CACHE,
    C3_TEST_CSV,
    C3_TRAIN_CSV,
    C3_VAL_CSV,
    CROSS_SPECIES_DIR,
    CROSS_SPECIES_SEQ_CACHE,
    ESMC_DEFAULT_SEQ_CACHE,
    FEATURE_TABLE,
    PAIR_CACHES,
    PDB_PPI_SAE_CACHE,
    PIC_DATASET_PKL,
    PIC_HUMAN_SAE_CACHE,
    PPI_DATA,
    PRING_HUMAN_SAE_CACHE,
    PRING_ROOT,
    PRING_SPECIES_SAE_CACHES,
    RESULTS_MISC,
    RESULTS_PAIR,
    RESULTS_PROTEIN,
    RESULTS_RESIDUE,
    ROSETTA_SEQ_CACHE,
)

# --- Layer ordering (main-trunk route: protein -> pair -> residue) ----------
# Producers before consumers. ``prep`` (the GPU forward-pass cache builders)
# feeds everything else, so it sorts first. The user's requested trunk is then
# protein -> pair -> residue; analysis/interpretability/baseline are
# cross-cutting and figures are pure sinks, so they sort to the end. The runner
# uses this as the primary sort key; readiness probing over declared inputs
# handles the finer producer->consumer edges within and across layers. Note prep
# cells are ``auto=False`` (see below), so ``--all`` never triggers a GPU
# encode; they are declared for matrix completeness and to light up the audit
# cells that consume their caches.
LAYER_ORDER = (
    "prep",
    "protein",
    "pair",
    "residue",
    "analysis",
    "interpretability",
    "baseline",
    "figures",
)


@dataclass(frozen=True)
class Experiment:
    """One concrete script invocation in the matrix.

    Attributes
    ----------
    name
        Stable, unique id used on the runner CLI (``--experiment <name>``) and
        as the ``experiment`` field in the history. kebab/underscore, no spaces.
    layer
        One of :data:`LAYER_ORDER`. Sets execution priority and grouping.
    script
        Repo-relative path to the entrypoint under ``scripts/``.
    args
        Extra CLI args appended after the script path. Never includes the
        interpreter or the script itself.
    inputs
        Paths that must exist for this cell to run: input caches, and -- for
        consumer layers -- the upstream products it reads. Probed at runtime;
        see :meth:`missing_inputs`.
    products
        Representative output paths this cell writes. Used by the runner for
        ``--skip-existing`` and post-run verification. Not exhaustive; one or
        two load-bearing files are enough.
    auto
        Whether ``--all`` includes this cell. Cache/prep/feature-extraction
        entrypoints are the expensive, human-managed prerequisites (GPU
        forward passes, external downloads); they are declared for completeness
        and readiness bookkeeping but excluded from ``--all`` (``auto=False``).
        The audit trunk and its consumers are ``auto=True``.
    """

    name: str
    layer: str
    script: str
    args: tuple[str, ...] = ()
    inputs: tuple[Path, ...] = ()
    products: tuple[Path, ...] = ()
    auto: bool = True

    def missing_inputs(self) -> list[Path]:
        """Declared input paths that do not exist on disk right now."""
        return [p for p in self.inputs if not Path(p).exists()]

    def is_ready(self) -> bool:
        """True when every declared input is present."""
        return not self.missing_inputs()

    def existing_products(self) -> list[Path]:
        """Declared product paths that already exist (for ``--skip-existing``)."""
        return [p for p in self.products if Path(p).exists()]

    @property
    def layer_rank(self) -> int:
        try:
            return LAYER_ORDER.index(self.layer)
        except ValueError:  # unknown layer sorts last
            return len(LAYER_ORDER)


# ---------------------------------------------------------------------------
# Axis vocabularies (mirror the scripts' own argparse choices / module consts;
# verified against source, not guessed).
# ---------------------------------------------------------------------------
PAIR_REPS = ("sae_max", "binary")            # C3/PRING endpoint scripts REP_SUBDIR / REPS
PRING_METHODS = ("BFS", "DFS", "RANDOM_WALK")
PRING_SPECIES = ("yeast", "ecoli", "arath")  # cross-species generalization test graphs
PROTEIN_REPS = ("sae_max", "binary", "esmc_mean")  # ppi_fingerprint REPRESENTATIONS / FE.REPS

# Pair-cache root -> rep subdir layout ({rep}/{split}_embeddings.pt). Only C3 is
# on disk today (PAIR_CACHES); other pair datasets are declared by convention so
# the cell lights up when its cache root is populated with the same layout.
PAIR_REP_SUBDIR = {"sae_max": "sae_max", "binary": "binary_thr0"}


def _pair_cache_inputs(cache_root: Path, reps: tuple[str, ...]) -> tuple[Path, ...]:
    """The train/val/test tensors a pair endpoint run needs for each rep."""
    out: list[Path] = []
    for rep in reps:
        sub = cache_root / PAIR_REP_SUBDIR[rep]
        out += [sub / f"{split}_embeddings.pt" for split in ("train", "val", "test")]
    return tuple(out)


# ===========================================================================
# Layer 0 -- PREP (feature-cache builders: the GPU forward passes)
# ===========================================================================
# These are the expensive, human-managed prerequisites: each runs an ESM-C (+SAE)
# forward pass over a dataset's sequences and writes the pooled cache the audit
# layers consume. They are declared for matrix completeness and readiness
# bookkeeping -- their ``products`` are exactly the ``inputs`` of downstream
# cells, so a prep cell's output auto-lights the consumers that need it -- but
# they are ``auto=False`` so ``--all`` never kicks off a multi-hour GPU job
# implicitly. Run them explicitly with ``--experiment prep.<name>``.
#
# ``inputs`` here are the *raw* sources (FASTA / split CSVs / pickles / metadata
# dirs), so a prep cell is "not ready" only when its raw data is genuinely
# absent -- distinct from "cache not built yet" (which shows up as the cache
# being missing from the consumer's inputs, not the prep cell's).
def _prep_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # --- Sequence-level SAE fingerprint caches (per benchmark) --------------
    # Each encodes a benchmark's unique sequences once. Raw source -> seq cache.
    _seq_prep = (
        # name suffix, script, raw inputs, product cache
        ("c3", "cache_esmc_fingerprints.py",
         (C3_TRAIN_CSV, C3_VAL_CSV, C3_TEST_CSV), ESMC_DEFAULT_SEQ_CACHE),
        ("cross_species", "cache_cross_species_esmc_fingerprints.py",
         (CROSS_SPECIES_DIR,), CROSS_SPECIES_SEQ_CACHE),
        ("bernett", "cache_bernett_esmc_fingerprints.py",
         (BERNETT_DIR,), BERNETT_SEQ_CACHE),
    )
    for suffix, script, raw, product in _seq_prep:
        exps.append(
            Experiment(
                name=f"prep.seq_cache.{suffix}",
                layer="prep",
                script=f"scripts/cache/{script}",
                inputs=raw,
                products=(product,),
                auto=False,
            )
        )

    # --- Protein-level SAE caches (participation / essentiality) ------------
    # PRING human graph proteins (seeded from the Rosetta seq cache).
    exps.append(
        Experiment(
            name="prep.protein_cache.pring_human",
            layer="prep",
            script="scripts/cache/cache_pring_human_esmc_sae.py",
            inputs=(PRING_ROOT / "human" / "human_simple.fasta",),
            products=(PRING_HUMAN_SAE_CACHE,),
            auto=False,
        )
    )
    # PRING held-out species -- these three products are exactly what gates
    # protein.pring_cross_species_generalization; building them lights it up.
    for species in PRING_SPECIES:
        exps.append(
            Experiment(
                name=f"prep.protein_cache.pring_{species}",
                layer="prep",
                script="scripts/cache/cache_pring_species_esmc_sae.py",
                args=("--species", species),
                inputs=(PRING_ROOT / species / f"{species}_simple.fasta",),
                products=(PRING_SPECIES_SAE_CACHES[species],),
                auto=False,
            )
        )
    # PIC human essentiality proteins.
    exps.append(
        Experiment(
            name="prep.protein_cache.pic_human",
            layer="prep",
            script="scripts/cache/cache_pic_human_esmc_sae.py",
            inputs=(PIC_DATASET_PKL["human"],),
            products=(PIC_HUMAN_SAE_CACHE,),
            auto=False,
        )
    )

    # --- Residue-level SAE cache (interface grounding) ----------------------
    # PDB_PPI positive interface chains. Builds LMDB-style dir under residue_caches.
    exps.append(
        Experiment(
            name="prep.residue_cache.pdb_ppi",
            layer="prep",
            script="scripts/cache/cache_pdb_afdb_sae.py",
            inputs=(PPI_DATA,),
            products=(PDB_PPI_SAE_CACHE,),
            auto=False,
        )
    )

    return exps


# ===========================================================================
# Layer 1 -- PROTEIN (participation / hubness / essentiality)
# ===========================================================================
def _protein_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # C3 high-participation ("hubness") classifier -- reads the C3 seq cache.
    for rep in PROTEIN_REPS:
        exps.append(
            Experiment(
                name=f"protein.c3_high_participation.{rep}",
                layer="protein",
                script="scripts/audit_protein/run_c3_high_participation_classifier.py",
                args=("--rep", rep),
                inputs=(ESMC_DEFAULT_SEQ_CACHE,),
                products=(RESULTS_PROTEIN / "c3_high_participation",),
            )
        )

    # C3 sequence participation oracle w/ feature importance -- needs the SAE
    # feature table too (annotation join).
    for rep in PROTEIN_REPS:
        exps.append(
            Experiment(
                name=f"protein.c3_participation_oracle.{rep}",
                layer="protein",
                script="scripts/audit_protein/run_c3_seq_participation_oracle_with_importance.py",
                args=("--rep", rep),
                inputs=(ESMC_DEFAULT_SEQ_CACHE, FEATURE_TABLE),
                products=(RESULTS_PROTEIN / "c3_seq_participation_oracle",),
            )
        )

    # Generic participation oracle across the two protein families it supports.
    _family_cache = {"c3": ESMC_DEFAULT_SEQ_CACHE, "cross_species": CROSS_SPECIES_SEQ_CACHE}
    for family, cache in _family_cache.items():
        for rep in PROTEIN_REPS:
            exps.append(
                Experiment(
                    name=f"protein.participation_oracle.{family}.{rep}",
                    layer="protein",
                    script="scripts/audit_protein/run_participation_oracle.py",
                    args=("--family", family, "--rep", rep),
                    inputs=(cache,),
                    products=(RESULTS_PROTEIN / "participation_oracle",),
                )
            )

    # PIC essentiality classifier -- one cell per PIC dataset (human on disk;
    # mouse/cell declared, light up when their protein caches are built).
    _pic_cache = {
        "human": PIC_HUMAN_SAE_CACHE,
        # mouse/cell share the essentiality script via --cache-path; their SAE
        # caches are not built yet but the raw pkl exists. Declared for matrix
        # completeness; the cell is not ready until a cache-path is produced.
    }
    for dataset, cache in _pic_cache.items():
        exps.append(
            Experiment(
                name=f"protein.pic_essentiality.{dataset}",
                layer="protein",
                script="scripts/audit_protein/run_pic_essentiality_classifier.py",
                args=("--cache-path", str(cache)),
                inputs=(cache, PIC_DATASET_PKL[dataset]),
                products=(RESULTS_PROTEIN / "pic_essentiality",),
            )
        )

    # PRING high-participation classifier -- one cell per graph-construction
    # method, reads the human protein SAE cache.
    for method in PRING_METHODS:
        exps.append(
            Experiment(
                name=f"protein.pring_high_participation.{method.lower()}",
                layer="protein",
                script="scripts/audit_protein/run_pring_high_participation_classifier.py",
                args=("--method", method),
                inputs=(PRING_HUMAN_SAE_CACHE,),
                products=(RESULTS_PROTEIN / "pring_participation" / "high_p90_xgboost",),
            )
        )

    # PRING participation oracle -- per method.
    for method in PRING_METHODS:
        exps.append(
            Experiment(
                name=f"protein.pring_participation_oracle.{method.lower()}",
                layer="protein",
                script="scripts/audit_protein/run_pring_participation_oracle.py",
                args=("--method", method),
                inputs=(PRING_HUMAN_SAE_CACHE,),
                products=(RESULTS_PROTEIN / "pring_participation",),
            )
        )

    # PRING cross-species generalization -- trains on human, zero-shot tests on
    # held-out species graphs. Needs human cache + each species cache present.
    _species_caches = tuple(PRING_SPECIES_SAE_CACHES[s] for s in PRING_SPECIES)
    exps.append(
        Experiment(
            name="protein.pring_cross_species_generalization",
            layer="protein",
            script="scripts/audit_protein/run_pring_cross_species_generalization.py",
            inputs=(PRING_HUMAN_SAE_CACHE, *_species_caches),
            products=(RESULTS_PROTEIN / "pring_participation",),
        )
    )

    return exps


# ===========================================================================
# Layer 2 -- PAIR (endpoint-additive / TabPFN / neg-sampling)
# ===========================================================================
def _pair_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # C3 endpoint-additive MLP + EBM, per rep. Read the C3 pair cache.
    for rep in PAIR_REPS:
        exps.append(
            Experiment(
                name=f"pair.c3_endpoint_additive_mlp.{rep}",
                layer="pair",
                script="scripts/audit_pair/run_c3_sae_endpoint_additive_mlp.py",
                args=("--rep", rep),
                inputs=_pair_cache_inputs(PAIR_CACHES, (rep,)),
                products=(RESULTS_PAIR / "c3_endpoint_additive_mlp_sae",),
            )
        )
        exps.append(
            Experiment(
                name=f"pair.c3_endpoint_additive_ebm.{rep}",
                layer="pair",
                script="scripts/audit_pair/run_c3_sae_endpoint_additive_ebm.py",
                args=("--rep", rep),
                inputs=_pair_cache_inputs(PAIR_CACHES, (rep,)),
                products=(RESULTS_PAIR / "c3_endpoint_additive_ebm_sae",),
            )
        )

    # PRING endpoint-additive MLP -- one cell per rep; the script itself loops
    # over BFS/DFS/RANDOM_WALK. Reads the human protein SAE cache.
    for rep in PAIR_REPS:
        exps.append(
            Experiment(
                name=f"pair.pring_endpoint_additive_mlp.{rep}",
                layer="pair",
                script="scripts/audit_pair/run_pring_sae_endpoint_additive_mlp.py",
                args=("--rep", rep),
                inputs=(PRING_HUMAN_SAE_CACHE,),
                products=(RESULTS_PAIR / "pring_endpoint_additive_mlp_sae",),
            )
        )

    # C3 negative-sampling bias audit -- consumes the pair cache + the pair-id
    # alignment produced under negative_sampling_audit/ by the prep step.
    exps.append(
        Experiment(
            name="pair.c3_negative_sampling_bias",
            layer="pair",
            script="scripts/audit_pair/analyze_c3_negative_sampling_bias.py",
            inputs=_pair_cache_inputs(PAIR_CACHES, ("sae_max",)),
            products=(RESULTS_PAIR / "negative_sampling_audit",),
        )
    )

    # C3 localization confound / robustness -- read the pair cache + the
    # fetched UniProt localization table under negative_sampling_audit/.
    for kind in ("confound", "robustness"):
        exps.append(
            Experiment(
                name=f"pair.c3_localization_{kind}",
                layer="pair",
                script=f"scripts/audit_pair/analyze_c3_localization_{kind}.py",
                inputs=_pair_cache_inputs(PAIR_CACHES, ("sae_max",)),
                products=(RESULTS_PAIR / "negative_sampling_audit",),
            )
        )

    return exps


# ===========================================================================
# Layer 3 -- RESIDUE (interface grounding)
# ===========================================================================
def _residue_experiments() -> list[Experiment]:
    enrich_dir = RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae"
    return [
        Experiment(
            name="residue.interface_sae_enrichment",
            layer="residue",
            script="scripts/audit_residue/compute_pdb_ppi_interface_sae_enrichment.py",
            inputs=(PDB_PPI_SAE_CACHE,),
            products=(enrich_dir,),
        ),
        Experiment(
            name="residue.contact_compatibility",
            layer="residue",
            script="scripts/audit_residue/compute_pdb_ppi_sae_contact_compatibility.py",
            inputs=(PDB_PPI_SAE_CACHE, FEATURE_TABLE),
            products=(RESULTS_RESIDUE / "interface_grounding" / "pdb_ppi_pos_sae_contact_compat_top4",),
        ),
    ]


# ===========================================================================
# Cross-cutting -- BASELINE (fingerprint sweep + benchmark eval)
# ===========================================================================
def _baseline_experiments() -> list[Experiment]:
    exps: list[Experiment] = []
    fingerprint_dir = RESULTS_MISC / "ppi_fingerprint"

    # PPI fingerprint baseline sweep. Reps mirror ppi_fingerprint REPRESENTATIONS.
    for rep in PROTEIN_REPS:
        exps.append(
            Experiment(
                name=f"baseline.ppi_fingerprint.{rep}",
                layer="baseline",
                script="scripts/baseline/run_ppi_fingerprint_baseline.py",
                args=("--rep", rep),
                inputs=(ESMC_DEFAULT_SEQ_CACHE, CROSS_SPECIES_SEQ_CACHE, ROSETTA_SEQ_CACHE),
                products=(fingerprint_dir,),
            )
        )

    # Benchmark evaluation (embedding-norm sanity eval) over all benchmarks.
    exps.append(
        Experiment(
            name="baseline.eval_benchmark",
            layer="baseline",
            script="scripts/baseline/eval_benchmark.py",
            args=("--all",),
            inputs=(ROSETTA_SEQ_CACHE, ESMC_DEFAULT_SEQ_CACHE, CROSS_SPECIES_SEQ_CACHE),
            products=(RESULTS_MISC / "eval",),
        )
    )
    return exps


# ===========================================================================
# Cross-cutting -- ANALYSIS / INTERPRETABILITY (consume upstream products)
# ===========================================================================
def _analysis_experiments() -> list[Experiment]:
    return [
        Experiment(
            name="analysis.cross_species_tabpfn_topk",
            layer="analysis",
            script="scripts/analysis/run_cross_species_tabpfn_topk.py",
            inputs=(CROSS_SPECIES_SEQ_CACHE,),
            products=(RESULTS_PAIR / "tabpfn" / "cross_species_tabpfn_topk",),
        ),
    ]


def _interpretability_experiments() -> list[Experiment]:
    # Explains TabPFN retrieval; consumes the TabPFN ranking product produced by
    # the pair TabPFN step. Declared as an input so it stays gated until ready.
    from conf.paths import TABPFN_RANKING, TABPFN_RETRIEVAL

    return [
        Experiment(
            name="interp.tabpfn_retrieval",
            layer="interpretability",
            script="scripts/interpretability/explain_tabpfn_retrieval.py",
            inputs=(TABPFN_RANKING,),
            products=(TABPFN_RETRIEVAL,),
        ),
    ]


# ===========================================================================
# Cross-cutting -- FIGURES (pure sinks; read many upstream products)
# ===========================================================================
def _figure_experiments() -> list[Experiment]:
    # Figure scripts under scripts/figures/ were removed and will be rewritten.
    # Re-register cells here once the new figure entrypoints exist; until then
    # the figures layer is intentionally empty so readiness/smoke stay honest.
    return []


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def all_experiments() -> list[Experiment]:
    """Every declared cell, sorted by layer rank then name (stable order)."""
    exps: list[Experiment] = []
    exps += _prep_experiments()
    exps += _protein_experiments()
    exps += _pair_experiments()
    exps += _residue_experiments()
    exps += _analysis_experiments()
    exps += _interpretability_experiments()
    exps += _baseline_experiments()
    exps += _figure_experiments()
    exps.sort(key=lambda e: (e.layer_rank, e.name))
    _check_unique(exps)
    return exps


def _check_unique(exps: list[Experiment]) -> None:
    seen: set[str] = set()
    for e in exps:
        if e.name in seen:
            raise ValueError(f"duplicate experiment name: {e.name}")
        seen.add(e.name)


def experiments_by_layer(layer: str) -> list[Experiment]:
    return [e for e in all_experiments() if e.layer == layer]


def get_experiment(name: str) -> Experiment:
    for e in all_experiments():
        if e.name == name:
            return e
    raise KeyError(name)


# Materialised once for cheap import-time access; call all_experiments() when a
# fresh readiness view is needed (probes hit the filesystem lazily anyway).
EXPERIMENTS = all_experiments()
