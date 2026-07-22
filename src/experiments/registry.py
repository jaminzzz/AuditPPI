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

Because the consumer layers (analysis / interpretability) list their
*upstream products* as inputs, the same probe doubles as dependency
resolution: a consumer whose upstream JSON has not been produced yet is simply
not ready. Combined with the layer ordering (see ``LAYER_ORDER``) this gives a
topological execution order without a hand-maintained DAG.

Out-dir policy
--------------
The registry does **not** redirect ``--out-dir``. Every script writes to its
own established default location, because the analysis/interpretability
consumers read their upstream products from hard-coded paths (e.g.
``RESULTS_PROTEIN/pring_participation/high_p90_xgboost``). Per-run traceability
is provided by the sidecar + history written *next to* each product by the
runner (see ``src.experiments.history``), which does not depend on the directory
layout. Moving products into a faceted tree would sever the consumer chain for
zero traceability gain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from conf.paths import (
    BERNETT_SAE_CACHE,
    C1_SAE_CACHE,
    C2_SAE_CACHE,
    C3_PAIR_INDEX_CACHES,
    C3_SAE_CACHE,
    CROSS_SPECIES_PAIR_INDEX_CACHES,
    CROSS_SPECIES_SAE_CACHE,
    FEATURE_TABLE,
    PDB_PPI_SAE_CACHE,
    PIC_DATASET_PKL,
    PIC_HUMAN_CSV,
    PIC_HUMAN_SAE_CACHE,
    POOLED_ESM2_SEQ_CACHE,
    POOLED_ESMC_SEQ_CACHE,
    PPI_DATA,
    PRING_CROSS_SPECIES as PRING_SPECIES,
    PRING_HUMAN_SAE_CACHE,
    PRING_SPECIES_SAE_CACHES,
    RESULTS_MISC,
    RESULTS_PAIR,
    RESULTS_PROTEIN,
    RESULTS_RESIDUE,
)
from src.data.pring_graph import METHODS as PRING_METHODS

# --- Layer ordering (main-trunk route: protein -> pair -> residue) ----------
# Producers before consumers. ``prep`` (the GPU forward-pass cache builders)
# feeds everything else, so it sorts first. The user's requested trunk is then
# protein -> pair -> residue; analysis/interpretability/baseline are
# cross-cutting, so they sort to the end. The runner
# uses this as the primary sort key; readiness probing over declared inputs
# handles the finer producer->consumer edges within and across layers. Note prep
# cells are ``auto=False`` (see below), so ``--all`` never triggers a GPU
# encode; they are declared for matrix completeness and to light up the audit
# cells that consume their caches.
#
# Figure rendering is deliberately *not* a layer here: the plotting scripts live
# under ``manuscripts/scripts/`` and are run by hand during manuscript prep, not
# through the audit reproduction matrix.
LAYER_ORDER = (
    "prep",
    "protein",
    "pair",
    "residue",
    "analysis",
    "interpretability",
    "baseline",
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
# PRING_METHODS / PRING_SPECIES imported above (single source: pring_graph / paths).
PROTEIN_REPS = ("sae_max", "binary", "esmc_mean")  # mirrors conf.model.REPRESENTATIONS

# A pair consumer's inputs are now the lightweight pair-index caches (endpoint
# row indices + labels) plus the one v1 protein cache they gather channels from.
# One index cache serves every (backbone, layer, rep) channel and pair mode, so
# the rep axis no longer multiplies the declared inputs.
def _pair_index_inputs(
    index_caches: dict[str, Path], protein_cache: Path
) -> tuple[Path, ...]:
    """The pair-index split caches + protein cache a pair endpoint run needs."""
    return (*index_caches.values(), protein_cache)


# ===========================================================================
# Layer 0 -- PREP (feature-cache builders: the GPU forward passes)
# ===========================================================================
# Active prep cells only. Per-benchmark ESM-C fingerprint builders, GPU PRING
# protein-cache scripts, and the PIC pickle-wrapper slicer were retired:
# features now come from pooled seq caches under ``data/sae/seq_caches`` via
# ``scripts/prep/slice_dataset_protein_cache.py`` (PIC uses the exported CSV
# ``data/raw/pic/pic_human.csv`` from ``export_pic_human_csv.py``). Retired
# scripts live under ``backups/scripts/cache/``.
#
# Remaining cells are ``auto=False`` so ``--all`` never kicks off multi-hour
# work implicitly. Run them with ``--experiment prep.<name>``.
def _prep_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # PIC human essentiality: pure-CPU slice from pooled seq caches (generic slicer).
    exps.append(
        Experiment(
            name="prep.protein_cache.pic_human",
            layer="prep",
            script="scripts/prep/slice_dataset_protein_cache.py",
            args=(
                "--input", str(PIC_HUMAN_CSV),
                "--sequence-cols", "sequence",
                "--id-cols", "ID",
                "--esmc-pooled", str(POOLED_ESMC_SEQ_CACHE),
                "--esm2-pooled", str(POOLED_ESM2_SEQ_CACHE),
                "--output", str(PIC_HUMAN_SAE_CACHE),
            ),
            inputs=(PIC_HUMAN_CSV, POOLED_ESMC_SEQ_CACHE, POOLED_ESM2_SEQ_CACHE),
            products=(PIC_HUMAN_SAE_CACHE,),
            auto=False,
        )
    )

    # Residue-level SAE cache (interface grounding). Builds LMDB-style dir under
    # residue_caches -- not the protein pooled pipeline.
    exps.append(
        Experiment(
            name="prep.residue_cache.pdb_ppi",
            layer="prep",
            script="scripts/features/cache_pdb_afdb_sae.py",
            inputs=(PPI_DATA,),
            products=(PDB_PPI_SAE_CACHE,),
            auto=False,
        )
    )

    return exps


# ===========================================================================
# Ladder 1 -- PROTEIN (participation / hubness / essentiality)
# ===========================================================================
def _protein_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # C3 high-participation ("hubness") classifier -- assembles per-protein rows
    # on the fly from the C3 v1 protein cache (seq2idx gather, backbone/layer axes).
    for rep in PROTEIN_REPS:
        exps.append(
            Experiment(
                name=f"protein.c3_high_participation.{rep}",
                layer="protein",
                script="scripts/audit_protein/run_c3_high_participation_classifier.py",
                args=("--rep", rep),
                inputs=(C3_SAE_CACHE,),
                products=(RESULTS_PROTEIN / "c3_high_participation",),
            )
        )

    # C3 sequence participation oracle w/ feature importance -- reads the C3 v1
    # protein cache plus the SAE feature table (annotation join).
    for rep in PROTEIN_REPS:
        exps.append(
            Experiment(
                name=f"protein.c3_participation_oracle.{rep}",
                layer="protein",
                script="scripts/audit_protein/run_c3_seq_participation_oracle_with_importance.py",
                args=("--rep", rep),
                inputs=(C3_SAE_CACHE, FEATURE_TABLE),
                products=(RESULTS_PROTEIN / "c3_seq_participation_oracle",),
            )
        )

    # Generic participation oracle across the two protein families it supports;
    # each family reads its own v1 protein cache (backbone/layer axes).
    _family_cache = {"c3": C3_SAE_CACHE, "cross_species": CROSS_SPECIES_SAE_CACHE}
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
# Ladder 2 -- PAIR (endpoint-additive / TabPFN / neg-sampling)
# ===========================================================================
def _pair_experiments() -> list[Experiment]:
    exps: list[Experiment] = []

    # C3 endpoint-additive MLP + EBM, per rep. Both now assemble endpoints on the
    # fly from the C3 v1 protein cache (sequence-keyed, all backbone/layer/rep
    # channels in one file); the EBM gathers endpoint rows through the lightweight
    # C3 pair-index caches.
    for rep in PAIR_REPS:
        exps.append(
            Experiment(
                name=f"pair.c3_endpoint_additive_mlp.{rep}",
                layer="pair",
                script="scripts/audit_pair/run_c3_sae_endpoint_additive_mlp.py",
                args=("--rep", rep),
                inputs=(C3_SAE_CACHE,),
                products=(RESULTS_PAIR / "c3_endpoint_additive_mlp_sae",),
            )
        )
        exps.append(
            Experiment(
                name=f"pair.c3_endpoint_additive_ebm.{rep}",
                layer="pair",
                script="scripts/audit_pair/run_c3_sae_endpoint_additive_ebm.py",
                args=("--rep", rep),
                inputs=_pair_index_inputs(C3_PAIR_INDEX_CACHES, C3_SAE_CACHE),
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
            inputs=_pair_index_inputs(C3_PAIR_INDEX_CACHES, C3_SAE_CACHE),
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
                inputs=_pair_index_inputs(C3_PAIR_INDEX_CACHES, C3_SAE_CACHE),
                products=(RESULTS_PAIR / "negative_sampling_audit",),
            )
        )

    # PPI-fingerprint pair-scale predictor (in-house participation-channel
    # baseline): train XGB on each family's native-train endpoints, score its
    # eval split. Backbone/layer is a formal matrix axis: each (family, axis)
    # is one cell (6 families x 3 axes = 18), so all three backbone/layer
    # results enter runs.jsonl + provenance history. The runner sweeps every
    # rep within a cell and writes a per-cell summary. Each family reads its own
    # v1 protein cache (PPI_PREDICTION_CACHES); PRING additionally needs its
    # per-species caches. All three axis channels co-exist in one cache payload
    # (ESM-C L60/L80 + ESM-2 L33), so a family's inputs are axis-independent.
    fingerprint_family_inputs = {
        "c1": (C1_SAE_CACHE,),
        "c2": (C2_SAE_CACHE,),
        "c3": (C3_SAE_CACHE,),
        "cross_species": (CROSS_SPECIES_SAE_CACHE,),
        "bernett": (BERNETT_SAE_CACHE,),
        "pring": tuple(PRING_SPECIES_SAE_CACHES.values()),
    }
    # (backbone, layer) axes -- the product tag matches the script's
    # ``summary_{family}_{model}_{backbone}L{layer}.json`` naming.
    fingerprint_axes = (("esmc", 60), ("esmc", 80), ("esm2", 33))
    for family, cache_inputs in fingerprint_family_inputs.items():
        for backbone, layer in fingerprint_axes:
            b_tag = f"{backbone}L{layer}"
            exps.append(
                Experiment(
                    name=f"pair.ppi_fingerprint.{family}.{b_tag}",
                    layer="pair",
                    script="scripts/audit_pair/run_ppi_fingerprint_baseline.py",
                    args=(
                        "--model", "xgb", "--family", family,
                        "--backbone", backbone, "--layer", str(layer),
                    ),
                    inputs=cache_inputs,
                    products=(
                        RESULTS_MISC / "ppi_fingerprint" / f"summary_{family}_xgb_{b_tag}.json",
                    ),
                )
            )

    return exps


# ===========================================================================
# Ladder 3 -- RESIDUE (interface grounding)
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
# Cross-cutting -- BASELINE (external published methods)
# ===========================================================================
def _baseline_experiments() -> list[Experiment]:
    # Feature extractors live under scripts/baseline/features/. Train/eval runners
    # that consume those caches will be registered here when added.
    return []


# ===========================================================================
# Cross-cutting -- ANALYSIS / INTERPRETABILITY (consume upstream products)
# ===========================================================================
def _analysis_experiments() -> list[Experiment]:
    from conf.paths import TABPFN_RANKING

    return [
        # Top-K SAE-feature probes: train on cross-species human_train endpoints,
        # zero-shot score the 5 held-out species. Gathers endpoint channels from
        # the shared cross-species v1 protein cache via per-graph pair-index caches
        # (val is carved from human_train in-memory); needs the TabPFN feature
        # ranking produced by the pair TabPFN step.
        Experiment(
            name="analysis.cross_species_tabpfn_topk",
            layer="analysis",
            script="scripts/analysis/run_cross_species_tabpfn_topk.py",
            inputs=(
                TABPFN_RANKING,
                *_pair_index_inputs(
                    CROSS_SPECIES_PAIR_INDEX_CACHES, CROSS_SPECIES_SAE_CACHE
                ),
            ),
            products=(RESULTS_PAIR / "tabpfn" / "cross_species_tabpfn_topk",),
        ),
        # Model-free t(p) diagnostic: how participation-prone each pair benchmark is.
        Experiment(
            name="analysis.diagnose_benchmark_participation",
            layer="analysis",
            script="scripts/analysis/diagnose_benchmark_participation.py",
            args=("--all",),
            inputs=(),
            products=(RESULTS_MISC / "eval",),
        ),
    ]


def _interpretability_experiments() -> list[Experiment]:
    # Explains TabPFN retrieval on C3: consumes the TabPFN ranking product from
    # the pair TabPFN step and materializes C3 endpoint features on the fly
    # through the C3 pair-index caches + v1 protein cache. All declared as
    # inputs so the cell stays gated until every dependency is ready.
    from conf.paths import TABPFN_RANKING, TABPFN_RETRIEVAL

    return [
        Experiment(
            name="interp.tabpfn_retrieval",
            layer="interpretability",
            script="scripts/analysis/explain_tabpfn_retrieval.py",
            inputs=(
                TABPFN_RANKING,
                *_pair_index_inputs(C3_PAIR_INDEX_CACHES, C3_SAE_CACHE),
            ),
            products=(TABPFN_RETRIEVAL,),
        ),
    ]


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
