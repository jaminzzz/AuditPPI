"""Configuration for the v1 PPI fingerprint baseline protocol.

The Ladder-2 baseline reads the per-dataset ``auditppi_protein_features_v1``
protein caches (``conf.paths.PPI_PREDICTION_CACHES``): one cache per benchmark
family holds every endpoint sequence across that family's splits, with BOTH
backbone lines (ESM-C L60/L80 + ESM-2 L33) and all four channels in one payload.
Backbone/layer are selected within the cache at read time. PRING is per-species
(human train graph + yeast/ecoli/arath zero-shot test graphs), so it resolves
its cache by species via ``PRING_SPECIES_SAE_CACHES`` rather than one family key.

Result layout (under :data:`OUT_DIR`)::

    {family}/{model}/seed_{S}/cells/{rep}_{backbone}L{layer}[_{pair_mode}]_{eval}.json
    {family}/{model}/seed_{S}/summaries/{backbone}L{layer}[_{pair_mode}].json

Every run writes under ``seed_{S}/`` (including the default seed 42).
"""

from __future__ import annotations

from pathlib import Path

from conf.model import DEFAULT_SEED, REPRESENTATIONS
from conf.paths import (
    PPI_PREDICTION_CACHES as CACHE,
    PRING_SPECIES_SAE_CACHES,
    RESULTS_MAIN,
)

MODEL_NAMES = ("xgb", "tabpfn", "mlp_pair", "tabm_pair")
PAIR_MODELS = frozenset({"mlp_pair", "tabm_pair"})

# Fingerprint-baseline representation vocabulary. Extends the three SAE views
# (:data:`~conf.model.REPRESENTATIONS`) with the backbone-agnostic eSIG-Net 573-D
# physicochemical fingerprint (``esig``). Kept separate from ``REPRESENTATIONS``
# so the participation oracle / SAE line stays pinned to the three SAE views; only
# this baseline opts into the extra comparison rep. ``esig`` ignores backbone/layer
# (served from the global eSIG cache), so it need only be run on a single nominal
# axis rather than swept across all backbone-layer combinations.
FINGERPRINT_REPS = (*REPRESENTATIONS, "esig")

# eSIG is backbone/layer-agnostic, so it collapses the ``{backbone}L{layer}`` axis
# tag onto a single fixed token. This keeps its results in their own summary/cell
# files (``esig[_mode].json``) instead of clobbering the SAE ``esmcL60`` summary,
# which carries a different rep set on the same nominal axis.
ESIG_AXIS_TAG = "esig"


def axis_tag(rep: str, backbone: str, layer: int) -> str:
    """The ``b_tag`` a rep writes under: ``{backbone}L{layer}`` for SAE reps,
    the fixed :data:`ESIG_AXIS_TAG` for the backbone-agnostic eSIG fingerprint."""
    if rep == "esig":
        return ESIG_AXIS_TAG
    return f"{backbone}L{layer}"

# Multi-seed matrix for stability reporting. Seed 42 is DEFAULT_SEED and is
# treated as already completed for the primary xgb matrix; re-runs typically
# only schedule the remaining seeds.
FINGERPRINT_SEEDS = (42, 43, 44)

# Benchmark family -> its own native train split. Each family trains on its own
# train set and is scored on its eval split (user decision). PRING is handled
# specially in the runner: it trains on the human train graph of the SAME
# sampling method as the eval (default BFS for the zero-shot cross-species
# graphs), so it is not keyed here -- see ``PRING_DEFAULT_METHOD``.
NATIVE_TRAIN = {
    "c1": "c1:train",
    "c2": "c2:train",
    "c3": "c3:train",
    "cross_species": "cross_species:human_train",
    "bernett": "bernett:train",
}

# PRING human-graph sampling method used for the train side when the eval graph
# does not itself pin one (the zero-shot cross-species test graphs).
PRING_DEFAULT_METHOD = "BFS"

OUT_DIR = RESULTS_MAIN / "ppi_fingerprint"


def family_of(name: str) -> str:
    """Top-level family token of a benchmark name (``c3:test`` → ``c3``)."""
    return name.split(":")[0]


def eval_suffix(eval_name: str) -> str:
    """Path-safe eval stem with the family prefix stripped.

    ``c3:test`` → ``test``; ``pring:human:test:BFS`` → ``human_test_BFS``;
    ``cross_species:ecoli`` → ``ecoli``; a bare name maps to itself.
    """
    if ":" not in eval_name:
        return eval_name.replace("/", "_")
    family, rest = eval_name.split(":", 1)
    return rest.replace(":", "_") if rest else family


def seed_dir(
    family: str,
    model: str,
    seed: int = DEFAULT_SEED,
    *,
    root: Path = OUT_DIR,
) -> Path:
    """``{root}/{family}/{model}/seed_{S}``."""
    return root / family / model / f"seed_{int(seed)}"


def cell_dir(
    family: str,
    model: str,
    seed: int = DEFAULT_SEED,
    *,
    root: Path = OUT_DIR,
) -> Path:
    return seed_dir(family, model, seed, root=root) / "cells"


def summary_dir(
    family: str,
    model: str,
    seed: int = DEFAULT_SEED,
    *,
    root: Path = OUT_DIR,
) -> Path:
    return seed_dir(family, model, seed, root=root) / "summaries"


def cell_stem(
    model: str,
    rep: str,
    b_tag: str,
    eval_name: str,
    *,
    pair_mode: str = "sym",
) -> str:
    """Filename stem (no ``.json``) for one (model, rep, axis, mode, eval) cell.

    Pair models always carry the mode token. Tabular models (xgb/tabpfn) stay
    bare for the default ``sym`` (so existing products are untouched) but carry
    the token for the ``product`` / ``absdiff`` feature ablations.
    """
    suffix = eval_suffix(eval_name)
    if model in PAIR_MODELS or pair_mode != "sym":
        return f"{rep}_{b_tag}_{pair_mode}_{suffix}"
    return f"{rep}_{b_tag}_{suffix}"


def summary_stem(
    model: str,
    b_tag: str,
    *,
    pair_mode: str = "sym",
) -> str:
    """Filename stem for one CLI summary (per family×model×axis[×pair_mode]).

    Same rule as :func:`cell_stem`: bare for tabular ``sym`` (untouched legacy
    layout), mode-tagged for pair models and the ``product`` / ``absdiff``
    tabular ablations.
    """
    if model in PAIR_MODELS or pair_mode != "sym":
        return f"{b_tag}_{pair_mode}"
    return b_tag


def cell_path(
    family: str,
    model: str,
    rep: str,
    b_tag: str,
    eval_name: str,
    *,
    pair_mode: str = "sym",
    seed: int = DEFAULT_SEED,
    root: Path = OUT_DIR,
) -> Path:
    return cell_dir(family, model, seed, root=root) / (
        f"{cell_stem(model, rep, b_tag, eval_name, pair_mode=pair_mode)}.json"
    )


def summary_path(
    family: str,
    model: str,
    b_tag: str,
    *,
    pair_mode: str = "sym",
    seed: int = DEFAULT_SEED,
    root: Path = OUT_DIR,
) -> Path:
    return summary_dir(family, model, seed, root=root) / (
        f"{summary_stem(model, b_tag, pair_mode=pair_mode)}.json"
    )


__all__ = [
    "CACHE",
    "ESIG_AXIS_TAG",
    "FINGERPRINT_REPS",
    "FINGERPRINT_SEEDS",
    "MODEL_NAMES",
    "NATIVE_TRAIN",
    "OUT_DIR",
    "PAIR_MODELS",
    "PRING_DEFAULT_METHOD",
    "PRING_SPECIES_SAE_CACHES",
    "REPRESENTATIONS",
    "axis_tag",
    "cell_dir",
    "cell_path",
    "cell_stem",
    "eval_suffix",
    "family_of",
    "seed_dir",
    "summary_dir",
    "summary_path",
    "summary_stem",
]
