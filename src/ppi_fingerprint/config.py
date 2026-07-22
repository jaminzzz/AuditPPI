"""Configuration for the v1 PPI fingerprint baseline protocol.

The Ladder-2 baseline reads the per-dataset ``auditppi_protein_features_v1``
protein caches (``conf.paths.PPI_PREDICTION_CACHES``): one cache per benchmark
family holds every endpoint sequence across that family's splits, with BOTH
backbone lines (ESM-C L60/L80 + ESM-2 L33) and all four channels in one payload.
Backbone/layer are selected within the cache at read time. PRING is per-species
(human train graph + yeast/ecoli/arath zero-shot test graphs), so it resolves
its cache by species via ``PRING_SPECIES_SAE_CACHES`` rather than one family key.

Result layout (under :data:`OUT_DIR`)::

    {family}/{model}/cells/{rep}_{backbone}L{layer}[_{pair_mode}]_{eval}.json
    {family}/{model}/summaries/{backbone}L{layer}[_{pair_mode}].json
"""

from __future__ import annotations

from pathlib import Path

from conf.model import REPRESENTATIONS
from conf.paths import (
    PPI_PREDICTION_CACHES as CACHE,
    PRING_SPECIES_SAE_CACHES,
    RESULTS_MAIN,
)

MODEL_NAMES = ("xgb", "tabpfn", "mlp_pair", "tabm_pair")
PAIR_MODELS = frozenset({"mlp_pair", "tabm_pair"})

# Benchmark family -> its own native train split. Each family trains on its own
# train set and is scored on its eval split (user decision). RF2-PPI has no train
# split, so it borrows C3's (zero-shot eval). PRING is handled specially in the
# runner: it trains on the human train graph of the SAME sampling method as the
# eval (default BFS for the zero-shot cross-species graphs), so it is not keyed
# here -- see ``PRING_DEFAULT_METHOD``.
NATIVE_TRAIN = {
    "c1": "c1:train",
    "c2": "c2:train",
    "c3": "c3:train",
    "cross_species": "cross_species:human_train",
    "bernett": "bernett:train",
    "rf2ppi": "c3:train",
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
    ``cross_species:ecoli`` → ``ecoli``; bare ``rf2ppi`` → ``rf2ppi``.
    """
    if ":" not in eval_name:
        return eval_name.replace("/", "_")
    family, rest = eval_name.split(":", 1)
    return rest.replace(":", "_") if rest else family


def cell_dir(family: str, model: str, *, root: Path = OUT_DIR) -> Path:
    return root / family / model / "cells"


def summary_dir(family: str, model: str, *, root: Path = OUT_DIR) -> Path:
    return root / family / model / "summaries"


def cell_stem(
    model: str,
    rep: str,
    b_tag: str,
    eval_name: str,
    *,
    pair_mode: str = "sym",
) -> str:
    """Filename stem (no ``.json``) for one (model, rep, axis, mode, eval) cell."""
    suffix = eval_suffix(eval_name)
    if model in PAIR_MODELS:
        return f"{rep}_{b_tag}_{pair_mode}_{suffix}"
    return f"{rep}_{b_tag}_{suffix}"


def summary_stem(
    model: str,
    b_tag: str,
    *,
    pair_mode: str = "sym",
) -> str:
    """Filename stem for one CLI summary (per family×model×axis[×pair_mode])."""
    if model in PAIR_MODELS:
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
    root: Path = OUT_DIR,
) -> Path:
    return cell_dir(family, model, root=root) / (
        f"{cell_stem(model, rep, b_tag, eval_name, pair_mode=pair_mode)}.json"
    )


def summary_path(
    family: str,
    model: str,
    b_tag: str,
    *,
    pair_mode: str = "sym",
    root: Path = OUT_DIR,
) -> Path:
    return summary_dir(family, model, root=root) / (
        f"{summary_stem(model, b_tag, pair_mode=pair_mode)}.json"
    )


__all__ = [
    "CACHE",
    "MODEL_NAMES",
    "NATIVE_TRAIN",
    "OUT_DIR",
    "PAIR_MODELS",
    "PRING_DEFAULT_METHOD",
    "PRING_SPECIES_SAE_CACHES",
    "REPRESENTATIONS",
    "cell_dir",
    "cell_path",
    "cell_stem",
    "eval_suffix",
    "family_of",
    "summary_dir",
    "summary_path",
    "summary_stem",
]
