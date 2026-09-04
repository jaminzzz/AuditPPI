"""Centralized model / SAE architecture constants for AuditPPI.

Companion to ``conf/paths.py``: where ``paths.py`` is the single source of truth
for every *path*, this module is the single source of truth for the *numeric
constants* of the frozen backbones and their SAEs — dimensions, sequence caps,
and the checkpoint's available layers. Downstream consumers (figure scripts,
analyses, audit scripts) previously re-declared these as bare literals
(``DIM = 16384``, ``LAYER = 60``, ``max_residues = 1022``, ...) in ~15 files;
import them from here instead.

Not everything here is a fixed invariant. Dimensions and codebook sizes are
architecture facts; the *probed layer* is a configurable audit choice with a
default (``ESMC_SAE_DEFAULT_LAYER``) that any run may override, kept distinct
from the fixed set the checkpoint actually provides
(``ESMC_SAE_AVAILABLE_LAYERS``).

Truth-source contract
----------------------
These constants are the *declared expectation*, NOT the primary source of truth.
The real source is the checkpoint: the cache-builder scripts read the SAE width
from ``w_enc.shape`` and top-k from ``layer.params.k`` at load time. That is
correct and must stay — do not replace those dynamic reads with these constants.
Instead, builders should **assert agreement**, e.g. ``assert dim == ESMC_SAE_DIM``,
so a checkpoint swap fails loudly here rather than silently drifting a hardcoded
copy elsewhere.

Scope
-----
Architecture invariants and project-wide defaults ONLY. Per-experiment tuning
knobs (XGBoost lr / n_estimators, EBM bins, MLP epochs, pair-MLP widths, ...)
do NOT belong here — they live in their estimator dataclasses / CLI defaults so
each experiment owns its own configuration. Audit *scientific definitions*
(participation quantile, contact-distance cutoff, surface rSASA, FDR alpha) are
a separate concern; keep those in ``conf/audit.py``.

Two backbone lines
------------------
    ESM-C  (ESMC-6B)  -- the primary audit line: 16384-codebook SAE shipping
                         weights for layers 60 and 80 (audit defaults to 60).
    ESM-2  (650M)     -- legacy InterPLM fingerprint line: layer-33, 10240 codebook.
Names are prefixed ``ESMC_`` / ``ESM2_`` so the two never collide.
"""
from __future__ import annotations

# === ESM-C (ESMC-6B) — primary audit line ==================================
# Probed transformer layer(s).
#
# TWO KINDS of value here, do not conflate them:
#   * ESMC_SAE_AVAILABLE_LAYERS -- an INVARIANT: the layers the SAE checkpoint
#     actually ships weights for. The ESMC-6B-sae-k64-codebook16384 snapshot has
#     layer_60.safetensors AND layer_80.safetensors, so the SAE is NOT single-
#     layer. Extend this tuple only when a new layer_*.safetensors is added.
#   * ESMC_SAE_DEFAULT_LAYER -- a CHOICE, not an invariant: the layer the cache
#     builders read when the caller does not pass --layer. The whole audit was
#     built on layer 60, so that is the default; pass --layer 80 to a cache
#     builder (and mind the layer-tagged output filename) to cache layer 80.
ESMC_SAE_AVAILABLE_LAYERS: tuple[int, ...] = (60, 80)  # checkpoint ships both (invariant)
ESMC_SAE_DEFAULT_LAYER = 60         # builder default when --layer is omitted (a choice)

# Dense-pooling layers the multi-layer extractor emits by default (esmc_l{L}_*).
# Dense features need no SAE weights, so this can span any valid transformer
# layer independent of ESMC_SAE_AVAILABLE_LAYERS.
ESMC_LAYERS: tuple[int, ...] = (60, 80)

# Backbone hidden size (SAE input / decoder-reconstruction dim). ESM-C is
# uniform-width, so layer 60 and layer 80 share this d_model. Determined by the
# model; assert against ``w_enc.shape[1]`` in cache builders.
ESMC_DIM = 2560

# SAE geometry. Declared expectation; builders read these from the checkpoint
# (``w_enc.shape[1]`` for the codebook — ``w_enc`` is ``(d_model, codebook)``,
# ``layer.params.k`` for top-k) and should assert equality rather than importing
# these as the source.
ESMC_SAE_DIM = 16384                # codebook size (# features)
ESMC_SAE_K = 64                     # top-k active features per residue (k64 codebook16384)


# === ESM-2 (650M) — legacy InterPLM fingerprint line =======================
# facebook/esm2_t33_650M_UR50D: 33 layers, last-layer hidden 1280. The InterPLM
# ReLU-SAE for layer 33 projects to a 10240-d codebook. Unlike ESM-C, ESM-2 ships
# a single SAE layer, so its "available layers" set is the singleton (33,).
ESM2_LAYER = 33                     # last layer of esm2_t33 (== last_hidden_state)
ESM2_DIM = 1280                     # backbone hidden size (SAE activation_dim)
ESM2_SAE_DIM = 10240                # InterPLM SAE codebook size
ESM2_SAE_AVAILABLE_LAYERS: tuple[int, ...] = (33,)  # InterPLM ships layer 33 only
ESM2_SAE_DEFAULT_LAYER = 33


# === eSIG-Net physicochemical fingerprint — backbone-agnostic comparison line =
# A pure sequence -> 573-D descriptor (AAC-20 + CTriad-343 + AC-210 over 7
# normalized physicochemical properties, lag 30). No backbone, no SAE, no layer:
# the width is fixed by the descriptor definition. Computed on the FULL sequence
# (eSIG's native behaviour), unlike the 1022-residue-capped ESM channels. Serves
# as the physicochemical baseline the SAE fingerprint is compared against, and is
# small enough to feed TabPFN's native feature budget directly.
ESIG_DIM = 573


# === Residue truncation (BOS/EOS sit on top of these budgets) ===============
# Callers use ``max_length = max_residues + 2`` (or equivalent), so these are
# *residue* caps, not tokenizer max_length.
#
# Backbone-native defaults follow each model's training / published context:
#   * ESM-2  -- 1024-token window  => 1022 residues + BOS/EOS
#   * ESM-C  -- 2048-token window  => 2046 residues + BOS/EOS
# ESM-C may also be run at 1022 (``ESMC_MAX_RESIDUES_COMPAT``) when aligning to
# legacy L60 pooled caches or to the ESM-2 length budget for fair comparison.
ESM2_MAX_RESIDUES = 1022
ESMC_MAX_RESIDUES = 2046
ESMC_MAX_RESIDUES_COMPAT = 1022  # optional shorter ESM-C budget

# Back-compat alias used by legacy single-layer cache builders and older call
# sites. Equals the ESM-2 / historical 1022 cap; new formal extractors should
# import ``ESMC_MAX_RESIDUES`` / ``ESM2_MAX_RESIDUES`` instead of this name.
MAX_RESIDUES = ESM2_MAX_RESIDUES

# Length variants that a v1 protein cache may be sliced at. This is a *which-file*
# axis (it names the on-disk cache variant, ``..._max{N}.pt``), NOT a within-cache
# selector like backbone/layer. ESM-C can be pooled at either budget; ESM-2's SAE
# is 1022-native, so an ``esm2_*`` channel only exists in the max1022 variant.
CACHE_MAX_RESIDUES_VARIANTS: tuple[int, ...] = (ESMC_MAX_RESIDUES_COMPAT, ESMC_MAX_RESIDUES)  # (1022, 2046)
CACHE_MAX_RESIDUES_DEFAULT = ESMC_MAX_RESIDUES_COMPAT  # 1022 — the cross-backbone-comparable slice on disk


def cache_max_residues_tag(max_residues: int) -> str:
    """Filename tag for a length variant, e.g. ``max1022`` (the v1 slicer suffix)."""
    return f"max{int(max_residues)}"

# "Active feature" definition for the binary SAE view: a feature fires on a
# residue iff ``esmc_sae_max > SAE_BINARY_THRESHOLD``. Encoded on disk as the
# ``binary_thr0`` cache name; surfaced here so ">0" is a named choice, not a
# scattered literal.
SAE_BINARY_THRESHOLD = 0.0

# The three pooled per-protein representations the SAE fingerprint line supports,
# shared verbatim by the fingerprint baseline and the participation predictors
# (previously re-declared in src/ppi_fingerprint/config.py and echoed in the
# sequence participation oracle):
#   binary    -- (esmc_sae_max > SAE_BINARY_THRESHOLD), the participation channel
#   sae_max   -- continuous pooled SAE max   [ESMC_SAE_DIM]
#   esmc_mean -- raw ESM-C layer mean        [ESMC_DIM]
REPRESENTATIONS: tuple[str, ...] = ("binary", "sae_max", "esmc_mean")

# === Backbone selector for v1 protein feature caches =======================
# A v1 cache (``auditppi_protein_features_v1``) holds channels for BOTH backbone
# lines at once: ``esmc_l{60,80}_{channel}`` AND ``esm2_l33_{channel}``, where
# channel in {dense_mean, dense_max, sae_max, sae_binary}. `backbone` selects
# which family a consumer reads — a within-cache axis, orthogonal to the
# length/which-file axis (CACHE_MAX_RESIDUES_*) above.
#
# The three REPRESENTATIONS names are backbone-agnostic aliases onto channels:
#   sae_max   -> {backbone}_l{layer}_sae_max     (continuous pooled SAE max)
#   binary    -> {backbone}_l{layer}_sae_binary  (== sae_max > 0, bit-identical)
#   esmc_mean -> {backbone}_l{layer}_dense_mean  (raw layer mean; name kept for
#                back-compat even when backbone == esm2)
BACKBONES: tuple[str, ...] = ("esmc", "esm2")
DEFAULT_BACKBONE = "esmc"

# Layers a v1 cache carries per backbone. ESM-C ships two SAE layers; ESM-2's
# InterPLM SAE is layer-33 only.
BACKBONE_LAYERS: dict[str, tuple[int, ...]] = {
    "esmc": ESMC_SAE_AVAILABLE_LAYERS,   # (60, 80)
    "esm2": (ESM2_LAYER,),               # (33,)
}
BACKBONE_DEFAULT_LAYER: dict[str, int] = {
    "esmc": ESMC_SAE_DEFAULT_LAYER,      # 60
    "esm2": ESM2_LAYER,                  # 33
}
# Per-backbone column counts, keyed by (backbone, is_dense). Used by rep_dim.
BACKBONE_DENSE_DIM: dict[str, int] = {"esmc": ESMC_DIM, "esm2": ESM2_DIM}
BACKBONE_SAE_DIM: dict[str, int] = {"esmc": ESMC_SAE_DIM, "esm2": ESM2_SAE_DIM}


def feature_cache_key(backbone: str, layer: int, channel: str) -> str:
    """v1 feature-cache key, e.g. ``esmc_l60_sae_max`` / ``esm2_l33_dense_mean``."""
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    return f"{backbone}_l{int(layer)}_{channel}"


def resolve_backbone_layer(backbone: str, layer: int | None) -> int:
    """Validate ``layer`` against a backbone (default when ``None``)."""
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    if layer is None:
        return BACKBONE_DEFAULT_LAYER[backbone]
    if int(layer) not in BACKBONE_LAYERS[backbone]:
        raise ValueError(
            f"backbone {backbone!r} has no layer {layer}; "
            f"available: {BACKBONE_LAYERS[backbone]}"
        )
    return int(layer)

# Project-wide RNG seed. Every model now uses this; the EBM/endpoint-additive
# line previously pinned 7 to reproduce its first cached manuscript fits, but that
# carve-out was retired so there is a single seed across the audit (see
# src/models/estimators/ebm.py). Re-running the EBM line reseeds on 42 and so
# differs from those original cached results.
DEFAULT_SEED = 42


# Export every module-level constant (UPPER_CASE names), mirroring paths.py so
# the export list never goes stale as constants are added.
__all__ = sorted(name for name in dict(globals()) if name.isupper())
