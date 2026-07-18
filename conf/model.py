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
knobs (XGBoost lr / n_estimators, EBM bins, MLP epochs, dual-tower widths, ...)
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
# (``w_enc.shape[0]`` for the codebook, ``layer.params.k`` for top-k) and should
# assert equality rather than importing these as the source.
ESMC_SAE_DIM = 16384                # codebook size (# features)
ESMC_SAE_K = 64                     # top-k active features per residue (k64 codebook16384)


# === ESM-2 (650M) — legacy InterPLM fingerprint line =======================
# facebook/esm2_t33_650M_UR50D: 33 layers, last-layer hidden 1280. The InterPLM
# ReLU-SAE for layer 33 projects to a 10240-d codebook.
ESM2_LAYER = 33                     # last layer of esm2_t33 (== last_hidden_state)
ESM2_DIM = 1280                     # backbone hidden size (SAE activation_dim)
ESM2_SAE_DIM = 10240                # InterPLM SAE codebook size


# === Shared / project-wide defaults ========================================
# Residue truncation cap shared by every extractor. BOS/EOS are added on top
# (callers use ``max_length = MAX_RESIDUES + 2``), so this is the residue budget,
# not the tokenizer's max_length.
MAX_RESIDUES = 1022

# "Active feature" definition for the binary SAE view: a feature fires on a
# residue iff ``esmc_sae_max > SAE_BINARY_THRESHOLD``. Encoded on disk as the
# ``binary_thr0`` cache name; surfaced here so ">0" is a named choice, not a
# scattered literal.
SAE_BINARY_THRESHOLD = 0.0

# Project-wide RNG seed. Every model now uses this; the EBM/endpoint-additive
# line previously pinned 7 to reproduce its first cached manuscript fits, but that
# carve-out was retired so there is a single seed across the audit (see
# src/models/estimators/ebm.py). Re-running the EBM line reseeds on 42 and so
# differs from those original cached results.
DEFAULT_SEED = 42


# Export every module-level constant (UPPER_CASE names), mirroring paths.py so
# the export list never goes stale as constants are added.
__all__ = sorted(name for name in dict(globals()) if name.isupper())
