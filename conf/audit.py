"""Centralized audit *scientific definitions* for AuditPPI.

Third companion to ``conf/paths.py`` (every path) and ``conf/model.py`` (frozen
backbone / SAE architecture constants). Where those two own *where things live*
and *how big the tensors are*, this module owns the **numeric choices that define
the audit's science**: the participation quantile that splits hubs from non-hubs,
the Cβ-Cβ distance that calls a residue pair a contact, the rSASA that calls a
residue solvent-exposed, and the BH-FDR level. These were previously re-declared
as bare CLI defaults / literals (``default=0.9``, ``default=8.0``,
``default=0.05``, ``np.quantile(..., 0.9)``) across the audit scripts.

Not invariants — they are *defaults for a scientific choice*
------------------------------------------------------------
Unlike ``conf/model.py`` (architecture facts the checkpoint fixes), everything
here is a deliberate analysis parameter with a defensible default that a run may
override. The point of centralizing is NOT to freeze them — CLI ``--fdr-alpha``,
``--contact-threshold``, etc. must keep working — it is to give the default a
single named home so the *same* number is not silently re-typed in eight scripts
and so changing the audit's definition of "contact" is a one-line edit here.

Contract for consumers
-----------------------
Import the default and pass it as the argparse ``default=`` (or use it directly
where the literal was hard-coded and there is no flag, e.g.
``src/eval/participation.py``'s degree-p90 threshold). Do NOT re-type the
literal. A script that needs a different value still passes it via its flag; the
constant is only the default.

Scope
-----
Audit science ONLY. Path constants -> ``conf/paths.py``. Backbone/SAE dimensions,
layers, seeds, the binary-view threshold -> ``conf/model.py``. Per-experiment
model hyperparameters (lr, n_estimators, epochs) stay in their estimator
dataclasses / CLI defaults, same rule as ``conf/model.py``.
"""
from __future__ import annotations

# === Participation / hubness (Ladder 1) =====================================
# Quantile of the (train-set) participation-degree distribution above which a
# protein is labelled "high participation" / hub. Default for the PRING and C3
# high-participation classifiers (``--quantile``) and the degree-p90 threshold
# reported by ``src/eval/participation.py`` (which had no flag: it hard-
# coded 0.9, now sourced from here so the two definitions cannot drift).
PARTICIPATION_QUANTILE = 0.9

# === Interface grounding (Ladder 3) =========================================
# Positive interface residue pair: inter-chain Cβ-Cβ distance <= this, in Å.
# Default for ``--contact-threshold`` in the enrichment / contact-compatibility
# audits and the interface-mask builder.
CONTACT_DISTANCE_ANGSTROM = 8.0

# Negative control residue pair: Cβ-Cβ distance >= this (in Å), sampled within
# the same PDB. Default for ``--control-min-distance`` in the contact-
# compatibility audit. Kept well above CONTACT_DISTANCE_ANGSTROM so the positive
# and control sets do not touch.
CONTROL_MIN_DISTANCE_ANGSTROM = 12.0

# Surface residue call: relative SASA (single-chain Shrake-Rupley SASA / max ASA)
# at or above this fraction is "solvent-exposed". Default for ``--rsasa-threshold``
# when adding surface non-interface controls. Deliberately coarse (0.20).
SURFACE_RSASA_THRESHOLD = 0.20

# === Statistics ============================================================
# Benjamini-Hochberg FDR level: a feature is called enriched iff its BH q-value
# <= this. Default for ``--fdr-alpha`` in both residue-level audits.
FDR_ALPHA = 0.05


# Export every module-level constant (UPPER_CASE names), mirroring paths.py /
# model.py so the export list never goes stale as constants are added.
__all__ = sorted(name for name in dict(globals()) if name.isupper())
