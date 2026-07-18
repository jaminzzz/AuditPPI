"""AuditPPI configuration package.

Three single-source-of-truth modules, imported explicitly (never via this
package root) so each consumer names exactly what it depends on:

    conf.paths   -- every filesystem path (roots, caches, datasets, weights)
    conf.model   -- frozen backbone / SAE architecture constants + project seed
    conf.audit   -- audit scientific-choice defaults (quantile, contact Å, FDR)

This file is intentionally empty of re-exports: import from the specific module
(``from conf.model import ESMC_SAE_DIM``), not from ``conf`` itself. Its only job
is to make ``conf`` a real package so it is discovered by setuptools
``packages.find`` and shipped in a built wheel (previously it resolved only as an
implicit namespace package on the dev path, which a wheel build would drop).
"""
