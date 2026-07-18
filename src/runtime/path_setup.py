"""Helpers for importing vendored upstream code that is not an installed package.

AuditPPI itself is an editable install (``conf`` / ``src``), so project-owned
modules never need ``sys.path`` surgery. A few workflows still import *upstream*
source trees that live under ``external/`` or ``baselines/`` and are deliberately
not packaged:

  - InterPLM ReLU-SAE (``external/InterPLM``)
  - PPLM / MINT baseline models (``baselines/PPLM``, ``baselines/mint``)
  - TabPFN source fallback when the package is not installed (``external/TabPFN/src``)

All such sites go through :func:`ensure_on_sys_path` so the bootstrap is one
named helper instead of ad-hoc ``sys.path.insert`` calls scattered across
scripts. This is intentional, not a regression of the editable-install layout.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_on_sys_path(root: str | Path) -> Path:
    """Prepend ``root`` to ``sys.path`` once, if it exists on disk.

    Returns the resolved path (whether or not it was newly inserted) so callers
    can log it. A missing root is a no-op on the path list -- the subsequent
    import will raise the normal ``ModuleNotFoundError`` with a clear message.
    """
    path = Path(root).expanduser().resolve()
    as_str = str(path)
    if path.exists() and as_str not in sys.path:
        sys.path.insert(0, as_str)
    return path
