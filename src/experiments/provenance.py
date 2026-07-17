"""Run provenance capture for reproducible audit outputs.

`capture()` records *how* a result was produced — code version, environment,
and the exact input files consumed — so a JSON metrics file can be traced back
to a rerunnable state. It is deliberately best-effort and side-effect free: a
missing git repo, an unimportable package, or a vanished input path degrades to
``None``/``False`` rather than raising, because provenance must never be the
thing that crashes an experiment.
"""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

# Packages whose versions are worth pinning to a result. Only those actually
# installed are recorded; the rest are silently skipped.
_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "torch",
    "interpret",
    "tabpfn",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_rev() -> str | None:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return rev.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return bool(status.stdout.strip())


def _package_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def _describe_input(path: Path) -> dict:
    p = Path(path)
    info: dict = {"path": str(p)}
    try:
        st = p.stat()
    except OSError:
        info["exists"] = False
        return info
    info["exists"] = True
    info["bytes"] = int(st.st_size)
    info["mtime_utc"] = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
        timespec="seconds"
    )
    return info


def capture(input_paths: Iterable[str | Path] = ()) -> dict:
    """Return a provenance record for the current run.

    Parameters
    ----------
    input_paths:
        Files this run reads (feature caches, split tables, ...). Each is
        recorded with its size and mtime so a stale rerun is detectable. A path
        that does not exist is recorded with ``exists=False`` rather than
        dropped, so the absence is itself part of the record.

    The returned dict is JSON-serialisable and safe to embed under a
    ``"provenance"`` key in any result payload.
    """
    return {
        "timestamp_utc": _utcnow(),
        "git_rev": _git_rev(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "packages": _package_versions(),
        "inputs": [_describe_input(Path(p)) for p in input_paths],
    }
