"""Provenance sidecars + an append-only run history.

The audit scripts write *frozen* result payloads (manuscript numbers) via their
own ``json.dumps``; this module never touches those files. Instead it records
*how* each run was produced, alongside the untouched result:

  - ``write_sidecar(result_path, prov)`` -- drops a ``<stem>.prov.json`` next to a
    result file (git rev, dirty flag, package versions, input-file fingerprints).
  - ``append_run(record)`` -- appends one JSON line to ``results/runs.jsonl``,
    the durable history of *every* run (the "track each run" spine, since the
    result files themselves are overwritten in place).

Both are best-effort and side-effect-safe: a failed sidecar/history write must
never sink an experiment whose real output already landed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from conf.paths import RESULTS
from src.experiments.metadata import capture

# Append-only run history. One JSON object per line. Lives under results/ (so it
# is git-ignored like the products it tracks) but is durable across result-file
# overwrites: each run appends a line recording what was run and against which
# code, without ever rewriting the frozen result payloads.
RUN_HISTORY = RESULTS / "runs.jsonl"

# Suffix for the per-result provenance sidecar. ``foo.json`` -> ``foo.prov.json``.
SIDECAR_SUFFIX = ".prov.json"


def new_run_id() -> str:
    """A short, sortable-ish run id: ``YYYYmmddTHHMMSSZ-<6 hex>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def sidecar_path(result_path: str | Path) -> Path:
    """Return the ``.prov.json`` path that pairs with ``result_path``.

    The sidecar sits beside the result and reuses its full name (extension
    included) so ``a/b/x.json`` and ``a/b/x.tsv`` get distinct sidecars
    (``x.json.prov.json`` / ``x.tsv.prov.json``) rather than colliding.
    """
    p = Path(result_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def write_sidecar(
    result_path: str | Path,
    prov: dict[str, Any] | None = None,
    *,
    input_paths: Iterable[str | Path] = (),
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Write a provenance sidecar next to ``result_path``.

    Pass a ready ``prov`` dict (e.g. captured once before the run and reused for
    every product), or omit it to ``capture(input_paths)`` here. ``extra`` is
    merged into the top level (run_id, experiment name, ...). Returns the sidecar
    path, or ``None`` if writing failed -- provenance must not crash a run.
    """
    try:
        payload = dict(prov) if prov is not None else capture(input_paths)
        if extra:
            payload = {**payload, **extra}
        out = sidecar_path(result_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        return out
    except OSError:
        return None


def append_run(record: dict[str, Any], *, path: str | Path = RUN_HISTORY) -> bool:
    """Append one run record to the JSONL history. Returns success.

    The write is atomic per line: a single ``write`` of one ``json.dumps`` +
    newline. Concurrent runners appending to the same file may interleave whole
    lines but never corrupt one (POSIX append semantics for small writes).
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(line)
        return True
    except OSError:
        return False


def run_record(
    *,
    run_id: str,
    experiment: str,
    layer: str,
    status: str,
    duration_s: float,
    git_rev: str | None,
    git_dirty: bool | None,
    argv: list[str],
    products: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble one run-history line. Kept flat + JSON-serialisable for easy grep/pandas."""
    rec: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "layer": layer,
        "status": status,
        "duration_s": round(float(duration_s), 3),
        "git_rev": git_rev,
        "git_dirty": git_dirty,
        "argv": list(argv),
    }
    if products is not None:
        rec["products"] = products
    if metrics:
        rec["metrics"] = metrics
    if error:
        rec["error"] = error
    return rec
