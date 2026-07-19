"""Uniform experiment envelope for audit outputs.

Audits emit bespoke result payloads (frozen manuscript numbers). This package
adds a thin, common *spine* around them so heterogeneous outputs can be
discovered and compared in batch, without reshaping the payloads:

  - `results`  — :class:`ExperimentRecord` + :func:`dump_experiment`, the
                 common spine that wraps each audit's payload + JSON dump.
  - `metadata` — :func:`capture`, best-effort run metadata (code version,
                 environment, input file metadata).
"""

from src.experiments.history import (
    RUN_HISTORY,
    append_run,
    run_record,
    new_run_id,
    sidecar_path,
    write_sidecar,
)
from src.experiments.metadata import capture
from src.experiments.results import SCHEMA_VERSION, ExperimentRecord, dump_experiment

__all__ = [
    "ExperimentRecord",
    "dump_experiment",
    "SCHEMA_VERSION",
    "capture",
    "RUN_HISTORY",
    "append_run",
    "run_record",
    "new_run_id",
    "sidecar_path",
    "write_sidecar",
]
