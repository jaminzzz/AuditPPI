"""Uniform experiment envelope for audit outputs.

Audits emit bespoke result payloads (frozen manuscript numbers). This package
adds a thin, common *spine* around them so heterogeneous outputs can be
discovered and compared in batch, without reshaping the payloads:

  - `results`     — :class:`ExperimentRecord`, the common spine + JSON dump.
  - `provenance`  — :func:`capture`, best-effort run provenance (code version,
                    environment, input file metadata).
"""

from src.experiments.provenance import capture
from src.experiments.results import SCHEMA_VERSION, ExperimentRecord

__all__ = ["ExperimentRecord", "SCHEMA_VERSION", "capture"]
