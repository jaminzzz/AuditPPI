"""Uniform envelope for audit result files.

Every audit writes a bespoke ``result`` dict (the EBM pair audit carries
``top_k``/``outputs``; the MLP carries ``best_epoch``/``dim``; protein audits
carry their own splits). Those payloads are frozen manuscript numbers and this
module does **not** reshape them. What it does add is a thin, common *spine* so
that heterogeneous outputs can be discovered and compared in batch:

    task · dataset · features · split · model · seed · metrics ·
    hyperparameters · artifacts · provenance

`ExperimentRecord` is that spine. ``metrics`` and ``hyperparameters`` are free
dicts — the payload stays whatever each audit already emits. The only invariant
the envelope enforces is ``schema_version`` plus the presence of the spine keys,
so a reader can trust the outer shape without knowing the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class ExperimentRecord:
    """The common spine shared by every audit result.

    Model-specific fields (formula, alpha_summary, best_epoch, ...) do not live
    here; they belong in ``metrics``/``hyperparameters`` or alongside the record
    at the payload level. This keeps the envelope stable while the payload stays
    free.
    """

    task: str
    dataset: str
    features: str
    split: str
    model: str
    seed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict with ``schema_version`` prepended."""
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def dump(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the record to ``path`` as JSON and return the path.

        Matches the ``json.dumps(..., indent=2)`` convention already used across
        the audit scripts.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent))
        return out
