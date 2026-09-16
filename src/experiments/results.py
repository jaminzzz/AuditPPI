"""Uniform envelope for audit result files.

Every audit writes a bespoke ``result`` dict (the EBM pair audit carries
``top_k``/``outputs``; the MLP carries ``best_epoch``/``dim``; protein audits
carry their own splits). Those payloads are frozen manuscript numbers and this
module does **not** reshape them: the original dict is embedded *verbatim* under
``payload``. What the envelope adds is a thin, common *spine* around it so that
heterogeneous outputs can be discovered and compared in batch:

    task · dataset · features · split · model · seed · metrics ·
    hyperparameters · payload

`ExperimentRecord` is that spine. ``metrics``/``hyperparameters`` are free
dicts (optional headline scalars lifted out for easy cross-run comparison), and
``payload`` holds the audit's own result dict untouched. The only invariant the
envelope enforces is ``schema_version`` plus the presence of the spine keys, so
a reader can trust the outer shape without knowing the model.

Relationship to provenance
---------------------------
The envelope answers *what this experiment is* (its scientific coordinates + the
frozen payload). It deliberately carries no run metadata: *how* a run was
produced (git rev, environment, input-file fingerprints) is recorded separately
by the runner as a ``.prov.json`` sidecar next to the product and a line in
``results/runs.jsonl`` (see :mod:`src.experiments.history`). The two are
complementary, not competing -- so ``provenance`` is left out of the record and
scripts never populate it by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class ExperimentRecord:
    """The common spine wrapped around an audit's own result payload.

    The audit's bespoke result dict is embedded verbatim under ``payload`` and
    never reshaped. Spine fields are the scientific coordinates that make
    heterogeneous results comparable in batch; ``metrics``/``hyperparameters``
    optionally lift a few headline values out of the payload for convenience.
    """

    task: str
    dataset: str
    features: str
    split: str
    model: str
    seed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

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


def dump_experiment(
    path: str | Path,
    *,
    task: str,
    dataset: str,
    features: str,
    split: str,
    model: str,
    seed: int,
    payload: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Wrap an audit's own ``payload`` dict in the spine and write it to ``path``.

    This is the one-liner the audit scripts call in place of
    ``path.write_text(json.dumps(result, indent=2))``. The ``result`` dict goes
    in verbatim as ``payload``; the spine keyword args supply the scientific
    coordinates. ``metrics``/``hyperparameters`` are optional headline lifts
    (leave them out when the payload has no single-scalar headline, e.g. an
    enrichment table or a multi-cell sweep).

    Conventions used by callers (kept loose; the envelope enforces only shape):

    - ``features`` -- the representation (``sae_max`` / ``binary`` / ...), or
      ``"na"`` when the audit is representation-agnostic.
    - ``seed`` -- the run seed, or ``-1`` when the audit is deterministic / has
      no seed argument.
    - ``split`` -- the headline split the ``metrics`` refer to (usually
      ``"test"``); ``"multi"`` for a sweep that spans many cells.
    - ``model`` -- a short stable label; the audit's own richer ``model`` field
      (which may be a dict) stays inside ``payload``.
    """
    return ExperimentRecord(
        task=task,
        dataset=dataset,
        features=features,
        split=split,
        model=model,
        seed=int(seed),
        metrics=dict(metrics or {}),
        hyperparameters=dict(hyperparameters or {}),
        payload=payload,
    ).dump(path, indent=indent)
