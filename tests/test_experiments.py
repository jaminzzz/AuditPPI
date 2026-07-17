"""Tests for the uniform experiment envelope (`src.experiments`).

The envelope is *additive* infrastructure: it wraps bespoke audit payloads
without reshaping them. These tests pin the outer shape (spine keys +
schema_version) and the best-effort, never-raising contract of provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.experiments import SCHEMA_VERSION, ExperimentRecord, capture


def _record(**overrides) -> ExperimentRecord:
    base = dict(
        task="pair_ppi",
        dataset="rappid_c3",
        features="sae_l60",
        split="c3",
        model="endpoint_additive_ebm",
        seed=0,
    )
    base.update(overrides)
    return ExperimentRecord(**base)


def test_spine_keys_present_and_ordered() -> None:
    d = _record().to_dict()
    assert list(d.keys()) == [
        "schema_version",
        "task",
        "dataset",
        "features",
        "split",
        "model",
        "seed",
        "metrics",
        "hyperparameters",
        "artifacts",
        "provenance",
    ]
    assert d["schema_version"] == SCHEMA_VERSION


def test_free_dicts_are_passed_through_verbatim() -> None:
    # The envelope must not touch payload contents — a bespoke metrics/hparam
    # dict comes back byte-identical.
    metrics = {"test": {"auroc": 0.5, "auprc": 0.5}, "best_epoch": 7}
    hparams = {"hidden": 128, "layers": 3, "dropout": 0.1}
    d = _record(metrics=metrics, hyperparameters=hparams).to_dict()
    assert d["metrics"] == metrics
    assert d["hyperparameters"] == hparams


def test_defaults_are_empty_dicts_not_shared() -> None:
    a, b = _record(), _record()
    a.metrics["x"] = 1
    assert b.metrics == {}  # no shared mutable default


def test_dump_writes_indent2_json_and_roundtrips(tmp_path: Path) -> None:
    rec = _record(metrics={"test": {"auroc": 0.5}})
    out = rec.dump(tmp_path / "nested" / "r.json")
    assert out.exists()
    text = out.read_text()
    assert "\n  " in text  # indent=2
    assert json.loads(text) == rec.to_dict()


def test_capture_is_json_serialisable_and_has_spine() -> None:
    prov = capture()
    json.dumps(prov)  # must not raise
    for key in ("timestamp_utc", "python", "hostname", "packages", "inputs"):
        assert key in prov
    # git_rev/git_dirty are best-effort: present as key, value may be None.
    assert "git_rev" in prov
    assert "git_dirty" in prov


def test_capture_records_existing_and_missing_inputs(tmp_path: Path) -> None:
    real = tmp_path / "data.npz"
    real.write_bytes(b"0123456789")
    missing = tmp_path / "gone.npz"

    prov = capture([real, missing])
    by_path = {i["path"]: i for i in prov["inputs"]}

    assert by_path[str(real)]["exists"] is True
    assert by_path[str(real)]["bytes"] == 10
    assert "mtime_utc" in by_path[str(real)]

    assert by_path[str(missing)]["exists"] is False
    assert "bytes" not in by_path[str(missing)]
