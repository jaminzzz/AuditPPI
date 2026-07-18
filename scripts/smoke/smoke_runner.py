#!/usr/bin/env python3
"""CPU smoke test for the audit runner + experiment registry + history spine.

Fast, GPU-free, side-effect-isolated. It proves the *plumbing* is intact --
that the matrix is coherent, that the runner selects/orders/probes correctly,
and that the run-tracking chain (subprocess -> product verify -> sidecar ->
history) works end to end -- **without** running any real (expensive) audit or
touching the paper's frozen results / real history.

Run it directly with the project interpreter (the editable-install ``conf``/
``src`` packages must resolve):

    python scripts/smoke/smoke_runner.py

Exit code 0 = all checks pass; non-zero = first failure is printed. This is a
plumbing check, not a science check: it deliberately never executes a real
matrix cell (those need caches + GPUs). It runs the runner only in --dry-run,
and exercises the execution/tracking machinery against a throwaway synthetic
cell in a temp dir.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from conf.paths import ROOT
from src.experiments import (
    SCHEMA_VERSION,
    append_run,
    capture,
    dump_experiment,
    run_record,
    new_run_id,
    sidecar_path,
    write_sidecar,
)
from src.experiments.registry import EXPERIMENTS, LAYER_ORDER, all_experiments

RUNNER = ROOT / "scripts" / "run_experiments.py"


class SmokeError(AssertionError):
    """A failed smoke check (kept distinct so the driver can label it)."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SmokeError(msg)


# --- 1. registry coherence --------------------------------------------------
def check_registry() -> str:
    exps = all_experiments()  # re-derive (also re-runs the uniqueness guard)
    _check(bool(exps), "registry is empty")
    _check(len(exps) == len(EXPERIMENTS), "all_experiments() disagrees with EXPERIMENTS")

    names = [e.name for e in exps]
    _check(len(names) == len(set(names)), "duplicate experiment names in registry")

    for e in exps:
        _check(e.layer in LAYER_ORDER, f"{e.name}: unknown layer {e.layer!r}")
        script = ROOT / e.script
        _check(script.exists(), f"{e.name}: script not found: {e.script}")
        # readiness API must be internally consistent.
        _check(e.is_ready() == (not e.missing_inputs()),
               f"{e.name}: is_ready() disagrees with missing_inputs()")
        for p in e.missing_inputs():
            _check(p in e.inputs, f"{e.name}: missing_inputs() returned a non-input path")

    # Order is (layer_rank, name), producers before consumers.
    keys = [(e.layer_rank, e.name) for e in exps]
    _check(keys == sorted(keys), "EXPERIMENTS is not in (layer_rank, name) order")

    ready = sum(1 for e in exps if e.is_ready())
    return f"{len(exps)} cells, {ready} ready, {len(exps) - ready} gated; names unique, scripts present, order OK"


# --- 2. runner dry-run (selection / ordering / readiness probe path) --------
def check_runner_dry_run() -> str:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--all", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True,
    )
    _check(proc.returncode == 0, f"runner --dry-run exited {proc.returncode}\n{proc.stderr}")
    out = proc.stdout
    _check("Selected" in out and "Done." in out, "runner --dry-run produced no summary")
    # Every printed cell is either would-run or a clearly-labelled skip.
    _check("[would-run]" in out, "runner --dry-run selected nothing to run")
    return out.strip().splitlines()[-1]  # the "Done. would run=.. skipped=.. failed=.." line


# --- 3. tracking chain: subprocess resolves conf/src, capture/sidecar/history -
def check_tracking_chain() -> str:
    """Exercise the engine's machinery against a throwaway cell in a temp dir.

    Isolated: writes its product, sidecar, and history line under a TemporaryDirectory,
    so the real results tree and results/runs.jsonl are never touched.
    """
    with tempfile.TemporaryDirectory(prefix="auditppi-smoke-") as tmp:
        tmpdir = Path(tmp)
        product = tmpdir / "smoke_product.json"
        script = tmpdir / "smoke_cell.py"
        history = tmpdir / "runs.jsonl"

        # A synthetic 'cell' that (a) proves conf/src import inside a subprocess
        # launched from ROOT -- i.e. the editable install resolves the same way
        # the real runner relies on -- and (b) writes a declared product.
        script.write_text(textwrap.dedent(f"""
            import json, sys
            from conf.paths import ROOT          # editable-install import must work
            from src.experiments import capture  # (both packages)
            json.dump({{"ok": True, "root": str(ROOT)}}, open({str(product)!r}, "w"))
            sys.exit(0)
        """))

        run_id = new_run_id()
        prov = capture([script])
        _check({"git_rev", "git_dirty", "python", "inputs"} <= set(prov),
               "capture() missing expected keys")

        argv = [sys.executable, str(script)]
        t0 = time.monotonic()
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        duration = time.monotonic() - t0
        _check(proc.returncode == 0,
               f"synthetic cell failed (conf/src import from subprocess?):\n{proc.stderr}")
        _check(product.exists(), "synthetic cell produced no product")

        # sidecar next to the product
        side = write_sidecar(product, prov,
                             extra={"run_id": run_id, "experiment": "smoke.cell", "layer": "smoke"})
        _check(side is not None and Path(side).exists(), "sidecar not written")
        _check(Path(side) == sidecar_path(product), "sidecar path mismatch")
        sc = json.loads(Path(side).read_text())
        _check(sc.get("run_id") == run_id and sc.get("experiment") == "smoke.cell",
               "sidecar missing run_id/experiment extras")

        # history line (isolated temp path)
        rec = run_record(
            run_id=run_id, experiment="smoke.cell", layer="smoke", status="ok",
            duration_s=duration, git_rev=prov.get("git_rev"), git_dirty=prov.get("git_dirty"),
            argv=argv, products=[str(product)],
        )
        _check(append_run(rec, path=history), "append_run returned False")
        lines = history.read_text().splitlines()
        _check(len(lines) == 1, f"expected 1 history line, got {len(lines)}")
        back = json.loads(lines[0])
        _check(back["run_id"] == run_id and back["status"] == "ok" and back["experiment"] == "smoke.cell",
               "round-tripped history record mismatch")

    return f"subprocess import OK, product+sidecar+history round-trip OK ({duration:.2f}s cell)"


# --- 4. experiment envelope: spine + verbatim payload -------------------------
def check_dump_experiment() -> str:
    """Prove dump_experiment wraps a payload without reshaping it."""
    with tempfile.TemporaryDirectory(prefix="auditppi-smoke-env-") as tmp:
        path = Path(tmp) / "metrics.json"
        payload = {
            "model": "endpoint_additive_mlp_sae_no_global_bias",
            "rep": "sae_max",
            "seed": 7,
            "test": {"auroc": 0.91, "auprc": 0.77, "brier": 0.12},
            "hyperparameters": {"hidden": 256, "layers": 1},
            "nested": {"alpha_summary": {"test_alpha_a_mean": 0.3}},
        }
        out = dump_experiment(
            path,
            task="pair.endpoint_additive_mlp",
            dataset="c3",
            features="sae_max",
            split="test",
            model="endpoint_additive_mlp",
            seed=7,
            payload=payload,
            metrics=payload["test"],
            hyperparameters=payload["hyperparameters"],
        )
        _check(out == path and path.exists(), "dump_experiment did not write the target path")
        rec = json.loads(path.read_text())

        required = {
            "schema_version",
            "task",
            "dataset",
            "features",
            "split",
            "model",
            "seed",
            "metrics",
            "hyperparameters",
            "payload",
        }
        _check(required <= set(rec), f"envelope missing keys: {required - set(rec)}")
        _check(rec["schema_version"] == SCHEMA_VERSION, "schema_version mismatch")
        _check(rec["task"] == "pair.endpoint_additive_mlp", "task not preserved")
        _check(rec["dataset"] == "c3", "dataset not preserved")
        _check(rec["features"] == "sae_max", "features not preserved")
        _check(rec["split"] == "test", "split not preserved")
        _check(rec["model"] == "endpoint_additive_mlp", "model not preserved")
        _check(rec["seed"] == 7, "seed not preserved")
        # Payload must be byte-for-byte the original dict (no reshaping).
        _check(rec["payload"] == payload, "payload was reshaped or mutated")
        _check(rec["metrics"] == payload["test"], "headline metrics not lifted")
        _check(rec["hyperparameters"] == payload["hyperparameters"], "hyperparameters not lifted")
        # No provenance field: run metadata lives in the sidecar/history, not here.
        _check("provenance" not in rec, "envelope unexpectedly contains provenance")
        # No artifacts field either (removed; outputs stay inside payload).
        _check("artifacts" not in rec, "envelope unexpectedly contains artifacts")

    return "spine keys present, payload byte-identical, no provenance leakage"


CHECKS = (
    ("registry coherence", check_registry),
    ("runner --dry-run", check_runner_dry_run),
    ("tracking chain", check_tracking_chain),
    ("dump_experiment envelope", check_dump_experiment),
)


def main() -> int:
    print("AuditPPI runner smoke test (CPU, no real cells)\n")
    failures = 0
    for label, fn in CHECKS:
        try:
            detail = fn()
        except SmokeError as exc:
            print(f"  [FAIL] {label}: {exc}")
            failures += 1
        except Exception as exc:  # unexpected -> still isolate & report
            print(f"  [ERROR] {label}: {type(exc).__name__}: {exc}")
            failures += 1
        else:
            print(f"  [pass] {label}: {detail}")

    print()
    if failures:
        print(f"SMOKE FAILED: {failures}/{len(CHECKS)} check(s) failed.")
        return 1
    print(f"SMOKE OK: {len(CHECKS)}/{len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
