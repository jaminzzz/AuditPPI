#!/usr/bin/env python3
"""The audit runner: execute experiment matrix cells, track every run.

This is the *engine* over :mod:`src.experiments.registry`. The registry declares
*what* exists; this script decides *what is runnable right now* and *runs it*,
recording provenance for every invocation so the whole matrix stays paper-ready
and per-run-trackable.

What it does, per selected cell, in order:

  1. **Select** cells from the registry (``--all`` / ``--layer`` / ``--experiment``).
  2. **Order** them by ``(layer_rank, name)`` -- the main-trunk route
     protein -> pair -> residue, producers before consumers. Readiness probing
     over declared inputs supplies the finer producer->consumer edges, so this
     is a topological order without a hand-maintained DAG.
  3. **Probe readiness**: a cell runs only when every declared input exists on
     disk *now*. Not-ready cells are skipped with a clear reason (which inputs
     are missing) -- this is how the full dataset matrix stays *configured*
     before its caches exist, and auto-lights when a cache is dropped in.
  4. **Capture** provenance once (git rev/dirty, env, input fingerprints).
  5. **Execute** the script as a subprocess with *this* interpreter from the
     repo root, so the editable-install ``conf``/``src`` packages resolve.
  6. **Verify** the declared products exist afterwards (a script that exits 0
     but writes nothing is a failure worth surfacing).
  7. **Record**: a provenance sidecar next to a product + one JSON line in the
     append-only history (``results/runs.jsonl``). Result payloads themselves are
     never touched -- frozen manuscript numbers stay byte-for-byte.

A single cell's failure is isolated: it is logged (status=error in the history)
and the run continues. ``--all`` never halts on one bad cell.

Examples
--------
    python scripts/run_experiments.py --all
    python scripts/run_experiments.py --all --dry-run
    python scripts/run_experiments.py --layer pair --skip-existing
    python scripts/run_experiments.py --experiment pair.c3_endpoint_additive_mlp.sae_max
    python scripts/run_experiments.py --all --cpu-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from conf.paths import ROOT
from src.experiments import (
    append_run,
    capture,
    new_run_id,
    run_record,
    write_sidecar,
)
from src.experiments.registry import (
    EXPERIMENTS,
    LAYER_ORDER,
    Experiment,
    experiments_by_layer,
    get_experiment,
)


# --- selection --------------------------------------------------------------
def select(args: argparse.Namespace) -> list[Experiment]:
    """Resolve the CLI selectors into an ordered, de-duplicated cell list."""
    chosen: list[Experiment] = []
    if args.experiment:
        for name in args.experiment:
            chosen.append(get_experiment(name))  # KeyError -> clear crash
    if args.layer:
        for layer in args.layer:
            chosen += experiments_by_layer(layer)
    if args.all or not (args.experiment or args.layer):
        # --all, or no selector at all, means the whole auto matrix.
        chosen += [e for e in EXPERIMENTS if e.auto]

    # De-dup by name, preserve a single stable (layer_rank, name) order.
    seen: set[str] = set()
    unique: list[Experiment] = []
    for e in sorted(chosen, key=lambda e: (e.layer_rank, e.name)):
        if e.name not in seen:
            seen.add(e.name)
            unique.append(e)
    return unique


# --- one cell ---------------------------------------------------------------
def run_one(exp: Experiment, *, cpu_only: bool) -> dict:
    """Execute one cell as a subprocess; return a history record dict."""
    run_id = new_run_id()
    prov = capture(exp.inputs)
    argv = [sys.executable, exp.script, *exp.args]

    env = dict(os.environ)
    if cpu_only:
        # Hide every GPU so torch/xgboost fall back to CPU. Additive: scripts
        # that never touch a GPU are unaffected.
        env["CUDA_VISIBLE_DEVICES"] = ""

    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=ROOT, env=env)
        rc = proc.returncode
        run_error: str | None = None if rc == 0 else f"exit code {rc}"
    except (OSError, KeyboardInterrupt) as exc:
        rc = -1
        run_error = f"{type(exc).__name__}: {exc}"
    duration = time.monotonic() - t0

    # Post-run product verification: a clean exit that produced nothing is a
    # silent failure worth flagging.
    existing = exp.existing_products()
    if run_error is None and exp.products and not existing:
        run_error = "no declared products found after run"

    status = "ok" if run_error is None else "error"
    products = [str(p) for p in existing] if existing else None

    # Sidecar next to the first existing product (best-effort; never fatal).
    if existing:
        write_sidecar(
            existing[0],
            prov,
            extra={"run_id": run_id, "experiment": exp.name, "layer": exp.layer},
        )

    rec = run_record(
        run_id=run_id,
        experiment=exp.name,
        layer=exp.layer,
        status=status,
        duration_s=duration,
        git_rev=prov.get("git_rev"),
        git_dirty=prov.get("git_dirty"),
        argv=argv,
        products=products,
        error=run_error,
    )
    append_run(rec)
    return rec


# --- driver -----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run audit experiment matrix cells and track every run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sel = ap.add_argument_group("selection (default: whole auto matrix)")
    sel.add_argument("--all", action="store_true", help="Run the whole auto matrix.")
    sel.add_argument(
        "--layer",
        action="append",
        choices=LAYER_ORDER,
        help="Run all cells in this layer (repeatable).",
    )
    sel.add_argument(
        "--experiment",
        action="append",
        metavar="NAME",
        help="Run one cell by registry name (repeatable).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cells whose declared products already exist on disk.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected/ready cells and what would run; execute nothing.",
    )
    ap.add_argument(
        "--cpu-only",
        action="store_true",
        help="Hide GPUs (CUDA_VISIBLE_DEVICES='') so runs fall back to CPU.",
    )
    args = ap.parse_args()

    try:
        selected = select(args)
    except KeyError as exc:
        ap.error(f"unknown experiment name: {exc}")

    if not selected:
        print("No experiments selected.")
        return 0

    print(f"Selected {len(selected)} cell(s):\n")
    ran = skipped = failed = 0
    for exp in selected:
        missing = exp.missing_inputs()
        existing = exp.existing_products()

        if missing:
            print(f"  [skip:not-ready] {exp.name}")
            for m in missing:
                print(f"      missing input: {m}")
            skipped += 1
            continue

        if args.skip_existing and existing:
            print(f"  [skip:existing]  {exp.name}  ({len(existing)} product(s) present)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [would-run]      {exp.name}  ->  {exp.script} {' '.join(exp.args)}")
            ran += 1
            continue

        print(f"  [run]            {exp.name}  ->  {exp.script} {' '.join(exp.args)}")
        rec = run_one(exp, cpu_only=args.cpu_only)
        if rec["status"] == "ok":
            print(f"      ok  ({rec['duration_s']}s)")
            ran += 1
        else:
            print(f"      FAILED  ({rec['duration_s']}s): {rec.get('error')}")
            failed += 1

    verb = "would run" if args.dry_run else "ran"
    print(f"\nDone. {verb}={ran}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
