#!/usr/bin/env python3
"""Aggregate baseline downstream results into a mean±std comparison table.

Scans ``results/main/baselines/{baseline}/{family}/seed_{S}/summaries/*.json``,
groups each metric by (baseline, eval_split) across seeds, and prints AUROC /
AUPRC as ``mean±std`` (n = number of seeds contributing). One row per eval
split, one column block per baseline, so the four LM baselines are directly
comparable on every benchmark.

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/baseline/runs/aggregate_baseline_results.py
    $PY scripts/baseline/runs/aggregate_baseline_results.py --metric auprc --csv out.csv
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASELINES = ("deepnano", "mint", "pplm", "flashppi")
ROOT = Path("results/main/baselines")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--metric", choices=("auroc", "auprc"), default="auroc")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--csv", default=None, help="also write the table to this CSV path")
    return p.parse_args()


def collect(root: Path, metric: str):
    """-> {eval_split: {baseline: [values across seeds]}}, ordered eval list."""
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    eval_order: list[str] = []
    for baseline in BASELINES:
        # Standard per-run summaries only; skip variant dirs like flashppi _contact.
        for sf in sorted((root / baseline).glob("*/seed_*/summaries/*.json")):
            if "_contact" in str(sf):
                continue
            data = json.loads(sf.read_text())
            for key, m in data.get("metrics", {}).items():
                # key = model/features/pair_mode/seedS/<eval_split>; eval_split may hold ':'
                eval_split = key.split("/", 4)[-1]
                if metric not in m:
                    continue
                if eval_split not in table:
                    eval_order.append(eval_split)
                table[eval_split][baseline].append(float(m[metric]))
    return table, eval_order


def fmt(vals: list[float]) -> str:
    if not vals:
        return "     -     "
    if len(vals) == 1:
        return f"{vals[0]:.4f}    "
    return f"{statistics.mean(vals):.4f}±{statistics.pstdev(vals):.4f}"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    table, eval_order = collect(root, args.metric)

    w_eval = max((len(e) for e in eval_order), default=10) + 1
    header = f"{'eval_split':<{w_eval}}" + "".join(f"  {b:<15}" for b in BASELINES)
    print(f"\n=== baseline comparison ({args.metric.upper()}, mean±std across seeds) ===")
    print(header)
    print("-" * len(header))
    for ev in eval_order:
        row = f"{ev:<{w_eval}}"
        for b in BASELINES:
            row += f"  {fmt(table[ev][b]):<15}"
        print(row)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["eval_split", *[f"{b}_{args.metric}" for b in BASELINES]])
            for ev in eval_order:
                wr.writerow([ev, *[fmt(table[ev][b]).strip() for b in BASELINES]])
        print(f"\n[csv] {args.csv}")


if __name__ == "__main__":
    main()
