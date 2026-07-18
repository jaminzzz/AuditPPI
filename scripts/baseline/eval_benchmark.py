#!/usr/bin/env python3
"""Run the participation diagnostic (and optionally a model scorer) against a PPI benchmark.

The diagnostic needs **no trained model** — it quantifies how participation-prone the benchmark itself
is (DESIGN §7), the honesty meter every model number is read against.

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/eval_benchmark.py --benchmark rf2ppi
    $PY scripts/eval_benchmark.py --benchmark c3:test
    $PY scripts/eval_benchmark.py --benchmark cross_species:ecoli
    $PY scripts/eval_benchmark.py --all          # rf2ppi + c3:test + all cross-species splits

A trained AuditPPI model plugs in later via ``src.eval.evaluate_scorer(scorer, bench)`` where
``scorer(a, b) -> float`` (wrap the model to look up sequences from ``bench.seqs``).
"""

import argparse
from pathlib import Path

from conf.paths import RESULTS_MISC
from src.eval.participation import benchmark_diagnostic
from src.data.pairs import list_cross_species, load_benchmark
from src.experiments.results import dump_experiment


def run_one(name: str, out_dir: Path) -> dict:
    bench = load_benchmark(name, attach_seqs=False)  # diagnostic needs only pairs+labels
    diag = benchmark_diagnostic(bench.pairs, bench.labels)
    ts = diag["t_std"] or 0.0
    verdict = ("PARTICIPATION-PRONE → read absolute AUROC with care"
               if ts >= 0.2 else "relatively balanced → absolute AUROC trustworthy")
    print(f"\n[{bench.name}] n_pairs={diag['n_pairs']} n_proteins={diag['n_proteins']} "
          f"pos_rate={diag['pos_rate']}", flush=True)
    print(f"           t_std={diag['t_std']}  → {verdict}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{bench.name}_participation_diagnostic.json"
    payload = {"benchmark": bench.name, "diagnostic": diag}
    dump_experiment(
        out,
        task="baseline.participation_diagnostic",
        dataset=bench.name,
        features="na",
        split="na",
        model="na",
        seed=-1,
        payload=payload,
        metrics={
            "t_std": diag.get("t_std"),
            "pos_rate": diag.get("pos_rate"),
            "n_pairs": diag.get("n_pairs"),
            "n_proteins": diag.get("n_proteins"),
        },
    )
    return diag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="rf2ppi",
                   help="rf2ppi | c3[:split] | cross_species:species")
    p.add_argument("--all", action="store_true", help="run rf2ppi + c3:test + all cross-species")
    p.add_argument("--out-dir", type=Path, default=RESULTS_MISC / "eval")
    args = p.parse_args()

    names = (["rf2ppi", "c3:test"] + [f"cross_species:{s}" for s in list_cross_species()]
             if args.all else [args.benchmark])
    summary = {}
    for name in names:
        try:
            d = run_one(name, args.out_dir)
            summary[name] = {"t_std": d["t_std"],
                             "pos_rate": d["pos_rate"], "n_proteins": d["n_proteins"]}
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}", flush=True)
            summary[name] = {"error": str(e)}
    if len(names) > 1:
        print("\n=== participation summary (t(p) distribution per benchmark) ===", flush=True)
        for n, s in summary.items():
            print(f"  {n:28s} {s}", flush=True)
        dump_experiment(
            args.out_dir / "participation_summary.json",
            task="baseline.participation_diagnostic",
            dataset="multi",
            features="na",
            split="multi",
            model="na",
            seed=-1,
            payload=summary,
            metrics=summary,
        )
    print(f"\n[done] diagnostics in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
