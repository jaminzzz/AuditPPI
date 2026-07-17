#!/usr/bin/env python3
"""Run the pooled-SAE fingerprint + classifier baseline and report AUROC/AUPRC.

    PY=/data/wmzhu/anaconda3/envs/genmol/bin/python
    $PY scripts/run_ppi_fingerprint_baseline.py --model xgb --rep binary --eval c3:test
    $PY scripts/run_ppi_fingerprint_baseline.py --model tabpfn --rep sae_max --eval rf2ppi
    $PY scripts/run_ppi_fingerprint_baseline.py --all                 # 3 models × 3 reps × {c3,cs,rf2ppi}
    $PY scripts/run_ppi_fingerprint_baseline.py --all --species       # also each cross-species test set

Per-benchmark native training (c3→c3:train, cross_species→human_train, rf2ppi→c3:train).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ppi_fingerprint.config import MODEL_NAMES as MODELS  # noqa: E402
from src.ppi_fingerprint.config import REPRESENTATIONS as REPS  # noqa: E402

HEADLINE_EVALS = ("c3:test", "cross_species:human_test", "rf2ppi")
SPECIES_EVALS = tuple(f"cross_species:{s}" for s in ("ecoli", "fly", "mouse", "worm", "yeast"))


def pick_free_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]).decode()
        return sorted(((int(l.split(",")[1]), l.split(",")[0].strip())
                       for l in out.strip().splitlines()), reverse=True)[0][1]
    except Exception:  # noqa: BLE001
        return "0"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="xgb")
    p.add_argument("--rep", choices=REPS, default="binary")
    p.add_argument("--eval", default="c3:test", help="c3:test | cross_species:<sp> | rf2ppi")
    p.add_argument("--top-k", type=int, default=500, help="TabPFN feature cap")
    p.add_argument("--train-subsample", type=int, default=100000)
    p.add_argument("--all", action="store_true", help="sweep MODELS × REPS × headline evals")
    p.add_argument("--species", action="store_true", help="add the 5 cross-species test sets to --all")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dev = str(args.device_id) if args.device_id is not None else pick_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    print(f"[device] CUDA_VISIBLE_DEVICES={dev}", flush=True)

    from src.ppi_fingerprint import OUT_DIR, run_baseline  # after env pin

    if not args.all:
        run_baseline(args.model, args.rep, args.eval, top_k=args.top_k,
                     train_subsample=args.train_subsample, seed=args.seed)
        return

    evals = list(HEADLINE_EVALS) + (list(SPECIES_EVALS) if args.species else [])
    summary = {}
    for model in MODELS:
        for rep in REPS:
            for ev in evals:
                key = f"{model}/{rep}/{ev}"
                try:
                    r = run_baseline(model, rep, ev, top_k=args.top_k,
                                     train_subsample=args.train_subsample, seed=args.seed)
                    summary[key] = {"auroc": r["auroc"], "auprc": r["auprc"],
                                    "n_skip": r["n_skipped_eval"]}
                except Exception as e:  # noqa: BLE001
                    print(f"[skip] {key}: {e}", flush=True)
                    summary[key] = {"error": str(e)}
    print("\n=== ppi_fingerprint baseline: AUROC / AUPRC ===", flush=True)
    for k, v in summary.items():
        print(f"  {k:42s} {v}", flush=True)
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {OUT_DIR/'baseline_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
