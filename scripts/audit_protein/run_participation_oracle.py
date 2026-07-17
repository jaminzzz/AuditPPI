#!/usr/bin/env python
"""Run the sequence→t(p) participation oracle (DESIGN §7).

Fits a per-protein regressor t̂ = f(pooled SAE fingerprint) on the C3 train+val participation rate t(p),
predicts on the disjoint C3 test proteins, and scores pairs by min(t̂_A, t̂_B). The regressor is trained
only on train+val proteins and never sees test labels, so its pair-level AUROC is an honest measure of
how far a label-free, sequence-only predictor can recover the participation structure.

Run: PYTHONPATH=. /data/wmzhu/anaconda3/envs/genmol/bin/python scripts/run_participation_oracle.py
     [--family c3|cross_species] [--rep sae_max|binary|esmc_mean] [--no-write]
"""
import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.participation.predictor import run_participation_oracle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="c3", choices=["c3", "cross_species"])
    ap.add_argument("--rep", default="sae_max", choices=["sae_max", "binary", "esmc_mean"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    run_participation_oracle(family=args.family, rep=args.rep, seed=args.seed, write=not args.no_write)


if __name__ == "__main__":
    main()
