#!/usr/bin/env python3
"""PPLM endpoint-embedding MLP-pair baseline over any PPI family.

PPLM-PPI emits both per-chain embeddings and pair-interaction attention features.
For a fair, apples-to-apples comparison against the gated AuditPPI model (and the
MINT baseline), this script uses PPLM's **endpoint embeddings only**
(``pplm_{pool}_embed_a`` / ``pplm_{pool}_embed_b``, ``pool`` in ``mean``/``max``),
assembled through the shared AuditPPI pair vocabulary and scored with the same
``mlp_pair`` head -- so the only axis versus the audit model is the
representation, not the pair-construction machinery.

Feature source: ``data/sae/baseline_features/pplm/{stem}.pt`` (one per benchmark
name), an ``auditppi_baseline_pair_features_v1`` cache with inline labels
row-aligned to the full benchmark order. Evaluation follows the audit protocol
exactly. Families/splits whose per-pair caches are not yet extracted are skipped
per-eval with a clear message.

Results land under ``results/main/baselines/pplm/{family}/seed_{S}/``.

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/baseline/runs/train_pplm_baseline.py --family c3
    $PY scripts/baseline/runs/train_pplm_baseline.py --family c3 --pool mean --pair-mode concat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from conf.model import DEFAULT_SEED  # noqa: E402
from src.features.pairs import PAIR_MODES  # noqa: E402
from src.runtime import setup_device  # noqa: E402

_POOLS = ("mean", "max")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    from scripts.baseline.runs._protocol import FAMILIES

    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--pool", choices=_POOLS, default="mean",
                   help="which PPLM endpoint-embedding pooling to use")
    p.add_argument("--pair-mode", choices=PAIR_MODES, default="sym")
    p.add_argument("--train-subsample", type=int, default=100000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device-id", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = setup_device(args.device_id)

    from scripts.baseline.runs._protocol import make_pair_loader, run_trainable_baseline

    key_a = f"pplm_{args.pool}_embed_a"
    key_b = f"pplm_{args.pool}_embed_b"
    loader = make_pair_loader("pplm", key_a, key_b)

    print(f"[plan] baseline=pplm family={args.family} pool={args.pool} "
          f"pair_mode={args.pair_mode} seed={args.seed}", flush=True)
    run_trainable_baseline(
        baseline="pplm",
        family=args.family,
        loader=loader,
        feat_tag=f"pplm_{args.pool}_embed",
        pair_mode=args.pair_mode,
        seed=args.seed,
        train_subsample=args.train_subsample,
        val_frac=args.val_frac,
        device=device,
        hparams={
            "backbone": "pplm_t33_650M", "layer": 33, "pool": args.pool,
            "embedding_dim": 1280, "pair_mode": args.pair_mode, "seed": args.seed,
            "train_subsample": args.train_subsample, "val_frac": args.val_frac,
        },
    )


if __name__ == "__main__":
    main()
