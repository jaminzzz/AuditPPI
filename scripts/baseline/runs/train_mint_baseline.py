#!/usr/bin/env python3
"""MINT joint-embedding MLP-pair baseline over any PPI family.

MINT jointly contextualizes both chains and mean-pools each into a 1280-d
embedding (``mint_embed_a`` / ``mint_embed_b``). This baseline feeds those two
endpoint vectors through the shared AuditPPI pair vocabulary + the same
``mlp_pair`` head the gated model is scored with, so the only axis versus the
audit model is the representation.

Feature source: ``data/sae/baseline_features/mint/{stem}.pt`` (one per benchmark
name, e.g. ``c3_train.pt`` / ``c3_test.pt``), an
``auditppi_baseline_pair_features_v1`` cache with inline labels row-aligned to
the full benchmark order. Evaluation follows the audit protocol exactly (family
-> eval, native train, official val, train-once-multi-eval, AUROC/AUPRC via
``safe_auroc``/``safe_auprc``). Families whose per-pair caches are not yet
extracted are skipped per-eval with a clear message.

Results land under ``results/main/baselines/mint/{family}/seed_{S}/``.

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/baseline/runs/train_mint_baseline.py --family c3
    $PY scripts/baseline/runs/train_mint_baseline.py --family c3 --pair-mode sym
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from conf.model import DEFAULT_SEED  # noqa: E402
from src.features.pairs import PAIR_MODES  # noqa: E402
from src.runtime import setup_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    from scripts.baseline.runs._protocol import FAMILIES

    p.add_argument("--family", choices=FAMILIES, default="c3")
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

    loader = make_pair_loader("mint", "mint_embed_a", "mint_embed_b")

    print(f"[plan] baseline=mint family={args.family} pair_mode={args.pair_mode} "
          f"seed={args.seed}", flush=True)
    run_trainable_baseline(
        baseline="mint",
        family=args.family,
        loader=loader,
        feat_tag="mint_embed",
        pair_mode=args.pair_mode,
        seed=args.seed,
        train_subsample=args.train_subsample,
        val_frac=args.val_frac,
        device=device,
        hparams={
            "backbone": "esm2_t33_650M", "layer": 33, "embedding_dim": 1280,
            "pair_mode": args.pair_mode, "seed": args.seed,
            "train_subsample": args.train_subsample, "val_frac": args.val_frac,
        },
    )


if __name__ == "__main__":
    main()
