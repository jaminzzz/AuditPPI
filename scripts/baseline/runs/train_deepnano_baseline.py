#!/usr/bin/env python3
"""DeepNano-seq frozen-embedding ensemble MLP-pair baseline over any PPI family.

Faithful to the upstream ``DeepNano_seq`` (``baselines/DeepNano/models/models.py``):
each protein is pooled three ways (mean/min/max) and **each pool gets its own
prediction head** consuming that pool's ``[A‖B]`` concat; the final score is the
mean of the three heads' probabilities (upstream trains all three jointly under an
averaged BCE loss). Features are frozen here, so the three heads share no
parameters or gradients -- averaged-loss joint training is then mathematically
equivalent to training each head independently and averaging their probabilities,
which is what :func:`run_ensemble_baseline` does. Each per-pool head is the shared
AuditPPI ``mlp_pair`` under ``pair_mode=concat`` (the AB/BA protocol), matching the
upstream per-branch ``cat((seq1, seq2))`` and the MINT/PPLM head so the only axis
versus those baselines is the representation, not head capacity.

Feature source: ``data/sae/baseline_features/deepnano/{family}_esm2.pt`` (one per
family; PRING is per-species), an ``auditppi_protein_features_v1`` cache holding
``deepnano_esm2_l33_{mean,min,max}`` for every endpoint sequence, keyed by
sequence. Evaluation follows the audit protocol exactly: family -> eval
benchmark(s), train on the family's native train, val on its official val (carved
from train for cross_species), train-once-multi-eval, AUROC/AUPRC via
``safe_auroc``/``safe_auprc``.

Results land under ``results/main/baselines/deepnano/{family}/seed_{S}/``.

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/baseline/runs/train_deepnano_baseline.py --family c3
    for f in c1 c2 c3 cross_species bernett pring; do
        $PY scripts/baseline/runs/train_deepnano_baseline.py --family $f; done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from conf.model import DEFAULT_SEED  # noqa: E402
from src.runtime import setup_device  # noqa: E402

# Upstream DeepNano-seq keeps one head per pool and ensembles the three; both the
# pool set and the per-branch [A‖B] concat are fixed by that protocol.
_POOLS = ("mean", "min", "max")
_PAIR_MODE = "concat"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    from scripts.baseline.runs._protocol import FAMILIES

    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--train-subsample", type=int, default=100000)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="val carved from train when a family has no official val")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device-id", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = setup_device(args.device_id)

    from scripts.baseline.runs._protocol import make_protein_loader, run_ensemble_baseline

    # One loader (hence one mlp_pair head) per residue pool; ensembled downstream.
    loaders = {
        pool: make_protein_loader("deepnano", [f"deepnano_esm2_l33_{pool}"], suffix="_esm2")
        for pool in _POOLS
    }
    feat_tag = "deepnano_esm2_l33_ensemble"

    print(f"[plan] baseline=deepnano family={args.family} pools={list(_POOLS)} "
          f"pair_mode={_PAIR_MODE} (per-pool head ensemble) seed={args.seed}", flush=True)
    run_ensemble_baseline(
        baseline="deepnano",
        family=args.family,
        loaders=loaders,
        feat_tag=feat_tag,
        pair_mode=_PAIR_MODE,
        seed=args.seed,
        train_subsample=args.train_subsample,
        val_frac=args.val_frac,
        device=device,
        hparams={
            "backbone": "esm2", "layer": 33, "pools": list(_POOLS),
            "ensemble": "per-pool mlp_pair heads, mean of probabilities",
            "pair_mode": _PAIR_MODE, "seed": args.seed,
            "train_subsample": args.train_subsample, "val_frac": args.val_frac,
        },
    )


if __name__ == "__main__":
    main()
