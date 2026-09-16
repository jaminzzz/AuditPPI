#!/usr/bin/env python3
"""FlashPPI zero-training retrieval baseline over any PPI family.

FlashPPI reframes PPI as dense retrieval: each protein is encoded to asymmetric
Query/Key CLIP vectors and the interaction score is ``Q(A)·K(B)``. This is the
*published checkpoint used as a zero-training scorer* -- there is no head to
train, so unlike the other three baselines there is no ``--pair-mode`` and no
train/val split. Because PPI pairs are undirected, every pair is scored in both
orders and symmetrised::

    clip = 0.5 * ( q(A)·k(B) + q(B)·k(A) )

Feature source: ``data/sae/baseline_features/flashppi/{family}.pt`` (one per
family; PRING per-species), an ``auditppi_protein_features_v1`` cache holding
``flashppi_query`` / ``flashppi_key`` for every endpoint sequence, keyed by
sequence. Evaluation follows the audit protocol's family -> eval expansion and
reports AUROC/AUPRC via ``safe_auroc``/``safe_auprc`` on each eval benchmark.

Results land under ``results/main/baselines/flashppi/{family}/seed_{S}/``
(seed is a label only -- the scorer is deterministic).

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/baseline/runs/eval_flashppi_baseline.py --family c3
    for f in c1 c2 c3 cross_species bernett pring; do
        $PY scripts/baseline/runs/eval_flashppi_baseline.py --family $f; done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from conf.model import DEFAULT_SEED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    from scripts.baseline.runs._protocol import FAMILIES

    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--normalize", action="store_true",
                   help="L2-normalize Q/K before the dot product (cosine CLIP)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="label only; the retrieval scorer is deterministic")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    from scripts.baseline.runs._protocol import (
        family_evals, make_protein_loader, run_scoring_baseline,
    )

    # Two loaders over the SAME per-protein cache: one gathers query vectors for
    # both endpoints, one gathers key vectors. gather_pair_endpoints returns
    # (A, B, y, kept) with a stable kept order, so q_a/k_b line up row-for-row.
    q_loader = make_protein_loader("flashppi", ["flashppi_query"], suffix="")
    k_loader = make_protein_loader("flashppi", ["flashppi_key"], suffix="")

    def maybe_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=1) if args.normalize else x

    def score_eval(name: str):
        qb = q_loader(name)   # qb.A = q(A), qb.B = q(B)
        kb = k_loader(name)   # kb.A = k(A), kb.B = k(B)
        qA, qB = maybe_norm(qb.A), maybe_norm(qb.B)
        kA, kB = maybe_norm(kb.A), maybe_norm(kb.B)
        clip_ab = (qA * kB).sum(dim=1)
        clip_ba = (qB * kA).sum(dim=1)
        scores = (0.5 * (clip_ab + clip_ba)).cpu().numpy().astype(np.float64)
        return scores, qb.y, qb.n_total, qb.n_scored

    print(f"[plan] baseline=flashppi family={args.family} "
          f"normalize={args.normalize} evals={family_evals(args.family)}", flush=True)
    run_scoring_baseline(
        baseline="flashppi",
        family=args.family,
        score_eval=score_eval,
        feat_tag="flashppi_glm2_clip",
        seed=args.seed,
        hparams={
            "backbone": "gLM2-650M", "clip_dim": 1024,
            "score": "0.5*(q(A)·k(B) + q(B)·k(A))",
            "normalize": args.normalize, "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
