#!/usr/bin/env python3
"""Full-dim sym feature ranking for the minimal-interpretable-fingerprint sweep.

Produces the per-representation feature ranking that the Fig 1 "minimal
interpretable fingerprint" K-sweep consumes. For one family (default C3) and one
representation (``sae_max`` or ``binary``) on ESM-C L60, it fits an XGBoost on
the full ``sym = [A*B, |A-B|]`` matrix (2 x sae_dim columns), then ranks every
column by mean|TreeSHAP| on the official val split -- identical semantics to the
TabPFN Top-K runner's ranking, but rep-tagged so the two representations never
clobber each other's CSV, and with no TabPFN probe attached.

Products (one per rep) land under a dedicated dir so nothing in tabpfn_topk/ is
touched::

    results/audit_pair/minimal_fingerprint/{family}/
        feature_ranking_{rep}_sym.csv

Cross-species special case
--------------------------
``cross_species`` has no official val CSV. Matching ``run_cross_species_tabpfn_topk.py``
(and NOT the fingerprint baseline's subsample-then-carve order):

  1. Load full ``cross_species:human_train``.
  2. Carve a stratified val of ``val_frac`` (default 0.1) with seed+1.
  3. Cap the remaining train with ``--train-subsample`` (seed).

Run both reps (GPU 0 default; ranking XGB matches the fingerprint baseline)::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py --rep sae_max
    $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py --rep binary
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.model import DEFAULT_BACKBONE, DEFAULT_SEED, resolve_backbone_layer
from conf.paths import (
    CROSS_SPECIES_PAIR_INDEX_CACHES,
    CROSS_SPECIES_SAE_CACHE,
    RESULTS_PAIR,
)
from src.runtime import setup_device
from src.features.pairs import load_protein_feature_cache, sym_features
from src.features.sampling import stratified_subsample
from src.interp.pair_probe import (
    carve_cross_species_train_val,
    compute_sym_shap_ranking,
    sae_dim_for_backbone,
    write_feature_ranking,
)

FAMILIES = ("c1", "c2", "c3", "bernett", "pring", "cross_species")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--family", choices=FAMILIES, default="c3")
    p.add_argument("--rep", choices=["sae_max", "binary"], default="sae_max")
    p.add_argument(
        "--method",
        choices=["BFS", "DFS", "RANDOM_WALK"],
        default="BFS",
        help="PRING human sampling method (ignored for non-PRING families).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="cross_species only: fraction of human_train carved as val "
             "(stratified, seed+1). Matches run_cross_species_tabpfn_topk.",
    )
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="SAE layer; defaults to the backbone's default layer.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="override product dir "
                        "(default: RESULTS_PAIR/minimal_fingerprint/{family_tag})")
    p.add_argument("--train-subsample", type=int, default=100000,
                   help="class-stratified train cap (C3 train < this, so a no-op there).")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Ranking XGB hyperparameters (match the fingerprint-baseline classifier).
    p.add_argument("--xgb-trees", type=int, default=1000)
    p.add_argument("--xgb-depth", type=int, default=4)
    p.add_argument("--xgb-lr", type=float, default=0.05)
    p.add_argument("--xgb-cpu", action="store_true")
    return p.parse_args()


def family_tag(family: str, method: str = "BFS") -> str:
    """Product-dir tag: bare family for BFS (legacy), method-suffixed otherwise."""
    if family != "pring" or method == "BFS":
        return family
    return f"pring_{method.lower()}"


def assemble_split(family: str, split: str, rep: str, *, backbone: str, layer: int, method: str = "BFS"):
    """train/val/test endpoints for a family via the fingerprint baseline path."""
    from src.ppi_fingerprint.baseline import _assemble

    # PRING's eval-name grammar is ``pring:species:split:method`` (per-species
    # caches, method-tagged splits); every other family uses ``family:split``.
    name = f"pring:human:{split}:{method}" if family == "pring" else f"{family}:{split}"
    _, A, B, y, _ = _assemble(name, rep, backbone=backbone, layer=layer)
    return A, B, y


def assemble_cross_species_train_val(
    rep: str, *, backbone: str, layer: int,
    train_subsample: int, val_frac: float, seed: int,
):
    """TabPFN-aligned cross_species protocol: carve val first, then cap train.

    Delegates to :func:`carve_cross_species_train_val`, which selects train/val
    ROW INDICES from the cheap pair-index labels FIRST and gathers only the
    selected endpoint rows at the protein cache's native dtype -- never the full
    421k-pair float32 matrix. Order (val carve seed+1, then train cap seed) and
    selection are identical to ``run_cross_species_tabpfn_topk``.
    """
    protein_cache = load_protein_feature_cache(CROSS_SPECIES_SAE_CACHE)
    return carve_cross_species_train_val(
        CROSS_SPECIES_PAIR_INDEX_CACHES["human_train"], protein_cache,
        rep=rep, backbone=backbone, layer=layer,
        train_subsample=train_subsample, val_frac=val_frac, seed=seed,
    )


def main() -> None:
    args = parse_args()
    setup_device(args.device_id)
    layer = resolve_backbone_layer(args.backbone, args.layer)
    sae_dim = sae_dim_for_backbone(args.backbone)

    tag = family_tag(args.family, args.method)
    out_dir = args.out_dir or (RESULTS_PAIR / "minimal_fingerprint" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_tag = f"{tag}_val"

    print(
        f"[plan] family={args.family} method={args.method} tag={tag} "
        f"rep={args.rep} {args.backbone}L{layer} "
        f"sae_dim={sae_dim} feature_dim={2 * sae_dim} out={out_dir}",
        flush=True,
    )

    import torch

    if args.family == "cross_species":
        train_a, train_b, train_y, val_a, val_b, val_y = assemble_cross_species_train_val(
            args.rep, backbone=args.backbone, layer=layer,
            train_subsample=args.train_subsample, val_frac=args.val_frac,
            seed=args.seed,
        )
    else:
        train_a, train_b, train_y = assemble_split(
            args.family, "train", args.rep, backbone=args.backbone, layer=layer,
            method=args.method,
        )
        sub = stratified_subsample(train_y, args.train_subsample, args.seed)
        if sub is not None:
            ti = torch.as_tensor(sub, dtype=torch.long)
            train_a = train_a.index_select(0, ti).contiguous()
            train_b = train_b.index_select(0, ti).contiguous()
            train_y = train_y[sub]
        val_a, val_b, val_y = assemble_split(
            args.family, "val", args.rep, backbone=args.backbone, layer=layer,
            method=args.method,
        )
    print(
        f"[data] train={len(train_y)} (pos={float(np.asarray(train_y).mean()):.3f}) "
        f"val={len(val_y)} (pos={float(np.asarray(val_y).mean()):.3f})",
        flush=True,
    )

    print("[rank] fitting XGBoost on full sym features + TreeSHAP on val", flush=True)
    t0 = time.time()
    full_train_x = sym_features(train_a, train_b)
    full_val_x = sym_features(val_a, val_b)
    del train_a, train_b, val_a, val_b
    gc.collect()
    ranking_rows = compute_sym_shap_ranking(
        full_train_x, np.asarray(train_y), full_val_x, np.asarray(val_y),
        sae_dim=sae_dim,
        trees=args.xgb_trees, depth=args.xgb_depth, lr=args.xgb_lr,
        seed=args.seed, cpu=args.xgb_cpu,
    )
    del full_train_x, full_val_x
    gc.collect()

    ranking_csv = out_dir / f"feature_ranking_{args.rep}_sym.csv"
    write_feature_ranking(ranking_csv, ranking_rows, split_tag)
    print(
        f"[rank] wrote {ranking_csv} ({len(ranking_rows)} features, "
        f"{time.time() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
