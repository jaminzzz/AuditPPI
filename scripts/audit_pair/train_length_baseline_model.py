#!/usr/bin/env python3
"""Train-once XGBoost baselines for the test-length robustness sweep.

The length experiment fixes the *training* side and varies only how much of each
*test* sequence ESM-C is allowed to see (``max_residues`` truncation). This
script produces the fixed side: for one family it fits the canonical ``sym``
XGBoost pair classifier on the existing ``max1022`` train (+ official val)
endpoints -- exactly the fingerprint-baseline protocol (``sae_max`` / ESM-C L60
/ ``sym``, official val for early stopping) -- and saves the booster so every
per-length test cache can be scored against the same frozen model.

Train and val are **never truncated**: they are read straight from the family's
existing ``auditppi_protein_features_v1`` protein cache (``max1022`` budget) via
the shared ``_assemble`` path, so the training features here are byte-identical
to the main C3/bernett ``xgb`` baseline cells.

Supported families (``--family``):

  c1, c2, c3, bernett  -- native train/val via ``{family}:train`` / ``{family}:val``
  pring:human:{METHOD} -- pring human graph of that sampling method; trains on
                          ``pring:human:train:{METHOD}`` / val on
                          ``pring:human:val:{METHOD}`` (METHOD ∈ BFS/DFS/RANDOM_WALK).
                          Three methods = three independent boosters (their pair
                          topologies differ), all scored against the SAME shared
                          test-length pool by run_length_robustness_sweep.py.
  pring:{species}      -- pring cross-species (yeast/ecoli/arath); trains on the
                          human BFS graph (PRING_DEFAULT_METHOD) and is scored
                          zero-shot on the species test graph.

Products (per family)::

    results/audit_pair/length_robustness/{family}/model_sae_max_esmcL60_sym.ubj
    results/audit_pair/length_robustness/{family}/model_sae_max_esmcL60_sym.meta.json

``{family}`` is the ``--family`` argument with ``:`` replaced by ``_`` (so
``pring:human:BFS`` -> ``pring_human_BFS``).

The meta sidecar records the protocol, the full-length (max1022) train/val
positive rates and pair counts, and the val AUROC at fit time -- the natural
reference point the per-length test curve is read against.

Run (CPU-only is fine; XGB will use CUDA if free and fall back automatically)::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/train_length_baseline_model.py --family c3
    $PY scripts/audit_pair/train_length_baseline_model.py --family bernett
    $PY scripts/audit_pair/train_length_baseline_model.py --family pring:human:BFS
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.model import DEFAULT_BACKBONE, DEFAULT_SEED, resolve_backbone_layer
from conf.paths import RESULTS_PAIR
from src.eval.classification import probe_classification_metrics
from src.features.pairs import sym_features
from src.interp.pair_probe import (
    rank_booster_by_shap,
    sae_dim_for_backbone,
    write_feature_ranking,
)
from src.models.estimators.xgboost import fit_xgb
from src.ppi_fingerprint.baseline import _assemble
from src.ppi_fingerprint.config import family_of, PRING_DEFAULT_METHOD

REP = "sae_max"
PAIR_MODE = "sym"


def _train_val_names(family_arg: str) -> tuple[str, str]:
    """Resolve a ``--family`` argument to (train_name, val_name) benchmark keys.

    c1/c2/c3/bernett  -> ``{family}:train`` / ``{family}:val``
    pring:human:{M}   -> ``pring:human:train:{M}`` / ``pring:human:val:{M}``
                        (the method is taken from the arg, so each of BFS/DFS/
                        RANDOM_WALK gets its OWN booster -- their pair topologies
                        differ; the test side still shares one pool)
    pring:{species}   -> ``pring:human:train:{PRING_DEFAULT_METHOD}`` /
                        ``pring:human:val:{PRING_DEFAULT_METHOD}`` (zero-shot
                        cross-species: train on human BFS, test on the species)
    """
    family = family_of(family_arg)
    if family == "pring":
        parts = family_arg.split(":")
        species = parts[1] if len(parts) > 1 and parts[1] else "human"
        if species != "human":
            method = PRING_DEFAULT_METHOD
        else:
            method = parts[2] if len(parts) > 2 and parts[2] else PRING_DEFAULT_METHOD
        return (f"pring:human:train:{method}", f"pring:human:val:{method}")
    return (f"{family}:train", f"{family}:val")


def _family_slug(family_arg: str) -> str:
    """Path-safe directory name for a family argument (``pring:human:BFS`` -> ``pring_human_BFS``)."""
    return family_arg.replace(":", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--family", default="c3",
        help=("family key: c1 | c2 | c3 | bernett | pring:human:{BFS|DFS|RANDOM_WALK} "
              "| pring:{yeast|ecoli|arath}"))
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="SAE layer; defaults to the backbone's default layer (ESM-C L60).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-root", type=Path, default=RESULTS_PAIR / "length_robustness")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layer = resolve_backbone_layer(args.backbone, args.layer)
    sae_dim = sae_dim_for_backbone(args.backbone)
    family_slug = _family_slug(args.family)
    out_dir = args.out_root / family_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model_{REP}_esmcL{layer}_{PAIR_MODE}.ubj"
    meta_path = model_path.with_suffix(".meta.json")
    ranking_csv = out_dir / f"feature_ranking_{REP}_sym.csv"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"model exists: {model_path}; pass --overwrite to replace it")

    train_name, val_name = _train_val_names(args.family)

    print(f"[train] assembling {train_name} / {val_name} "
          f"({REP}, {args.backbone} L{layer}, {PAIR_MODE})", flush=True)
    _, Atr, Btr, ytr, _ = _assemble(train_name, REP, backbone=args.backbone, layer=layer)
    _, Ava, Bva, yva, _ = _assemble(val_name, REP, backbone=args.backbone, layer=layer)

    Xtr = sym_features(Atr, Btr)
    Xva = sym_features(Ava, Bva)
    print(f"[train] Xtr={Xtr.shape} pos={float(ytr.mean()):.4f} | "
          f"Xva={Xva.shape} pos={float(yva.mean()):.4f}", flush=True)

    clf = fit_xgb(Xtr, ytr, Xva, yva, seed=args.seed)
    clf.save_model(str(model_path))
    print(f"[saved] {model_path}", flush=True)

    # Val AUROC at fit time -- the frozen model's in-domain reference point.
    val_prob = clf.predict_proba(Xva)[:, 1]
    val_metrics = probe_classification_metrics(yva, val_prob)

    # Rank the FROZEN booster's sym columns by mean|TreeSHAP| on val -- the
    # anchor set for the downstream "does length-L extraction still cover the
    # model's important features?" analysis. Derived from the same booster we
    # saved (no re-fit), so the ranking corresponds exactly to the frozen model.
    print(f"[rank] TreeSHAP over frozen booster on {val_name} (sae_dim={sae_dim})", flush=True)
    ranking_rows = rank_booster_by_shap(clf.get_booster(), Xva, sae_dim=sae_dim)
    write_feature_ranking(ranking_csv, ranking_rows, f"{args.family}_val")
    print(f"[saved] {ranking_csv} ({len(ranking_rows)} features)", flush=True)

    meta = {
        "family": args.family,
        "rep": REP,
        "pair_mode": PAIR_MODE,
        "backbone": args.backbone,
        "layer": int(layer),
        "seed": int(args.seed),
        "train_name": train_name,
        "val_name": val_name,
        "n_train_pairs": int(len(ytr)),
        "n_val_pairs": int(len(yva)),
        "train_pos_rate": round(float(ytr.mean()), 6),
        "val_pos_rate": round(float(yva.mean()), 6),
        "train_feature_dim": int(Xtr.shape[1]),
        "sae_dim": int(sae_dim),
        "ranking_csv": str(ranking_csv),
        "val_metrics_at_fit": {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in val_metrics.items()},
        "note": ("train/val read from the family's existing max1022 protein cache "
                 "(never truncated); this booster is scored against per-length "
                 "test caches by run_length_robustness_sweep.py. feature_ranking "
                 "is mean|TreeSHAP| over the frozen booster on val -- the anchor "
                 "set for the length-coverage analysis."),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[saved] {meta_path}  val_auroc={val_metrics.get('auroc')}", flush=True)


if __name__ == "__main__":
    main()
