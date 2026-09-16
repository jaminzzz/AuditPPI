#!/usr/bin/env python3
"""Compute + cache full TreeSHAP explanation bundles for the frozen sym boosters.

Reuses the boosters already trained for the length-robustness sweep
(``results/audit_pair/length_robustness/{family}/model_sae_max_esmcL60_sym.ubj``),
so the SHAP values explain exactly the models whose numbers are in the manuscript --
no re-fitting, no protocol drift. Each booster is explained on its own official val
split, the same split its stored ``feature_ranking_sae_max_sym.csv`` was ranked on
(verified to reproduce it to ~1e-16).

Only the columns a booster actually splits on are kept (~4k of 32768), so a cached
bundle is <100 MB instead of up to 8 GB.

Note on PRING: ``pring:{arath,ecoli,yeast}`` and ``pring:human:BFS`` share one
booster (cross-species runs train on human BFS and are scored zero-shot), verified
by md5. They therefore share one explanation; only ``pring:human:BFS`` is computed
and the species families are skipped as aliases.

Products::

    results/analysis/shap_interpretation/{family}/shap_bundle.npz
    results/analysis/shap_interpretation/{family}/top_features_annotated.tsv

Run::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/analysis/build_shap_explanations.py --family c3
    $PY scripts/analysis/build_shap_explanations.py --all
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
from conf.paths import RESULTS_ANALYSIS, RESULTS_PAIR
from src.features.pairs import sym_features
from src.interp.pair_probe import sae_dim_for_backbone
from src.interp.shap_explain import ShapBundle, compute_shap_bundle
from src.ppi_fingerprint.baseline import _assemble
from src.ppi_fingerprint.config import PRING_DEFAULT_METHOD, family_of

REP = "sae_max"
PAIR_MODE = "sym"
DEFAULT_ROW_CAP = 12000

# Families with an independently trained booster, in leakage-gradient order.
INDEPENDENT_FAMILIES = [
    "c1",
    "c2",
    "c3",
    "pring:human:BFS",
    "pring:human:DFS",
    "pring:human:RANDOM_WALK",
    "bernett",
]

# Cross-species PRING runs reuse the human-BFS booster (md5-identical) and are
# scored zero-shot, so their explanation on human val is the BFS one.
ALIAS_FAMILIES = {
    "pring:arath": "pring:human:BFS",
    "pring:ecoli": "pring:human:BFS",
    "pring:yeast": "pring:human:BFS",
}


def _train_val_names(family_arg: str) -> tuple[str, str]:
    """Mirror of train_length_baseline_model._train_val_names (same resolution)."""
    family = family_of(family_arg)
    if family == "pring":
        parts = family_arg.split(":")
        species = parts[1] if len(parts) > 1 and parts[1] else "human"
        method = (
            PRING_DEFAULT_METHOD
            if species != "human"
            else (parts[2] if len(parts) > 2 and parts[2] else PRING_DEFAULT_METHOD)
        )
        return (f"pring:human:train:{method}", f"pring:human:val:{method}")
    return (f"{family}:train", f"{family}:val")


def _slug(family_arg: str) -> str:
    return family_arg.replace(":", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--family", default=None, help="single family key, e.g. c3 or pring:human:BFS")
    p.add_argument("--all", action="store_true", help="process every independent family")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--row-cap", type=int, default=DEFAULT_ROW_CAP,
                   help="max val rows to explain (label-stratified subsample); 0 = all")
    p.add_argument("--model-root", type=Path, default=RESULTS_PAIR / "length_robustness")
    p.add_argument("--out-root", type=Path, default=RESULTS_ANALYSIS / "shap_interpretation")
    p.add_argument("--top-n", type=int, default=300, help="rows written to the annotated TSV")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_one(family: str, args: argparse.Namespace, layer: int) -> None:
    import xgboost as xgb

    slug = _slug(family)
    out_dir = args.out_root / slug
    bundle_path = out_dir / "shap_bundle.npz"
    tsv_path = out_dir / "top_features_annotated.tsv"
    meta_path = out_dir / "shap_bundle.meta.json"

    if bundle_path.exists() and not args.overwrite:
        print(f"[skip] {bundle_path} exists (pass --overwrite)", flush=True)
        return

    model_path = args.model_root / slug / f"model_{REP}_esmcL{layer}_{PAIR_MODE}.ubj"
    if not model_path.exists():
        raise FileNotFoundError(f"no frozen booster for {family}: {model_path}")

    _, val_name = _train_val_names(family)
    print(f"[{family}] assembling {val_name} ({REP}, {args.backbone} L{layer}, {PAIR_MODE})", flush=True)
    _, Ava, Bva, yva, _ = _assemble(val_name, REP, backbone=args.backbone, layer=layer)
    Xva = sym_features(Ava, Bva)
    del Ava, Bva
    print(f"[{family}] Xva={Xva.shape} pos={float(yva.mean()):.4f}", flush=True)

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    row_cap = None if args.row_cap in (0, None) else int(args.row_cap)
    print(f"[{family}] TreeSHAP (row_cap={row_cap})", flush=True)
    bundle = compute_shap_bundle(
        booster, Xva, yva,
        family=family,
        sae_dim=sae_dim_for_backbone(args.backbone),
        row_cap=row_cap,
        seed=args.seed,
    )
    print(f"[{family}] shap={bundle.shap_values.shape} active_cols={len(bundle.columns)} "
          f"base={bundle.base_value:.5f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(bundle_path)
    print(f"[saved] {bundle_path} ({bundle_path.stat().st_size/1e6:.1f} MB)", flush=True)

    annotated = bundle.annotated_frame()
    annotated.head(args.top_n).to_csv(tsv_path, sep="\t", index=False)
    print(f"[saved] {tsv_path}", flush=True)

    meta_path.write_text(json.dumps({
        "family": family,
        "rep": REP,
        "pair_mode": PAIR_MODE,
        "backbone": args.backbone,
        "layer": int(layer),
        "seed": int(args.seed),
        "val_name": val_name,
        "booster": str(model_path),
        "n_val_pairs_total": int(len(yva)),
        "n_rows_explained": bundle.n_rows,
        "row_cap": row_cap,
        "n_active_columns": int(len(bundle.columns)),
        "n_total_columns": int(bundle.n_total_columns),
        "base_value": float(bundle.base_value),
        "note": ("full TreeSHAP matrix over the frozen length-robustness booster on its "
                 "official val split; columns trimmed to those the booster splits on "
                 "(zero-SHAP columns dropped). Reproduces the stored "
                 "feature_ranking_sae_max_sym.csv mean|shap| to ~1e-16."),
    }, indent=2, sort_keys=True))
    print(f"[saved] {meta_path}", flush=True)

    del Xva, bundle
    import gc; gc.collect()


def main() -> None:
    args = parse_args()
    layer = resolve_backbone_layer(args.backbone, args.layer)

    if args.all:
        families = list(INDEPENDENT_FAMILIES)
    elif args.family:
        fam = args.family
        if fam in ALIAS_FAMILIES:
            print(f"[alias] {fam} shares the {ALIAS_FAMILIES[fam]} booster; "
                  f"build that family instead", flush=True)
            return
        families = [fam]
    else:
        raise SystemExit("pass --family or --all")

    for fam in families:
        build_one(fam, args, layer)

    if args.all:
        alias_path = args.out_root / "pring_species_aliases.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.write_text(json.dumps({
            "note": ("PRING cross-species families reuse the human-BFS booster "
                     "(md5-identical) and are scored zero-shot, so they share its "
                     "SHAP explanation on human val."),
            "aliases": ALIAS_FAMILIES,
        }, indent=2, sort_keys=True))
        print(f"[saved] {alias_path}", flush=True)


if __name__ == "__main__":
    main()
