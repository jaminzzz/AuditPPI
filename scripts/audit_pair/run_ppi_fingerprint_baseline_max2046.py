#!/usr/bin/env python3
"""Ladder-2 PPI fingerprint baseline on the max2046 ESM-C residue budget.

Identical protocol to ``run_ppi_fingerprint_baseline.py`` but points every
family's v1 protein cache at the ``*_max2046.pt`` slices (sliced from
``pooled_esmc_l60_l80_max2046_features.pt``) and writes results under a
separate ``results/main/ppi_fingerprint_max2046/`` tree, so the existing
max1022 products are untouched.

The max2046 pooled cache only carries ESM-C (no ESM-2 line), so this runner is
scoped to the ESM-C backbones/rep views -- ``--backbone esmc`` and the three
SAE/dense reps. eSIG is unaffected (its own global cache) and is left out.

Memory-safe cross_species
------------------------
``cross_species:human_train`` holds 421k pairs; the fingerprint-baseline
``_assemble`` path force-casts the FULL graph endpoints to float32 before
subsampling (~55 GB just for the sym sae_max matrix, plus ~52 GB for the A/B
gather -- right at a 120 GB host's ceiling). So for cross_species only, this
runner swaps ``_prepare_train_val`` for the memory-safe
:func:`carve_cross_species_train_val` path used by
``run_minimal_fingerprint_sweep.py`` / ``run_cross_species_tabpfn_topk.py``:
it selects train/val ROW INDICES from the cheap pair-index labels FIRST, then
gathers ONLY the selected rows. Order (val carve seed+1, then train cap seed)
and selection are bit-identical to the ``_assemble`` carve, so results match
the 1022 protocol -- only the peak memory differs. Eval graphs are small
(human_test 52k pairs ~6.9 GB; species tests ~7 GB) and stay on ``_assemble``.

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/run_ppi_fingerprint_baseline_max2046.py --model xgb
    $PY scripts/audit_pair/run_ppi_fingerprint_baseline_max2046.py --model xgb --family c3 --seed 43
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    resolve_backbone_layer,
)
from conf.paths import (
    CROSS_SPECIES_PAIR_INDEX_CACHES,
    PRING_CROSS_SPECIES,
    PROTEIN_SAE_CACHES,
    RESULTS_MAIN,
)
from src.data.pairs import list_cross_species
from src.data.pring_graph import METHODS as PRING_METHODS
from src.experiments.results import dump_experiment
from src.features.pairs import PAIR_MODES, load_protein_feature_cache

# Separate output tree so max1022 results stay pristine.
OUT_DIR = RESULTS_MAIN / "ppi_fingerprint_max2046"

# max2046 per-dataset caches (ESM-C only). Sibling names to the max1022 ones
# under data/sae/protein_caches/.
CACHES_2046 = {
    "c1": PROTEIN_SAE_CACHES / "c1_protein_features_max2046.pt",
    "c2": PROTEIN_SAE_CACHES / "c2_protein_features_max2046.pt",
    "c3": PROTEIN_SAE_CACHES / "c3_protein_features_max2046.pt",
    "cross_species": PROTEIN_SAE_CACHES / "cross_species_protein_features_max2046.pt",
    "bernett": PROTEIN_SAE_CACHES / "bernett_protein_features_max2046.pt",
    "pring": PROTEIN_SAE_CACHES / "pring_human_protein_features_max2046.pt",
}
PRING_SPECIES_2046 = {
    "human": CACHES_2046["pring"],
    **{
        sp: PROTEIN_SAE_CACHES / f"pring_{sp}_protein_features_max2046.pt"
        for sp in PRING_CROSS_SPECIES
    },
}

from src.ppi_fingerprint.config import (
    FINGERPRINT_REPS,
    MODEL_NAMES as MODELS,
    PRING_DEFAULT_METHOD,
    REPRESENTATIONS as REPS,
    axis_tag,
)


def _summary_path(family, model, b_tag, *, pair_mode="sym", seed=DEFAULT_SEED):
    return OUT_DIR / family / model / f"seed_{int(seed)}" / "summaries" / f"{b_tag}.json"


def family_evals(family: str) -> list[str]:
    if family in {"c1", "c2", "c3"}:
        return [f"{family}:test"]
    if family == "cross_species":
        species = [s for s in list_cross_species() if s != "human_train"]
        return [f"cross_species:{s}" for s in species]
    if family == "bernett":
        return ["bernett:test"]
    if family == "pring":
        evals = [f"pring:human:test:{m}" for m in PRING_METHODS]
        evals += [f"pring:{sp}:test" for sp in PRING_CROSS_SPECIES]
        return evals
    raise ValueError(f"unknown family {family!r}")


FAMILIES = ("c1", "c2", "c3", "cross_species", "bernett", "pring")


def _install_cross_species_safe_prepare(bl, cross_species_cache_path: Path) -> None:
    """Monkeypatch ``baseline._prepare_train_val`` to route cross_species train
    through the memory-safe pair-index carve; other families stay unchanged.

    ``cross_species`` ships no official val split, so the original ``_prepare_train_val``
    carves one from human_train in memory -- but only AFTER gathering the full
    421k-pair float32 matrix. This patched version delegates the cross_species
    train/val carve to :func:`carve_cross_species_train_val`, which selects
    indices from the cheap pair-index labels FIRST and gathers only the kept
    rows. Selection (val seed+1, then train cap seed) is bit-identical.
    """
    import torch

    from src.interp.pair_probe import carve_cross_species_train_val

    orig_prepare = bl._prepare_train_val

    # Load the cross_species protein cache once for the carve path. Memoized on
    # the baseline module's _loaded so it is shared with the eval-side _assemble.
    protein_cache = load_protein_feature_cache(cross_species_cache_path)

    def patched_prepare(train_name, val_name, rep, *, backbone, layer,
                        train_subsample, val_frac, seed):
        if train_name == "cross_species:human_train":
            train_a, train_b, train_y, val_a, val_b, val_y = carve_cross_species_train_val(
                CROSS_SPECIES_PAIR_INDEX_CACHES["human_train"], protein_cache,
                rep=rep, backbone=backbone, layer=layer,
                train_subsample=train_subsample, val_frac=val_frac, seed=seed,
            )
            # carve returns native-dtype (float16/bool) endpoint tensors; the
            # downstream _fit_scorer -> pair_features_np force-casts to float32,
            # so cast here to match the original _assemble contract (A/B float32).
            Atr_ = train_a.float(); Btr_ = train_b.float()
            Ava_ = val_a.float(); Bva_ = val_b.float()
            return Atr_, Btr_, train_y, Ava_, Bva_, val_y, int(len(train_y))
        return orig_prepare(
            train_name, val_name, rep,
            backbone=backbone, layer=layer,
            train_subsample=train_subsample, val_frac=val_frac, seed=seed,
        )

    bl._prepare_train_val = patched_prepare


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="xgb")
    p.add_argument("--family", choices=FAMILIES, default=None,
                   help="benchmark family; omit to sweep ALL families")
    p.add_argument("--families", nargs="*", default=None,
                   help="explicit family list (overrides --family)")
    p.add_argument("--backbone", choices=BACKBONES, default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="within-cache layer (default: backbone default)")
    p.add_argument("--reps", nargs="*", default=list(REPS),
                   help="pooling views (default: the three SAE views)")
    p.add_argument("--pair-mode", choices=PAIR_MODES, default="sym")
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--train-subsample", type=int, default=100000)
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    layer = resolve_backbone_layer(args.backbone, args.layer)
    if layer not in BACKBONE_LAYERS[args.backbone]:
        raise ValueError(f"backbone {args.backbone!r} has no layer {layer}")

    for rep in args.reps:
        if rep not in REPS:
            raise SystemExit(
                f"max2046 cache is ESM-C only; rep {rep!r} unavailable "
                f"(choose from {list(REPS)})"
            )

    # Verify all 2046 caches exist before starting (fail fast, not mid-run).
    missing = [p for p in CACHES_2046.values() if not p.exists()]
    missing += [p for p in PRING_SPECIES_2046.values() if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} max2046 cache(s); slice them first with "
            f"scripts/prep/slice_dataset_protein_cache.py:\n  "
            + "\n  ".join(str(m) for m in missing)
        )

    dev = str(args.device_id) if args.device_id is not None else str(_pick_free_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    print(f"[device] CUDA_VISIBLE_DEVICES={dev}", flush=True)

    # Inject the max2046 cache paths into the baseline module BEFORE it resolves
    # any benchmark. baseline._cache_path_for reads the module-level CACHE and
    # PRING_SPECIES_SAE_CACHES at call time, so patching them in-place is enough.
    import src.ppi_fingerprint.baseline as bl

    bl.CACHE = CACHES_2046
    bl.PRING_SPECIES_SAE_CACHES = PRING_SPECIES_2046
    # Clear the loaded-cache memo so a stale max1022 load can't leak in.
    bl._loaded.clear()

    # Route cross_species train through the memory-safe carve (421k-pair human
    # graph would otherwise materialize ~107 GB of float32 on a 120 GB host).
    _install_cross_species_safe_prepare(bl, CACHES_2046["cross_species"])

    from src.ppi_fingerprint.baseline import run_baseline_evals

    fams = args.families if args.families else ([args.family] if args.family else list(FAMILIES))
    b_tag = axis_tag(args.reps[0], args.backbone, layer)

    all_summary: dict[str, dict] = {}
    for fam in fams:
        evals = family_evals(fam)
        print(
            f"\n[plan] family={fam} model={args.model} backbone={b_tag} "
            f"seed={args.seed} reps={args.reps} pair_mode={args.pair_mode} evals={evals}",
            flush=True,
        )
        summary: dict[str, dict] = {}
        for rep in args.reps:
            try:
                rows = run_baseline_evals(
                    args.model, rep, evals,
                    backbone=args.backbone, layer=layer,
                    pair_mode=args.pair_mode,
                    top_k=args.top_k, train_subsample=args.train_subsample,
                    seed=args.seed, out_dir=OUT_DIR,
                )
                for r in rows:
                    key = f"{args.model}/{rep}/{b_tag}/{args.pair_mode}/seed{args.seed}/{r['eval']}"
                    summary[key] = {
                        "auroc": r["auroc"], "auprc": r["auprc"],
                        "n_skip": r["n_skipped_eval"], "train": r["train"],
                        "pair_mode": r.get("pair_mode"), "seed": r.get("seed", args.seed),
                    }
            except Exception as exc:  # noqa: BLE001
                for ev in evals:
                    key = f"{args.model}/{rep}/{b_tag}/{args.pair_mode}/seed{args.seed}/{ev}"
                    print(f"[skip] {key}: {exc}", flush=True)
                    summary[key] = {"error": str(exc)}

        print(
            f"\n=== ppi_fingerprint_max2046 {fam}/{args.model}/{b_tag}/seed{args.seed}: "
            f"AUROC / AUPRC ===",
            flush=True,
        )
        for key, value in summary.items():
            print(f"  {key:52s} {value}", flush=True)
        all_summary.update(summary)

        out_path = _summary_path(fam, args.model, b_tag,
                                pair_mode=args.pair_mode, seed=args.seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_experiment(
            out_path,
            task="pair.ppi_fingerprint_max2046",
            dataset=fam,
            features="multi" if len(args.reps) > 1 else args.reps[0],
            split="multi" if len(evals) > 1 else evals[0],
            model=args.model,
            seed=args.seed,
            payload=summary,
            metrics=summary,
            hyperparameters={
                "backbone": args.backbone, "layer": layer,
                "top_k": args.top_k, "train_subsample": args.train_subsample,
                "reps": list(args.reps), "pair_mode": args.pair_mode,
                "seed": args.seed, "residue_budget": 2046,
            },
        )
        print(f"[done] {out_path}", flush=True)

    print("\n=== ALL FAMILIES SUMMARY (max2046) ===", flush=True)
    for key, value in all_summary.items():
        print(f"  {key:52s} {value}", flush=True)


def _pick_free_gpu() -> int:
    try:
        from src.runtime.device import pick_free_gpu
        return pick_free_gpu()
    except Exception:
        return 0


if __name__ == "__main__":
    main()
