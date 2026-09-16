#!/usr/bin/env python3
"""Score a family's frozen baseline against its per-length TEST feature caches.

The evaluation side of the test-length robustness sweep. The training side is
fixed (``train_length_baseline_model.py`` saved one booster on the max1022 train
+ official val endpoints); this script loads that booster and, for each
truncation length ``L``, assembles the test-pair ``sym`` matrix from the matching
``test_features_max{L}.pt`` cache and reports test AUROC/AUPRC.

Two cache sources:

* per-family (default): ``{out_root}/{family}/test_features_max{L}.pt`` -- the
  c3/bernett caches produced by ``extract_length_test_features.py``.
* shared pool (``--shared-pool``): ``{shared_pool}/test_features_max{L}.pt`` --
  ONE deduplicated union of c1/c2/pring test endpoint sequences produced by
  ``extract_length_shared_pool.py``. Every c1/c2/pring family reads its rows from
  this same pool (``pair_feature_rows`` is sequence-keyed, so the family only
  picks up the pairs whose endpoints are in the pool -- all of them, by
  construction).

Family keys (``--family``): ``c1 | c2 | c3 | bernett | pring:human:{BFS|DFS|
RANDOM_WALK} | pring:{yeast|ecoli|arath}``. The booster is loaded from
``{out_root}/{family_slug}/model_*.ubj`` where ``family_slug`` replaces ``:`` with
``_`` (matching ``train_length_baseline_model.py``).

Per length ``L`` (all reading the SAME booster + SAME pair topology/labels):
  1. ``load_benchmark(family_test)`` -> pair endpoints + labels (topology fixed).
  2. ``load_protein_feature_cache(cache_path)`` -> truncated features.
  3. ``pair_feature_rows(bench, cache, sae_max, L60)`` gathers endpoints by the
     FULL sequence key (truncation changed values, not keys) -> ``sym_features``.
  4. ``booster.predict_proba`` -> ``probe_classification_metrics``.

The kept-pair count MUST be identical across every length (same keys present);
the script records ``n_scored`` per length and warns if any drift is seen.

Product::

    results/audit_pair/length_robustness/{family_slug}/length_sweep.json
      (dump_experiment spine; payload.cells = one row per length + a "full"
       reference row lifted from the training meta sidecar's val metrics)

Run (CPU; the booster was re-homed to CPU at save time)::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/run_length_robustness_sweep.py --family c3
    $PY scripts/audit_pair/run_length_robustness_sweep.py --family pring:human:BFS \
        --shared-pool results/audit_pair/length_robustness/_shared_pool
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

from conf.model import DEFAULT_BACKBONE, resolve_backbone_layer
from conf.paths import RESULTS_PAIR
from src.data import pairs as D
from src.eval.classification import probe_classification_metrics
from src.experiments.results import dump_experiment
from src.features.pairs import load_protein_feature_cache, sym_features
from src.features.protein_cache import pair_feature_rows
from src.ppi_fingerprint.config import family_of, PRING_DEFAULT_METHOD

REP = "sae_max"
PAIR_MODE = "sym"
DEFAULT_LENGTHS = (100, 200, 400, 600, 800, 1000, 1500, 2046)


def _family_slug(family_arg: str) -> str:
    """Path-safe directory name (``pring:human:BFS`` -> ``pring_human_BFS``)."""
    return family_arg.replace(":", "_")


def _test_benchmark_name(family_arg: str) -> str:
    """The ``load_benchmark`` key for a family's TEST split.

    pring:human:{M} -> ``pring:human:test:{M}``
    pring:{species}  -> ``pring:{species}:test:{PRING_DEFAULT_METHOD}``
    others          -> ``{family}:test``
    """
    family = family_of(family_arg)
    if family == "pring":
        parts = family_arg.split(":")
        species = parts[1] if len(parts) > 1 and parts[1] else "human"
        if species == "human":
            method = parts[2] if len(parts) > 2 and parts[2] else PRING_DEFAULT_METHOD
            return f"pring:human:test:{method}"
        return f"pring:{species}:test:{PRING_DEFAULT_METHOD}"
    return f"{family}:test"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--family", default="c3",
        help=("family key: c1 | c2 | c3 | bernett | pring:human:{BFS|DFS|RANDOM_WALK} "
              "| pring:{yeast|ecoli|arath}"))
    p.add_argument("--split", default="test")
    p.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="SAE layer; defaults to the backbone's default layer (ESM-C L60).")
    p.add_argument("--out-root", type=Path, default=RESULTS_PAIR / "length_robustness")
    p.add_argument(
        "--shared-pool", type=Path, default=None,
        help=("read per-length test caches from a SHARED pool directory "
              "(produced by extract_length_shared_pool.py) instead of the "
              "per-family directory. c1/c2/pring use this."))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_booster(model_path: Path):
    import xgboost as xgb

    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    return clf


def score_length(clf, bench, cache_path: Path, *, backbone: str, layer: int) -> dict:
    cache = load_protein_feature_cache(cache_path)
    out = pair_feature_rows(bench, cache, REP, layer=layer, backbone=backbone)
    if out is None:
        raise RuntimeError(f"no cached test pairs for {cache_path}")
    A, B, y, kept = out
    X = sym_features(A, B)
    proba = clf.predict_proba(X)[:, 1]
    m = probe_classification_metrics(y, proba)
    n_trunc = cache.get("meta", {}).get("extractor", {}).get(
        "length_robustness", {}
    ).get("n_truncated")
    return {
        "n_scored": int(len(kept)),
        "n_pairs_total": int(len(bench.pairs)),
        "pos_rate": round(float(np.mean(y)), 6),
        "auroc": round(float(m["auroc"]), 6),
        "auprc": round(float(m["auprc"]), 6),
        "n_truncated_unique_seqs": n_trunc,
    }


def main() -> None:
    args = parse_args()
    layer = resolve_backbone_layer(args.backbone, args.layer)
    family_slug = _family_slug(args.family)
    fam_dir = args.out_root / family_slug
    model_path = fam_dir / f"model_{REP}_esmcL{layer}_{PAIR_MODE}.ubj"
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing frozen baseline: {model_path}\n"
            f"run train_length_baseline_model.py --family {args.family} first."
        )
    out_path = fam_dir / "length_sweep.json"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {out_path}; pass --overwrite to replace it")

    # Where per-length test caches live: shared pool (c1/c2/pring) or per-family
    # (c3/bernett).
    cache_dir = args.shared_pool if args.shared_pool is not None else fam_dir

    meta_side = json.loads(model_path.with_suffix(".meta.json").read_text())
    clf = load_booster(model_path)
    test_name = _test_benchmark_name(args.family)
    bench = D.load_benchmark(test_name, attach_seqs=True)
    print(f"[eval] {test_name} pairs={len(bench.pairs)} "
          f"model={model_path.name} cache_dir={cache_dir}", flush=True)

    cells = []
    n_scored_ref = None
    for length in sorted(args.lengths):
        cache_path = cache_dir / f"test_features_max{length}.pt"
        if not cache_path.exists():
            print(f"[skip] missing cache for L={length}: {cache_path}", flush=True)
            continue
        cell = score_length(clf, bench, cache_path, backbone=args.backbone, layer=layer)
        cell["max_residues"] = int(length)
        cells.append(cell)
        if n_scored_ref is None:
            n_scored_ref = cell["n_scored"]
        drift = "" if cell["n_scored"] == n_scored_ref else "  !! n_scored DRIFT"
        print(f"[L={length:>4}] auroc={cell['auroc']:.4f} auprc={cell['auprc']:.4f} "
              f"n_scored={cell['n_scored']} trunc_seqs={cell['n_truncated_unique_seqs']}{drift}",
              flush=True)

    if not cells:
        raise RuntimeError("no length caches found; run the extractor first.")

    payload = {
        "family": args.family,
        "family_slug": family_slug,
        "split": args.split,
        "test_benchmark": test_name,
        "rep": REP,
        "pair_mode": PAIR_MODE,
        "backbone": args.backbone,
        "layer": int(layer),
        "lengths": [c["max_residues"] for c in cells],
        "cells": cells,
        "model": model_path.name,
        "cache_dir": str(cache_dir),
        "shared_pool": args.shared_pool is not None,
        "train_meta": meta_side,
        "note": ("test-length robustness: fixed booster (max1022 train+val) scored "
                 "against test features re-extracted at each max_residues; train/val "
                 "never truncated. n_scored is identical across lengths by design "
                 "(truncation changes feature values, not seq2idx keys)."),
    }
    best = max(cells, key=lambda c: c["max_residues"])
    dump_experiment(
        out_path,
        task="length_robustness",
        dataset=args.family,
        features=REP,
        split=args.split,
        model="xgb",
        seed=int(meta_side.get("seed", -1)),
        payload=payload,
        metrics={"auroc": best["auroc"], "auprc": best["auprc"],
                 "max_residues": best["max_residues"]},
        hyperparameters={"rep": REP, "pair_mode": PAIR_MODE,
                         "backbone": args.backbone, "layer": int(layer)},
    )
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
