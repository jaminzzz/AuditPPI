#!/usr/bin/env python
"""Train-fit ceiling for panel c: score the human FIT set in-sample.

Panel c contrasts a lenient split (cross_species) against a strict topological
split (PRING) for the shared species. Both are trained ONLY on human, so the
honest "how well can this fingerprint fit what it saw" ceiling is a single human
number per split protocol -- the in-sample train AUROC. Held-out species
(ecoli/yeast) are human-model zero-shot transfers and have no own train, so the
ceiling is human-specific by construction.

MEMORY SAFETY (the earlier OOM, root cause):
    ``_assemble`` builds features for EVERY pair in a split *before* any
    subsampling. cross_species:human_train has 523k pairs and sae_max is 16384-D,
    so A+B alone = 523420 * 16384 * 4 bytes * 2 ~ 68 GB -- it blew the 100 GB
    shared-server ceiling before the subsample to 100k ever ran.

    Fix: subsample the pair LIST first (class-stratified), build a small
    Benchmark holding only those <=100k pairs, and assemble features for those
    rows only. Peak features ~ 100000 * 16384 * 4 * 2 ~ 13 GB. We never
    materialise the full graph.

Fixed to the main config (xgb / sae_max / ESM-C L60 / sym), 3 seeds.
Writes results/main/eval/panelc_train_ceiling.json (mean/std over seeds).

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/audit_pair/eval_train_ceiling.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from conf.paths import RESULTS_MAIN

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.data import pairs as D  # noqa: E402
from src.data.pairs import Benchmark  # noqa: E402
from src.features.pairs import pair_features_np  # noqa: E402
from src.features.protein_cache import pair_feature_rows  # noqa: E402
from src.features.sampling import stratified_subsample  # noqa: E402
from src.models.estimators.xgboost import fit_xgb  # noqa: E402
from src.models.estimators.tabpfn import predict_proba_chunked  # noqa: E402
from src.ppi_fingerprint.baseline import _get_cache, _train_name_for, _val_name_for  # noqa: E402

SEEDS = (42, 43, 44)
# (label, an eval name whose native human train we score in-sample)
CASES = [
    ("cross_species", "cross_species:human_test"),
    ("pring_bfs", "pring:human:test:BFS"),
]

REP = "sae_max"
BACKBONE = "esmc"
LAYER = 60
PAIR_MODE = "sym"
TRAIN_SUBSAMPLE = 100000   # matches the production main-config cap
VAL_SUBSAMPLE = 20000      # early-stopping val; capped so it is memory-safe too


def _subsample_bench(name: str, max_rows: int, seed: int) -> Benchmark:
    """Load a benchmark and return a class-stratified <=max_rows pair subset.

    Crucially this subsamples the pair LIST before any feature assembly, so we
    never build the full-graph feature matrix.
    """
    bench = D.load_benchmark(name, attach_seqs=True)
    y = bench.labels
    idx = stratified_subsample(y, max_rows, seed)
    if idx is None:
        return bench  # already <= max_rows: keep all
    pairs = [bench.pairs[i] for i in idx]
    labels = y[idx]
    return Benchmark(name=bench.name, pairs=pairs, labels=labels, seqs=bench.seqs)


def _assemble_small(name: str, max_rows: int, seed: int):
    """Subsample pairs first, then assemble features for that subset only."""
    bench = _subsample_bench(name, max_rows, seed)
    out = pair_feature_rows(bench, _get_cache(name), REP, layer=LAYER, backbone=BACKBONE)
    if out is None:
        raise RuntimeError(f"no cached proteins for {name!r}")
    A, B, y, _ = out
    return A, B, y


def main() -> None:
    out = {}
    for label, ref_eval in CASES:
        train_name = _train_name_for(ref_eval)
        val_name = _val_name_for(ref_eval)  # None for cross_species
        aurocs, auprcs, n_fits = [], [], []
        for seed in SEEDS:
            # Fit set: <=100k class-stratified human-train pairs (assembled small).
            Atr, Btr, ytr = _assemble_small(train_name, TRAIN_SUBSAMPLE, seed)
            if val_name is not None:
                Ava, Bva, yva = _assemble_small(val_name, VAL_SUBSAMPLE, seed)
            else:
                # cross_species has no official val: carve a stratified 10% from
                # the fit set (indices into the already-small matrices).
                n_val = max(1, int(len(ytr) * 0.1))
                vidx = stratified_subsample(ytr, n_val, seed + 1)
                mask = np.ones(len(ytr), dtype=bool)
                mask[vidx] = False
                import torch
                mt = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
                mv = torch.as_tensor(vidx, dtype=torch.long)
                Ava, Bva, yva = Atr.index_select(0, mv), Btr.index_select(0, mv), ytr[vidx]
                Atr, Btr, ytr = Atr.index_select(0, mt), Btr.index_select(0, mt), ytr[mask]

            Xtr = pair_features_np(Atr, Btr, PAIR_MODE)
            Xva = pair_features_np(Ava, Bva, PAIR_MODE)
            clf = fit_xgb(Xtr, ytr, Xva, yva, seed=seed)

            # In-sample scoring on the SAME fit matrix (chunked -> memory safe).
            scores = predict_proba_chunked(clf, Xtr)
            yv = ytr.numpy() if hasattr(ytr, "numpy") else np.asarray(ytr)
            auroc = float(roc_auc_score(yv, scores))
            auprc = float(average_precision_score(yv, scores))
            aurocs.append(auroc)
            auprcs.append(auprc)
            n_fits.append(int(len(yv)))
            print(f"[{label}·seed{seed}] in-sample AUROC={auroc:.4f} "
                  f"AUPRC={auprc:.4f} n_fit={len(yv)} (train={train_name})",
                  flush=True)
        out[label] = {
            "train_name": train_name,
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auprc_mean": float(np.mean(auprcs)),
            "auprc_std": float(np.std(auprcs)),
            "n_fit": int(np.round(np.mean(n_fits))),
            "seeds": list(SEEDS),
        }

    path = RESULTS_MAIN / "eval" / "panelc_train_ceiling.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {path}\n{json.dumps(out, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
