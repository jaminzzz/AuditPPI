"""Orchestration for the pooled-SAE fingerprint + classifier baseline.

Trains a classifier (XGB / TabPFN / dual-tower MLP) on a representation (binary / sae_max / esmc_mean) of
the **native train set** for each eval benchmark, then scores the eval benchmark and reports AUROC/AUPRC
via :func:`src.eval.evaluate_scorer`. This is the pooled-fingerprint "participation channel" baseline the
gated AuditPPI model is judged against.

Training protocol (user decision — per-benchmark native train):
  c3:*           ← train on c3:train
  cross_species:*← train on cross_species:human_train
  rf2ppi         ← train on c3:train (RF2-PPI has no train split), zero-shot eval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from conf.model import DEFAULT_SEED

from src.eval import evaluate_scorer
from src.data import pairs as D
from src.models.architectures.dual_tower import train_dual_tower
from src.models.estimators.tabpfn import fit_tabpfn, predict_proba_chunked
from src.models.estimators.xgboost import fit_xgb
from src.ppi_fingerprint import features as FE
from src.ppi_fingerprint.config import CACHE, MODEL_NAMES, NATIVE_TRAIN, OUT_DIR

_loaded: Dict[str, Dict] = {}  # cache of loaded pooled caches (large files)


def _family(name: str) -> str:
    return name.split(":")[0]


def _get_cache(family: str) -> Dict:
    if family not in _loaded:
        _loaded[family] = FE.load_pooled_cache(CACHE[family])
    return _loaded[family]


def _assemble(name: str, rep: str):
    bench = D.load_benchmark(name, attach_seqs=True)
    out = FE.assemble_pairs(bench, _get_cache(_family(name)), rep)
    if out is None:
        raise RuntimeError(f"no cached proteins for benchmark {name!r} rep {rep!r}")
    A, B, y, kept = out
    return bench, A, B, y, kept


def run_baseline(model: str, rep: str, eval_name: str, *, top_k: int = 500,
                 train_subsample: Optional[int] = 100000, val_frac: float = 0.1, seed: int = DEFAULT_SEED,
                 out_dir: Path = OUT_DIR, write: bool = True) -> Dict:
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; choose from {MODEL_NAMES}")
    if rep not in FE.REPS:
        raise ValueError(f"unknown representation {rep!r}; choose from {FE.REPS}")
    train_name = NATIVE_TRAIN[_family(eval_name)]

    # ---- train (native) ----
    _, Atr, Btr, ytr, _ = _assemble(train_name, rep)
    sub = FE.stratified_subsample(ytr, train_subsample, seed)
    if sub is not None:
        import torch
        ti = torch.as_tensor(sub, dtype=torch.long)
        Atr, Btr, ytr = Atr.index_select(0, ti), Btr.index_select(0, ti), ytr[sub]
    # stratified train/val split for early stopping
    val_idx = FE.stratified_subsample(ytr, max(1, int(len(ytr) * val_frac)), seed + 1)
    mask = np.ones(len(ytr), dtype=bool)
    mask[val_idx] = False
    import torch
    mt = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
    mv = torch.as_tensor(val_idx, dtype=torch.long)
    Atr_, Btr_, ytr_ = Atr.index_select(0, mt), Btr.index_select(0, mt), ytr[mask]
    Ava_, Bva_, yva_ = Atr.index_select(0, mv), Btr.index_select(0, mv), ytr[val_idx]

    # ---- eval (target benchmark) ----
    ebench, Ae, Be, ye, kept = _assemble(eval_name, rep)
    n_skipped = len(ebench.pairs) - len(kept)

    # ---- fit + predict ----
    cols = None
    if model == "xgb":
        Xtr, Xva = FE.sym_features(Atr_, Btr_), FE.sym_features(Ava_, Bva_)
        clf = fit_xgb(Xtr, ytr_, Xva, yva_, seed=seed)
        scores = predict_proba_chunked(clf, FE.sym_features(Ae, Be))
    elif model == "tabpfn":
        Xtr_full = FE.sym_features(Atr_, Btr_)
        cols = FE.xgb_topk_columns(Xtr_full, ytr_, top_k, seed=seed)
        clf = fit_tabpfn(Xtr_full[:, cols], ytr_, seed=seed)
        scores = predict_proba_chunked(clf, FE.sym_features(Ae, Be, cols))
    else:  # dualtower
        tower = train_dual_tower(Atr_, Btr_, ytr_, Ava_, Bva_, yva_, seed=seed)
        scores = tower.predict_proba_pairs(Ae, Be)

    # ---- score the eval benchmark (AUROC / AUPRC) ----
    score_map = {ebench.pairs[i]: float(s) for i, s in zip(kept, scores)}
    res = evaluate_scorer(lambda a, b: score_map.get((a, b)), ebench, name=f"{model}_{rep}")
    res.update({"model": model, "rep": rep, "eval": eval_name, "train": train_name,
                "top_k": top_k if model == "tabpfn" else None,
                "n_train": int(len(ytr_)), "n_skipped_eval": int(n_skipped)})
    print(f"[{model}·{rep}·{eval_name}] AUROC={res['auroc']} AUPRC={res['auprc']} "
          f"(n_train={res['n_train']}, skip={n_skipped})", flush=True)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{model}_{rep}_{eval_name.replace(':', '_')}.json").write_text(json.dumps(res, indent=2))
    return res
