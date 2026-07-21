"""Orchestration for the pooled-SAE fingerprint + classifier baseline.

Trains a classifier (XGB / TabPFN / dual-tower MLP) on a representation (binary / sae_max / esmc_mean) of
the **native train set** for each eval benchmark, then scores the eval benchmark and reports AUROC/AUPRC
via :func:`src.eval.evaluate_scorer`. This is the pooled-fingerprint "participation channel" baseline the
gated AuditPPI model is judged against.

Training protocol (user decision — per-benchmark native train):
  c1:*           ← train on c1:train
  c2:*           ← train on c2:train
  c3:*           ← train on c3:train
  cross_species:*← train on cross_species:human_train
  bernett:*      ← train on bernett:train
  pring:*        ← train on pring:human:train:<method> (zero-shot cross-species)
  rf2ppi         ← train on c3:train (RF2-PPI has no train split), zero-shot eval

Feature source (v1): each family's ``auditppi_protein_features_v1`` protein cache
(``conf.paths.PPI_PREDICTION_CACHES``) holds every endpoint sequence for that
family across all its splits, keyed by sequence via ``seq2idx``. The
``(backbone, layer, rep)`` channel is selected inside the cache at read time by
:func:`~src.features.protein_cache.representation_matrix`. PRING is per-species,
so its cache is resolved by species via ``PRING_SPECIES_SAE_CACHES``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from conf.model import DEFAULT_BACKBONE, DEFAULT_SEED, REPRESENTATIONS, resolve_backbone_layer

from src.eval import evaluate_scorer
from src.data import pairs as D
from src.features.feature_selection import xgb_topk_columns
from src.features.pairs import load_protein_feature_cache, sym_features
from src.features.protein_cache import pair_feature_rows
from src.features.sampling import stratified_subsample
from src.models.architectures.dual_tower import train_dual_tower
from src.models.estimators.tabpfn import fit_tabpfn, predict_proba_chunked
from src.models.estimators.xgboost import fit_xgb
from src.ppi_fingerprint.config import (
    CACHE,
    MODEL_NAMES,
    NATIVE_TRAIN,
    OUT_DIR,
    PRING_DEFAULT_METHOD,
    PRING_SPECIES_SAE_CACHES,
)

_loaded: Dict[Path, Dict] = {}  # cache of loaded v1 caches, keyed by resolved path


def _family(name: str) -> str:
    return name.split(":")[0]


def _cache_path_for(name: str) -> Path:
    """Resolve a benchmark name to its v1 protein feature cache path.

    All families map through ``PPI_PREDICTION_CACHES`` except PRING, whose cache
    is per-species (``name = pring:species[:split[:method]]``): human train graph
    and each held-out species test graph live in their own cache.
    """
    family = _family(name)
    if family == "pring":
        parts = name.split(":")
        species = parts[1] if len(parts) > 1 and parts[1] else "human"
        return PRING_SPECIES_SAE_CACHES[species]
    return CACHE[family]


def _get_cache(name: str) -> Dict:
    path = _cache_path_for(name)
    if path not in _loaded:
        _loaded[path] = load_protein_feature_cache(path)
    return _loaded[path]


def _train_name_for(eval_name: str) -> str:
    """The native-train benchmark name for an eval benchmark.

    PRING trains on the human graph of the SAME sampling method as the eval graph
    (or ``PRING_DEFAULT_METHOD`` for the cross-species test graphs, which carry no
    method); every other family uses its ``NATIVE_TRAIN`` split.
    """
    family = _family(eval_name)
    if family == "pring":
        parts = eval_name.split(":")
        method = parts[3] if len(parts) > 3 and parts[3] else PRING_DEFAULT_METHOD
        return f"pring:human:train:{method}"
    return NATIVE_TRAIN[family]


def _assemble(name: str, rep: str, *, backbone: str, layer: Optional[int]):
    bench = D.load_benchmark(name, attach_seqs=True)
    out = pair_feature_rows(bench, _get_cache(name), rep, layer=layer, backbone=backbone)
    if out is None:
        raise RuntimeError(f"no cached proteins for benchmark {name!r} rep {rep!r}")
    A, B, y, kept = out
    return bench, A, B, y, kept


def run_baseline(model: str, rep: str, eval_name: str, *, top_k: int = 500,
                 backbone: str = DEFAULT_BACKBONE, layer: Optional[int] = None,
                 train_subsample: Optional[int] = 100000, val_frac: float = 0.1, seed: int = DEFAULT_SEED,
                 out_dir: Path = OUT_DIR, write: bool = True) -> Dict:
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; choose from {MODEL_NAMES}")
    if rep not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {rep!r}; choose from {REPRESENTATIONS}")
    train_name = _train_name_for(eval_name)

    # ---- train (native) ----
    import torch

    _, Atr, Btr, ytr, _ = _assemble(train_name, rep, backbone=backbone, layer=layer)
    sub = stratified_subsample(ytr, train_subsample, seed)
    if sub is not None:
        ti = torch.as_tensor(sub, dtype=torch.long)
        Atr, Btr, ytr = Atr.index_select(0, ti), Btr.index_select(0, ti), ytr[sub]
    # stratified train/val split for early stopping.
    # stratified_subsample returns None when max_rows >= len(y); never index with None
    # (numpy treats None as newaxis and would silently corrupt the split).
    n_val = max(1, int(len(ytr) * val_frac))
    val_idx = stratified_subsample(ytr, n_val, seed + 1)
    if val_idx is None:
        # Degenerate tiny train set: hold out a single stratified-or-first row.
        if len(ytr) <= 1:
            raise ValueError(
                f"need at least 2 training pairs for a val split; got {len(ytr)}"
            )
        val_idx = np.array([0], dtype=np.int64)
    mask = np.ones(len(ytr), dtype=bool)
    mask[val_idx] = False
    mt = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
    mv = torch.as_tensor(val_idx, dtype=torch.long)
    Atr_, Btr_, ytr_ = Atr.index_select(0, mt), Btr.index_select(0, mt), ytr[mask]
    Ava_, Bva_, yva_ = Atr.index_select(0, mv), Btr.index_select(0, mv), ytr[val_idx]

    # ---- eval (target benchmark) ----
    ebench, Ae, Be, ye, kept = _assemble(eval_name, rep, backbone=backbone, layer=layer)
    n_skipped = len(ebench.pairs) - len(kept)

    # ---- fit + predict ----
    cols = None
    if model == "xgb":
        Xtr, Xva = sym_features(Atr_, Btr_), sym_features(Ava_, Bva_)
        clf = fit_xgb(Xtr, ytr_, Xva, yva_, seed=seed)
        scores = predict_proba_chunked(clf, sym_features(Ae, Be))
    elif model == "tabpfn":
        Xtr_full = sym_features(Atr_, Btr_)
        cols = xgb_topk_columns(Xtr_full, ytr_, top_k, seed=seed)
        clf = fit_tabpfn(Xtr_full[:, cols], ytr_, seed=seed)
        scores = predict_proba_chunked(clf, sym_features(Ae, Be, cols))
    else:  # dualtower
        tower = train_dual_tower(Atr_, Btr_, ytr_, Ava_, Bva_, yva_, seed=seed)
        scores = tower.predict_proba_pairs(Ae, Be)

    # ---- score the eval benchmark (AUROC / AUPRC) ----
    resolved_layer = resolve_backbone_layer(backbone, layer)
    score_map = {ebench.pairs[i]: float(s) for i, s in zip(kept, scores)}
    res = evaluate_scorer(lambda a, b: score_map.get((a, b)), ebench, name=f"{model}_{rep}")
    res.update({"model": model, "rep": rep, "eval": eval_name, "train": train_name,
                "backbone": backbone, "layer": resolved_layer,
                "top_k": top_k if model == "tabpfn" else None,
                "n_train": int(len(ytr_)), "n_skipped_eval": int(n_skipped)})
    print(f"[{model}·{rep}·{backbone}L{resolved_layer}·{eval_name}] "
          f"AUROC={res['auroc']} AUPRC={res['auprc']} "
          f"(n_train={res['n_train']}, skip={n_skipped})", flush=True)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        b_tag = f"{backbone}L{resolved_layer}"
        stem = f"{model}_{rep}_{b_tag}_{eval_name.replace(':', '_')}"
        (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
    return res
