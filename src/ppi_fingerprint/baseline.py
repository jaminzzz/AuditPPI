"""Orchestration for the pooled-SAE fingerprint + classifier baseline.

Trains a classifier (XGB / TabPFN / MLP-pair / TabM-pair) on a representation
(binary / sae_max / esmc_mean) of the **native train set** for each eval
benchmark, then scores the eval benchmark and reports AUROC/AUPRC via
:func:`src.eval.evaluate_scorer`. This is the pooled-fingerprint "participation
channel" baseline the gated AuditPPI model is judged against.

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

``mlp_pair`` and ``tabm_pair`` share the :data:`~src.features.pairs.PAIR_MODES`
vocabulary (``sym`` / ``concat`` / ``rich`` / ablations); only ``concat`` uses
AB/BA train/eval.
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
from src.features.pairs import PAIR_MODES, load_protein_feature_cache, sym_features
from src.features.protein_cache import pair_feature_rows
from src.features.sampling import stratified_subsample
from src.models.architectures.mlp_pair import train_mlp_pair
from src.models.architectures.tabm_pair import train_tabm_pair
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


def _val_name_for(eval_name: str) -> Optional[str]:
    """The OFFICIAL validation benchmark for early stopping, or ``None``.

    Every family ships a held-out val split except ``cross_species`` (only a
    human train graph + per-species test graphs on disk) -- for that family the
    caller carves a stratified val from train in memory. PRING validates on the
    human val graph of the matching sampling method; ``rf2ppi`` (no train split
    of its own, trains on ``c3:train``) validates on ``c3:val``.
    """
    family = _family(eval_name)
    if family in {"c1", "c2", "c3"}:
        return f"{family}:val"
    if family == "bernett":
        return "bernett:val"
    if family == "rf2ppi":
        return "c3:val"
    if family == "pring":
        parts = eval_name.split(":")
        method = parts[3] if len(parts) > 3 and parts[3] else PRING_DEFAULT_METHOD
        return f"pring:human:val:{method}"
    return None  # cross_species: no official val split


def _assemble(name: str, rep: str, *, backbone: str, layer: Optional[int]):
    bench = D.load_benchmark(name, attach_seqs=True)
    out = pair_feature_rows(bench, _get_cache(name), rep, layer=layer, backbone=backbone)
    if out is None:
        raise RuntimeError(f"no cached proteins for benchmark {name!r} rep {rep!r}")
    A, B, y, kept = out
    return bench, A, B, y, kept


def run_baseline(model: str, rep: str, eval_name: str, *, top_k: int = 500,
                 backbone: str = DEFAULT_BACKBONE, layer: Optional[int] = None,
                 pair_mode: str = "sym",
                 train_subsample: Optional[int] = 100000, val_frac: float = 0.1, seed: int = DEFAULT_SEED,
                 out_dir: Path = OUT_DIR, write: bool = True) -> Dict:
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; choose from {MODEL_NAMES}")
    if rep not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {rep!r}; choose from {REPRESENTATIONS}")
    if pair_mode not in PAIR_MODES:
        raise ValueError(f"unknown pair_mode {pair_mode!r}; choose from {PAIR_MODES}")
    train_name = _train_name_for(eval_name)
    val_name = _val_name_for(eval_name)

    # ---- train (native) ----
    import torch

    _, Atr, Btr, ytr, _ = _assemble(train_name, rep, backbone=backbone, layer=layer)
    sub = stratified_subsample(ytr, train_subsample, seed)
    if sub is not None:
        ti = torch.as_tensor(sub, dtype=torch.long)
        Atr, Btr, ytr = Atr.index_select(0, ti), Btr.index_select(0, ti), ytr[sub]

    if val_name is not None:
        # Official held-out val split for early stopping -- all of train is used
        # for fitting; the val benchmark is loaded and featurised separately.
        _, Ava_, Bva_, yva_, _ = _assemble(val_name, rep, backbone=backbone, layer=layer)
        Atr_, Btr_, ytr_ = Atr, Btr, ytr
    else:
        # No official val (cross_species): carve a stratified val from train.
        # stratified_subsample returns None when max_rows >= len(y); never index
        # with None (numpy treats None as newaxis and silently corrupts the split).
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
    # xgb/tabpfn stay on the historical ``sym`` tabular features (order-invariant
    # trees/TabPFN do not need AB/BA). mlp_pair and tabm_pair share ``pair_mode``
    # so the only free axis between them is classifier capacity.
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
    elif model == "mlp_pair":
        mlp = train_mlp_pair(
            Atr_, Btr_, ytr_, Ava_, Bva_, yva_,
            pair_mode=pair_mode, seed=seed,
        )
        scores = mlp.predict_proba_pairs(Ae, Be)
    else:  # tabm_pair -- same pair_mode vocabulary as mlp_pair
        tabm = train_tabm_pair(
            Atr_, Btr_, ytr_, Ava_, Bva_, yva_,
            pair_mode=pair_mode, seed=seed,
        )
        scores = tabm.predict_proba_pairs(Ae, Be)

    # ---- score the eval benchmark (AUROC / AUPRC) ----
    resolved_layer = resolve_backbone_layer(backbone, layer)
    score_map = {ebench.pairs[i]: float(s) for i, s in zip(kept, scores)}
    res = evaluate_scorer(lambda a, b: score_map.get((a, b)), ebench, name=f"{model}_{rep}")
    pair_models = {"mlp_pair", "tabm_pair"}
    res.update({"model": model, "rep": rep, "eval": eval_name, "train": train_name,
                "backbone": backbone, "layer": resolved_layer,
                "pair_mode": pair_mode if model in pair_models else "sym",
                "top_k": top_k if model == "tabpfn" else None,
                "n_train": int(len(ytr_)), "n_skipped_eval": int(n_skipped)})
    pm_tag = f"·{pair_mode}" if model in pair_models else ""
    print(f"[{model}·{rep}·{backbone}L{resolved_layer}{pm_tag}·{eval_name}] "
          f"AUROC={res['auroc']} AUPRC={res['auprc']} "
          f"(n_train={res['n_train']}, skip={n_skipped})", flush=True)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        b_tag = f"{backbone}L{resolved_layer}"
        if model in pair_models:
            stem = f"{model}_{rep}_{b_tag}_{pair_mode}_{eval_name.replace(':', '_')}"
        else:
            stem = f"{model}_{rep}_{b_tag}_{eval_name.replace(':', '_')}"
        (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
    return res
