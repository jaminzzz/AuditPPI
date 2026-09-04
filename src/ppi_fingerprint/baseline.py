"""Orchestration for the pooled-SAE fingerprint + classifier baseline.

Trains a classifier (XGB / TabPFN / MLP-pair / TabM-pair) on a representation
(binary / sae_max / esmc_mean) of the **native train set**, then scores one or
more eval benchmarks and reports AUROC/AUPRC via :func:`src.eval.evaluate_scorer`.
This is the pooled-fingerprint "participation channel" baseline the gated
AuditPPI model is judged against.

Training protocol (user decision — per-benchmark native train):
  c1:*           ← train on c1:train
  c2:*           ← train on c2:train
  c3:*           ← train on c3:train
  cross_species:*← train on cross_species:human_train  (**once** per rep/axis)
  bernett:*      ← train on bernett:train
  pring:*        ← train on pring:human:train:<method> (zero-shot cross-species)

When several evals share the same native train (e.g. all ``cross_species:*``, or
PRING's BFS human test + yeast/ecoli/arath), :func:`run_baseline_evals` fits
**once** and only re-runs feature assembly + inference per eval.

Feature source (v1): each family's ``auditppi_protein_features_v1`` protein cache
(``conf.paths.PPI_PREDICTION_CACHES``) holds every endpoint sequence for that
family across all its splits, keyed by sequence via ``seq2idx``. The
``(backbone, layer, rep)`` channel is selected inside the cache at read time by
:func:`~src.features.protein_cache.representation_matrix`. PRING is per-species,
so its cache is resolved by species via ``PRING_SPECIES_SAE_CACHES``.

``mlp_pair`` and ``tabm_pair`` share the :data:`~src.features.pairs.PAIR_MODES`
vocabulary (``sym`` / ``concat`` / ``rich`` / ablations); only ``concat`` uses
AB/BA train/eval.

Results land under
``OUT_DIR/{family}/{model}/seed_{S}/cells|summaries/``
(see :mod:`src.ppi_fingerprint.config`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from conf.model import DEFAULT_BACKBONE, DEFAULT_SEED, resolve_backbone_layer

from src.eval import evaluate_scorer
from src.data import pairs as D
from src.features.feature_selection import xgb_topk_columns
from src.features.pairs import PAIR_MODES, load_protein_feature_cache, pair_features_np
from src.features.protein_cache import pair_feature_rows
from src.features.sampling import stratified_subsample
from src.models.architectures.mlp_pair import train_mlp_pair
from src.models.architectures.tabm_pair import train_tabm_pair
from src.models.estimators.tabpfn import fit_tabpfn, predict_proba_chunked
from src.models.estimators.xgboost import fit_xgb
from src.ppi_fingerprint.config import (
    CACHE,
    FINGERPRINT_REPS,
    MODEL_NAMES,
    NATIVE_TRAIN,
    OUT_DIR,
    PAIR_MODELS,
    PRING_DEFAULT_METHOD,
    PRING_SPECIES_SAE_CACHES,
    axis_tag,
    cell_path,
    family_of,
)

_loaded: Dict[Path, Dict] = {}  # cache of loaded v1 caches, keyed by resolved path


def _family(name: str) -> str:
    return family_of(name)


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
    human val graph of the matching sampling method.
    """
    family = _family(eval_name)
    if family in {"c1", "c2", "c3"}:
        return f"{family}:val"
    if family == "bernett":
        return "bernett:val"
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


def _prepare_train_val(
    train_name: str,
    val_name: Optional[str],
    rep: str,
    *,
    backbone: str,
    layer: Optional[int],
    train_subsample: Optional[int],
    val_frac: float,
    seed: int,
):
    """Load/subsample train (+ official or carved val). Returns tensors + n_train."""
    import torch

    _, Atr, Btr, ytr, _ = _assemble(train_name, rep, backbone=backbone, layer=layer)
    sub = stratified_subsample(ytr, train_subsample, seed)
    if sub is not None:
        ti = torch.as_tensor(sub, dtype=torch.long)
        Atr, Btr, ytr = Atr.index_select(0, ti), Btr.index_select(0, ti), ytr[sub]

    if val_name is not None:
        # Official held-out val -- all of (subsampled) train is used for fitting.
        _, Ava_, Bva_, yva_, _ = _assemble(val_name, rep, backbone=backbone, layer=layer)
        Atr_, Btr_, ytr_ = Atr, Btr, ytr
    else:
        # No official val (cross_species): carve a stratified val from train.
        n_val = max(1, int(len(ytr) * val_frac))
        val_idx = stratified_subsample(ytr, n_val, seed + 1)
        if val_idx is None:
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

    return Atr_, Btr_, ytr_, Ava_, Bva_, yva_, int(len(ytr_))


def _fit_scorer(
    model: str,
    Atr_,
    Btr_,
    ytr_,
    Ava_,
    Bva_,
    yva_,
    *,
    pair_mode: str,
    top_k: int,
    seed: int,
) -> Tuple[Callable, Optional[np.ndarray]]:
    """Fit once; return ``predict(Ae, Be) -> scores`` and optional TabPFN cols."""
    cols = None
    if model == "xgb":
        Xtr = pair_features_np(Atr_, Btr_, pair_mode)
        Xva = pair_features_np(Ava_, Bva_, pair_mode)
        clf = fit_xgb(Xtr, ytr_, Xva, yva_, seed=seed)

        def predict(Ae, Be, _clf=clf, _mode=pair_mode):
            return predict_proba_chunked(_clf, pair_features_np(Ae, Be, _mode))

        return predict, cols

    if model == "tabpfn":
        Xtr_full = pair_features_np(Atr_, Btr_, pair_mode)
        cols = xgb_topk_columns(Xtr_full, ytr_, top_k, seed=seed)
        clf = fit_tabpfn(Xtr_full[:, cols], ytr_, seed=seed)

        def predict(Ae, Be, _clf=clf, _cols=cols, _mode=pair_mode):
            return predict_proba_chunked(_clf, pair_features_np(Ae, Be, _mode, _cols))

        return predict, cols

    if model == "mlp_pair":
        mlp = train_mlp_pair(
            Atr_, Btr_, ytr_, Ava_, Bva_, yva_,
            pair_mode=pair_mode, seed=seed,
        )

        def predict(Ae, Be, _mlp=mlp):
            return _mlp.predict_proba_pairs(Ae, Be)

        return predict, cols

    # tabm_pair -- same pair_mode vocabulary as mlp_pair
    tabm = train_tabm_pair(
        Atr_, Btr_, ytr_, Ava_, Bva_, yva_,
        pair_mode=pair_mode, seed=seed,
    )

    def predict(Ae, Be, _tabm=tabm):
        return _tabm.predict_proba_pairs(Ae, Be)

    return predict, cols


def _score_eval(
    predict: Callable,
    eval_name: str,
    rep: str,
    *,
    model: str,
    backbone: str,
    layer: Optional[int],
    train_name: str,
    pair_mode: str,
    top_k: int,
    n_train: int,
    cols: Optional[np.ndarray],
    seed: int,
    write: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    ebench, Ae, Be, ye, kept = _assemble(eval_name, rep, backbone=backbone, layer=layer)
    n_skipped = len(ebench.pairs) - len(kept)
    scores = predict(Ae, Be)

    resolved_layer = resolve_backbone_layer(backbone, layer)
    score_map = {ebench.pairs[i]: float(s) for i, s in zip(kept, scores)}
    res = evaluate_scorer(lambda a, b: score_map.get((a, b)), ebench, name=f"{model}_{rep}")
    res.update({
        "model": model,
        "rep": rep,
        "eval": eval_name,
        "train": train_name,
        "backbone": backbone,
        "layer": resolved_layer,
        "pair_mode": pair_mode,
        "seed": int(seed),
        "top_k": top_k if model == "tabpfn" else None,
        "n_train": int(n_train),
        "n_skipped_eval": int(n_skipped),
    })
    pm_tag = f"·{pair_mode}"
    print(
        f"[{model}·{rep}·{backbone}L{resolved_layer}{pm_tag}·seed{seed}·{eval_name}] "
        f"AUROC={res['auroc']} AUPRC={res['auprc']} "
        f"(n_train={res['n_train']}, skip={n_skipped})",
        flush=True,
    )

    if write:
        family = _family(eval_name)
        b_tag = axis_tag(rep, backbone, resolved_layer)
        path = cell_path(
            family, model, rep, b_tag, eval_name,
            pair_mode=pair_mode,
            seed=seed,
            root=out_dir,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2))
    return res


def run_baseline_evals(
    model: str,
    rep: str,
    eval_names: Sequence[str],
    *,
    top_k: int = 500,
    backbone: str = DEFAULT_BACKBONE,
    layer: Optional[int] = None,
    pair_mode: str = "sym",
    train_subsample: Optional[int] = 100000,
    val_frac: float = 0.1,
    seed: int = DEFAULT_SEED,
    out_dir: Path = OUT_DIR,
    write: bool = True,
) -> List[Dict[str, Any]]:
    """Fit once per shared native-train key, then score every eval.

    Evals are grouped by :func:`_train_name_for`. Within a group the classifier
    is fit a single time (same train/val protocol) and only feature assembly +
    inference re-run per eval -- so ``cross_species`` and PRING zero-shot
    species no longer retrain N times on the same human graph.

    Results always land under ``seed_{seed}/`` (including the default 42).
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; choose from {MODEL_NAMES}")
    if rep not in FINGERPRINT_REPS:
        raise ValueError(f"unknown representation {rep!r}; choose from {FINGERPRINT_REPS}")
    if pair_mode not in PAIR_MODES:
        raise ValueError(f"unknown pair_mode {pair_mode!r}; choose from {PAIR_MODES}")
    if not eval_names:
        raise ValueError("eval_names must be non-empty")

    groups: Dict[str, List[str]] = defaultdict(list)
    for ev in eval_names:
        groups[_train_name_for(ev)].append(ev)

    results: List[Dict[str, Any]] = []
    for train_name, group_evals in groups.items():
        # Val protocol is a function of family/method, identical for a shared train.
        val_name = _val_name_for(group_evals[0])
        Atr_, Btr_, ytr_, Ava_, Bva_, yva_, n_train = _prepare_train_val(
            train_name, val_name, rep,
            backbone=backbone, layer=layer,
            train_subsample=train_subsample, val_frac=val_frac, seed=seed,
        )
        if len(group_evals) > 1:
            print(
                f"[fit] {model}/{rep}/seed{seed} train={train_name} "
                f"→ {len(group_evals)} evals (train-once)",
                flush=True,
            )
        predict, cols = _fit_scorer(
            model, Atr_, Btr_, ytr_, Ava_, Bva_, yva_,
            pair_mode=pair_mode, top_k=top_k, seed=seed,
        )
        for ev in group_evals:
            results.append(
                _score_eval(
                    predict, ev, rep,
                    model=model, backbone=backbone, layer=layer,
                    train_name=train_name, pair_mode=pair_mode,
                    top_k=top_k, n_train=n_train, cols=cols,
                    seed=seed, write=write, out_dir=out_dir,
                )
            )
    return results


def run_baseline(
    model: str,
    rep: str,
    eval_name: str,
    *,
    top_k: int = 500,
    backbone: str = DEFAULT_BACKBONE,
    layer: Optional[int] = None,
    pair_mode: str = "sym",
    train_subsample: Optional[int] = 100000,
    val_frac: float = 0.1,
    seed: int = DEFAULT_SEED,
    out_dir: Path = OUT_DIR,
    write: bool = True,
) -> Dict[str, Any]:
    """Single-eval convenience wrapper around :func:`run_baseline_evals`."""
    return run_baseline_evals(
        model, rep, [eval_name],
        top_k=top_k, backbone=backbone, layer=layer, pair_mode=pair_mode,
        train_subsample=train_subsample, val_frac=val_frac, seed=seed,
        out_dir=out_dir, write=write,
    )[0]
