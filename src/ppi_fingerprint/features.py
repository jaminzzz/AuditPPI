"""Pooled-SAE / ESM-C feature assembly for the ppi_fingerprint baseline.

Loads the per-protein **pooled** cache (``seq2idx`` + ``esmc_mean`` / ``esmc_sae_max`` / ``esmc_sae_mean``,
produced by the audit's ``cache_esmc_fingerprints.py``), turns each benchmark pair into per-protein vectors
at a chosen representation, and builds the symmetric pair features ``sym = [a⊙b, |a−b|]`` the audit used.
Top-k feature selection (TabPFN caps #features) is computed **self-contained** from XGBoost gain on the
train split — no dependency on the audit's ranking CSVs.

Representations (user-selected): ``binary`` = ``(esmc_sae_max > 0)`` (the participation channel),
``sae_max`` = continuous ``esmc_sae_max`` [16384], ``esmc_mean`` = raw ESM-C layer-60 mean [2560].
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from conf.model import DEFAULT_SEED, ESMC_SAE_DIM, REPRESENTATIONS
from src.features.protein_cache import load_pooled_cache, pair_feature_rows, rep_dim

DIM = ESMC_SAE_DIM  # SAE codebook size (source: conf.model)
REPS = REPRESENTATIONS

# The pooled-cache load, the representation->matrix switch, and the
# id/sequence->row assembly now live in ``src.features.protein_cache`` (shared
# with the participation predictors, which removes the old participation ->
# ppi_fingerprint back-dependency). ``load_pooled_cache`` / ``rep_dim`` are
# re-exported above so callers keep using ``FE.load_pooled_cache`` / ``FE.rep_dim``.


def assemble_pairs(bench, cache: Dict, rep: str):
    """Benchmark pairs → per-protein rep vectors (shared assembly).

    Thin wrapper over :func:`src.features.protein_cache.pair_feature_rows`; the
    id → sequence (``bench.seqs``) → ``seq2idx`` → row mapping is shared with the
    participation predictors. Returns ``(A, B, y, kept_idx)`` with A,B float32
    torch tensors (n_kept, rep_dim), y int64, and ``kept_idx`` the indices into
    ``bench.pairs`` that had both proteins cached (rest skipped); ``None`` when
    no pair had both endpoints cached.
    """
    return pair_feature_rows(bench, cache, rep)


def sym_features(A, B, cols: Optional[np.ndarray] = None) -> np.ndarray:
    """``sym = [A⊙B, |A−B|]`` (float32, n × 2·rep_dim). If ``cols`` given, return only those columns.

    Thin numpy adapter over the canonical torch ``pair_features(..., "sym")`` so the
    ``[A*B, abs(A-B)]`` definition lives in exactly one place (``src.features.pairs``).
    A/B arrive as float32 torch tensors, so this is bit-for-bit the old
    ``concatenate([(A*B), (A-B).abs()])``; the numpy cast + column select is the only
    ppi_fingerprint-specific bit (TabPFN's feature cap needs a fixed column subset).
    """
    from src.features.pairs import pair_features

    X = pair_features(A, B, "sym").numpy().astype(np.float32, copy=False)
    return X[:, cols] if cols is not None else X


def xgb_topk_columns(X: np.ndarray, y: np.ndarray, k: int, *, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Top-k sym columns by a quick XGBoost gain fit (self-contained; for TabPFN's feature cap)."""
    import xgboost as xgb

    try:
        import torch

        use_gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        use_gpu = False

    def _fit(device):
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=seed, device=device)
        clf.fit(X, y, verbose=False)
        return clf

    try:
        clf = _fit("cuda" if use_gpu else "cpu")
    except Exception:  # noqa: BLE001  (GPU OOM on a wide matrix → CPU)
        clf = _fit("cpu")
    imp = clf.feature_importances_
    k = min(k, X.shape[1])
    return np.argsort(imp)[::-1][:k].copy()


def stratified_subsample(y: np.ndarray, max_rows: Optional[int], seed: int) -> Optional[np.ndarray]:
    """Indices of a class-stratified subsample (None ⇒ keep all). Mirrors the audit's stratified_indices."""
    if max_rows is None or max_rows >= len(y):
        return None
    rng = np.random.default_rng(seed)
    chosen = []
    remaining = max_rows
    classes = np.unique(y)
    for i, cls in enumerate(classes):
        idx = np.flatnonzero(y == cls)
        n_take = remaining if i == len(classes) - 1 else min(
            int(round(max_rows * len(idx) / len(y))), len(idx), remaining)
        chosen.append(rng.choice(idx, size=n_take, replace=False))
        remaining -= n_take
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out
