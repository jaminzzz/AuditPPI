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

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from conf.model import DEFAULT_SEED, ESMC_DIM, ESMC_SAE_DIM, SAE_BINARY_THRESHOLD
from src.ppi_fingerprint.config import REPRESENTATIONS

DIM = ESMC_SAE_DIM  # SAE codebook size (source: conf.model)
REPS = REPRESENTATIONS


def load_pooled_cache(path: Path) -> Dict:
    """Read a per-protein pooled cache → {seq2idx, esmc_mean, esmc_sae_max, esmc_sae_mean} (torch tensors)."""
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    return {k: d[k] for k in ("seq2idx", "esmc_mean", "esmc_sae_max", "esmc_sae_mean")}


def rep_dim(rep: str) -> int:
    return ESMC_DIM if rep == "esmc_mean" else DIM


def _rep_matrix(cache: Dict, rep: str):
    """The per-protein matrix for a rep (torch tensor; binary is bool)."""
    if rep == "binary":
        return cache["esmc_sae_max"] > SAE_BINARY_THRESHOLD
    if rep == "sae_max":
        return cache["esmc_sae_max"]
    if rep == "esmc_mean":
        return cache["esmc_mean"]
    raise ValueError(f"unknown rep {rep!r}; choose from {REPS}")


def assemble_pairs(bench, cache: Dict, rep: str):
    """Benchmark pairs → per-protein rep vectors. Maps id → sequence (``bench.seqs``) → ``seq2idx`` → row.

    Returns ``(A, B, y, kept_idx)`` with A,B float32 torch tensors (n_kept, rep_dim), y int64, and
    ``kept_idx`` the indices into ``bench.pairs`` that had both proteins cached (rest skipped)."""
    import torch

    seq2idx = cache["seq2idx"]
    mat = _rep_matrix(cache, rep)
    rows_a: List[int] = []
    rows_b: List[int] = []
    ys: List[int] = []
    kept: List[int] = []
    for i, ((a, b), y) in enumerate(zip(bench.pairs, bench.labels)):
        sa = bench.seqs.get(a)
        sb = bench.seqs.get(b)
        ja = seq2idx.get(sa) if sa is not None else None
        jb = seq2idx.get(sb) if sb is not None else None
        if ja is None or jb is None:
            continue
        rows_a.append(ja)
        rows_b.append(jb)
        ys.append(int(y))
        kept.append(i)
    if not kept:
        return None
    ia = torch.as_tensor(rows_a, dtype=torch.long)
    ib = torch.as_tensor(rows_b, dtype=torch.long)
    A = mat.index_select(0, ia).float()
    B = mat.index_select(0, ib).float()
    return A, B, np.asarray(ys, dtype=np.int64), kept


def sym_features(A, B, cols: Optional[np.ndarray] = None) -> np.ndarray:
    """``sym = [A⊙B, |A−B|]`` (float32, n × 2·rep_dim). If ``cols`` given, return only those columns."""
    prod = (A * B).numpy()
    diff = (A - B).abs().numpy()
    X = np.concatenate([prod, diff], axis=1).astype(np.float32, copy=False)
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
