"""Compact SAE pair-feature probes for analysis / interpretability scripts.

Owns ranking I/O, selected-column ``[A*B, |A-B|]`` materialization, embedding-split
loading, and thin probe fit wrappers. Classification metrics live in
:mod:`src.eval.classification`
(:func:`~src.eval.classification.probe_classification_metrics`,
:func:`~src.eval.classification.expected_calibration_error`).

Lives under :mod:`src.interp` (not :mod:`src.features`): these helpers
assemble and score already-extracted pair embeddings; they do not extract or
cache protein features.
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np

from conf.model import BACKBONE_SAE_DIM, DEFAULT_BACKBONE, ESMC_SAE_DIM
from src.eval.classification import probe_classification_metrics
from src.features.sampling import stratified_subsample
from src.models.estimators.tabpfn import fit_tabpfn as _fit_tabpfn
from src.models.estimators.tabpfn import predict_proba_chunked as _predict_proba_chunked
from src.models.estimators.xgboost import fit_xgb

SAE_DIM = ESMC_SAE_DIM  # ESM-C default; prefer sae_dim_for_backbone() / explicit sae_dim=
BLOCK_PRODUCT = "AND(a*b)"
BLOCK_ABSDIFF = "|a-b|"


def sae_dim_for_backbone(backbone: str = DEFAULT_BACKBONE) -> int:
    """Codebook width for the given backbone (ESM-C 16384 / ESM-2 10240)."""
    try:
        return int(BACKBONE_SAE_DIM[backbone])
    except KeyError as exc:
        raise ValueError(
            f"unknown backbone {backbone!r}; expected one of {tuple(BACKBONE_SAE_DIM)}"
        ) from exc


def _apply_subsample(
    emb_a,
    emb_b,
    labels_full: np.ndarray,
    max_rows: int | None,
    seed: int,
    *,
    return_indices: bool,
):
    """Shared class-stratified subsample tail for embedding / pair-index loaders."""
    import torch

    indices = stratified_subsample(labels_full, max_rows, seed)
    if indices is None:
        emb_a = emb_a.contiguous()
        emb_b = emb_b.contiguous()
        labels = labels_full
        kept = np.arange(len(labels_full), dtype=np.int64)
    else:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        emb_a = emb_a.index_select(0, tensor_indices).contiguous()
        emb_b = emb_b.index_select(0, tensor_indices).contiguous()
        labels = labels_full[indices]
        kept = indices.astype(np.int64, copy=False)
    if return_indices:
        return emb_a, emb_b, labels, kept
    return emb_a, emb_b, labels


def load_pair_embedding_split(
    path: Path,
    max_rows: int | None,
    seed: int,
    *,
    return_indices: bool = False,
):
    """Load endpoint tensors and labels from a legacy ``{split}_embeddings.pt`` dump.

    Prefer :func:`materialize_pair_split` for v1 pair-index + protein caches.
    When ``return_indices`` is true, also returns the kept original row indices
    (identity when no subsample is applied). Still used by
    :func:`src.interp.tabpfn_retrieval.load_split_with_indices`.
    """
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    labels_full = payload["label"].numpy().astype(np.int64, copy=False)
    return _apply_subsample(
        payload["emb_a"],
        payload["emb_b"],
        labels_full,
        max_rows,
        seed,
        return_indices=return_indices,
    )


def materialize_pair_split(
    index_cache_path: Path,
    protein_cache: dict,
    *,
    rep: str,
    backbone: str,
    layer: int | None,
    max_rows: int | None = None,
    seed: int = 0,
    return_indices: bool = False,
):
    """v1 analogue of :func:`load_pair_embedding_split`.

    Gathers endpoint rows from a lightweight ``auditppi_pair_index_v1`` cache +
    an ``auditppi_protein_features_v1`` protein cache (one channel picked by
    ``rep``/``backbone``/``layer``), then applies the same class-stratified
    subsample and returns the identical ``(emb_a, emb_b, labels[, kept])``
    contract the old per-rep ``{split}_embeddings.pt`` dumps did.
    """
    from src.features.pairs import load_pair_index_cache, materialize_pair_endpoints

    index_cache = load_pair_index_cache(index_cache_path)
    emb_a, emb_b, labels_t = materialize_pair_endpoints(
        index_cache, protein_cache, rep=rep, backbone=backbone, layer=layer
    )
    labels_full = labels_t.numpy().astype(np.int64, copy=False)
    return _apply_subsample(
        emb_a,
        emb_b,
        labels_full,
        max_rows,
        seed,
        return_indices=return_indices,
    )


def read_feature_ranking(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            row["rank"] = int(row["rank"])
            row["flat_feature"] = int(row["flat_feature"])
            row["sae_feature"] = int(row["sae_feature"])
            row["rank_score"] = float(row["rank_score"])
            rows.append(row)
    rows.sort(key=lambda row: row["rank"])
    return rows


def select_top_features(
    ranking: list[dict],
    top_k: int,
    mode: str,
    *,
    sae_dim: int = SAE_DIM,
) -> tuple[list[int], list[dict]]:
    """Select ranked flat columns or paired product/absdiff SAE columns."""
    if mode not in {"flat", "sae-id"}:
        raise ValueError("mode must be flat or sae-id")
    if mode == "flat":
        flat_features = [int(row["flat_feature"]) for row in ranking[:top_k]]
        metadata = []
        for flat_feature in flat_features:
            block = BLOCK_PRODUCT if flat_feature < sae_dim else BLOCK_ABSDIFF
            metadata.append(
                {
                    "flat_feature": flat_feature,
                    "sae_feature": flat_feature % sae_dim,
                    "block": block,
                }
            )
        return flat_features, metadata

    sae_features = []
    seen = set()
    for row in ranking:
        feature_id = int(row["sae_feature"])
        if feature_id in seen:
            continue
        seen.add(feature_id)
        sae_features.append(feature_id)
        if len(sae_features) == top_k:
            break
    flat_features = sae_features + [feature_id + sae_dim for feature_id in sae_features]
    metadata = [
        {"flat_feature": feature_id, "sae_feature": feature_id, "block": BLOCK_PRODUCT}
        for feature_id in sae_features
    ] + [
        {
            "flat_feature": feature_id + sae_dim,
            "sae_feature": feature_id,
            "block": BLOCK_ABSDIFF,
        }
        for feature_id in sae_features
    ]
    return flat_features, metadata


def build_dense_sym_topk(
    endpoint_a,
    endpoint_b,
    flat_features: list[int],
    *,
    sae_dim: int = SAE_DIM,
) -> np.ndarray:
    """Materialize selected columns from ``[A*B, abs(A-B)]``."""
    product_ids = [index for index in flat_features if index < sae_dim]
    absdiff_ids = [index - sae_dim for index in flat_features if index >= sae_dim]
    blocks = []
    if product_ids:
        a = endpoint_a[:, product_ids].numpy().astype(np.float32, copy=False)
        b = endpoint_b[:, product_ids].numpy().astype(np.float32, copy=False)
        blocks.append(a * b)
    if absdiff_ids:
        a = endpoint_a[:, absdiff_ids].numpy().astype(np.float32, copy=False)
        b = endpoint_b[:, absdiff_ids].numpy().astype(np.float32, copy=False)
        blocks.append(np.abs(a - b))
    if not blocks:
        return np.empty((endpoint_a.shape[0], 0), dtype=np.float32)
    grouped = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    grouped_order = product_ids + [feature_id + sae_dim for feature_id in absdiff_ids]
    if grouped_order == flat_features:
        return grouped
    positions = {feature_id: index for index, feature_id in enumerate(grouped_order)}
    return grouped[:, [positions[feature_id] for feature_id in flat_features]]


def predict_proba_chunked(model, matrix: np.ndarray, batch_size: int) -> np.ndarray:
    """Positive-class probabilities with progress prints for long probes.

    Delegates the actual ``predict_proba`` call to
    :func:`src.models.estimators.tabpfn.predict_proba_chunked` (``batch_size=0``
    means "no further chunking") and only owns the progress reporting the
    analysis scripts rely on.
    """
    if batch_size <= 0 or matrix.shape[0] <= batch_size:
        return _predict_proba_chunked(model, matrix, batch_size=0)
    chunks = []
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        chunks.append(_predict_proba_chunked(model, matrix[start:end], batch_size=0))
        print(f"    predict rows {end}/{matrix.shape[0]}", flush=True)
    return np.concatenate(chunks)


def fit_tabpfn_probe(
    train_x,
    train_y,
    *,
    n_estimators: int,
    subsample_samples: int,
    ignore_limits: bool,
    seed: int,
    device: str = "cuda",
):
    return _fit_tabpfn(
        train_x,
        train_y,
        n_estimators=n_estimators,
        subsample_samples=subsample_samples,
        ignore_limits=ignore_limits,
        seed=seed,
        device=device,
    )


def fit_xgb_probe(
    train_x,
    train_y,
    val_x,
    val_y,
    *,
    trees: int,
    depth: int,
    learning_rate: float,
    seed: int,
    cpu: bool,
):
    """Fit the analysis XGB probe via the shared pair-classifier factory."""
    return fit_xgb(
        train_x,
        train_y,
        val_x,
        val_y,
        trees=trees,
        depth=depth,
        lr=learning_rate,
        seed=seed,
        cpu=cpu,
        verbose=50,
    )


def fit_logistic_probe(train_x, train_y, *, seed: int):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=seed)
    model.fit(train_x, train_y)
    return model


def _score_pair_endpoints(
    model,
    endpoint_a,
    endpoint_b,
    labels: np.ndarray,
    flat_features: list[int],
    *,
    predict_batch_size: int,
    sae_dim: int = SAE_DIM,
) -> dict:
    """Build top-k sym features, score, and free intermediates (shared by evaluate_*)."""
    matrix = build_dense_sym_topk(
        endpoint_a, endpoint_b, flat_features, sae_dim=sae_dim
    )
    del endpoint_a, endpoint_b
    gc.collect()
    probabilities = predict_proba_chunked(model, matrix, predict_batch_size)
    output = {
        "n": int(len(labels)),
        "pos_rate": float(labels.mean()),
        **probe_classification_metrics(labels, probabilities),
    }
    del matrix, labels, probabilities
    gc.collect()
    return output


def evaluate_species(
    model,
    species: str,
    embedding_dir: Path,
    flat_features: list[int],
    *,
    test_subsample: int | None,
    seed: int,
    predict_batch_size: int,
    sae_dim: int = SAE_DIM,
) -> dict:
    """Legacy path: per-species ``{species}_embeddings.pt``. Prefer :func:`evaluate_species_v1`."""
    endpoint_a, endpoint_b, labels = load_pair_embedding_split(
        embedding_dir / f"{species}_embeddings.pt", test_subsample, seed + 17
    )
    return _score_pair_endpoints(
        model,
        endpoint_a,
        endpoint_b,
        labels,
        flat_features,
        predict_batch_size=predict_batch_size,
        sae_dim=sae_dim,
    )


def compute_sym_shap_ranking(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    sae_dim: int,
    trees: int = 1000,
    depth: int = 4,
    lr: float = 0.05,
    seed: int = 0,
    cpu: bool = False,
) -> list[dict]:
    """Fit XGB on full sym features, rank every column by mean|TreeSHAP| on val.

    Shared by the C-level and cross-species Top-K runners so both derive their
    own ranking with identical semantics. ``train_x``/``val_x`` are the full
    ``[A*B, |A-B|]`` matrices (``2 * sae_dim`` columns). Returns backup-format
    rows sorted by descending mean|shap|::

        rank, flat_feature, sae_feature, block, rank_score, xgb_importance,
        mean_abs_shap, mean_signed_shap
    """
    import xgboost as xgb

    feature_dim = train_x.shape[1]  # 2 * sae_dim
    clf = fit_xgb(
        train_x, train_y, val_x, val_y,
        trees=trees, depth=depth, lr=lr, seed=seed, cpu=cpu, verbose=50,
    )
    booster = clf.get_booster()

    # TreeSHAP on val. pred_contribs returns (n, feature_dim + 1); last col is the
    # bias term. booster was re-homed to CPU by fit_xgb, so this runs on CPU.
    dval = xgb.DMatrix(val_x)
    contribs = booster.predict(dval, pred_contribs=True)
    shap = contribs[:, :feature_dim]  # drop bias column
    mean_abs = np.abs(shap).mean(axis=0)
    mean_signed = shap.mean(axis=0)
    del contribs, shap, dval
    gc.collect()

    # XGB gain per flat feature (booster keys like "f123").
    gain = np.zeros(feature_dim, dtype=np.float64)
    for key, value in booster.get_score(importance_type="gain").items():
        if key.startswith("f"):
            idx = int(key[1:])
            if 0 <= idx < feature_dim:
                gain[idx] = float(value)

    order = np.argsort(mean_abs)[::-1]
    rows: list[dict] = []
    for rank, flat in enumerate(order, start=1):
        flat = int(flat)
        block = BLOCK_PRODUCT if flat < sae_dim else BLOCK_ABSDIFF
        rows.append({
            "rank": rank,
            "flat_feature": flat,
            "sae_feature": flat % sae_dim,
            "block": block,
            "rank_score": float(mean_abs[flat]),
            "xgb_importance": float(gain[flat]),
            "mean_abs_shap": float(mean_abs[flat]),
            "mean_signed_shap": float(mean_signed[flat]),
        })
    return rows


def write_feature_ranking(path: Path, rows: list[dict], split_tag: str) -> None:
    """Write a ranking in the backup column layout (split-tagged shap columns)."""
    fields = [
        "rank", "flat_feature", "sae_feature", "block", "rank_score",
        "xgb_importance",
        f"{split_tag}_mean_abs_shap", f"{split_tag}_mean_signed_shap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for r in rows:
            writer.writerow([
                r["rank"], r["flat_feature"], r["sae_feature"], r["block"],
                r["rank_score"], r["xgb_importance"],
                r["mean_abs_shap"], r["mean_signed_shap"],
            ])


def evaluate_species_v1(
    model,
    index_cache_path: Path,
    protein_cache: dict,
    flat_features: list[int],
    *,
    rep: str,
    backbone: str,
    layer: int | None,
    test_subsample: int | None,
    seed: int,
    predict_batch_size: int,
    sae_dim: int | None = None,
) -> dict:
    """v1 analogue of :func:`evaluate_species`.

    Scores a held-out species graph from its pair-index cache + the shared
    cross-species protein cache instead of a per-species ``{species}_embeddings.pt``.
    """
    if sae_dim is None:
        sae_dim = sae_dim_for_backbone(backbone)
    endpoint_a, endpoint_b, labels = materialize_pair_split(
        index_cache_path,
        protein_cache,
        rep=rep,
        backbone=backbone,
        layer=layer,
        max_rows=test_subsample,
        seed=seed + 17,
    )
    return _score_pair_endpoints(
        model,
        endpoint_a,
        endpoint_b,
        labels,
        flat_features,
        predict_batch_size=predict_batch_size,
        sae_dim=sae_dim,
    )


__all__ = [
    "BLOCK_ABSDIFF",
    "BLOCK_PRODUCT",
    "SAE_DIM",
    "build_dense_sym_topk",
    "compute_sym_shap_ranking",
    "evaluate_species",
    "evaluate_species_v1",
    "fit_logistic_probe",
    "fit_tabpfn_probe",
    "fit_xgb_probe",
    "load_pair_embedding_split",
    "materialize_pair_split",
    "predict_proba_chunked",
    "read_feature_ranking",
    "sae_dim_for_backbone",
    "select_top_features",
    "write_feature_ranking",
]
