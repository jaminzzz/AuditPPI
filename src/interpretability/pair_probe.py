"""Compact SAE pair-feature probes for analysis / interpretability scripts.

Owns ranking I/O, selected-column ``[A*B, |A-B|]`` materialization, embedding-split
loading, and thin probe fit wrappers. Classification metrics live in
:mod:`src.eval.classification`
(:func:`~src.eval.classification.probe_classification_metrics`,
:func:`~src.eval.classification.expected_calibration_error`).

Lives under :mod:`src.interpretability` (not :mod:`src.features`): these helpers
assemble and score already-extracted pair embeddings; they do not extract or
cache protein features.
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np

from conf.model import ESMC_SAE_DIM
from src.eval.classification import probe_classification_metrics
from src.features.sampling import stratified_subsample
from src.models.estimators.tabpfn import fit_tabpfn as _fit_tabpfn
from src.models.estimators.tabpfn import predict_proba_chunked as _predict_proba_chunked
from src.models.estimators.xgboost import fit_xgb

SAE_DIM = ESMC_SAE_DIM  # SAE codebook size; kept as a module alias for back-compat
BLOCK_PRODUCT = "AND(a*b)"
BLOCK_ABSDIFF = "|a-b|"


def load_pair_embedding_split(
    path: Path,
    max_rows: int | None,
    seed: int,
    *,
    return_indices: bool = False,
):
    """Load endpoint tensors and labels from an AuditPPI embedding split.

    When ``return_indices`` is true, also returns the kept original row indices
    (identity when no subsample is applied). Shared by probe scripts and
    :func:`src.interpretability.tabpfn_retrieval.load_split_with_indices`.
    """
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    labels_full = payload["label"].numpy().astype(np.int64, copy=False)
    indices = stratified_subsample(labels_full, max_rows, seed)
    if indices is None:
        emb_a = payload["emb_a"].contiguous()
        emb_b = payload["emb_b"].contiguous()
        labels = labels_full
        kept = np.arange(len(labels_full), dtype=np.int64)
    else:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        emb_a = payload["emb_a"].index_select(0, tensor_indices).contiguous()
        emb_b = payload["emb_b"].index_select(0, tensor_indices).contiguous()
        labels = labels_full[indices]
        kept = indices.astype(np.int64, copy=False)
    if return_indices:
        return emb_a, emb_b, labels, kept
    return emb_a, emb_b, labels


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
    import torch

    from src.features.pairs import load_pair_index_cache, materialize_pair_endpoints

    index_cache = load_pair_index_cache(index_cache_path)
    emb_a, emb_b, labels_t = materialize_pair_endpoints(
        index_cache, protein_cache, rep=rep, backbone=backbone, layer=layer
    )
    labels_full = labels_t.numpy().astype(np.int64, copy=False)
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


def evaluate_species(
    model,
    species: str,
    embedding_dir: Path,
    flat_features: list[int],
    *,
    test_subsample: int | None,
    seed: int,
    predict_batch_size: int,
) -> dict:
    endpoint_a, endpoint_b, labels = load_pair_embedding_split(
        embedding_dir / f"{species}_embeddings.pt", test_subsample, seed + 17
    )
    matrix = build_dense_sym_topk(endpoint_a, endpoint_b, flat_features)
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
) -> dict:
    """v1 analogue of :func:`evaluate_species`.

    Scores a held-out species graph from its pair-index cache + the shared
    cross-species protein cache instead of a per-species ``{species}_embeddings.pt``.
    """
    endpoint_a, endpoint_b, labels = materialize_pair_split(
        index_cache_path,
        protein_cache,
        rep=rep,
        backbone=backbone,
        layer=layer,
        max_rows=test_subsample,
        seed=seed + 17,
    )
    matrix = build_dense_sym_topk(endpoint_a, endpoint_b, flat_features)
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


__all__ = [
    "BLOCK_ABSDIFF",
    "BLOCK_PRODUCT",
    "SAE_DIM",
    "build_dense_sym_topk",
    "evaluate_species",
    "evaluate_species_v1",
    "fit_logistic_probe",
    "fit_tabpfn_probe",
    "fit_xgb_probe",
    "load_pair_embedding_split",
    "materialize_pair_split",
    "predict_proba_chunked",
    "read_feature_ranking",
    "select_top_features",
]
