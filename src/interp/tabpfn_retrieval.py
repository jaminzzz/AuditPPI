"""TabPFN retrieval and SAE-feature-overlap interpretation utilities."""

from __future__ import annotations

import math
import typing
from pathlib import Path

import numpy as np

from src.interp.pair_probe import load_pair_embedding_split


def load_split_with_indices(path: Path, max_rows: int | None, seed: int):
    """Load an embedding split and return kept original row indices.

    Thin wrapper over
    :func:`src.interp.pair_probe.load_pair_embedding_split` with
    ``return_indices=True`` so retrieval audits can map subsampled rows back to
    the source CSV.
    """
    return load_pair_embedding_split(path, max_rows, seed, return_indices=True)


def read_split_csv(c3_dir: Path, split: str):
    import pandas as pd

    return pd.read_csv(c3_dir / f"c3.{split}.csv")


def short_sequence(sequence: str, n: int = 24) -> str:
    sequence = str(sequence)
    return sequence if len(sequence) <= n * 2 + 3 else f"{sequence[:n]}...{sequence[-n:]}"


def embeddings_with_configs(model, matrix: np.ndarray, data_source: str):
    """Return TabPFN ensemble embeddings together with their configs."""
    import torch
    from tabpfn.base import fix_dtypes
    from tabpfn.validation import ensure_compatible_predict_input_sklearn

    selected_data = {"train": "train_embeddings", "test": "test_embeddings"}[data_source]
    matrix = ensure_compatible_predict_input_sklearn(matrix, model)
    matrix = fix_dtypes(matrix, cat_indices=model.categorical_features_indices)
    matrix = model.ordinal_encoder_.transform(matrix)
    embeddings = []
    configs = []
    for output, config in model.executor_.iter_outputs(
        matrix,
        autocast=model.use_autocast_,
        task_type="multiclass",
        only_return_standard_out=False,
    ):
        output_dict = typing.cast("dict[str, torch.Tensor]", output)
        embedding = output_dict[selected_data].squeeze(1)
        embeddings.append(embedding.squeeze().cpu().numpy())
        configs.append(config)
    return np.array(embeddings), configs


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), eps)


def decoder_attention_weights(
    model,
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    configs,
) -> np.ndarray | None:
    """Average TabPFN-3 decoder attention over ensemble members."""
    import torch

    if train_embeddings.ndim != 3 or query_embeddings.ndim != 3:
        return None
    if train_embeddings.shape[0] != query_embeddings.shape[0]:
        return None
    device = model.executor_.get_devices()[0]
    accumulated = None
    used = 0
    with torch.inference_mode():
        for estimator_index, config in enumerate(configs):
            model_index = getattr(config, "_model_index", 0)
            architecture = model.executor_.model_caches[model_index].get(device)
            decoder = getattr(architecture, "many_class_decoder", None)
            if decoder is None:
                return None
            query_input = torch.as_tensor(
                query_embeddings[estimator_index], dtype=torch.float32, device=device
            )
            key_input = torch.as_tensor(
                train_embeddings[estimator_index], dtype=torch.float32, device=device
            )
            query = decoder.q_projection(query_input).view(
                query_input.shape[0], decoder.num_heads, decoder.head_dim
            )
            key = decoder.k_projection(key_input).view(
                key_input.shape[0], decoder.num_heads, decoder.head_dim
            )
            if decoder.softmax_scaling_layer is not None:
                query = decoder.softmax_scaling_layer(query.unsqueeze(0), key.shape[0]).squeeze(0)
            scores = torch.einsum("mhd,nhd->mnh", query, key) / math.sqrt(
                float(decoder.head_dim)
            )
            weights = torch.softmax(scores, dim=1).mean(dim=2).detach().cpu().numpy()
            accumulated = weights if accumulated is None else accumulated + weights
            used += 1
    return None if used == 0 else accumulated / used


def input_overlap_stats(
    query: np.ndarray,
    train: np.ndarray,
    neighbor_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_active = query > 0
    neighbor_active = train[neighbor_indices] > 0
    intersection = np.logical_and(neighbor_active, query_active[None, :]).sum(axis=1)
    union = np.logical_or(neighbor_active, query_active[None, :]).sum(axis=1)
    jaccard = intersection / np.maximum(union, 1)
    return intersection.astype(int), union.astype(int), jaccard.astype(float)


def top_shared_features(
    query: np.ndarray,
    neighbor: np.ndarray,
    feature_metadata: list[dict],
    limit: int = 12,
) -> str:
    shared = np.flatnonzero((query > 0) & (neighbor > 0))
    items = []
    for position in shared[:limit]:
        metadata = feature_metadata[int(position)]
        block = str(metadata["block"]).replace("AND(a*b)", "AND").replace("|a-b|", "XOR")
        items.append(f"{block}:{metadata['sae_feature']}")
    return ";".join(items)


def choose_queries(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    query_indices: list[int] | None,
    min_probability: float,
    prefer_true_positive: bool,
    num_queries: int,
) -> np.ndarray:
    if query_indices:
        return np.asarray(query_indices, dtype=np.int64)
    mask = probabilities >= min_probability
    if prefer_true_positive:
        mask &= labels == 1
    candidates = np.flatnonzero(mask)
    if len(candidates) < num_queries:
        candidates = np.arange(len(labels))
    order = candidates[np.argsort(probabilities[candidates])[::-1]]
    return order[:num_queries].astype(np.int64, copy=False)


__all__ = [
    "choose_queries",
    "decoder_attention_weights",
    "embeddings_with_configs",
    "input_overlap_stats",
    "l2_normalize",
    "load_split_with_indices",
    "read_split_csv",
    "short_sequence",
    "top_shared_features",
]
