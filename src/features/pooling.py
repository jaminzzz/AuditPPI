"""Residue-to-protein pooling shared by all backbone extractors."""

from __future__ import annotations

from typing import Callable

import torch


@torch.no_grad()
def dense_mean_max(residue_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise mean and max over an ``(L, d)`` residue representation."""
    if residue_states.ndim != 2 or residue_states.shape[0] == 0:
        raise ValueError(f"expected non-empty (L,d) residue tensor, got {tuple(residue_states.shape)}")
    states = residue_states.float()
    return states.mean(dim=0), states.max(dim=0).values


@torch.no_grad()
def pool_esmc_topk_sae(
    residue_states: torch.Tensor,
    *,
    w_enc: torch.Tensor,
    b_dec: torch.Tensor,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the Biohub ESM-C TopK SAE without materializing ``L x 16384``.

    Returns ``(sae_max, sae_binary)``. SAE mean-pooling is intentionally omitted:
    the audit uses max (continuous fingerprint) and binary (active-feature set).
    """
    n_tokens = int(residue_states.shape[0])
    if n_tokens == 0:
        raise ValueError("cannot pool an empty protein")
    dim = int(w_enc.shape[1])
    pooled_max = torch.zeros(dim, device=residue_states.device, dtype=torch.float32)
    for start in range(0, n_tokens, max(1, chunk_size)):
        h = residue_states[start : start + chunk_size].float()
        h = h - h.mean(dim=-1, keepdim=True)
        h = h / (h.std(dim=-1, keepdim=True) + 1e-5)
        pre = torch.relu((h - b_dec) @ w_enc)
        values, indices = pre.topk(k, dim=-1)
        flat_idx = indices.reshape(-1)
        flat_val = values.reshape(-1)
        pooled_max.scatter_reduce_(0, flat_idx, flat_val, reduce="amax", include_self=True)
    return pooled_max, pooled_max > 0


@torch.no_grad()
def pool_relu_sae(
    residue_states: torch.Tensor,
    *,
    encode: Callable[[torch.Tensor], torch.Tensor],
    feature_dim: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool a dense non-negative ReLU-SAE representation in residue chunks.

    Returns ``(sae_max, sae_binary)`` only (no SAE mean).
    """
    n_tokens = int(residue_states.shape[0])
    if n_tokens == 0:
        raise ValueError("cannot pool an empty protein")
    pooled_max = torch.zeros(feature_dim, device=residue_states.device, dtype=torch.float32)
    for start in range(0, n_tokens, max(1, chunk_size)):
        features = encode(residue_states[start : start + chunk_size].float()).float()
        pooled_max = torch.maximum(pooled_max, features.max(dim=0).values)
    return pooled_max, pooled_max > 0
