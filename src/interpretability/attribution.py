"""Gradient-based attribution for endpoint-additive protein models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def endpoint_gradient_input_attribution(
    model,
    endpoint_a: torch.Tensor,
    endpoint_b: torch.Tensor,
    *,
    device: torch.device | str,
    batch_size: int,
    max_endpoints: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Aggregate ``x * d alpha(x)/dx`` over both endpoints of every pair."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = model.to(device)
    model.eval()
    feature_dim = int(endpoint_a.shape[1])
    endpoints = torch.cat([endpoint_a, endpoint_b], dim=0)
    n_endpoints = int(endpoints.shape[0])
    if max_endpoints > 0 and max_endpoints < n_endpoints:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(n_endpoints, size=max_endpoints, replace=False))
        endpoints = endpoints.index_select(0, torch.as_tensor(chosen, dtype=torch.long))
        n_endpoints = int(endpoints.shape[0])
    if n_endpoints == 0:
        raise ValueError("attribution requires at least one endpoint")

    signed_sum = torch.zeros(feature_dim, dtype=torch.float64)
    absolute_sum = torch.zeros(feature_dim, dtype=torch.float64)
    active_count = torch.zeros(feature_dim, dtype=torch.float64)
    alpha_sum = 0.0
    for start in range(0, n_endpoints, batch_size):
        inputs = endpoints[start : start + batch_size].to(device).float()
        inputs.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        endpoint_score = model.alpha(inputs)
        endpoint_score.sum().backward()
        attribution = (inputs.grad * inputs).detach().cpu().double()
        signed_sum += attribution.sum(dim=0)
        absolute_sum += attribution.abs().sum(dim=0)
        active_count += (inputs.detach().cpu() != 0).double().sum(dim=0)
        alpha_sum += float(endpoint_score.detach().sum().cpu())

    frame = pd.DataFrame(
        {
            "feature_id": np.arange(feature_dim, dtype=int),
            "mean_signed_attr": (signed_sum / n_endpoints).numpy(),
            "mean_abs_attr": (absolute_sum / n_endpoints).numpy(),
            "active_fraction": (active_count / n_endpoints).numpy(),
        }
    )
    frame["mean_endpoint_alpha"] = alpha_sum / n_endpoints
    return frame


__all__ = ["endpoint_gradient_input_attribution"]
