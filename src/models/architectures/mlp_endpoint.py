"""No-bias endpoint-additive MLP used by the C3 and PRING audits."""

from __future__ import annotations

import torch


class MLPEndpoint(torch.nn.Module):
    """Compute ``logit(A,B) = alpha(A) + alpha(B)`` with a shared MLP."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        *,
        layer_bias: bool = False,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        blocks: list[torch.nn.Module] = []
        in_dim = dim
        for _ in range(layers):
            blocks.append(torch.nn.Linear(in_dim, hidden, bias=layer_bias))
            blocks.append(torch.nn.ReLU())
            if dropout > 0:
                blocks.append(torch.nn.Dropout(dropout))
            in_dim = hidden
        blocks.append(torch.nn.Linear(in_dim, 1, bias=layer_bias))
        self.endpoint = torch.nn.Sequential(*blocks)

    def alpha(self, x: torch.Tensor) -> torch.Tensor:
        return self.endpoint(x.float()).squeeze(-1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.alpha(a) + self.alpha(b)


__all__ = ["MLPEndpoint"]
