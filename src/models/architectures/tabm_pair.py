"""TabM architecture wrapper for protein-pair feature matrices."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

PAIR_MODE_MULTIPLIER = {
    "concat": 2,
    "sym": 2,
    "product": 1,
    "absdiff": 1,
    "rich": 4,
}


class TabMPair(nn.Module):
    """Reconstruct the TabM pair model used by the saved AuditPPI checkpoints."""

    def __init__(
        self,
        feat_dim: int,
        *,
        pair_mode: str = "sym",
        k: int = 32,
        n_blocks: int = 2,
        d_block: int = 512,
        dropout: float = 0.1,
        arch_type: str = "tabm",
        input_norm: bool = False,
    ) -> None:
        super().__init__()
        if pair_mode not in PAIR_MODE_MULTIPLIER:
            raise ValueError(f"pair_mode must be one of {tuple(PAIR_MODE_MULTIPLIER)}")
        from tabm import TabM

        self.feat_dim = int(feat_dim)
        self.pair_mode = pair_mode
        self.k = int(k)
        in_dim = PAIR_MODE_MULTIPLIER[pair_mode] * self.feat_dim
        self.norm = nn.BatchNorm1d(in_dim) if input_norm else None
        self.tabm = TabM.make(
            n_num_features=in_dim,
            cat_cardinalities=[],
            d_out=1,
            k=self.k,
            n_blocks=int(n_blocks),
            d_block=int(d_block),
            dropout=float(dropout),
            arch_type=arch_type,
        )

    def assemble(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.pair_mode == "concat":
            x = torch.cat([a, b], dim=1)
        elif self.pair_mode == "sym":
            x = torch.cat([a * b, (a - b).abs()], dim=1)
        elif self.pair_mode == "product":
            x = a * b
        elif self.pair_mode == "absdiff":
            x = (a - b).abs()
        elif self.pair_mode == "rich":
            x = torch.cat([a, b, a * b, (a - b).abs()], dim=1)
        else:
            raise ValueError(self.pair_mode)
        return self.norm(x) if self.norm is not None else x

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits = self.tabm(x_num=self.assemble(a, b), x_cat=None)
        return logits.mean(dim=1).squeeze(-1)

    @classmethod
    def from_hparams(cls, hparams: Mapping) -> "TabMPair":
        return cls(
            feat_dim=int(hparams["feat_dim"]),
            pair_mode=hparams["pair_mode"],
            k=int(hparams["k"]),
            n_blocks=int(hparams["n_blocks"]),
            d_block=int(hparams["d_block"]),
            dropout=float(hparams["dropout"]),
            arch_type=hparams["arch_type"],
            input_norm=bool(hparams.get("input_norm", False)),
        )


__all__ = ["PAIR_MODE_MULTIPLIER", "TabMPair"]
