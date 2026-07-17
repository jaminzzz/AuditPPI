"""Siamese dual-tower MLP for protein-pair classification.

This is the organized implementation of the model historically used by the
PPI fingerprint baseline. Defaults and training behavior are kept identical
for backward-compatible experiments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FUSE_MULTIPLIER = {"concat": 2, "hadamard": 1, "sym": 2, "sumprod": 2}
FUSE_MODES = (*FUSE_MULTIPLIER, "cosine")


def make_tower(feat_dim, hidden, out_dim, n_layers, dropout, input_norm):
    layers = [nn.BatchNorm1d(feat_dim)] if input_norm else []
    dim = feat_dim
    for _ in range(max(0, n_layers)):
        layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class DualTowerNet(nn.Module):
    """Shared protein tower followed by a configurable pair-fusion head."""

    def __init__(
        self,
        feat_dim: int,
        *,
        fuse: str = "sym",
        tower_dim: int = 512,
        tower_hidden: int = 512,
        tower_layers: int = 2,
        head_dim: int = 256,
        dropout: float = 0.1,
        input_norm: bool = False,
    ) -> None:
        super().__init__()
        if fuse not in FUSE_MODES:
            raise ValueError(f"fuse must be one of {FUSE_MODES}")
        self.fuse = fuse
        self.tower = make_tower(
            feat_dim, tower_hidden, tower_dim, tower_layers, dropout, input_norm
        )
        if fuse == "cosine":
            self.head = None
            self.cos_scale = nn.Parameter(torch.tensor(10.0))
            self.cos_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.head = nn.Sequential(
                nn.Linear(FUSE_MULTIPLIER[fuse] * tower_dim, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x)

    def fuse_embeddings(self, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
        if self.fuse == "cosine":
            return F.cosine_similarity(za, zb, dim=1)
        if self.fuse == "concat":
            return torch.cat([za, zb], dim=1)
        if self.fuse == "hadamard":
            return za * zb
        if self.fuse == "sym":
            return torch.cat([za * zb, (za - zb).abs()], dim=1)
        if self.fuse == "sumprod":
            return torch.cat([za + zb, za * zb], dim=1)
        raise ValueError(self.fuse)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        embedded = self.tower(torch.cat([a, b], dim=0))
        za, zb = embedded[: a.size(0)], embedded[a.size(0) :]
        fused = self.fuse_embeddings(za, zb)
        if self.fuse == "cosine":
            return self.cos_scale * fused + self.cos_bias
        assert self.head is not None
        return self.head(fused).squeeze(1)


class DualTowerPredictor:
    """Trained network wrapper exposing ``predict_proba_pairs``."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_proba_pairs(self, a, b, batch_size: int = 4096) -> np.ndarray:
        import torch

        self.model.eval()
        output = []
        with torch.no_grad():
            for start in range(0, a.shape[0], batch_size):
                aa = a[start : start + batch_size].to(self.device)
                bb = b[start : start + batch_size].to(self.device)
                output.append(torch.sigmoid(self.model(aa, bb)).float().cpu().numpy())
        return np.concatenate(output)


def train_dual_tower(
    Atr,
    Btr,
    ytr,
    Ava,
    Bva,
    yva,
    *,
    fuse: str = "sym",
    tower_dim: int = 512,
    tower_hidden: int = 512,
    tower_layers: int = 2,
    head_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 20,
    patience: int = 10,
    input_norm: bool = False,
    seed: int = 42,
    device: Optional[str] = None,
) -> DualTowerPredictor:
    """Train with the historical AuditPPI defaults and early stopping."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DualTowerNet(
        int(Atr.shape[1]),
        fuse=fuse,
        tower_dim=tower_dim,
        tower_hidden=tower_hidden,
        tower_layers=tower_layers,
        head_dim=head_dim,
        dropout=dropout,
        input_norm=input_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.7, patience=5
    )
    loss_fn = nn.BCEWithLogitsLoss()
    targets = torch.as_tensor(ytr, dtype=torch.float32)
    n = Atr.shape[0]
    best_auroc, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(seed)

    for _epoch in range(max_epochs):
        model.train()
        permutation = rng.permutation(n)
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            a = Atr[indices].to(device)
            b = Btr[indices].to(device)
            y = targets[indices].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(a, b), y)
            loss.backward()
            optimizer.step()

        predictor = DualTowerPredictor(model, device)
        val_prob = predictor.predict_proba_pairs(Ava, Bva)
        val_auroc = float(roc_auc_score(yva, val_prob)) if len(np.unique(yva)) > 1 else 0.0
        scheduler.step(val_auroc)
        if val_auroc > best_auroc:
            best_auroc, bad = val_auroc, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return DualTowerPredictor(model, device)


# Historical private name used by a few notebooks/scripts.
_DualTower = DualTowerPredictor

__all__ = [
    "DualTowerNet",
    "DualTowerPredictor",
    "FUSE_MODES",
    "train_dual_tower",
]
