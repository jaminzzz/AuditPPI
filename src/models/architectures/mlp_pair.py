"""Plain MLP pair classifier: assemble endpoints once, then a feed-forward head.

Pair assembly is the shared :func:`src.features.pairs.pair_features` vocabulary
(``sym`` / ``concat`` / ``rich`` / ``product`` / ``absdiff`` / ``sum``); only
``concat`` uses the AB/BA train/eval protocol (:func:`~src.features.pairs.needs_abba`).
``tabm_pair`` consumes the same modes so the only axis between the two models is
classifier capacity (single MLP vs TabM ensemble).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from conf.model import DEFAULT_SEED
from src.features.pairs import PAIR_MODES, needs_abba, pair_features, pair_mode_dim


def make_mlp(
    in_dim: int,
    *,
    hidden: int = 512,
    n_layers: int = 2,
    dropout: float = 0.1,
    input_norm: bool = False,
) -> nn.Sequential:
    """``BatchNorm? → (Linear→ReLU→Dropout)×n_layers → Linear→1``."""
    layers: list[nn.Module] = []
    if input_norm:
        layers.append(nn.BatchNorm1d(in_dim))
    dim = in_dim
    for _ in range(max(0, n_layers)):
        layers += [nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        dim = hidden
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


class MLPPair(nn.Module):
    """Endpoint assemble → MLP → logit."""

    def __init__(
        self,
        feat_dim: int,
        *,
        pair_mode: str = "sym",
        hidden: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        input_norm: bool = False,
    ) -> None:
        super().__init__()
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}")
        self.feat_dim = int(feat_dim)
        self.pair_mode = pair_mode
        self.mlp = make_mlp(
            pair_mode_dim(self.feat_dim, pair_mode),
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            input_norm=input_norm,
        )

    def assemble(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return pair_features(a, b, self.pair_mode)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.assemble(a, b)).squeeze(-1)


class MLPPairPredictor:
    """Trained MLP pair model exposing ``predict_proba_pairs``.

    For ``concat`` the score is the mean of the AB and BA sigmoid outputs; every
    other mode is a single forward pass.
    """

    def __init__(self, model: MLPPair, device):
        self.model = model
        self.device = device
        self.pair_mode = model.pair_mode
        self._abba = needs_abba(model.pair_mode)

    def predict_proba_pairs(self, a, b, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        output = []
        with torch.no_grad():
            for start in range(0, a.shape[0], batch_size):
                aa = a[start : start + batch_size].to(self.device)
                bb = b[start : start + batch_size].to(self.device)
                if self._abba:
                    p_ab = torch.sigmoid(self.model(aa, bb))
                    p_ba = torch.sigmoid(self.model(bb, aa))
                    prob = 0.5 * (p_ab + p_ba)
                else:
                    prob = torch.sigmoid(self.model(aa, bb))
                output.append(prob.float().cpu().numpy())
        return np.concatenate(output)


def train_mlp_pair(
    Atr,
    Btr,
    ytr,
    Ava,
    Bva,
    yva,
    *,
    pair_mode: str = "sym",
    hidden: int = 512,
    n_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 20,
    patience: int = 10,
    input_norm: bool = False,
    seed: int = DEFAULT_SEED,
    device: Optional[str] = None,
) -> MLPPairPredictor:
    """Train the plain MLP pair model.

    ``pair_mode="concat"`` doubles each training pair into ``(A,B)`` and ``(B,A)``
    (shared label) and early-stops / predicts under AB/BA-averaged probabilities.
    All other modes train and score a single ordered forward pass.
    """
    from sklearn.metrics import roc_auc_score

    if pair_mode not in PAIR_MODES:
        raise ValueError(f"pair_mode must be one of {PAIR_MODES}")

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPair(
        int(Atr.shape[1]),
        pair_mode=pair_mode,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
        input_norm=input_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.7, patience=5
    )
    loss_fn = nn.BCEWithLogitsLoss()

    if needs_abba(pair_mode):
        Afit = torch.cat([Atr, Btr], dim=0)
        Bfit = torch.cat([Btr, Atr], dim=0)
        yfit = torch.as_tensor(np.concatenate([ytr, ytr]), dtype=torch.float32)
    else:
        Afit, Bfit = Atr, Btr
        yfit = torch.as_tensor(ytr, dtype=torch.float32)

    n = Afit.shape[0]
    best_auroc, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(seed)

    for _epoch in range(max_epochs):
        model.train()
        permutation = rng.permutation(n)
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            a = Afit[indices].to(device)
            b = Bfit[indices].to(device)
            y = yfit[indices].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(a, b), y)
            loss.backward()
            optimizer.step()

        predictor = MLPPairPredictor(model, device)
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
    return MLPPairPredictor(model, device)


__all__ = [
    "MLPPair",
    "MLPPairPredictor",
    "make_mlp",
    "train_mlp_pair",
]
