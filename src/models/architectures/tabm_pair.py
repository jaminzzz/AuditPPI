"""TabM architecture wrapper for protein-pair feature matrices.

Pair assembly is the shared :func:`src.features.pairs.pair_features` vocabulary
used by :mod:`src.models.architectures.mlp_pair` -- same modes, same AB/BA rule
(only ``concat`` is order-sensitive). The sole axis between the two models is
classifier capacity: a single MLP vs a TabM ensemble (``k`` members).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from conf.model import DEFAULT_SEED
from src.features.pairs import PAIR_MODES, needs_abba, pair_features, pair_mode_dim


class TabMPair(nn.Module):
    """TabM ensemble over a shared pair-feature assemble."""

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
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}")
        from tabm import TabM

        self.feat_dim = int(feat_dim)
        self.pair_mode = pair_mode
        self.k = int(k)
        in_dim = pair_mode_dim(self.feat_dim, pair_mode)
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
        x = pair_features(a, b, self.pair_mode)
        return self.norm(x) if self.norm is not None else x

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits = self.tabm(x_num=self.assemble(a, b), x_cat=None)
        return logits.mean(dim=1).squeeze(-1)

    def member_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-ensemble-member logits ``(batch, k)`` -- the training target.

        TabM is an efficient MLP ensemble: one forward yields ``k`` member
        predictions. The correct objective trains *every* member against the
        label (mean-then-loss would collapse the ensemble), so training reads
        this while :meth:`forward` (mean-over-members) is used for prediction.
        """
        return self.tabm(x_num=self.assemble(a, b), x_cat=None).squeeze(-1)

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


class TabMPairPredictor:
    """Trained TabM pair model exposing ``predict_proba_pairs``.

    For ``concat`` the score is the mean of the AB and BA sigmoid outputs; every
    other mode (including ``rich``) is a single forward pass.
    """

    def __init__(self, model: TabMPair, device):
        self.model = model
        self.device = device
        self.pair_mode = model.pair_mode
        self._abba = needs_abba(model.pair_mode)

    def predict_proba_pairs(self, a, b, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        output = []
        with torch.no_grad():
            for start in range(0, a.shape[0], batch_size):
                aa = a[start : start + batch_size].to(self.device)
                bb = b[start : start + batch_size].to(self.device)
                if self._abba:
                    # forward() averages over ensemble members; symmetrise AB/BA.
                    p_ab = torch.sigmoid(self.model(aa, bb))
                    p_ba = torch.sigmoid(self.model(bb, aa))
                    prob = 0.5 * (p_ab + p_ba)
                else:
                    prob = torch.sigmoid(self.model(aa, bb))
                output.append(prob.float().cpu().numpy())
        return np.concatenate(output)


def train_tabm_pair(
    Atr,
    Btr,
    ytr,
    Ava,
    Bva,
    yva,
    *,
    pair_mode: str = "sym",
    k: int = 32,
    n_blocks: int = 2,
    d_block: int = 512,
    dropout: float = 0.1,
    arch_type: str = "tabm",
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 20,
    patience: int = 10,
    input_norm: bool = False,
    seed: int = DEFAULT_SEED,
    device: Optional[str] = None,
) -> TabMPairPredictor:
    """Train TabM-pair on a shared pair-feature mode.

    ``pair_mode="concat"`` doubles each training pair into ``(A,B)`` and ``(B,A)``
    (shared label) and early-stops / predicts under AB/BA-averaged probabilities.
    Loss is computed per ensemble member. Default ``pair_mode`` is ``sym`` so the
    fingerprint baseline matches ``mlp_pair`` / xgb's primary representation
    unless the caller opts into ``concat``. ``arch_type`` is the upstream TabM
    library knob (default ``"tabm"``), not the fingerprint model name.
    """
    from sklearn.metrics import roc_auc_score

    if pair_mode not in PAIR_MODES:
        raise ValueError(f"pair_mode must be one of {PAIR_MODES}")

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TabMPair(
        int(Atr.shape[1]),
        pair_mode=pair_mode,
        k=k,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        arch_type=arch_type,
        input_norm=input_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.7, patience=5
    )
    loss_fn = nn.BCEWithLogitsLoss()

    if needs_abba(pair_mode):
        # AB/BA augmentation at the pair level (concat's assemble does the ordered
        # cat, so augment the raw endpoints -- not the assembled features).
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
            logits = model.member_logits(a, b)  # (batch, k)
            loss = loss_fn(logits, y.unsqueeze(1).expand_as(logits))
            loss.backward()
            optimizer.step()

        predictor = TabMPairPredictor(model, device)
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
    return TabMPairPredictor(model, device)


__all__ = [
    "TabMPair",
    "TabMPairPredictor",
    "train_tabm_pair",
]
