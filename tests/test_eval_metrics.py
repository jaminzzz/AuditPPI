"""Regression guards for the shared endpoint-additive audit primitives.

``pair_score_metrics`` and ``seed_all`` were extracted verbatim from the C3 /
PRING endpoint-additive audit scripts (EBM / MLP), which all produced identical
copies. The scripts write ``pair_score_metrics``'s output straight to the
on-disk JSON that backs the manuscript numbers, so both the *field set* and the
*values* are a frozen contract. The golden values below were captured from the
original in-script implementations before extraction; if a change here shifts
them, the audit JSON would silently diverge from the published figures.
"""

import numpy as np

from src.eval.metrics import pair_score_metrics
from src.runtime import seed_all

# Fixed inputs: labels/scores drawn under a pinned seed (captured pre-extraction).
_RNG = np.random.default_rng(0)
_Y = (_RNG.random(2000) < 0.5).astype(np.int8)
_P = _RNG.random(2000).astype(np.float64)

# Golden output of the original in-script metrics() on (_Y, _P), recomputed from
# the verbatim pre-extraction formula on these exact pinned inputs. (The extracted
# pair_score_metrics was separately shown byte-identical to that formula.)
_GOLDEN = {
    "n": 2000,
    "pos_rate": 0.5035,
    "auroc": 0.4846297468575961,
    "auprc": 0.49749672204838374,
    "accuracy_at_0.5": 0.4915,
    "brier": 0.3391167841179201,
    "score_mean": 0.49432255018240007,
    "score_std": 0.28568243893258743,
}


def test_pair_score_metrics_field_set_is_frozen():
    """The JSON contract's keys must not drift."""
    assert set(pair_score_metrics(_Y, _P)) == set(_GOLDEN)


def test_pair_score_metrics_values_match_golden():
    """Values must match the pre-extraction in-script implementation exactly."""
    out = pair_score_metrics(_Y, _P)
    assert out["n"] == _GOLDEN["n"]
    for key in ("pos_rate", "auroc", "auprc", "accuracy_at_0.5", "brier",
                "score_mean", "score_std"):
        assert out[key] == _GOLDEN[key], key


def test_seed_all_is_deterministic_across_backends():
    """seed_all must seed random/numpy/torch reproducibly (captured golden)."""
    import random

    seed_all(0)
    assert random.random() == 0.8444218515250481
    assert float(np.random.random()) == 0.5488135039273248
    try:
        import torch
    except ImportError:
        return
    assert float(torch.rand(1).item()) == 0.49625658988952637
