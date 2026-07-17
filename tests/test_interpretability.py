import numpy as np
import pandas as pd
import torch

from src.interpretability.attribution import endpoint_gradient_input_attribution
from src.interpretability.ebm_effects import ebm_feature_tables
from src.models.architectures.endpoint_mlp import EndpointMLP


def test_gradient_input_attribution_matches_linear_endpoint_model():
    model = EndpointMLP(2, hidden=2, layers=1, dropout=0.0)
    model.endpoint = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model.endpoint[0].weight.copy_(torch.tensor([[2.0, -1.0]]))
    endpoint_a = torch.tensor([[1.0, 3.0]])
    endpoint_b = torch.tensor([[2.0, 1.0]])
    frame = endpoint_gradient_input_attribution(
        model, endpoint_a, endpoint_b, device="cpu", batch_size=2
    )
    assert np.allclose(frame["mean_signed_attr"], [3.0, -2.0])
    assert np.allclose(frame["mean_abs_attr"], [3.0, 2.0])


class _FakeEbm:
    term_names_ = ["f0", "f1"]
    term_scores_ = [np.array([0.0, 1.0]), np.array([-2.0, 0.0, 2.0])]

    def eval_terms(self, matrix):
        return np.column_stack([matrix[:, 0], matrix[:, 1] * 2.0])


def test_ebm_feature_tables_have_feature_and_bin_rows():
    split = {
        "a": np.array([[1.0, 2.0]], dtype=np.float32),
        "b": np.array([[3.0, 4.0]], dtype=np.float32),
        "y": np.array([1], dtype=np.int8),
    }
    selected = pd.DataFrame({"feature_id": [0, 1], "rank": [1, 2]})
    summary, effects = ebm_feature_tables(
        _FakeEbm(), split, np.array([0, 1]), selected, "dense"
    )
    assert set(summary["feature_id"]) == {0, 1}
    assert len(effects) == 5
