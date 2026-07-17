import torch

from src.models import ESTIMATORS
from src.models.architectures.dual_tower import DualTowerNet
from src.models.architectures.endpoint_mlp import EndpointMLP


def test_endpoint_mlp_is_pair_symmetric_and_has_expected_state_keys():
    torch.manual_seed(1)
    model = EndpointMLP(4, hidden=3, layers=1, dropout=0.0)
    endpoint_a = torch.randn(5, 4)
    endpoint_b = torch.randn(5, 4)
    assert torch.equal(model(endpoint_a, endpoint_b), model(endpoint_b, endpoint_a))
    assert list(model.state_dict()) == ["endpoint.0.weight", "endpoint.2.weight"]


def test_dual_tower_sym_fusion_is_order_invariant_in_eval_mode():
    torch.manual_seed(2)
    model = DualTowerNet(
        4,
        fuse="sym",
        tower_dim=3,
        tower_hidden=5,
        tower_layers=1,
        head_dim=2,
        dropout=0.0,
    ).eval()
    endpoint_a = torch.randn(6, 4)
    endpoint_b = torch.randn(6, 4)
    assert torch.allclose(model(endpoint_a, endpoint_b), model(endpoint_b, endpoint_a))


def test_xgboost_classifier_is_registered():
    assert ESTIMATORS["xgboost_classifier"].endswith(":fit_xgb_classifier")
