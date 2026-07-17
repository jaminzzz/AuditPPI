import torch

from src.features.pairs import (
    concat_evaluation_examples,
    concat_training_examples,
    pair_features,
)


def test_pair_feature_modes_and_symmetry():
    endpoint_a = torch.tensor([[1.0, 2.0]])
    endpoint_b = torch.tensor([[3.0, 1.0]])
    assert torch.equal(pair_features(endpoint_a, endpoint_b, "product"), torch.tensor([[3.0, 2.0]]))
    assert torch.equal(pair_features(endpoint_a, endpoint_b, "absdiff"), torch.tensor([[2.0, 1.0]]))
    symmetric = pair_features(endpoint_a, endpoint_b, "sym")
    assert torch.equal(symmetric, pair_features(endpoint_b, endpoint_a, "sym"))
    assert torch.equal(symmetric, torch.tensor([[3.0, 2.0, 2.0, 1.0]]))


def test_concat_ab_ba_protocol():
    endpoint_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    endpoint_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    labels = torch.tensor([0, 1])
    matrix, expanded_labels, original, direction = concat_training_examples(
        endpoint_a, endpoint_b, labels
    )
    ab, ba = concat_evaluation_examples(endpoint_a, endpoint_b)
    assert torch.equal(matrix[:2], ab)
    assert torch.equal(matrix[2:], ba)
    assert expanded_labels.tolist() == [0, 1, 0, 1]
    assert original.tolist() == [0, 1, 0, 1]
    assert direction.tolist() == [0, 0, 1, 1]
