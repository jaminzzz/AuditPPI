from pathlib import Path

import numpy as np

from src.participation.calibration import fit_degree_calibration
from src.participation.cache import stratified_degree_split
from src.participation.features import sequence_features
from src.participation.labels import full_graph_participation_labels


def test_sequence_features_have_stable_dimensions():
    assert sequence_features("ACDX", kmer=1).shape == (23,)
    assert sequence_features("ACDX", kmer=2).shape == (423,)


def test_graph_participation_drops_or_counts_self_loops(tmp_path: Path):
    species_dir = tmp_path / "human"
    species_dir.mkdir()
    (species_dir / "human_ppi.txt").write_text("A B\nB C\nC C\n")
    dropped = full_graph_participation_labels(
        root=tmp_path, species="human", self_loop_mode="drop", prefer_graph=False
    )
    counted = full_graph_participation_labels(
        root=tmp_path, species="human", self_loop_mode="once", prefer_graph=False
    )
    assert dropped.degree == {"A": 1, "B": 2, "C": 1}
    assert counted.degree["C"] == 2


def test_scale_calibration_matches_validation_degree_sum():
    params, calibrate = fit_degree_calibration(
        "scale", np.log1p(np.array([1.0, 2.0])), np.array([2.0, 4.0])
    )
    calibrated = calibrate(np.log1p(np.array([1.0, 2.0])))
    assert params["kind"] == "scale"
    assert np.isclose(calibrated.sum(), 6.0)


def test_degree_split_is_deterministic_and_disjoint():
    protein_ids = [f"P{index}" for index in range(30)]
    degree = {protein_id: index + 1 for index, protein_id in enumerate(protein_ids)}
    first = stratified_degree_split(protein_ids, degree, val_frac=0.2, seed=7)
    second = stratified_degree_split(protein_ids, degree, val_frac=0.2, seed=7)
    assert first == second
    train_ids, val_ids = first
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(protein_ids)
