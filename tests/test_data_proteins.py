import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conf.paths import BERNETT_DIR, BERNETT_SPLIT_CSVS, PIC_DATASET_PKL, PRING_ROOT
from src.data.pairs import (
    Benchmark,
    _bernett_clean_seq,
    _read_pring_pair_file,
    load_bernett,
    load_pring_pairs,
)
from src.data.proteins import ProteinDataset, load_pic, load_pring_participation


# --- ProteinDataset contract (synthetic; always runs) ---------------------


def test_protein_dataset_binary_contract():
    dataset = ProteinDataset("toy", ["a", "b", "c"], [1, 0, 1], label_kind="binary")
    assert len(dataset) == 3
    assert dataset.positive_rate == pytest.approx(2 / 3)
    with pytest.raises(ValueError):  # duplicate ids
        ProteinDataset("dup", ["a", "a"], [0, 1])
    with pytest.raises(ValueError):  # non-binary label under binary kind
        ProteinDataset("bad", ["a"], [2])
    with pytest.raises(ValueError):  # id/label length mismatch
        ProteinDataset("mismatch", ["a", "b"], [1])


def test_protein_dataset_continuous_keeps_floats_and_no_positive_rate():
    dataset = ProteinDataset("toy", ["a", "b"], [0.1, 0.9], label_kind="continuous")
    assert dataset.labels.dtype == float
    assert dataset.positive_rate is None
    with pytest.raises(ValueError):
        ProteinDataset("bad-kind", ["a"], [0.1], label_kind="ordinal")


# --- Pair parser / cleaner (synthetic; always runs) -----------------------


def test_read_pring_pair_file_matches_whitespace_contract(tmp_path: Path):
    path = tmp_path / "ppi.txt"
    # 3-col labelled rows; a short line is skipped, extra columns ignored.
    path.write_text("P1 P2 1\nP2 P3 0\nBADLINE\nP4 P5 1 extra\n")
    pairs, labels = _read_pring_pair_file(path)
    assert pairs == [("P1", "P2"), ("P2", "P3"), ("P4", "P5")]
    assert np.array_equal(labels, np.array([1, 0, 1]))


def test_bernett_clean_seq_mirrors_cache_script():
    # '*' and 'f' are dropped, 'J' -> 'L'; other characters (incl. spaces) survive.
    assert _bernett_clean_seq("ACD*fJ") == "ACDL"
    assert _bernett_clean_seq("MJJ*f") == "MLL"


# --- Real-data regression checks (skip when baseline data is absent) ------


def _skip_if_missing(path: Path) -> None:
    if not Path(path).exists():
        pytest.skip(f"baseline data not present: {path}")


def test_load_pic_human_matches_raw_pickle():
    pkl = PIC_DATASET_PKL["human"]
    _skip_if_missing(pkl)
    dataset = load_pic("human")
    with open(pkl, "rb") as handle:
        frame = pickle.load(handle)
    assert dataset.label_kind == "binary"
    assert len(dataset) == len(frame)
    assert int(dataset.labels.sum()) == int(frame["human"].sum())
    # sequences are upper-cased and keyed by id
    assert set(dataset.seqs) == set(dataset.ids)
    assert dataset.name == "pic_human"


def test_load_pic_cell_requires_label_col():
    pkl = PIC_DATASET_PKL["cell"]
    _skip_if_missing(pkl)
    with pytest.raises(ValueError):
        load_pic("cell")


def test_load_pring_participation_split_partitions_full_graph():
    graph = PRING_ROOT / "human" / "human_graph.pkl"
    _skip_if_missing(graph)
    full = load_pring_participation("human")
    train = load_pring_participation("human", split="BFS:train")
    test = load_pring_participation("human", split="BFS:test")
    assert full.label_kind == "continuous"
    assert len(train) + len(test) == len(full)
    assert set(train.ids).isdisjoint(test.ids)


def test_load_pring_pairs_reproduces_manuscript_parser():
    path = PRING_ROOT / "human" / "BFS" / "human_test_ppi.txt"
    _skip_if_missing(path)
    benchmark = load_pring_pairs("human", "test", "BFS", attach_seqs=False)
    pairs, labels = _read_pring_pair_file(path)
    assert benchmark.pairs == [tuple(pair) for pair in pairs]
    assert np.array_equal(benchmark.labels, labels)


def test_load_pring_pairs_cross_species_is_test_only():
    path = PRING_ROOT / "yeast" / "yeast_test_ppi.txt"
    _skip_if_missing(path)
    benchmark = load_pring_pairs("yeast", "test", attach_seqs=False)
    assert isinstance(benchmark, Benchmark)
    assert benchmark.name == "pring_yeast_test"
    with pytest.raises(ValueError):
        load_pring_pairs("yeast", "train")


def test_load_bernett_matches_raw_csv():
    csv = BERNETT_DIR / BERNETT_SPLIT_CSVS["test"]
    _skip_if_missing(csv)
    benchmark = load_bernett("test")
    frame = pd.read_csv(csv, usecols=["seq1", "seq2", "labels"])
    assert len(benchmark) == len(frame)
    assert int(benchmark.labels.sum()) == int(frame["labels"].sum())
    with pytest.raises(ValueError):
        load_bernett("nonexistent-split")
