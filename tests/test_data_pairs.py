from pathlib import Path

import numpy as np
import pytest

from src.data.pairs import Benchmark, load_rf2ppi
from src.data.sequences import normalize_sequence, read_fasta


def test_benchmark_validates_shape_and_binary_labels():
    benchmark = Benchmark("toy", [("a", "b"), ("b", "c")], [1, 0])
    assert len(benchmark) == 2
    assert benchmark.protein_ids == frozenset({"a", "b", "c"})
    assert benchmark.positive_rate == 0.5
    with pytest.raises(ValueError):
        Benchmark("bad", [("a", "b")], [0, 1])
    with pytest.raises(ValueError):
        Benchmark("bad", [("a", "b")], [2])


def test_rf2ppi_and_fasta_loading(tmp_path: Path):
    pairs = tmp_path / "pairs.tsv"
    fasta = tmp_path / "proteins.fasta"
    pairs.write_text("pair\tcategory\nP1_P2\tpositive\nP2_P3\tnegative\n")
    fasta.write_text(">P1 description\naa a\n>P2\nBB\n")
    benchmark = load_rf2ppi(pairs, fasta)
    assert benchmark.pairs == [("P1", "P2"), ("P2", "P3")]
    assert np.array_equal(benchmark.labels, np.array([1, 0]))
    assert benchmark.seqs == {"P1": "AAA", "P2": "BB"}
    assert read_fasta(fasta) == benchmark.seqs
    assert normalize_sequence(" aa\n b ") == "AAB"
