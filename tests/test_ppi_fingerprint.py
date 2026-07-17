import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.participation.predictor import assemble_protein_features, safe_spearman
from src.ppi_fingerprint.config import MODEL_NAMES, NATIVE_TRAIN, REPRESENTATIONS


def test_fingerprint_protocol_configuration():
    assert MODEL_NAMES == ("xgb", "tabpfn", "dualtower")
    assert REPRESENTATIONS == ("binary", "sae_max", "esmc_mean")
    assert NATIVE_TRAIN["rf2ppi"] == "c3:train"


def test_protein_assembly_uses_sequence_cache_rows():
    cache = {
        "seq2idx": {"AAA": 0, "BBB": 1},
        "esmc_sae_max": torch.tensor([[0.0, 2.0], [3.0, 0.0]]),
        "esmc_mean": torch.tensor([[1.0], [2.0]]),
    }
    matrix, kept = assemble_protein_features(
        ["p2", "missing", "p1"],
        {"p1": "AAA", "p2": "BBB"},
        cache,
        "binary",
    )
    assert kept == ["p2", "p1"]
    assert np.array_equal(matrix, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert safe_spearman(np.array([1, 2, 3]), np.array([1, 2, 3])) == 1.0


def test_fingerprint_package_does_not_eagerly_import_torch_in_fresh_process():
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import sys; import src.ppi_fingerprint; print('torch' in sys.modules)",
        ],
        cwd=root,
        text=True,
    ).strip()
    assert output == "False"
