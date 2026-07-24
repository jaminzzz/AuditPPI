#!/usr/bin/env python
"""Convert protein FASTA records to corrected eSIG-Net 573-D features."""

from __future__ import annotations

import argparse
from itertools import product
import math
from pathlib import Path

from Bio import SeqIO
import h5py
import numpy as np


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PCP_NAMES = ["H1", "H2", "NCI", "P1", "P2", "SASA", "V"]
PROPERTY_TABLE_VERSION = "corrected-2026-07"
AA_PCP_VALUES = {
    "A": (0.62, -0.5, 0.007187, 8.1, 0.046, 1.181, 27.5),
    "C": (0.29, -1.0, -0.036610, 5.5, 0.128, 1.461, 44.6),
    "D": (-0.90, 3.0, -0.023820, 13.0, 0.105, 1.587, 40.0),
    "E": (-0.74, 3.0, 0.006802, 12.3, 0.151, 1.862, 62.0),
    "F": (1.19, -2.5, 0.037552, 5.2, 0.290, 2.228, 115.5),
    "G": (0.48, 0.0, 0.179052, 9.0, 0.000, 0.881, 0.0),
    "H": (-0.40, -0.5, -0.010690, 10.4, 0.230, 2.025, 79.0),
    "I": (1.38, -1.8, 0.021631, 5.2, 0.186, 1.810, 93.5),
    "K": (-1.50, 3.0, 0.017708, 11.3, 0.219, 2.258, 100.0),
    "L": (1.06, -1.8, 0.051672, 4.9, 0.186, 1.931, 93.5),
    "M": (0.64, -1.3, 0.002683, 5.7, 0.221, 2.034, 94.1),
    "N": (-0.78, 2.0, 0.005392, 11.6, 0.134, 1.655, 58.7),
    "P": (0.12, 0.0, 0.239531, 8.0, 0.131, 1.468, 41.9),
    "Q": (-0.85, 0.2, 0.049211, 10.5, 0.180, 1.932, 80.7),
    "R": (-2.53, 3.0, 0.043587, 10.5, 0.291, 2.560, 105.0),
    "S": (-0.18, 0.3, 0.004627, 9.2, 0.062, 1.298, 29.3),
    "T": (-0.05, -0.4, 0.003352, 8.6, 0.108, 1.525, 51.3),
    "V": (1.08, -1.5, 0.057004, 5.9, 0.140, 1.645, 71.5),
    "W": (0.81, -3.4, 0.037977, 5.4, 0.409, 2.663, 145.5),
    "Y": (0.26, -2.3, 0.023599, 6.2, 0.298, 2.368, 117.3),
}
AA_PCP = {
    aa: dict(zip(PCP_NAMES, values)) for aa, values in AA_PCP_VALUES.items()
}


def aac_feature(sequence: str) -> list[float]:
    sequence = sequence.upper()
    length = len(sequence)
    return [sequence.count(aa) / length if length else 0.0 for aa in AMINO_ACIDS]


def ct_feature(sequence: str) -> list[float]:
    groups = {
        **dict.fromkeys("AGV", "0"),
        **dict.fromkeys("ILFP", "1"),
        **dict.fromkeys("YMTS", "2"),
        **dict.fromkeys("HNQW", "3"),
        **dict.fromkeys("RK", "4"),
        **dict.fromkeys("DE", "5"),
        "C": "6",
    }
    triads = ["".join(values) for values in product("0123456", repeat=3)]
    counts = dict.fromkeys(triads, 0)
    encoded = [groups[aa] for aa in sequence.upper() if aa in groups]
    for index in range(len(encoded) - 2):
        counts["".join(encoded[index:index + 3])] += 1
    total = sum(counts.values())
    return [counts[key] / total if total else 0.0 for key in triads]


def _normalized_properties() -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for property_name in PCP_NAMES:
        values = [properties[property_name] for properties in AA_PCP.values()]
        mean = sum(values) / len(values)
        standard_deviation = math.sqrt(
            sum((value - mean) ** 2 for value in values) / len(values)
        )
        for amino_acid, properties in AA_PCP.items():
            result.setdefault(amino_acid, {})[property_name] = (
                properties[property_name] - mean
            ) / standard_deviation
    return result


NORMALIZED_PROPERTIES = _normalized_properties()


def ac_feature(sequence: str, lag: int = 30) -> list[float]:
    residues = [aa for aa in sequence.upper() if aa in NORMALIZED_PROPERTIES]
    length = len(residues)
    if length < lag + 1:
        return [0.0] * (len(PCP_NAMES) * lag)
    output: list[float] = []
    for property_name in PCP_NAMES:
        values = [NORMALIZED_PROPERTIES[aa][property_name] for aa in residues]
        mean = sum(values) / length
        centered = [value - mean for value in values]
        for distance in range(1, lag + 1):
            output.append(
                sum(
                    centered[index] * centered[index + distance]
                    for index in range(length - distance)
                )
                / (length - distance)
            )
    return output


def extract_573_features(sequence: str) -> np.ndarray:
    vector = np.asarray(
        aac_feature(sequence) + ct_feature(sequence) + ac_feature(sequence),
        dtype=np.float64,
    )
    if vector.shape != (573,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid feature vector with shape {vector.shape}")
    return vector


def fasta_to_h5(fasta_path: str | Path, output_path: str | Path) -> None:
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    identifiers = [record.id for record in records]
    duplicates = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicates:
        raise ValueError(f"Duplicate FASTA IDs: {', '.join(duplicates[:20])}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["feature_dimension"] = 573
        handle.attrs["property_table_version"] = PROPERTY_TABLE_VERSION
        handle.attrs["generator"] = "eSIG-Net h5file.py"
        for record in records:
            sequence = str(record.seq).replace(" ", "").upper()
            if not sequence:
                raise ValueError(f"Empty sequence for FASTA ID {record.id}")
            handle.create_dataset(record.id, data=extract_573_features(sequence))
    print(f"Saved {len(records)} protein feature vectors to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, help="input FASTA containing mutant, wild-type, and partner sequences")
    parser.add_argument("--output", required=True, help="output HDF5 path")
    arguments = parser.parse_args()
    fasta_to_h5(arguments.fasta, arguments.output)


if __name__ == "__main__":
    main()
