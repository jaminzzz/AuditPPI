"""PRING graph-derived participation labels and protein splits."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from conf.paths import PRING_ROOT

Pair = Tuple[str, str]
METHODS = ("BFS", "DFS", "RANDOM_WALK")
SELF_LOOP_MODES = ("drop", "once")


@dataclass
class ParticipationLabels:
    degree: Dict[str, int]
    t: Dict[str, float]
    n_nodes: int
    n_edges_used: int
    n_self_loops_seen: int
    self_loop_mode: str


def _read_ppi_edges(path: Path) -> tuple[set[str], list[Pair]]:
    nodes: set[str] = set()
    edges: list[Pair] = []
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            endpoint_a, endpoint_b = parts[0], parts[1]
            nodes.update((endpoint_a, endpoint_b))
            edges.append((endpoint_a, endpoint_b))
    return nodes, edges


def full_graph_participation_labels(
    *,
    root: Path = PRING_ROOT,
    species: str = "human",
    self_loop_mode: str = "drop",
    prefer_graph: bool = True,
) -> ParticipationLabels:
    """Compute degree and normalized participation from a full PRING graph."""
    if self_loop_mode not in SELF_LOOP_MODES:
        raise ValueError(f"self_loop_mode must be one of {SELF_LOOP_MODES}")
    species = species.lower()
    graph_path = root / species / f"{species}_graph.pkl"
    pair_path = root / species / f"{species}_ppi.txt"
    if prefer_graph and graph_path.exists():
        with graph_path.open("rb") as handle:
            graph = pickle.load(handle)
        nodes = {str(node) for node in graph.nodes()}
        raw_edges = [(str(a), str(b)) for a, b in graph.edges()]
    else:
        nodes, raw_edges = _read_ppi_edges(pair_path)

    adjacency: dict[str, set[str]] = {protein: set() for protein in nodes}
    self_loops = 0
    edges_used = 0
    for endpoint_a, endpoint_b in raw_edges:
        adjacency.setdefault(endpoint_a, set())
        adjacency.setdefault(endpoint_b, set())
        nodes.update((endpoint_a, endpoint_b))
        if endpoint_a == endpoint_b:
            self_loops += 1
            if self_loop_mode == "once":
                adjacency[endpoint_a].add(endpoint_a)
                edges_used += 1
            continue
        adjacency[endpoint_a].add(endpoint_b)
        adjacency[endpoint_b].add(endpoint_a)
        edges_used += 1

    n_nodes = len(nodes)
    denominator = n_nodes if self_loop_mode == "once" else max(1, n_nodes - 1)
    degree = {protein: len(adjacency.get(protein, set())) for protein in nodes}
    participation = {protein: degree[protein] / denominator for protein in nodes}
    return ParticipationLabels(
        degree=degree,
        t=participation,
        n_nodes=n_nodes,
        n_edges_used=edges_used,
        n_self_loops_seen=self_loops,
        self_loop_mode=self_loop_mode,
    )


def load_pring_human_split(
    method: str,
    *,
    root: Path = PRING_ROOT,
) -> tuple[set[str], set[str]]:
    method = method.upper()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    split_path = root / "human" / method / f"human_{method}_split.pkl"
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    return set(split["train"]), set(split["test"])


__all__ = [
    "METHODS",
    "ParticipationLabels",
    "SELF_LOOP_MODES",
    "full_graph_participation_labels",
    "load_pring_human_split",
]
