"""PRING human-to-species participation generalization protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from conf.model import DEFAULT_SEED
from conf.paths import PRING_ROOT
from src.data.sequences import read_fasta
from src.participation.cache import (
    features_from_cache,
    load_feature_cache,
    species_cache_path,
    stratified_degree_split,
)
from src.participation.calibration import fit_degree_calibration
from src.participation.config import CROSS_SPECIES, OUT_DIR, TRAIN_SPECIES
from src.participation.evaluation import (
    participation_node_metrics,
    participation_pair_metrics,
)
from src.participation.features import normalize_feature_kind
from src.participation.importance import select_topk_features
from src.participation.labels import ParticipationLabels, full_graph_participation_labels
from src.participation.models import MODEL_KINDS, fit_participation_model


def load_full_species_nodes(
    labels: ParticipationLabels,
    sequences: Mapping[str, str],
) -> List[str]:
    """Return proteins present in both a species FASTA and its graph labels."""
    return sorted(protein_id for protein_id in labels.degree if protein_id in sequences)


def run_pring_cross_species_generalization(
    *,
    root: Path = PRING_ROOT,
    out_dir: Path = OUT_DIR,
    test_species: Sequence[str] = CROSS_SPECIES,
    self_loop_mode: str = "drop",
    drop_self_pairs: bool = True,
    val_frac: float = 0.1,
    seed: int = DEFAULT_SEED,
    kmer: int = 2,
    feature_kind: str = "sae_max",
    train_cache_path: Optional[Path] = None,
    species_cache_paths: Optional[Mapping[str, Path]] = None,
    model_kind: str = "xgboost",
    calibration: str = "log_linear",
    n_estimators: int = 800,
    max_depth: int = 4,
    lr: float = 0.05,
    device: str = "cpu",
    top_k_features: int = 0,
    early_stopping_rounds: int = 50,
    tabpfn_estimators: int = 8,
    tabpfn_subsample_samples: int = 50000,
    pair_eval: bool = True,
    write: bool = True,
) -> dict:
    """Train on all usable human proteins and zero-shot test other species."""
    feature_kind = normalize_feature_kind(feature_kind)
    model_kind = model_kind.lower()
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}")

    train_labels = full_graph_participation_labels(
        root=root, species=TRAIN_SPECIES, self_loop_mode=self_loop_mode
    )
    train_sequences = read_fasta(
        root / TRAIN_SPECIES / f"{TRAIN_SPECIES}_simple.fasta"
    )
    train_candidates = sorted(
        protein_id
        for protein_id in train_labels.degree
        if protein_id in train_sequences
    )

    needs_cache = feature_kind != "sequence_basic"
    train_cache = None
    if needs_cache:
        cache_path = (
            Path(train_cache_path)
            if train_cache_path
            else species_cache_path(TRAIN_SPECIES)
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"human training cache not found: {cache_path}. Build it with "
                "scripts/cache/cache_pring_human_esmc_sae.py "
                "(or pass --train-cache-path)."
            )
        train_cache = load_feature_cache(cache_path)

    full_train_x, train_kept, train_missing = features_from_cache(
        train_candidates,
        train_sequences,
        feature_kind=feature_kind,
        kmer=kmer,
        cache=train_cache,
    )
    row_by_id = {protein_id: row for row, protein_id in enumerate(train_kept)}
    train_ids, val_ids = stratified_degree_split(
        train_kept, train_labels.degree, val_frac=val_frac, seed=seed
    )
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"empty human train/val split: train={len(train_ids)} val={len(val_ids)}"
        )
    train_rows = np.asarray([row_by_id[protein_id] for protein_id in train_ids], dtype=int)
    val_rows = np.asarray([row_by_id[protein_id] for protein_id in val_ids], dtype=int)
    train_x, val_x = full_train_x[train_rows], full_train_x[val_rows]
    train_y = np.asarray(
        [np.log1p(train_labels.degree[protein_id]) for protein_id in train_ids],
        dtype=np.float32,
    )
    val_y = np.asarray(
        [np.log1p(train_labels.degree[protein_id]) for protein_id in val_ids],
        dtype=np.float32,
    )

    feature_dim = train_x.shape[1]
    effective_top_k = top_k_features
    if model_kind == "tabpfn" and effective_top_k <= 0 and feature_dim > 500:
        effective_top_k = 500
        print(
            "    [tabpfn] auto-selecting top 500 features with XGBoost importance",
            flush=True,
        )
    selected_columns, _ = select_topk_features(
        train_x,
        train_y,
        val_x,
        val_y,
        top_k=effective_top_k,
        seed=seed,
        max_depth=max_depth,
        learning_rate=lr,
        device=device,
        importance_estimators=200,
        early_stopping_rounds=early_stopping_rounds,
    )

    def select(matrix: np.ndarray) -> np.ndarray:
        return matrix if selected_columns is None else matrix[:, selected_columns]

    regressor = fit_participation_model(
        model_kind,
        select(train_x),
        train_y,
        select(val_x),
        val_y,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        device=device,
        early_stopping_rounds=early_stopping_rounds,
        tabpfn_estimators=tabpfn_estimators,
        tabpfn_subsample_samples=tabpfn_subsample_samples,
    )
    predicted_val = np.asarray(regressor.predict(select(val_x)), dtype=float)
    calibration_params, calibrate = fit_degree_calibration(
        calibration,
        predicted_val,
        np.asarray(
            [train_labels.degree[protein_id] for protein_id in val_ids], dtype=float
        ),
    )

    per_species: Dict[str, dict] = {}
    for species_name in test_species:
        species_name = species_name.lower()
        species_labels = full_graph_participation_labels(
            root=root, species=species_name, self_loop_mode=self_loop_mode
        )
        species_sequences = read_fasta(
            root / species_name / f"{species_name}_simple.fasta"
        )
        species_cache = None
        if needs_cache:
            cache_path = (species_cache_paths or {}).get(
                species_name
            ) or species_cache_path(species_name)
            cache_path = Path(cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"[{species_name}] species cache not found: {cache_path}. "
                    "Build it with scripts/cache/cache_pring_species_esmc_sae.py "
                    f"--species {species_name}."
                )
            species_cache = load_feature_cache(cache_path)

        candidates = load_full_species_nodes(species_labels, species_sequences)
        test_x, test_ids, missing_test = features_from_cache(
            candidates,
            species_sequences,
            feature_kind=feature_kind,
            kmer=kmer,
            cache=species_cache,
        )
        if len(test_ids) < 3:
            raise RuntimeError(
                f"[{species_name}] too few usable test proteins: {len(test_ids)}"
            )

        predicted_log_degree = np.asarray(
            regressor.predict(select(test_x)), dtype=float
        )
        predicted_degree = calibrate(predicted_log_degree)
        denominator = (
            species_labels.n_nodes
            if self_loop_mode == "once"
            else max(1, species_labels.n_nodes - 1)
        )
        predicted_t = {
            protein_id: float(degree) / denominator
            for protein_id, degree in zip(test_ids, predicted_degree)
        }

        pair_metrics = {}
        if pair_eval:
            for tag in ("test_ppi", "all_test_ppi"):
                filename = f"{species_name}_{tag}.txt"
                pair_path = root / species_name / filename
                if pair_path.exists():
                    pair_metrics[tag] = participation_pair_metrics(
                        pair_path,
                        pred_t=predicted_t,
                        true_t=species_labels.t,
                        drop_self_pairs=drop_self_pairs,
                    )

        node_metrics = participation_node_metrics(
            test_ids,
            species_labels.degree,
            predicted_log_degree,
            predicted_degree,
        )
        per_species[species_name] = {
            "n_reference_nodes": species_labels.n_nodes,
            "n_reference_edges_used": species_labels.n_edges_used,
            "n_test_proteins": len(test_ids),
            "n_candidates": len(candidates),
            "missing_test_features": len(missing_test),
            "node_metrics": node_metrics,
            "pair_metrics": pair_metrics,
        }
        print(
            f"[PRING.xspecies.{species_name}.{feature_kind}.{model_kind}] "
            f"test Spearman={node_metrics['spearman_pred_degree']} "
            f"high_deg_AUROC={node_metrics['high_degree_auroc']} "
            f"n_test={len(test_ids)}",
            flush=True,
        )

    result = {
        "task": "pring_cross_species_participation_generalization",
        "train_species": TRAIN_SPECIES,
        "test_species": [species.lower() for species in test_species],
        "root": str(root),
        "label_definition": {
            "reference": "each species' own full {species}_graph.pkl",
            "target": "log1p(full_graph_degree)",
            "self_loop_mode": self_loop_mode,
            "note": "degrees are within-species; only rank metrics compare across species",
        },
        "train": {
            "n_human_candidates": len(train_candidates),
            "n_train_with_features": len(train_kept),
            "missing_train_features": len(train_missing),
            "n_train_nodes": len(train_ids),
            "n_val_nodes": len(val_ids),
            "val_frac": val_frac,
            "seed": seed,
        },
        "features": {
            "kind": feature_kind,
            "dim": int(feature_dim),
            "kmer": kmer if feature_kind == "sequence_basic" else None,
        },
        "feature_selection": {
            "requested_top_k_features": int(top_k_features),
            "effective_top_k_features": int(effective_top_k),
            "selected_dim": int(select(train_x).shape[1]),
        },
        "model": {
            "kind": model_kind,
            "target": "log1p(full_graph_degree)",
            "device": device,
        },
        "calibration": calibration_params,
        "drop_self_pairs_in_pair_eval": drop_self_pairs,
        "per_species": per_species,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"pring_crossspecies_{feature_kind}_{model_kind}"
        output_path = out_dir / f"{stem}.json"
        output_path.write_text(json.dumps(result, indent=2))
        print(f"[done] {output_path}", flush=True)
    return result


__all__ = ["load_full_species_nodes", "run_pring_cross_species_generalization"]
