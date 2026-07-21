#!/usr/bin/env python3
"""Run PRING cross-species participation generalization.

Definition:
  Train the sequence -> full-graph participation oracle on ALL human PRING
  proteins (an inner stratified val split drives early stopping + degree
  calibration), then zero-shot test on each held-out PRING species' full graph
  (yeast / ecoli / arath). Each species' degree labels come from its OWN
  {species}_graph.pkl, so absolute degrees are not comparable across species --
  the reported cross-species signal is the rank-based test Spearman and
  high-degree AUROC.

The workflow body lives here (not in a src package): it is one Ladder-1 audit
orchestration that composes the shared data/features/models/eval primitives.

Examples:
  # CPU smoke, no feature cache needed:
  /data/wmzhu/anaconda3/envs/E1/bin/python \
    scripts/audit_protein/run_pring_cross_species_generalization.py \
    --feature-kind sequence_basic

  # SAE features (needs per-species v1 protein caches from
  # scripts/prep/slice_dataset_protein_cache.py):
  /data/wmzhu/anaconda3/envs/E1/bin/python \
    scripts/audit_protein/run_pring_cross_species_generalization.py \
    --feature-kind sae_max --model-kind xgboost
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    CACHE_MAX_RESIDUES_DEFAULT,
    CACHE_MAX_RESIDUES_VARIANTS,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    resolve_backbone_layer,
)
from conf.paths import (
    PRING_ROOT,
    PRING_CROSS_SPECIES as CROSS_SPECIES,
    PRING_PARTICIPATION_DIR as OUT_DIR,
    PRING_TRAIN_SPECIES as TRAIN_SPECIES,
)
from src.data.pring_graph import ParticipationLabels, full_graph_participation_labels
from src.data.sequences import read_fasta
from src.eval.participation import (
    participation_node_metrics,
    participation_pair_metrics,
)
from src.experiments.results import dump_experiment
from src.features.feature_selection import select_topk_features
from src.features.pooled_assembly import (
    features_from_cache,
    load_pooled_payload,
    species_cache_path,
    stratified_degree_split,
)
from src.features.sequence_composition import (
    FEATURE_KINDS,
    FORMAL_FEATURE_KINDS,
    normalize_feature_kind,
)
from src.models.calibration import fit_degree_calibration
from src.models.estimators.regressors import MODEL_KINDS, fit_participation_model


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
    backbone: str = DEFAULT_BACKBONE,
    layer: Optional[int] = None,
    max_residues: int = CACHE_MAX_RESIDUES_DEFAULT,
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
    resolved_layer = resolve_backbone_layer(backbone, layer)

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
            else species_cache_path(TRAIN_SPECIES, max_residues)
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"human training cache not found: {cache_path}. Build it with "
                "scripts/prep/slice_dataset_protein_cache.py from the pooled "
                "seq caches (or pass --train-cache-path)."
            )
        train_cache = load_pooled_payload(cache_path)

    full_train_x, train_kept, train_missing = features_from_cache(
        train_candidates,
        train_sequences,
        feature_kind=feature_kind,
        kmer=kmer,
        cache=train_cache,
        layer=layer,
        backbone=backbone,
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
            ) or species_cache_path(species_name, max_residues)
            cache_path = Path(cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"[{species_name}] species cache not found: {cache_path}. "
                    "Build it with scripts/prep/slice_dataset_protein_cache.py "
                    f"(species={species_name}) from the pooled seq caches."
                )
            species_cache = load_pooled_payload(cache_path)

        candidates = load_full_species_nodes(species_labels, species_sequences)
        test_x, test_ids, missing_test = features_from_cache(
            candidates,
            species_sequences,
            feature_kind=feature_kind,
            kmer=kmer,
            cache=species_cache,
            layer=layer,
            backbone=backbone,
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
            "backbone": backbone,
            "layer": int(resolved_layer),
            "max_residues": int(max_residues),
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
        stem = f"pring_crossspecies_{feature_kind}_{backbone}_l{resolved_layer}_{model_kind}"
        output_path = out_dir / f"{stem}.json"
        dump_experiment(
            output_path,
            task="protein.cross_species_generalization",
            dataset="pring_cross_species",
            features=feature_kind,
            split="multi",
            model=model_kind,
            seed=seed,
            payload=result,
            metrics={
                species: {
                    "spearman_pred_degree": cell.get("node_metrics", {}).get(
                        "spearman_pred_degree"
                    ),
                    "high_degree_auroc": cell.get("node_metrics", {}).get(
                        "high_degree_auroc"
                    ),
                }
                for species, cell in result.get("per_species", {}).items()
            },
            hyperparameters=result.get("model"),
        )
        print(f"[done] {output_path}", flush=True)
    return result


def _metric(res: dict, path: list[str]):
    cur = res
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-species", nargs="+", default=list(CROSS_SPECIES))
    p.add_argument("--root", type=Path, default=PRING_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--feature-kind",
        default="sae_max",
        choices=[*FEATURE_KINDS, "pooled_sae", "binary_sae", "esmc", "all"],
        help="'all' runs the formal inputs: sae_max, binary, esmc_mean",
    )
    p.add_argument("--model-kind", default="xgboost", choices=MODEL_KINDS)
    p.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=DEFAULT_BACKBONE,
        help="pLM line to read from the v1 feature cache (esmc or esm2).",
    )
    p.add_argument(
        "--layer",
        type=int,
        choices=sorted({layer for layers in BACKBONE_LAYERS.values() for layer in layers}),
        default=None,
        help="Backbone layer; defaults to the backbone's default (ESM-C 60, ESM-2 33).",
    )
    p.add_argument(
        "--max-residues",
        type=int,
        choices=CACHE_MAX_RESIDUES_VARIANTS,
        default=CACHE_MAX_RESIDUES_DEFAULT,
        help="On-disk length variant of the v1 cache (max1022 / max2046).",
    )
    p.add_argument("--train-cache-path", type=Path, default=None,
                   help="human pooled ESM-C/SAE cache; defaults to the standard PRING human cache")
    p.add_argument("--calibration", choices=["log_linear", "scale", "none"], default="log_linear")
    p.add_argument("--self-loop-mode", choices=["drop", "once"], default="drop")
    p.add_argument("--keep-self-pairs", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--kmer", type=int, default=2, choices=[1, 2])
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--top-k-features", type=int, default=0)
    p.add_argument("--tabpfn-estimators", type=int, default=8)
    p.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    p.add_argument("--no-pair-eval", action="store_true")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    feature_kinds = FORMAL_FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)

    summary = {}
    for feature_kind in feature_kinds:
        res = run_pring_cross_species_generalization(
            root=args.root,
            out_dir=args.out_dir,
            test_species=args.test_species,
            self_loop_mode=args.self_loop_mode,
            drop_self_pairs=not args.keep_self_pairs,
            val_frac=args.val_frac,
            seed=args.seed,
            kmer=args.kmer,
            feature_kind=feature_kind,
            backbone=args.backbone,
            layer=args.layer,
            max_residues=args.max_residues,
            train_cache_path=args.train_cache_path,
            model_kind=args.model_kind,
            calibration=args.calibration,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            lr=args.lr,
            device=args.device,
            top_k_features=args.top_k_features,
            early_stopping_rounds=args.early_stopping_rounds,
            tabpfn_estimators=args.tabpfn_estimators,
            tabpfn_subsample_samples=args.tabpfn_subsample_samples,
            pair_eval=not args.no_pair_eval,
            write=not args.no_write,
        )
        summary[feature_kind] = {
            sp: {
                "test_spearman": _metric(spec, ["node_metrics", "spearman_pred_degree"]),
                "test_high_degree_auroc": _metric(spec, ["node_metrics", "high_degree_auroc"]),
                "n_test": _metric(spec, ["n_test_proteins"]),
                "missing_test_features": _metric(spec, ["missing_test_features"]),
            }
            for sp, spec in res.get("per_species", {}).items()
        }

    print("\n=== PRING cross-species participation generalization summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        resolved_layer = resolve_backbone_layer(args.backbone, args.layer)
        stem = f"pring_crossspecies_{args.model_kind}_{args.backbone}_l{resolved_layer}_summary"
        dump_experiment(
            args.out_dir / f"{stem}.json",
            task="protein.cross_species_generalization",
            dataset="pring_cross_species",
            features="multi",
            split="multi",
            model=args.model_kind,
            seed=args.seed,
            payload=summary,
            metrics=summary,
        )
        print(f"\n[done] results in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
