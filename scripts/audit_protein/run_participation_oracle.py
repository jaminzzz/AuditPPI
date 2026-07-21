#!/usr/bin/env python
"""Run the sequence→t(p) participation oracle (DESIGN §7).

Fits a per-protein regressor t̂ = f(pooled SAE fingerprint) on the C3 train+val participation rate t(p),
predicts on the disjoint C3 test proteins, and scores pairs by min(t̂_A, t̂_B). The regressor is trained
only on train+val proteins and never sees test labels, so its pair-level AUROC is an honest measure of
how far a label-free, sequence-only predictor can recover the participation structure.

The workflow body lives here (not in a src package): it is one Layer-1 audit
orchestration that composes the shared data/features/models/eval primitives.

Run: /data/wmzhu/anaconda3/envs/E1/bin/python scripts/audit_protein/run_participation_oracle.py
     [--family c3|cross_species] [--rep sae_max|binary|esmc_mean] [--no-write]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    resolve_backbone_layer,
)
from conf.paths import PPI_PREDICTION_CACHES as CACHE, RESULTS_MISC
from src.data import pairs as benchmark_data
from src.eval import evaluate_scorer
from src.eval.metrics import participation_t, safe_spearman
from src.experiments.results import dump_experiment
from src.features.pairs import load_protein_feature_cache
from src.features.protein_cache import protein_feature_rows
from src.models.estimators.xgboost import fit_xgb_regressor

# Default write dir preserved byte-identically from the pre-refactor predictor
# (was ``ppi_fingerprint.config.OUT_DIR``); the registered runs rely on it.
OUT_DIR = RESULTS_MISC / "ppi_fingerprint"

TRAINVAL = {
    "c3": ("c3:train", "c3:val"),
    "cross_species": ("cross_species:human_train",),
}

TEST = {
    "c3": "c3:test",
    "cross_species": "cross_species:human_test",
}


def train_target_t(family: str):
    """Combine native train/validation splits into per-protein ``t(p)`` targets."""
    if family not in TRAINVAL:
        raise ValueError(f"unknown family {family!r}; choose from {tuple(TRAINVAL)}")
    pairs = []
    labels = []
    sequences: dict[str, str] = {}
    for split_name in TRAINVAL[family]:
        benchmark = benchmark_data.load_benchmark(split_name, attach_seqs=True)
        pairs.extend(benchmark.pairs)
        labels.extend(int(label) for label in benchmark.labels)
        sequences.update(benchmark.seqs)
    target, degree = participation_t(pairs, labels)
    return target, degree, sequences


def _fit_participation_regressor(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    holdout_fraction: float,
):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(targets))
    n_holdout = max(1, int(len(permutation) * holdout_fraction))
    if n_holdout >= len(permutation):
        raise ValueError("participation training requires at least two proteins")
    holdout, train = permutation[:n_holdout], permutation[n_holdout:]
    model = fit_xgb_regressor(
        features[train],
        targets[train],
        features[holdout],
        targets[holdout],
        sample_weight=weights[train],
        sample_weight_eval=weights[holdout],
        seed=seed,
    )
    return model, train, holdout


def run_participation_oracle(
    *,
    family: str = "c3",
    rep: str = "sae_max",
    backbone: str = DEFAULT_BACKBONE,
    layer: int | None = None,
    seed: int = DEFAULT_SEED,
    holdout_fraction: float = 0.1,
    out_dir: Path = OUT_DIR,
    write: bool = True,
) -> dict:
    """Train a sequence-only ``t_hat(p)`` predictor and score pairs by endpoint minimum."""
    if family not in TEST:
        raise ValueError(f"unknown family {family!r}; choose from {tuple(TEST)}")
    layer = resolve_backbone_layer(backbone, layer)
    cache = load_protein_feature_cache(CACHE[family])
    train_t, train_degree, train_sequences = train_target_t(family)
    train_ids = list(train_t)
    train_x, kept_train = protein_feature_rows(
        train_ids, train_sequences, cache, rep, layer=layer, backbone=backbone
    )
    if train_x is None:
        raise RuntimeError(f"no cached training proteins for {family}/{rep}")
    targets = np.asarray([train_t[protein_id] for protein_id in kept_train], dtype=np.float32)
    weights = np.asarray([train_degree[protein_id] for protein_id in kept_train], dtype=np.float32)
    model, train_indices, holdout_indices = _fit_participation_regressor(
        train_x,
        targets,
        weights,
        seed=seed,
        holdout_fraction=holdout_fraction,
    )

    test_benchmark = benchmark_data.load_benchmark(TEST[family], attach_seqs=True)
    test_ids = sorted(test_benchmark.protein_ids)
    test_x, kept_test = protein_feature_rows(
        test_ids, test_benchmark.seqs, cache, rep, layer=layer, backbone=backbone
    )
    if test_x is None:
        raise RuntimeError(f"no cached test proteins for {family}/{rep}")
    predictions = np.clip(model.predict(test_x), 0.0, 1.0)
    predicted_t = {
        protein_id: float(value) for protein_id, value in zip(kept_test, predictions)
    }

    def predicted_scorer(endpoint_a, endpoint_b):
        value_a = predicted_t.get(endpoint_a)
        value_b = predicted_t.get(endpoint_b)
        return None if value_a is None or value_b is None else min(value_a, value_b)

    result = evaluate_scorer(
        predicted_scorer,
        test_benchmark,
        name=f"seq_participation_oracle_{rep}",
    )
    true_t, test_degree = participation_t(test_benchmark.pairs, test_benchmark.labels)
    oracle_result = evaluate_scorer(
        lambda endpoint_a, endpoint_b: min(true_t[endpoint_a], true_t[endpoint_b]),
        test_benchmark,
        name="participation_oracle",
    )
    common = [protein_id for protein_id in kept_test if protein_id in true_t]
    degree_three = [protein_id for protein_id in common if test_degree[protein_id] >= 3]
    result.update(
        {
            "participation_oracle_auroc": oracle_result["auroc"],
            "participation_oracle_auprc": oracle_result["auprc"],
            "lift_over_oracle": round(result["auroc"] - oracle_result["auroc"], 4),
            "lift_over_oracle_auprc": round(result["auprc"] - oracle_result["auprc"], 4),
            "family": family,
            "rep": rep,
            "backbone": backbone,
            "layer": layer,
            "train_splits": list(TRAINVAL[family]),
            "test_split": TEST[family],
            "n_train_proteins": int(len(train_indices)),
            "n_holdout_proteins": int(len(holdout_indices)),
            "n_test_proteins_scored": int(len(kept_test)),
            "spearman_that_vs_test_t": safe_spearman(
                np.asarray([predicted_t[protein_id] for protein_id in common]),
                np.asarray([true_t[protein_id] for protein_id in common]),
            ),
            "spearman_that_vs_test_t_deg_ge3": safe_spearman(
                np.asarray([predicted_t[protein_id] for protein_id in degree_three]),
                np.asarray([true_t[protein_id] for protein_id in degree_three]),
            ),
            "n_deg_ge3": int(len(degree_three)),
        }
    )
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"seq_participation_oracle_{rep}_{family}_{backbone}_l{layer}.json"
        dump_experiment(
            output_path,
            task="protein.participation_oracle",
            dataset=family,
            features=rep,
            split="test",
            model="xgboost_regressor",
            seed=seed,
            payload=result,
            metrics={
                "auroc": result.get("auroc"),
                "auprc": result.get("auprc"),
                "participation_oracle_auroc": result.get("participation_oracle_auroc"),
                "spearman_that_vs_test_t": result.get("spearman_that_vs_test_t"),
            },
        )
        print(f"[done] wrote {output_path}", flush=True)
    print(
        f"[seq_oracle.{rep}.{family}] AUROC={result['auroc']} AUPRC={result['auprc']} "
        f"oracle={result['participation_oracle_auroc']}",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="c3", choices=["c3", "cross_species"])
    ap.add_argument("--rep", default="sae_max", choices=["sae_max", "binary", "esmc_mean"])
    ap.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=DEFAULT_BACKBONE,
        help="Backbone family to read from the v1 feature cache (esmc or esm2).",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        choices=sorted({layer for layers in BACKBONE_LAYERS.values() for layer in layers}),
        help="Backbone layer; defaults to the backbone's default (ESM-C 60, ESM-2 33).",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    run_participation_oracle(
        family=args.family,
        rep=args.rep,
        backbone=args.backbone,
        layer=args.layer,
        seed=args.seed,
        write=not args.no_write,
    )


if __name__ == "__main__":
    main()
