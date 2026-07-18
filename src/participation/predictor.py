"""Sequence-to-participation predictors for C3 and cross-species benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from conf.model import DEFAULT_SEED
from src.data import pairs as benchmark_data
from src.eval import evaluate_scorer
from src.eval.metrics import participation_t
from src.experiments.results import dump_experiment
from src.features.protein_cache import load_pooled_cache, protein_feature_rows
from src.models.estimators.xgboost import fit_xgb_regressor
from src.ppi_fingerprint.config import CACHE, OUT_DIR

TRAINVAL = {
    "c3": ("c3:train", "c3:val"),
    "cross_species": ("cross_species:human_train",),
}

TEST = {
    "c3": "c3:test",
    "cross_species": "cross_species:human_test",
}


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    """Finite Spearman correlation rounded for experiment summaries."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return None if not np.isfinite(value) else round(value, 4)


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


def assemble_protein_features(
    protein_ids,
    sequences: dict[str, str],
    cache: dict,
    representation: str,
):
    """Map protein IDs through sequence strings into pooled fingerprint rows.

    Shared assembly: delegates to
    :func:`src.features.protein_cache.protein_feature_rows` (the same primitive
    the fingerprint baseline's :func:`assemble_pairs` uses), so the
    representation switch and the id/sequence → row logic live in exactly one
    place. Returns ``(rows, kept_ids)`` with ``rows`` a float32 numpy array or
    ``None`` when nothing was cached.
    """
    return protein_feature_rows(protein_ids, sequences, cache, representation)


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
    seed: int = DEFAULT_SEED,
    holdout_fraction: float = 0.1,
    out_dir: Path = OUT_DIR,
    write: bool = True,
) -> dict:
    """Train a sequence-only ``t_hat(p)`` predictor and score pairs by endpoint minimum."""
    if family not in TEST:
        raise ValueError(f"unknown family {family!r}; choose from {tuple(TEST)}")
    cache = load_pooled_cache(CACHE[family])
    train_t, train_degree, train_sequences = train_target_t(family)
    train_ids = list(train_t)
    train_x, kept_train = assemble_protein_features(train_ids, train_sequences, cache, rep)
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
    test_x, kept_test = assemble_protein_features(test_ids, test_benchmark.seqs, cache, rep)
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
        output_path = out_dir / f"seq_participation_oracle_{rep}_{family}.json"
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


__all__ = [
    "CACHE",
    "TEST",
    "TRAINVAL",
    "assemble_protein_features",
    "safe_spearman",
    "run_participation_oracle",
    "train_target_t",
]
