#!/usr/bin/env python3
"""Run PRING full-graph sequence -> participation prediction.

Definition:
  Use PRING's full Human reference graph only to define protein-level labels
  (full-graph degree / participation), then train and evaluate under PRING's
  protein-disjoint Human BFS/DFS/RANDOM_WALK splits.

The workflow body lives here (not in a src package): it is one Ladder-1 audit
orchestration that composes the shared data/features/models/eval primitives.

Examples:
  /data/wmzhu/anaconda3/envs/E1/bin/python scripts/audit_protein/run_pring_participation_oracle.py \
    --method BFS --feature-kind all --cache-path data/sae/protein_caches/pring_human_esmc_sae_cache.pt

  /data/wmzhu/anaconda3/envs/E1/bin/python scripts/audit_protein/run_pring_participation_oracle.py \
    --method all --feature-kind sae_max --model-kind xgboost --pair-eval human_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    resolve_backbone_layer,
)
from conf.paths import (
    PRING_ROOT,
    PRING_HUMAN_SAE_CACHE as DEFAULT_PRING_CACHE,
    PRING_PARTICIPATION_DIR as OUT_DIR,
)
from src.data.pring_graph import (
    METHODS,
    full_graph_participation_labels,
    load_pring_human_split,
)
from src.data.sequences import read_fasta
from src.eval.participation import (
    participation_node_metrics,
    participation_pair_metrics,
    write_participation_predictions,
)
from src.experiments.results import dump_experiment
from src.features.feature_selection import (
    build_importance_rows,
    extract_xgb_importance,
    select_topk_features,
    write_feature_importance,
)
from src.features.pooled_assembly import prepare_features
from src.features.sequence_composition import (
    FEATURE_KINDS,
    FORMAL_FEATURE_KINDS,
    normalize_feature_kind,
)
from src.models.calibration import fit_degree_calibration
from src.models.estimators.regressors import MODEL_KINDS, fit_participation_model
from src.models.estimators.xgboost import fit_xgb_logdegree

PAIR_EVALS = ("none", "human_test", "all")


def run_pring_participation_oracle(
    *,
    method: str = "BFS",
    root: Path = PRING_ROOT,
    out_dir: Path = OUT_DIR,
    self_loop_mode: str = "drop",
    drop_self_pairs: bool = True,
    val_frac: float = 0.1,
    seed: int = DEFAULT_SEED,
    kmer: int = 2,
    feature_kind: str = "sequence_basic",
    backbone: str = DEFAULT_BACKBONE,
    layer: int | None = None,
    cache_path: Path | None = None,
    model_kind: str = "xgboost",
    calibration: str = "log_linear",
    n_estimators: int = 800,
    max_depth: int = 4,
    lr: float = 0.05,
    device: str = "cpu",
    top_k_features: int = 0,
    importance: bool = True,
    importance_top_n: int = 100,
    importance_estimators: int = 200,
    early_stopping_rounds: int = 50,
    tabpfn_estimators: int = 8,
    tabpfn_subsample_samples: int = 50000,
    pair_eval: str = "none",
    write: bool = True,
) -> dict:
    """Train and evaluate sequence-to-full-graph participation on PRING human."""
    method = method.upper()
    feature_kind = normalize_feature_kind(feature_kind)
    model_kind = model_kind.lower()
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}")
    if pair_eval not in PAIR_EVALS:
        raise ValueError(f"pair_eval must be one of {PAIR_EVALS}")
    resolved_layer = resolve_backbone_layer(backbone, layer)

    labels = full_graph_participation_labels(
        root=root, self_loop_mode=self_loop_mode
    )
    sequences = read_fasta(root / "human" / "human_simple.fasta")
    train_nodes, test_nodes = load_pring_human_split(method, root=root)
    usable_train = sorted(
        protein_id
        for protein_id in train_nodes
        if protein_id in sequences and protein_id in labels.degree
    )
    usable_test = sorted(
        protein_id
        for protein_id in test_nodes
        if protein_id in sequences and protein_id in labels.degree
    )

    feature_pack = prepare_features(
        feature_kind=feature_kind,
        seqs=sequences,
        labels=labels,
        candidate_train=usable_train,
        candidate_test=usable_test,
        kmer=kmer,
        val_frac=val_frac,
        seed=seed,
        cache_path=cache_path,
        backbone=backbone,
        layer=layer,
    )
    train_x, val_x, test_x = (
        feature_pack.Xtr,
        feature_pack.Xva,
        feature_pack.Xte,
    )
    train_ids, val_ids, test_ids = (
        feature_pack.train_ids,
        feature_pack.val_ids,
        feature_pack.test_ids,
    )
    train_y = np.asarray(
        [np.log1p(labels.degree[protein_id]) for protein_id in train_ids],
        dtype=np.float32,
    )
    val_y = np.asarray(
        [np.log1p(labels.degree[protein_id]) for protein_id in val_ids],
        dtype=np.float32,
    )

    effective_top_k = top_k_features
    if model_kind == "tabpfn" and effective_top_k <= 0 and train_x.shape[1] > 500:
        effective_top_k = 500
        print(
            "    [tabpfn] auto-selecting top 500 features with XGBoost importance",
            flush=True,
        )
    selected_columns, selection_model = select_topk_features(
        train_x,
        train_y,
        val_x,
        val_y,
        top_k=effective_top_k,
        seed=seed,
        max_depth=max_depth,
        learning_rate=lr,
        device=device,
        importance_estimators=importance_estimators,
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

    predicted_log_train = np.asarray(regressor.predict(select(train_x)), dtype=float)
    predicted_log_val = np.asarray(regressor.predict(select(val_x)), dtype=float)
    predicted_log_test = np.asarray(regressor.predict(select(test_x)), dtype=float)
    calibration_params, calibrate = fit_degree_calibration(
        calibration,
        predicted_log_val,
        np.asarray([labels.degree[protein_id] for protein_id in val_ids], dtype=float),
    )
    predicted_degree_train = calibrate(predicted_log_train)
    predicted_degree_val = calibrate(predicted_log_val)
    predicted_degree_test = calibrate(predicted_log_test)

    denominator = labels.n_nodes if self_loop_mode == "once" else max(1, labels.n_nodes - 1)
    all_ids = train_ids + val_ids + test_ids
    all_log = np.concatenate(
        [predicted_log_train, predicted_log_val, predicted_log_test]
    )
    all_degree = np.concatenate(
        [predicted_degree_train, predicted_degree_val, predicted_degree_test]
    )
    predicted_log_by_id = {
        protein_id: float(value) for protein_id, value in zip(all_ids, all_log)
    }
    predicted_degree_by_id = {
        protein_id: float(value) for protein_id, value in zip(all_ids, all_degree)
    }
    predicted_t = {
        protein_id: degree / denominator
        for protein_id, degree in predicted_degree_by_id.items()
    }

    pair_metrics = {}
    if pair_eval != "none":
        pair_dir = root / "human" / method
        pair_metrics["human_test_ppi"] = participation_pair_metrics(
            pair_dir / "human_test_ppi.txt",
            pred_t=predicted_t,
            true_t=labels.t,
            drop_self_pairs=drop_self_pairs,
        )
        if pair_eval == "all":
            pair_metrics["all_test_ppi"] = participation_pair_metrics(
                pair_dir / "all_test_ppi.txt",
                pred_t=predicted_t,
                true_t=labels.t,
                drop_self_pairs=drop_self_pairs,
            )

    importance_summary = None
    importance_rows = None
    if importance:
        arrays = extract_xgb_importance(
            regressor, len(feature_pack.feature_names), selected_columns
        )
        importance_source = "model" if arrays is not None else None
        if arrays is None and selection_model is not None:
            arrays = extract_xgb_importance(
                selection_model, len(feature_pack.feature_names)
            )
            importance_source = "feature_selection_xgboost"
        elif arrays is None and feature_kind != "sequence_basic":
            auxiliary = fit_xgb_logdegree(
                train_x,
                train_y,
                val_x,
                val_y,
                seed=seed,
                n_estimators=importance_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                device=device,
                early_stopping_rounds=early_stopping_rounds,
            )
            arrays = extract_xgb_importance(
                auxiliary, len(feature_pack.feature_names)
            )
            importance_source = "auxiliary_xgboost"
        if arrays is not None:
            importance_rows = build_importance_rows(
                feature_names=feature_pack.feature_names,
                arrays=arrays,
                selected_columns=selected_columns,
            )
            top_rows = importance_rows[: max(0, importance_top_n)]
            importance_summary = {
                "source": importance_source,
                "n_features_ranked": len(importance_rows),
                "top_n_in_json": len(top_rows),
                "top_features": top_rows,
            }

    best_iteration = getattr(regressor, "best_iteration", None)
    result = {
        "task": "pring_full_graph_sequence_participation",
        "method": method,
        "root": str(root),
        "label_definition": {
            "reference": "full human_graph.pkl",
            "target": "log1p(full_graph_degree)",
            "t": "full_graph_degree / denominator",
            "self_loop_mode": self_loop_mode,
            "denominator": denominator,
            "n_reference_nodes": labels.n_nodes,
            "n_reference_edges_used": labels.n_edges_used,
            "n_self_loops_seen": labels.n_self_loops_seen,
        },
        "split": {
            "source": f"human/{method}/human_{method}_split.pkl",
            "val": "stratified inner protein split sampled from PRING train proteins",
            "val_frac": val_frac,
            "seed": seed,
            "n_train_nodes_raw": len(train_nodes),
            "n_test_nodes_raw": len(test_nodes),
            "n_train_nodes_with_sequence_and_label": len(usable_train),
            "n_test_nodes_with_sequence_and_label": len(usable_test),
            "n_train_nodes": len(train_ids),
            "n_val_nodes": len(val_ids),
            "n_test_nodes": len(test_ids),
            "missing_train_seq_or_label": len(train_nodes) - len(usable_train),
            "missing_test_seq_or_label": len(test_nodes) - len(usable_test),
        },
        "features": feature_pack.info,
        "feature_selection": {
            "requested_top_k_features": int(top_k_features),
            "effective_top_k_features": int(effective_top_k),
            "selected_dim": int(select(train_x).shape[1]),
            "selected_columns": (
                selected_columns.tolist() if selected_columns is not None else None
            ),
        },
        "model": {
            "kind": model_kind,
            "target": "log1p(full_graph_degree)",
            "n_estimators": (
                n_estimators if model_kind != "tabpfn" else tabpfn_estimators
            ),
            "max_depth": max_depth,
            "learning_rate": lr if model_kind == "xgboost" else None,
            "early_stopping_rounds": (
                early_stopping_rounds if model_kind == "xgboost" else None
            ),
            "device": device,
            "best_iteration": (
                int(best_iteration) if best_iteration is not None else None
            ),
        },
        "calibration": calibration_params,
        "drop_self_pairs_in_pair_eval": drop_self_pairs,
        "node_metrics": {
            "train": participation_node_metrics(
                train_ids,
                labels.degree,
                predicted_log_train,
                predicted_degree_train,
            ),
            "val": participation_node_metrics(
                val_ids,
                labels.degree,
                predicted_log_val,
                predicted_degree_val,
            ),
            "test": participation_node_metrics(
                test_ids,
                labels.degree,
                predicted_log_test,
                predicted_degree_test,
            ),
        },
        "pair_eval": pair_eval,
        "pair_metrics": pair_metrics,
        "feature_importance": importance_summary,
    }
    result["backbone"] = backbone
    result["layer"] = int(resolved_layer)

    print(
        f"[PRING.{method}.{feature_kind}.{backbone}L{resolved_layer}.{model_kind}] "
        f"test Spearman={result['node_metrics']['test']['spearman_pred_degree']} "
        f"high_deg_AUROC={result['node_metrics']['test']['high_degree_auroc']} "
        f"n_train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ids)}",
        flush=True,
    )

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"pring_human_{method.lower()}_{feature_kind}_"
            f"{backbone}_l{resolved_layer}_{model_kind}"
        )
        dump_experiment(
            out_dir / f"{stem}.json",
            task="protein.participation_oracle",
            dataset=f"pring_human_{method.lower()}",
            features=feature_kind,
            split="test",
            model=model_kind,
            seed=seed,
            payload=result,
            metrics=result.get("node_metrics", {}).get("test"),
            hyperparameters=result.get("model"),
        )
        write_participation_predictions(
            out_dir / f"{stem}_protein_predictions.tsv",
            split_ids={"train": train_ids, "val": val_ids, "test": test_ids},
            degree=labels.degree,
            true_t=labels.t,
            pred_log=predicted_log_by_id,
            pred_degree=predicted_degree_by_id,
            pred_t=predicted_t,
        )
        if importance_rows is not None:
            write_feature_importance(
                out_dir / f"{stem}_feature_importance.tsv", importance_rows
            )
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
    p.add_argument("--method", default="BFS", choices=[*METHODS, "all"])
    p.add_argument("--root", type=Path, default=PRING_ROOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--self-loop-mode", choices=["drop", "once"], default="drop")
    p.add_argument("--keep-self-pairs", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--kmer", type=int, default=2, choices=[1, 2])
    p.add_argument(
        "--feature-kind",
        default="sae_max",
        choices=[*FEATURE_KINDS, "pooled_sae", "binary_sae", "esmc", "all"],
        help="'all' runs the formal inputs: sae_max, binary, esmc_mean",
    )
    p.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_PRING_CACHE,
        help="v1 protein feature cache with seq2idx and preferably uniprotid2idx",
    )
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
    p.add_argument("--model-kind", default="xgboost", choices=MODEL_KINDS)
    p.add_argument("--calibration", choices=["log_linear", "scale", "none"], default="log_linear")
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument(
        "--top-k-features",
        type=int,
        default=0,
        help="optional XGBoost top-k feature selection before the final regressor; useful for TabPFN",
    )
    p.add_argument("--no-importance", action="store_true", help="disable feature-importance output")
    p.add_argument("--importance-top-n", type=int, default=100)
    p.add_argument("--importance-estimators", type=int, default=200)
    p.add_argument("--tabpfn-estimators", type=int, default=8)
    p.add_argument("--tabpfn-subsample-samples", type=int, default=50000)
    p.add_argument("--pair-eval", choices=PAIR_EVALS, default="none")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    methods = METHODS if args.method == "all" else (args.method,)
    feature_kinds = FORMAL_FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)

    summary = {}
    for method in methods:
        summary[method] = {}
        for feature_kind in feature_kinds:
            res = run_pring_participation_oracle(
                method=method,
                root=args.root,
                out_dir=args.out_dir,
                self_loop_mode=args.self_loop_mode,
                drop_self_pairs=not args.keep_self_pairs,
                val_frac=args.val_frac,
                seed=args.seed,
                kmer=args.kmer,
                feature_kind=feature_kind,
                backbone=args.backbone,
                layer=args.layer,
                cache_path=args.cache_path,
                model_kind=args.model_kind,
                calibration=args.calibration,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                lr=args.lr,
                device=args.device,
                early_stopping_rounds=args.early_stopping_rounds,
                top_k_features=args.top_k_features,
                importance=not args.no_importance,
                importance_top_n=args.importance_top_n,
                importance_estimators=args.importance_estimators,
                tabpfn_estimators=args.tabpfn_estimators,
                tabpfn_subsample_samples=args.tabpfn_subsample_samples,
                pair_eval=args.pair_eval,
                write=not args.no_write,
            )
            summary[method][feature_kind] = {
                "test_spearman": _metric(res, ["node_metrics", "test", "spearman_pred_degree"]),
                "test_high_degree_auroc": _metric(res, ["node_metrics", "test", "high_degree_auroc"]),
                "n_train": _metric(res, ["split", "n_train_nodes"]),
                "n_val": _metric(res, ["split", "n_val_nodes"]),
                "n_test": _metric(res, ["split", "n_test_nodes"]),
                "missing_train_features": _metric(res, ["features", "missing_train_features"]),
                "missing_test_features": _metric(res, ["features", "missing_test_features"]),
                "backbone": _metric(res, ["backbone"]),
                "layer": _metric(res, ["layer"]),
                "pair_human_test_auroc": _metric(
                    res, ["pair_metrics", "human_test_ppi", "pred_min_t_auroc"]
                ),
                "pair_human_test_auprc": _metric(
                    res, ["pair_metrics", "human_test_ppi", "pred_min_t_auprc"]
                ),
            }

    print("\n=== PRING full-graph sequence -> participation summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        resolved_layer = resolve_backbone_layer(args.backbone, args.layer)
        stem = f"pring_human_{args.model_kind}_{args.backbone}_l{resolved_layer}_summary"
        dump_experiment(
            args.out_dir / f"{stem}.json",
            task="protein.participation_oracle",
            dataset="pring_human",
            features="multi",
            split="multi",
            model=args.model_kind,
            seed=args.seed,
            payload=summary,
            metrics={
                method: {
                    feature: {
                        "test_spearman": cell.get("test_spearman"),
                        "test_high_degree_auroc": cell.get("test_high_degree_auroc"),
                    }
                    for feature, cell in method_cells.items()
                }
                for method, method_cells in summary.items()
            },
        )
        print(f"\n[done] results in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
