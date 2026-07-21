#!/usr/bin/env python3
"""Layer-1 audit: characterize the C3 negative-sampling construction bias.

The C3 benchmark hands us `(pair, label)` with no record of HOW each negative was
constructed. This script reverse-engineers the construction bias from the data itself,
asking two questions that directly bound the concordance / participation audits:

  (1) SEMANTIC DISTANCE bias. Are negative pairs systematically LESS similar in SAE
      feature space than positive pairs? If so, a "semantic concordance" signal
      (cosine / Jaccard) separates the classes partly because the negative sampler
      glued "far apart in feature space" to the negative label -- not because the
      model learned interaction. We report the positive-vs-negative distributions of
        - dense cosine on the continuous sae_max fingerprint
        - binary Jaccard on the fired-feature sets
      and the AUROC of each *model-free* similarity used as a lone classifier.

  (2) DEGREE bias. Random negative sampling pairs two proteins drawn ~uniformly, so
      negatives skew toward low-degree proteins (hubs are rarely hit by chance), while
      positives concentrate on hubs. We report the degree distribution of proteins on
      the positive vs negative side, and the AUROC of endpoint degree (min / max)
      predicting the label -- the participation shortcut, measured on THIS split's own
      construction.

Inputs (all row-aligned, verified by export_c3_pair_id_alignment.py):
  - pair-index cache  c3/{split}_pairs.pt  (rows_a/rows_b/labels into the C3 v1
    protein cache; one file serves both sae_max and binary channels)
  - C3 v1 protein cache                    (auditppi_protein_features_v1)
  - c3_{split}_pair_ids.parquet            (row -> id_a, id_b, label)

Run in E1:
  python scripts/audit_pair/analyze_c3_negative_sampling_bias.py --split test
  python scripts/audit_pair/analyze_c3_negative_sampling_bias.py --split train
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auditppi")

import numpy as np
import pandas as pd

from conf.paths import C3_PAIR_INDEX_CACHES, C3_SAE_CACHE, RESULTS_PAIR
from src.experiments.results import dump_experiment
from src.features.pairs import load_pair_index_cache, load_protein_feature_cache, materialize_pair_endpoints

ALIGN_DIR = RESULTS_PAIR / "negative_sampling_audit"
OUT_DIR = RESULTS_PAIR / "negative_sampling_audit"


def _load_rep(rep: str, split: str, *, index_cache: dict, protein_cache: dict):
    """Materialize (emb_a, emb_b, label) for a rep from the pair-index cache."""
    emb_a, emb_b, labels = materialize_pair_endpoints(
        index_cache, protein_cache, rep=rep
    )
    return emb_a, emb_b, labels.numpy().astype(np.int8)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between paired rows of a and b."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros_like(num, dtype=np.float64)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _row_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Jaccard over fired-feature sets (a, b are 0/1)."""
    inter = np.minimum(a, b).sum(axis=1).astype(np.float64)
    union = np.maximum(a, b).sum(axis=1).astype(np.float64)
    out = np.zeros_like(inter)
    nz = union > 0
    out[nz] = inter[nz] / union[nz]
    return out


def _dist_summary(x: np.ndarray) -> dict:
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p5": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "p95": float(q[4]),
    }


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _mann_whitney(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U with a rank-biserial effect size (= 2*AUC - 1)."""
    from scipy.stats import mannwhitneyu

    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(u) / (pos.size * neg.size)  # P(pos > neg)
    return {"u": float(u), "p_value": float(p), "auc_pos_gt_neg": auc,
            "rank_biserial": 2.0 * auc - 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    split = args.split

    # --- load row-aligned inputs -----------------------------------------------------
    index_cache = load_pair_index_cache(C3_PAIR_INDEX_CACHES[split])
    protein_cache = load_protein_feature_cache(C3_SAE_CACHE)
    sae_a, sae_b, y_sae = _load_rep("sae_max", split, index_cache=index_cache, protein_cache=protein_cache)
    bin_a, bin_b, y_bin = _load_rep("binary", split, index_cache=index_cache, protein_cache=protein_cache)
    align = pd.read_parquet(ALIGN_DIR / f"c3_{split}_pair_ids.parquet")
    y = align["label"].to_numpy().astype(np.int8)

    # alignment sanity: reps labels must equal parquet labels row-for-row
    assert len(align) == y_sae.size == y_bin.size, (len(align), y_sae.size, y_bin.size)
    assert np.array_equal(y, y_sae) and np.array_equal(y, y_bin), "label misalignment"

    sae_a = sae_a.float().numpy()
    sae_b = sae_b.float().numpy()
    bin_a = bin_a.numpy().astype(np.int8)
    bin_b = bin_b.numpy().astype(np.int8)

    # --- (1) semantic-distance bias --------------------------------------------------
    cos = _row_cosine(sae_a, sae_b)
    jac = _row_jaccard(bin_a, bin_b)
    pos_mask = y == 1
    neg_mask = y == 0

    similarity = {}
    for name, s in (("cosine_sae_max", cos), ("jaccard_binary", jac)):
        similarity[name] = {
            "positive": _dist_summary(s[pos_mask]),
            "negative": _dist_summary(s[neg_mask]),
            "auroc_similarity_alone": _auroc(y, s),
            "mann_whitney_pos_vs_neg": _mann_whitney(s[pos_mask], s[neg_mask]),
        }

    # --- (2) degree bias -------------------------------------------------------------
    from src.eval.metrics import participation_t

    pairs = list(zip(align["id_a"], align["id_b"]))
    _, degree = participation_t(pairs, y)  # degree[p] = #pairs touching p in THIS split
    deg_a = np.array([degree[a] for a, _ in pairs], dtype=np.float64)
    deg_b = np.array([degree[b] for _, b in pairs], dtype=np.float64)
    deg_min = np.minimum(deg_a, deg_b)
    deg_max = np.maximum(deg_a, deg_b)

    degree_bias = {}
    for name, s in (("degree_min", deg_min), ("degree_max", deg_max)):
        degree_bias[name] = {
            "positive": _dist_summary(s[pos_mask]),
            "negative": _dist_summary(s[neg_mask]),
            "auroc_degree_alone": _auroc(y, s),
            "mann_whitney_pos_vs_neg": _mann_whitney(s[pos_mask], s[neg_mask]),
        }

    # per-protein degree split by whether the protein ever appears in a positive pair
    pos_proteins = {p for (a, b), yy in zip(pairs, y) if yy == 1 for p in (a, b)}
    neg_only_proteins = {p for p in degree if p not in pos_proteins}
    degree_bias["protein_level"] = {
        "n_proteins": len(degree),
        "n_in_any_positive": len(pos_proteins),
        "n_negative_only": len(neg_only_proteins),
        "degree_of_positive_proteins": _dist_summary(
            np.array([degree[p] for p in pos_proteins], dtype=np.float64)
        ),
        "degree_of_negative_only_proteins": (
            _dist_summary(np.array([degree[p] for p in neg_only_proteins], dtype=np.float64))
            if neg_only_proteins else None
        ),
    }

    result = {
        "split": split,
        "n_pairs": int(y.size),
        "n_positive": int(pos_mask.sum()),
        "n_negative": int(neg_mask.sum()),
        "pos_rate": float(y.mean()),
        "semantic_distance_bias": similarity,
        "degree_bias": degree_bias,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"c3_{split}_negative_sampling_bias.json"
    dump_experiment(
        out_json,
        task="pair.negative_sampling_bias",
        dataset="c3",
        features="sae_max+binary",
        split=split,
        model="na",
        seed=-1,
        payload=result,
    )

    # per-pair table for downstream plotting
    pd.DataFrame({
        "label": y,
        "cosine_sae_max": cos,
        "jaccard_binary": jac,
        "degree_min": deg_min,
        "degree_max": deg_max,
    }).to_parquet(args.out_dir / f"c3_{split}_pair_similarity_degree.parquet")

    # --- console summary -------------------------------------------------------------
    print(f"[{split}] n={y.size} pos={pos_mask.sum()} neg={neg_mask.sum()} pos_rate={y.mean():.4f}")
    print("  --- semantic-distance bias (higher for positives => sampler glued 'far' to negative) ---")
    for name in ("cosine_sae_max", "jaccard_binary"):
        s = similarity[name]
        print(f"    {name:16s} pos_med={s['positive']['median']:.4f} "
              f"neg_med={s['negative']['median']:.4f} "
              f"AUROC(sim alone)={s['auroc_similarity_alone']:.4f} "
              f"MWU p={s['mann_whitney_pos_vs_neg']['p_value']:.2e}")
    print("  --- degree bias (higher for positives => hubs concentrate in positives) ---")
    for name in ("degree_min", "degree_max"):
        s = degree_bias[name]
        print(f"    {name:16s} pos_med={s['positive']['median']:.1f} "
              f"neg_med={s['negative']['median']:.1f} "
              f"AUROC(deg alone)={s['auroc_degree_alone']:.4f} "
              f"MWU p={s['mann_whitney_pos_vs_neg']['p_value']:.2e}")
    pl = degree_bias["protein_level"]
    print(f"  proteins: {pl['n_proteins']} total, {pl['n_in_any_positive']} in >=1 positive, "
          f"{pl['n_negative_only']} negative-only")
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
