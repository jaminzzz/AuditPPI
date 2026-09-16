#!/usr/bin/env python3
"""Ladder-2 audit: is the C3 semantic-concordance signal a subcellular-localization confound?

Builds on Ladder-1 (analyze_c3_negative_sampling_bias.py), which showed positive
pairs are more SAE-similar than negatives (cosine AUROC ~0.68). Ladder-2 asks WHAT
that similarity encodes, using UniProt subcellular-localization annotation:

  1. CO-LOCALIZATION bias — do positive pairs share compartments more often than
     negatives?  If the negative sampler paired proteins at random, negatives will
     co-localize less (refs 11-12: the localization shortcut).  Measured by
     "shares >=1 compartment" and compartment Jaccard, with single-feature AUROC.

  2. LOCALIZATION-MATCHED HARD-NEGATIVE CONTROL (the decisive test) — restrict to
     the CO-LOCALIZED subset (pairs sharing >=1 compartment) and re-measure whether
     SAE cosine still separates pos/neg. This strips away easy localization-mismatch
     negatives and leaves harder co-localized negatives. If the global cosine AUROC
     (~0.68) collapses toward 0.5 on co-localized pairs, the concordance signal WAS
     largely localization, not learned interaction. Any residual AUROC is the
     concordance that survives the co-localized hard-negative control.

Compartments are coarse-grained from GO cellular-component (structured, GO-id bearing;
falls back to the free-text subcellular_cc only when go_cc is empty).

Inputs:
  results/audit_pair/negative_sampling_audit/c3_{split}_pair_ids.parquet
  results/audit_pair/negative_sampling_audit/c3_uniprot_localization.parquet
  data/sae/pair_caches/c3/{split}_pairs.pt
  data/sae/protein_caches/c3_protein_features_max1022.pt

Outputs:
  results/audit_pair/negative_sampling_audit/c3_{split}_localization_confound.json
  results/audit_pair/negative_sampling_audit/c3_{split}_protein_compartments.parquet

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/analysis/analyze_c3_localization_confound.py --split test
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from conf.paths import C3_PAIR_INDEX_CACHES, C3_SAE_CACHE, RESULTS_PAIR
from src.experiments.results import dump_experiment
from src.features.pairs import load_pair_index_cache, load_protein_feature_cache, materialize_pair_endpoints

AUDIT_DIR = RESULTS_PAIR / "negative_sampling_audit"
LOC_PARQUET = AUDIT_DIR / "c3_uniprot_localization.parquet"

# Coarse compartment map: keyword substring (lowercased GO CC / subcellular text) -> compartment.
# Order matters — first match wins, so put specific organelles before the generic "membrane".
COMPARTMENT_RULES: list[tuple[str, str]] = [
    ("nucleol", "nucleus"),
    ("nucleoplasm", "nucleus"),
    ("chromatin", "nucleus"),
    ("chromosome", "nucleus"),
    ("nuclear", "nucleus"),
    ("nucleus", "nucleus"),
    ("mitochond", "mitochondrion"),
    ("endoplasmic reticulum", "ER"),
    ("golgi", "golgi"),
    ("lysosom", "lysosome"),
    ("endosom", "endosome"),
    ("peroxisom", "peroxisome"),
    ("exosome", "secreted"),
    ("extracellular", "secreted"),
    ("secreted", "secreted"),
    ("cell surface", "plasma_membrane"),
    ("plasma membrane", "plasma_membrane"),
    ("focal adhesion", "plasma_membrane"),
    ("cell junction", "plasma_membrane"),
    ("synapse", "synapse"),
    ("synaptic", "synapse"),
    ("postsynaptic", "synapse"),
    ("presynaptic", "synapse"),
    ("dendrit", "synapse"),
    ("axon", "synapse"),
    ("neuron", "synapse"),
    ("cytoskelet", "cytoskeleton"),
    ("microtubule", "cytoskeleton"),
    ("actin", "cytoskeleton"),
    ("centrosome", "cytoskeleton"),
    ("spindle", "cytoskeleton"),
    ("cytosol", "cytoplasm"),
    ("cytoplasm", "cytoplasm"),
    ("perinuclear", "cytoplasm"),
    # generic membrane last, so specific organelle membranes are captured above
    ("membrane", "membrane_other"),
]


def text_to_compartments(text: str) -> set[str]:
    """Map a GO-CC / subcellular-location string to a set of coarse compartments."""
    if not text:
        return set()
    low = text.lower()
    # split into individual terms so a generic 'membrane' in one term does not
    # swallow a specific organelle named in another term
    terms = re.split(r"[;.]", low)
    comps: set[str] = set()
    for term in terms:
        term = term.strip()
        if not term:
            continue
        for kw, comp in COMPARTMENT_RULES:
            if kw in term:
                comps.add(comp)
                break
    return comps


def build_protein_compartments() -> dict[str, set[str]]:
    df = pd.read_parquet(LOC_PARQUET)
    out: dict[str, set[str]] = {}
    for acc, go_cc, sub_cc in zip(df["accession"], df["go_cc"].fillna(""), df["subcellular_cc"].fillna("")):
        comps = text_to_compartments(go_cc)
        if not comps:  # fall back to free-text only when GO CC is empty
            comps = text_to_compartments(sub_cc)
        out[acc] = comps
    return out


def auroc_safe(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def mwu_greater(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    try:
        _, p = mannwhitneyu(pos, neg, alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    split = args.split

    align = pd.read_parquet(AUDIT_DIR / f"c3_{split}_pair_ids.parquet")
    index_cache = load_pair_index_cache(C3_PAIR_INDEX_CACHES[split])
    protein_cache = load_protein_feature_cache(C3_SAE_CACHE)
    a_max, b_max, labels = materialize_pair_endpoints(index_cache, protein_cache, rep="sae_max")
    y = labels.numpy().astype(int)
    align_y = align["label"].to_numpy().astype(int)
    assert len(align_y) == len(y), f"align {len(align_y)} vs reps {len(y)}"
    assert np.array_equal(align_y, y), "label misalignment between pair ids and pair-index cache"

    a_max = a_max.float()
    b_max = b_max.float()
    cos = torch.nn.functional.cosine_similarity(a_max, b_max, dim=1).numpy()

    prot2comp = build_protein_compartments()

    ids_a = list(align["id_a"])
    ids_b = list(align["id_b"])

    # per-pair co-localization metrics
    shares = np.zeros(len(y), dtype=float)      # 1.0 if pair shares >=1 compartment
    comp_jac = np.zeros(len(y), dtype=float)    # Jaccard over compartment sets
    both_annotated = np.zeros(len(y), dtype=bool)
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        ca = prot2comp.get(a, set())
        cb = prot2comp.get(b, set())
        both_annotated[i] = bool(ca) and bool(cb)
        if ca and cb:
            inter = len(ca & cb)
            union = len(ca | cb)
            shares[i] = 1.0 if inter > 0 else 0.0
            comp_jac[i] = inter / union if union else 0.0

    report: dict = {
        "split": split,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "n_both_annotated": int(both_annotated.sum()),
    }

    # ---- 1. co-localization bias (only over pairs where both ends are annotated) ----
    m = both_annotated
    ym = y[m]
    for name, s_all in [("shares_compartment", shares), ("compartment_jaccard", comp_jac)]:
        s = s_all[m]
        pos, neg = s[ym == 1], s[ym == 0]
        report[name] = {
            "pos_mean": float(pos.mean()) if len(pos) else float("nan"),
            "neg_mean": float(neg.mean()) if len(neg) else float("nan"),
            "auroc_alone": auroc_safe(ym, s),
            "mwu_p": mwu_greater(pos, neg),
        }
    # co-localization rate among positives vs negatives (fraction sharing >=1 compartment)
    report["coloc_rate"] = {
        "pos": float(shares[m & (y == 1)].mean()) if (m & (y == 1)).sum() else float("nan"),
        "neg": float(shares[m & (y == 0)].mean()) if (m & (y == 0)).sum() else float("nan"),
    }

    # ---- 2. localization-matched hard-negative control: cosine AUROC on co-localized pairs ----
    coloc = m & (shares > 0)          # both annotated AND sharing >=1 compartment
    noncoloc = m & (shares == 0)      # both annotated AND sharing none
    report["cosine_auroc"] = {
        "all_pairs": auroc_safe(y, cos),
        "both_annotated": auroc_safe(y[m], cos[m]),
        "colocalized_subset": auroc_safe(y[coloc], cos[coloc]),
        "noncolocalized_subset": auroc_safe(y[noncoloc], cos[noncoloc]),
        "n_coloc": int(coloc.sum()),
        "n_coloc_pos": int((coloc & (y == 1)).sum()),
        "n_coloc_neg": int((coloc & (y == 0)).sum()),
        "n_noncoloc": int(noncoloc.sum()),
    }

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / f"c3_{split}_localization_confound.json"
    dump_experiment(
        out,
        task="pair.localization_confound",
        dataset="c3",
        features="sae_max",
        split=split,
        model="na",
        seed=-1,
        payload=report,
        metrics=report.get("cosine_auroc"),
    )

    # persist per-protein compartments for this split (for downstream figures)
    uniq = sorted(set(ids_a) | set(ids_b))
    comp_df = pd.DataFrame(
        {"accession": uniq, "compartments": [";".join(sorted(prot2comp.get(p, set()))) for p in uniq]}
    )
    comp_df.to_parquet(AUDIT_DIR / f"c3_{split}_protein_compartments.parquet", index=False)

    # console summary
    cj = report["cosine_auroc"]
    print(f"[{split}] n={report['n']} both_annotated={report['n_both_annotated']}")
    print(f"  coloc rate: pos={report['coloc_rate']['pos']:.3f} neg={report['coloc_rate']['neg']:.3f}"
          f"  (shares>=1 compartment AUROC={report['shares_compartment']['auroc_alone']:.4f})")
    print(f"  cosine AUROC: all={cj['all_pairs']:.4f}  both_annot={cj['both_annotated']:.4f}"
          f"  COLOC={cj['colocalized_subset']:.4f} (n={cj['n_coloc']}, {cj['n_coloc_pos']}+/{cj['n_coloc_neg']}-)"
          f"  noncoloc={cj['noncolocalized_subset']:.4f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
