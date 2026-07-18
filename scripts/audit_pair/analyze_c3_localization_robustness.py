#!/usr/bin/env python3
"""Layer-2 ROBUSTNESS: is the co-localization confound driven by generic compartments?

Follow-up to analyze_c3_localization_confound.py. The baseline "shares >=1 compartment"
definition is lenient: cytoplasm (50.6% of test proteins) and nucleus (45.8%) are so
common that two random human proteins co-localize by generic-label collision alone
(~0.51^2 + ... chance on cytoplasm), inflating the co-localization rate and weakening
the hard-negative-stripping control.

This script re-runs the decisive test (cosine AUROC on the co-localized subset) under
three co-localization definitions of increasing stringency, all on the SAME pairs and
SAME compartment sets as the baseline audit:

  A. ANY        - shares >=1 compartment                       (baseline, reproduces the JSON)
  B. NO_GENERIC - shares >=1 compartment AFTER dropping        (removes cytoplasm/nucleus
                  cytoplasm + nucleus from every protein's set  generic-collision floor)
  C. RARE_ONLY  - shares >=1 RARE compartment                  (peroxisome/lysosome/endosome/
                                                                 golgi/mito/ER/cytoskeleton/synapse)

For each definition we report: co-localization rate (pos vs neg), shares-alone AUROC,
and — the point of the exercise — cosine AUROC restricted to the co-localized subset
vs the non-co-localized subset. If the concordance signal that "survives" the coloc
control is stable across A/B/C, the confound is not a generic-label artifact; if the
coloc-subset cosine AUROC rises toward the global value as we tighten the definition,
the baseline "shares>=1" was diluting the control with generic-collision pairs.

Read-only w.r.t. the baseline audit outputs; writes its own JSON.

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/audit_pair/analyze_c3_localization_robustness.py --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.experiments.results import dump_experiment

# reuse the exact loaders / metrics / compartment map from the baseline audit
from analyze_c3_localization_confound import (
    AUDIT_DIR,
    SAE_MAX_DIR,
    auroc_safe,
    build_protein_compartments,
    mwu_greater,
)

# Generic compartments whose high marginal frequency creates a co-localization floor.
GENERIC = {"cytoplasm", "nucleus"}

# Rare / specific compartments: sharing one of these is strong evidence of true
# functional co-localization rather than a generic-label collision.
RARE = {
    "peroxisome",
    "lysosome",
    "endosome",
    "golgi",
    "mitochondrion",
    "ER",
    "cytoskeleton",
    "synapse",
}


def coloc_flags(ca: set[str], cb: set[str], mode: str) -> tuple[bool, bool]:
    """Return (both_annotated, shares) under a given co-localization definition.

    both_annotated is recomputed per-mode: e.g. under NO_GENERIC a protein annotated
    ONLY as cytoplasm becomes unannotated, so pairs involving it drop out of the
    controlled subset instead of counting as a (spurious) non-co-localization.
    """
    if mode == "any":
        sa, sb = ca, cb
    elif mode == "no_generic":
        sa, sb = ca - GENERIC, cb - GENERIC
    elif mode == "rare_only":
        sa, sb = ca & RARE, cb & RARE
    else:
        raise ValueError(mode)
    both = bool(sa) and bool(sb)
    shares = both and len(sa & sb) > 0
    return both, shares


def eval_mode(
    mode: str,
    y: np.ndarray,
    cos: np.ndarray,
    ids_a: list[str],
    ids_b: list[str],
    prot2comp: dict[str, set[str]],
) -> dict:
    n = len(y)
    both = np.zeros(n, dtype=bool)
    shares = np.zeros(n, dtype=float)
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        bo, sh = coloc_flags(prot2comp.get(a, set()), prot2comp.get(b, set()), mode)
        both[i] = bo
        shares[i] = 1.0 if sh else 0.0

    m = both
    ym = y[m]
    sm = shares[m]
    pos, neg = sm[ym == 1], sm[ym == 0]

    coloc = m & (shares > 0)
    noncoloc = m & (shares == 0)
    return {
        "mode": mode,
        "n_both_annotated": int(m.sum()),
        "coloc_rate_pos": float(shares[m & (y == 1)].mean()) if (m & (y == 1)).sum() else float("nan"),
        "coloc_rate_neg": float(shares[m & (y == 0)].mean()) if (m & (y == 0)).sum() else float("nan"),
        "shares_auroc": auroc_safe(ym, sm),
        "shares_mwu_p": mwu_greater(pos, neg),
        "cosine_auroc": {
            "all_pairs": auroc_safe(y, cos),
            "both_annotated": auroc_safe(y[m], cos[m]),
            "colocalized_subset": auroc_safe(y[coloc], cos[coloc]),
            "noncolocalized_subset": auroc_safe(y[noncoloc], cos[noncoloc]),
            "n_coloc": int(coloc.sum()),
            "n_coloc_pos": int((coloc & (y == 1)).sum()),
            "n_coloc_neg": int((coloc & (y == 0)).sum()),
            "n_noncoloc": int(noncoloc.sum()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    split = args.split

    align = pd.read_parquet(AUDIT_DIR / f"c3_{split}_pair_ids.parquet")
    d_max = torch.load(SAE_MAX_DIR / f"{split}_embeddings.pt", map_location="cpu", weights_only=False)
    y = d_max["label"].numpy().astype(int)
    assert len(align) == len(y), f"align {len(align)} vs reps {len(y)}"

    a_max = d_max["emb_a"].float()
    b_max = d_max["emb_b"].float()
    cos = torch.nn.functional.cosine_similarity(a_max, b_max, dim=1).numpy()

    prot2comp = build_protein_compartments()
    ids_a = list(align["id_a"])
    ids_b = list(align["id_b"])

    report = {
        "split": split,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "generic_dropped": sorted(GENERIC),
        "rare_set": sorted(RARE),
        "modes": {},
    }
    for mode in ("any", "no_generic", "rare_only"):
        report["modes"][mode] = eval_mode(mode, y, cos, ids_a, ids_b, prot2comp)

    out = AUDIT_DIR / f"c3_{split}_localization_robustness.json"
    dump_experiment(
        out,
        task="pair.localization_robustness",
        dataset="c3",
        features="sae_max",
        split=split,
        model="na",
        seed=-1,
        payload=report,
    )

    print(f"[{split}] n={report['n']} ({report['n_pos']}+/{report['n_neg']}-)")
    print(f"{'mode':<12} {'both':>6} {'coloc+':>7} {'coloc-':>7} {'shrAUC':>7} "
          f"{'cos_all':>8} {'cos_COLOC':>10} {'cos_noncol':>11}  n_coloc(+/-)")
    for mode in ("any", "no_generic", "rare_only"):
        r = report["modes"][mode]
        c = r["cosine_auroc"]
        print(f"{mode:<12} {r['n_both_annotated']:>6} "
              f"{r['coloc_rate_pos']:>7.3f} {r['coloc_rate_neg']:>7.3f} {r['shares_auroc']:>7.4f} "
              f"{c['all_pairs']:>8.4f} {c['colocalized_subset']:>10.4f} {c['noncolocalized_subset']:>11.4f}"
              f"  {c['n_coloc']}({c['n_coloc_pos']}+/{c['n_coloc_neg']}-)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
