#!/usr/bin/env python3
"""Backfill the endpoint-min pair shortcut into the PRING high-degree products.

The high-degree classifiers under ``PRING_PARTICIPATION_DIR/high_p90_xgboost``
were run with ``--pair-eval none``, so their JSONs carry ``node_metrics`` but an
empty ``pair_metrics``. Supplementary Fig. 5 needs both halves: the node-level
AUROC for "sequence predicts hubness" and the pair-level AUROC for "the pair
labels largely suppress it".

Recovering the pair half needs no refitting. Each run already wrote every
protein's ``pred_high_prob`` to its ``_protein_predictions.tsv``, and the pair
shortcut is just ``min(p_a, p_b)`` scored against PRING's labelled test edge
list -- the same :func:`src.eval.participation.endpoint_min_pair_metrics` the
runner itself calls under ``--pair-eval human_test``.

Self-pairs are KEPT (``drop_self_pairs=False``): the C-level participation
products Fig. S5 plots alongside these keep them, and every PRING self-pair is a
positive (BFS 1891, DFS 1708, RW 1701), so dropping them on one side of the
figure only would make the two blocks non-comparable.

The script rewrites only the ``payload.pair_*`` keys, in place and idempotently;
``payload.node_metrics`` and the predictions TSVs are left untouched.

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_protein/backfill_pring_degree_pair_metrics.py
    $PY scripts/audit_protein/backfill_pring_degree_pair_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from conf.audit import PARTICIPATION_QUANTILE
from conf.model import DEFAULT_BACKBONE, resolve_backbone_layer
from conf.paths import PRING_PARTICIPATION_DIR, PRING_ROOT
from src.data.pring_graph import METHODS
from src.eval.participation import endpoint_min_pair_metrics, read_labeled_pairs
from src.features.sequence_composition import FORMAL_FEATURE_KINDS

# Matches the stem `run_pring_high_participation_classifier.py` builds.
STEM = "pring_human_{method}_{rep}_{backbone}_l{layer}_high_q{q:02d}_xgboost"
PAIR_FILE = "human_test_ppi.txt"
# Provenance marker: these numbers were derived from a finished run's saved
# per-protein predictions, not produced during that run.
SOURCE = "post-hoc from *_protein_predictions.tsv (no refit)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=PRING_ROOT)
    p.add_argument("--dir", type=Path, default=PRING_PARTICIPATION_DIR / "high_p90_xgboost",
                   help="directory holding the high-degree classifier products")
    p.add_argument("--method", choices=[*METHODS, "all"], default="all")
    p.add_argument("--feature-kind", choices=[*FORMAL_FEATURE_KINDS, "all"], default="all")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--quantile", type=float, default=PARTICIPATION_QUANTILE)
    p.add_argument("--keep-self-pairs", action="store_true", default=True,
                   help="keep a==b rows (default: on, matching the C-level products)")
    p.add_argument("--drop-self-pairs", dest="keep_self_pairs", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="compute and print, but do not write the JSONs")
    return p.parse_args()


def read_pred_prob(path: Path) -> dict[str, float]:
    """``{protein: pred_high_prob}`` across every split of one prediction TSV."""
    with path.open() as handle:
        return {
            row["protein"]: float(row["pred_high_prob"])
            for row in csv.DictReader(handle, delimiter="\t")
        }


def main() -> None:
    args = parse_args()
    layer = resolve_backbone_layer(args.backbone, args.layer)
    methods = METHODS if args.method == "all" else (args.method,)
    features = FORMAL_FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)
    drop_self = not args.keep_self_pairs

    written, missing = 0, []
    for method in methods:
        pair_path = args.root / "human" / method / PAIR_FILE
        pairs, labels = read_labeled_pairs(pair_path, drop_self_pairs=drop_self)
        for feature in features:
            stem = STEM.format(
                method=method.lower(), rep=feature, backbone=args.backbone,
                layer=layer, q=int(args.quantile * 100),
            )
            json_path = args.dir / f"{stem}.json"
            pred_path = args.dir / f"{stem}_protein_predictions.tsv"
            if not (json_path.exists() and pred_path.exists()):
                missing.append(stem)
                continue

            metrics = endpoint_min_pair_metrics(pairs, labels, read_pred_prob(pred_path))
            metrics["path"] = str(pair_path)
            metrics["drop_self_pairs"] = drop_self

            print(
                f"[{method}.{feature}] n_scored={metrics['n_scored']} "
                f"skipped={metrics['n_skipped']} pos_rate={metrics['pos_rate']} "
                f"auroc={metrics['auroc']:.4f} auprc={metrics['auprc']:.4f}",
                flush=True,
            )
            if args.dry_run:
                continue

            record = json.loads(json_path.read_text())
            payload = record["payload"]
            payload["pair_eval"] = "human_test"
            payload["pair_metrics"] = {"human_test_ppi": metrics}
            payload["pair_metrics_source"] = SOURCE
            json_path.write_text(json.dumps(record, indent=2))
            written += 1

    if missing:
        print(f"[warn] no product for: {', '.join(missing)}", flush=True)
    print(f"[done] {'dry-run, 0' if args.dry_run else written} JSONs updated", flush=True)


if __name__ == "__main__":
    main()
