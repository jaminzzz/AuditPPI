"""Ladder-2 PPI prediction from pooled per-protein SAE fingerprints.

Trains a classifier (XGB / TabPFN / MLP-pair / TabM-pair) on a pooled representation
(``binary`` / ``sae_max`` / ``esmc_mean``) of each benchmark's **own** native
train set, then scores its eval split(s) and reports AUROC/AUPRC. This is the
pooled-fingerprint "participation channel" pair-scale predictor -- endpoints
carry only per-protein features (no pair-interaction terms), so a high score
means the label is decidable from participation alone.

Feature source (v1): each family's ``auditppi_protein_features_v1`` protein
cache (``conf.paths.PPI_PREDICTION_CACHES`` / ``PRING_SPECIES_SAE_CACHES``) holds
every endpoint sequence across that family's splits, with BOTH backbone lines
(ESM-C L60/L80 + ESM-2 L33) and all channels in one payload. ``--backbone`` /
``--layer`` pick the channel within the cache; ``--rep`` picks the pooling view.
``mlp_pair`` / ``tabm_pair`` additionally take ``--pair-mode`` (shared vocabulary:
``sym`` / ``concat`` / ``rich`` / ``product`` / ``absdiff`` / ``sum``; only
``concat`` uses AB/BA train/eval).

Evals that share the same native train are fit **once** and scored many times
(e.g. ``cross_species``: one ``human_train`` fit → human_test + all species;
PRING BFS train → human BFS test + yeast/ecoli/arath).

Results::

    results/main/ppi_fingerprint/{family}/{model}/cells/...
    results/main/ppi_fingerprint/{family}/{model}/summaries/{backbone}L{layer}[_{pair_mode}].json

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/run_ppi_fingerprint_baseline.py --model xgb --family c3
    $PY scripts/audit_pair/run_ppi_fingerprint_baseline.py --model mlp_pair --family c3 \\
        --pair-mode sym
    $PY scripts/audit_pair/run_ppi_fingerprint_baseline.py --model tabm_pair --family c3 \\
        --pair-mode concat --backbone esm2 --layer 33
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from conf.model import (
    BACKBONE_LAYERS,
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_SEED,
    resolve_backbone_layer,
)
from conf.paths import PRING_CROSS_SPECIES
from src.data.pairs import list_cross_species
from src.data.pring_graph import METHODS as PRING_METHODS
from src.experiments.results import dump_experiment
from src.features.pairs import PAIR_MODES
from src.ppi_fingerprint.config import (
    MODEL_NAMES as MODELS,
    PAIR_MODELS,
    PRING_DEFAULT_METHOD,
    REPRESENTATIONS as REPS,
    summary_path,
)
from src.runtime.device import pick_free_gpu


def family_evals(family: str) -> list[str]:
    """Expand a benchmark family into its eval benchmark name(s)."""
    if family in {"c1", "c2", "c3"}:
        return [f"{family}:test"]
    if family == "cross_species":
        # Trains once on human; scores human self-test plus every held-out species
        # (ecoli/fly/mouse/worm/yeast) zero-shot -- the dataset's whole point.
        species = [s for s in list_cross_species() if s != "human_train"]
        return [f"cross_species:{s}" for s in species]
    if family == "bernett":
        return ["bernett:test"]
    if family == "pring":
        evals = [f"pring:human:test:{m}" for m in PRING_METHODS]
        evals += [f"pring:{sp}:test" for sp in PRING_CROSS_SPECIES]
        return evals
    if family == "rf2ppi":
        return ["rf2ppi"]
    raise ValueError(f"unknown family {family!r}")


FAMILIES = ("c1", "c2", "c3", "cross_species", "bernett", "pring", "rf2ppi")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="xgb")
    p.add_argument("--family", choices=FAMILIES, default="c3",
                   help="benchmark family; expands to its eval benchmark(s)")
    p.add_argument("--backbone", choices=BACKBONES, default=DEFAULT_BACKBONE)
    p.add_argument("--layer", type=int, default=None,
                   help="within-cache layer (default: backbone default)")
    p.add_argument("--reps", nargs="*", choices=REPS, default=list(REPS),
                   help="pooling views to sweep (default: all)")
    p.add_argument("--pair-mode", choices=PAIR_MODES, default="sym",
                   help="pair assemble for mlp_pair/tabm_pair (ignored by xgb/tabpfn; "
                        "only concat uses AB/BA)")
    p.add_argument("--top-k", type=int, default=500, help="TabPFN feature cap")
    p.add_argument("--train-subsample", type=int, default=100000)
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    layer = resolve_backbone_layer(args.backbone, args.layer)
    if layer not in BACKBONE_LAYERS[args.backbone]:
        raise ValueError(f"backbone {args.backbone!r} has no layer {layer}")

    dev = str(args.device_id) if args.device_id is not None else str(pick_free_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    print(f"[device] CUDA_VISIBLE_DEVICES={dev}", flush=True)

    # Import after env pin so CUDA device selection is stable.
    from src.ppi_fingerprint.baseline import run_baseline_evals

    evals = family_evals(args.family)
    b_tag = f"{args.backbone}L{layer}"
    effective_mode = args.pair_mode if args.model in PAIR_MODELS else "sym"
    print(
        f"[plan] family={args.family} model={args.model} backbone={b_tag} "
        f"reps={args.reps} pair_mode={effective_mode} evals={evals}",
        flush=True,
    )

    summary: dict[str, dict] = {}
    for rep in args.reps:
        try:
            rows = run_baseline_evals(
                args.model, rep, evals,
                backbone=args.backbone, layer=layer,
                pair_mode=args.pair_mode,
                top_k=args.top_k, train_subsample=args.train_subsample,
                seed=args.seed,
            )
            for r in rows:
                key = f"{args.model}/{rep}/{b_tag}/{effective_mode}/{r['eval']}"
                summary[key] = {
                    "auroc": r["auroc"],
                    "auprc": r["auprc"],
                    "n_skip": r["n_skipped_eval"],
                    "train": r["train"],
                    "pair_mode": r.get("pair_mode"),
                }
        except Exception as exc:  # noqa: BLE001  keep the sweep going
            for ev in evals:
                key = f"{args.model}/{rep}/{b_tag}/{effective_mode}/{ev}"
                print(f"[skip] {key}: {exc}", flush=True)
                summary[key] = {"error": str(exc)}

    print(f"\n=== ppi_fingerprint {args.family}/{args.model}/{b_tag}: AUROC / AUPRC ===", flush=True)
    for key, value in summary.items():
        print(f"  {key:52s} {value}", flush=True)

    out_path = summary_path(
        args.family, args.model, b_tag, pair_mode=effective_mode,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_experiment(
        out_path,
        task="pair.ppi_fingerprint",
        dataset=args.family,
        features="multi" if len(args.reps) > 1 else args.reps[0],
        split="multi" if len(evals) > 1 else evals[0],
        model=args.model,
        seed=args.seed,
        payload=summary,
        metrics=summary,
        hyperparameters={
            "backbone": args.backbone, "layer": layer,
            "top_k": args.top_k, "train_subsample": args.train_subsample,
            "reps": list(args.reps), "pair_mode": effective_mode,
        },
    )
    print(f"\n[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
