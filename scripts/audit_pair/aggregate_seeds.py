"""Aggregate PPI fingerprint summaries across seeds into mean±std tables.

Scans ``OUT_DIR/{family}/{model}/seed_{S}/summaries/*.json`` for every
seed in :data:`FINGERPRINT_SEEDS`, groups metric cells by their
seed-independent identity (the ``/seed{S}/`` segment is stripped from each
metric key), and writes one aggregate JSON per (family, model, axis[, mode])
holding mean/std/n over the seeds that were actually found.

Output layout::

    {family}/{model}/aggregate/{axis}[_{pair_mode}].json

Each aggregate mirrors the summary shape: a ``metrics`` map keyed by the
seed-independent cell id, each value carrying ``auroc_mean/auroc_std`` and
``auprc_mean/auprc_std`` plus ``seeds`` (the list found) and ``n_seeds``.
Missing seeds are reported, not fatal -- the aggregate records how many
seeds backed each cell so partial matrices remain usable.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from src.ppi_fingerprint.config import (
    FINGERPRINT_SEEDS,
    MODEL_NAMES,
    OUT_DIR,
)

# Segments stripped from a metric key to give a seed- and mode-independent cell
# identity. ``/seed{N}/`` is dropped so seeds group together; the pair-mode token
# is dropped too because each aggregate stem is already mode-specific and the
# migrated seed-42 keys omit it (e.g. ``xgb/binary/esmcL80/c1:test`` vs the newer
# ``xgb/binary/esmcL80/sym/seed43/c1:test``), so collapsing it lets all seeds
# land in one group.
_SEED_SEG = re.compile(r"/seed\d+/")
_MODE_SEG = re.compile(r"/(?:sym|concat|rich|product|absdiff|sum)(?=/)")


def _strip_seed(metric_key: str) -> str:
    """Canonical seed/mode-independent cell id.

    ``mlp_pair/binary/esm2L33/rich/seed42/c1:test`` → ``mlp_pair/binary/esm2L33/c1:test``;
    ``xgb/binary/esmcL80/sym/seed43/c1:test``        → ``xgb/binary/esmcL80/c1:test``;
    ``xgb/binary/esmcL80/c1:test`` (migrated)        → unchanged.
    """
    return _MODE_SEG.sub("", _SEED_SEG.sub("/", metric_key))


def _families(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_group(
    family: str,
    model: str,
    stem: str,
    seeds: tuple[int, ...],
    root: Path,
) -> dict | None:
    """Aggregate one summary stem (e.g. ``esm2L33_rich``) across seeds.

    Returns the aggregate dict, or ``None`` if no seed had this summary.
    """
    # cell id -> {"auroc": [...], "auprc": [...], "seeds": [...], "meta": {...}}
    cells: dict[str, dict] = {}
    seeds_found: list[int] = []

    for seed in seeds:
        path = root / family / model / f"seed_{seed}" / "summaries" / f"{stem}.json"
        if not path.exists():
            continue
        seeds_found.append(seed)
        doc = json.loads(path.read_text())
        for key, val in doc.get("metrics", {}).items():
            # Skip cells a run wrote as a failure sentinel (e.g. tabm_pair OOM
            # dumps ``{"error": "CUDA out of memory..."}`` instead of metrics).
            # A partial matrix stays aggregatable; only healthy cells contribute.
            if "auroc" not in val or "auprc" not in val:
                continue
            cid = _strip_seed(key)
            slot = cells.setdefault(
                cid,
                {"auroc": [], "auprc": [], "seeds": [], "meta": {}},
            )
            slot["auroc"].append(float(val["auroc"]))
            slot["auprc"].append(float(val["auprc"]))
            slot["seeds"].append(seed)
            # Keep the seed-independent metadata from the first sighting.
            if not slot["meta"]:
                slot["meta"] = {
                    k: v
                    for k, v in val.items()
                    if k not in ("auroc", "auprc", "seed")
                }

    if not seeds_found:
        return None

    # Reload one doc for header fields (task/dataset/model/hyperparameters).
    header_src = root / family / model / f"seed_{seeds_found[0]}" / "summaries" / f"{stem}.json"
    header = json.loads(header_src.read_text())

    metrics_out: dict[str, dict] = {}
    for cid, slot in sorted(cells.items()):
        au_mean, au_std = _mean_std(slot["auroc"])
        ap_mean, ap_std = _mean_std(slot["auprc"])
        entry = {
            "auroc_mean": round(au_mean, 6),
            "auroc_std": round(au_std, 6),
            "auprc_mean": round(ap_mean, 6),
            "auprc_std": round(ap_std, 6),
            "n_seeds": len(slot["seeds"]),
            "seeds": slot["seeds"],
        }
        entry.update(slot["meta"])
        metrics_out[cid] = entry

    hp = dict(header.get("hyperparameters", {}))
    hp.pop("seed", None)

    return {
        "schema_version": 1,
        "task": header.get("task"),
        "dataset": header.get("dataset", family),
        "features": header.get("features"),
        "split": header.get("split"),
        "model": model,
        "aggregate": "seed_mean_std",
        "seeds_requested": list(seeds),
        "seeds_found": seeds_found,
        "hyperparameters": hp,
        "metrics": metrics_out,
    }


def run(root: Path, seeds: tuple[int, ...], families: list[str] | None) -> None:
    fam_list = families or _families(root)
    written = 0
    for family in fam_list:
        fam_dir = root / family
        if not fam_dir.is_dir():
            continue
        for model in MODEL_NAMES:
            # Collect the union of summary stems across all seeds for this model.
            stems: set[str] = set()
            for seed in seeds:
                sdir = fam_dir / model / f"seed_{seed}" / "summaries"
                if sdir.is_dir():
                    stems.update(
                        p.stem
                        for p in sdir.glob("*.json")
                        if not p.name.endswith(".prov.json")
                    )
            if not stems:
                continue
            out_dir = fam_dir / model / "aggregate"
            out_dir.mkdir(parents=True, exist_ok=True)
            for stem in sorted(stems):
                agg = aggregate_group(family, model, stem, seeds, root)
                if agg is None:
                    continue
                out_path = out_dir / f"{stem}.json"
                out_path.write_text(json.dumps(agg, indent=2))
                n = len(agg["seeds_found"])
                flag = "" if n == len(seeds) else f"  (only {n}/{len(seeds)} seeds)"
                print(f"[agg] {family}/{model}/{stem}: n_seeds={n}{flag}")
                written += 1
    print(f"\nDone. wrote {written} aggregate summaries under {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=OUT_DIR,
        help=f"fingerprint results root (default: {OUT_DIR})",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(FINGERPRINT_SEEDS),
        help=f"seeds to aggregate (default: {list(FINGERPRINT_SEEDS)})",
    )
    ap.add_argument(
        "--family",
        nargs="+",
        default=None,
        help="restrict to these families (default: all under root)",
    )
    args = ap.parse_args()
    run(args.root, tuple(args.seeds), args.family)


if __name__ == "__main__":
    main()
