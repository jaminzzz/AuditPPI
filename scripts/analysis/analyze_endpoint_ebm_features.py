#!/usr/bin/env python3
"""Ladder-2 INTERPRETATION: which SAE features carry the endpoint shortcut?

The endpoint-additive EBM (``run_clevel_sae_endpoint_additive_ebm.py`` /
``run_pring_sae_endpoint_additive_ebm.py``) already writes a per-feature
importance table and a per-bin shape table for every family x seed, but nothing
reads them -- only ``metrics.auroc`` is consumed downstream (Fig. S3). This
script is the reader. It answers three questions the raw TSVs do not:

  1. WHAT are the top features? Rank by across-seed mean contribution and print
     the SAE annotation (category / summary / UniRef90 frequency) alongside.

  2. WHICH DIRECTION do they push? The stored tables give a magnitude
     (``importance_mean_abs_test_contribution``) but no sign that survives
     averaging. We recover an ordinal direction from the shape function: the
     effect in the lowest real bin (``effect_off`` -- for sparse ``sae_max`` this
     is the "feature absent" bin) vs the mean effect over the top decile of bins
     (``effect_on``). ``effect_on > effect_off`` means an active feature raises
     alpha(p), i.e. pushes the pair toward "interacting".

     NOTE: this is ordinal only. The effects TSV stores ``bin_index`` but not the
     EBM's bin cut points (``ebm.bins_`` is never exported), so we can say
     "high activation" but not "activation above X". A proper shape-function
     plot needs the exporter fixed first.

     The direction is a PARTIAL effect (the feature's own term, with the other
     299 held additively fixed), whereas ``endpoint_label_corr`` -- the statistic
     the top-300 were selected by -- is a MARGINAL association. SAE features
     co-activate heavily, so the two disagree for roughly 40% of the top
     features, and the fitted shapes stay monotone where they do (median
     |rho| ~0.76): these are genuine suppression flips, not shape artifacts. The
     report prints both plus a ``mono`` column and a per-setting disagreement
     count, so a flipped feature is visible rather than buried.

  3. IS IT THE SAME SHORTCUT everywhere? Per-setting top-K lists are compared
     pairwise, and features that reach the top-K of several settings are pulled
     out as a shared core. Each setting selects its own top-300 by endpoint-label
     correlation, so "absent from setting B" can mean either "not selected" or
     "selected but ranked low" -- the shared-core table distinguishes the two.

Settings whose mean test AUROC is below ``--null-auroc`` (Bernett, PRING -- the
degree-matched negative controls) have no shortcut to interpret; their feature
tables are noise ranked by magnitude. They are reported but flagged NULL and
excluded from shared-core voting.

Read-only w.r.t. the EBM run outputs; writes its own tables + JSON.

Run:
    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/analysis/analyze_endpoint_ebm_features.py
    ... --top-k 25
    ... --setting c3 --setting cross_species
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from conf.paths import RESULTS_ANALYSIS, RESULTS_PAIR
from src.experiments.results import dump_experiment

# Display order mirrors plot_si.py's S3_SETTINGS so the tables line up with the
# manuscript figure that reports these same runs' AUROCs.
SETTING_ORDER = (
    "c1",
    "c2",
    "c3",
    "pring_bfs",
    "pring_dfs",
    "pring_random_walk",
    "bernett",
    "cross_species",
)

# Below this mean test AUROC the additive endpoint model recovered no shortcut,
# so its top features are magnitude-ranked noise, not an explanation.
NULL_AUROC = 0.55

# Fraction of the highest bins averaged into ``effect_on``. One bin is too noisy
# (it is the extreme tail of a sparse activation); a decile is stable and still
# unambiguously "high activation".
HIGH_BIN_FRACTION = 0.1


@dataclass(frozen=True)
class Run:
    """One EBM fit on disk: a (setting, seed) pair and its four output files."""

    setting: str
    seed: int
    stem: str
    metrics_path: Path

    @property
    def summary_path(self) -> Path:
        return self.metrics_path.with_name(f"{self.stem}_feature_summary_annotated.tsv")

    @property
    def effects_path(self) -> Path:
        return self.metrics_path.with_name(f"{self.stem}_feature_effects_annotated.tsv")


def discover_runs(results_pair: Path) -> list[Run]:
    """Find every endpoint-additive EBM metrics JSON and key it by setting.

    C-levels / Bernett / cross_species carry ``family`` in the payload; PRING
    writes one file per sampling method into a shared seed dir and carries
    ``method`` instead. The resulting keys match plot_si.py (``pring_bfs``, ...).
    """
    runs: list[Run] = []
    for path in sorted(results_pair.glob("*_endpoint_additive_ebm_sae/seed_*/*_metrics.json")):
        payload = json.load(open(path)).get("payload", {})
        family = payload.get("family")
        if family is None:
            method = payload.get("method")
            if method is None:
                continue
            family = f"pring_{str(method).lower()}"
        stem = path.name[: -len("_metrics.json")]
        runs.append(Run(setting=family, seed=int(payload["seed"]), stem=stem, metrics_path=path))
    return runs


def setting_sort_key(setting: str) -> tuple[int, str]:
    order = SETTING_ORDER.index(setting) if setting in SETTING_ORDER else len(SETTING_ORDER)
    return (order, setting)


def shape_direction(effects: pd.DataFrame) -> pd.DataFrame:
    """Ordinal direction + monotonicity of each term's shape function.

    ``term_scores_`` carries two sentinel bins that the exporter kept: index 0 is
    the "missing" bin and the last index is the "unknown" bin, both scored 0 by a
    fit that never saw either. They are dropped here so a real all-positive shape
    function is not pinned to 0 at one end.
    """
    last = effects.groupby("term_index")["bin_index"].transform("max")
    real = effects[(effects["bin_index"] > 0) & (effects["bin_index"] < last)]
    rows = []
    for (term_index, feature_id), group in real.groupby(["term_index", "feature_id"], sort=True):
        effect = group.sort_values("bin_index")["effect"].to_numpy(dtype=float)
        n_bins = effect.size
        if n_bins == 0:
            continue
        n_high = max(1, int(round(HIGH_BIN_FRACTION * n_bins)))
        effect_off = float(effect[0])
        effect_on = float(effect[-n_high:].mean())
        rho = float(spearmanr(np.arange(n_bins), effect).statistic) if n_bins > 2 else np.nan
        rows.append(
            {
                "feature_id": int(feature_id),
                "n_bins": int(n_bins),
                "effect_off": effect_off,
                "effect_on": effect_on,
                "effect_on_minus_off": effect_on - effect_off,
                "shape_monotonicity_rho": rho,
            }
        )
    return pd.DataFrame(rows)


def load_run(run: Run) -> pd.DataFrame:
    """One run's feature table: importance, share, rank, and shape direction."""
    summary = pd.read_csv(run.summary_path, sep="\t")
    importance = summary["importance_mean_abs_test_contribution"]
    total = float(importance.sum())
    summary = summary.assign(
        setting=run.setting,
        seed=run.seed,
        importance=importance,
        # Share makes settings with different signal scales comparable: Bernett's
        # top feature is 30x smaller in absolute contribution than C3's.
        importance_share=importance / total if total > 0 else np.nan,
        rank=importance.rank(ascending=False, method="min").astype(int),
    )
    effects = pd.read_csv(
        run.effects_path, sep="\t", usecols=["feature_id", "term_index", "bin_index", "effect"]
    )
    return summary.merge(shape_direction(effects), on="feature_id", how="left")


def aggregate_over_seeds(per_run: pd.DataFrame) -> pd.DataFrame:
    """Collapse a setting's seeds into one row per feature.

    ``n_seeds_selected`` is the honest denominator: a feature the correlation
    filter picked in only one seed is averaged over that seed alone, so it is
    reported rather than silently treated as a 3-seed mean.
    """
    grouped = per_run.groupby(["setting", "feature_id"], sort=False)
    aggregated = grouped.agg(
        n_seeds_selected=("seed", "nunique"),
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        importance_share_mean=("importance_share", "mean"),
        rank_mean=("rank", "mean"),
        rank_best=("rank", "min"),
        rank_worst=("rank", "max"),
        endpoint_label_corr=("endpoint_label_corr", "mean"),
        effect_off=("effect_off", "mean"),
        effect_on=("effect_on", "mean"),
        effect_on_minus_off=("effect_on_minus_off", "mean"),
        shape_monotonicity_rho=("shape_monotonicity_rho", "mean"),
        effect_range=("effect_range", "mean"),
        n_bins=("n_bins", "mean"),
        category=("category", "first"),
        summary=("summary", "first"),
        activation_pattern=("activation_pattern", "first"),
        uniref90_frequency=("uniref90_frequency", "first"),
    ).reset_index()
    # Direction agreement across seeds: 1.0 = every seed's shape function moved
    # the same way. Anything below 1.0 means the sign is a coin flip, so the
    # feature's direction must not be quoted.
    sign_agreement = grouped["effect_on_minus_off"].apply(
        lambda s: float(np.abs(np.sign(s.dropna()).mean())) if s.notna().any() else np.nan
    )
    aggregated["direction_seed_agreement"] = sign_agreement.to_numpy()
    aggregated["direction"] = np.where(
        aggregated["effect_on_minus_off"] > 0, "up (toward interacting)", "down (toward non-interacting)"
    )
    aggregated.loc[aggregated["effect_on_minus_off"].isna(), "direction"] = "unknown"
    # Partial (shape) vs marginal (selection statistic) direction. A False here is
    # not an error: it is a feature whose own term points the opposite way once
    # the co-activating features are in the model.
    aggregated["dir_matches_corr"] = np.sign(aggregated["effect_on_minus_off"]) == np.sign(
        aggregated["endpoint_label_corr"]
    )
    aggregated = aggregated.sort_values(["setting", "importance_mean"], ascending=[True, False])
    return aggregated.reset_index(drop=True)


def setting_auroc(runs: list[Run]) -> dict[str, dict[str, float]]:
    """Mean/std test AUROC per setting, used to flag the NULL controls."""
    scores: dict[str, list[float]] = {}
    for run in runs:
        record = json.load(open(run.metrics_path))
        scores.setdefault(run.setting, []).append(float(record["metrics"]["auroc"]))
    return {
        setting: {
            "auroc_mean": float(np.mean(values)),
            "auroc_std": float(np.std(values)),
            "n_seeds": len(values),
        }
        for setting, values in scores.items()
    }


def top_k_sets(aggregated: pd.DataFrame, top_k: int) -> dict[str, set[int]]:
    return {
        setting: set(group.nlargest(top_k, "importance_mean")["feature_id"].astype(int))
        for setting, group in aggregated.groupby("setting")
    }


def overlap_matrix(tops: dict[str, set[int]]) -> pd.DataFrame:
    settings = sorted(tops, key=setting_sort_key)
    matrix = pd.DataFrame(index=settings, columns=settings, dtype=int)
    for a in settings:
        for b in settings:
            matrix.loc[a, b] = len(tops[a] & tops[b])
    return matrix


def shared_core(
    aggregated: pd.DataFrame, tops: dict[str, set[int]], settings: list[str], min_settings: int
) -> pd.DataFrame:
    """Features reaching the top-K of >= min_settings settings.

    For every such feature each setting gets two columns: its rank there, and
    whether it was even selected. That separates "this setting also relies on it"
    from "this setting's correlation filter never offered it to the EBM".
    """
    votes: dict[int, list[str]] = {}
    for setting in settings:
        for feature_id in tops.get(setting, set()):
            votes.setdefault(feature_id, []).append(setting)
    core_ids = [fid for fid, hits in votes.items() if len(hits) >= min_settings]
    if not core_ids:
        return pd.DataFrame()

    indexed = aggregated.set_index(["setting", "feature_id"])
    rows = []
    for feature_id in core_ids:
        hits = votes[feature_id]
        first = aggregated[aggregated["feature_id"] == feature_id].iloc[0]
        row = {
            "feature_id": feature_id,
            "n_settings_in_top_k": len(hits),
            "settings_in_top_k": ",".join(sorted(hits, key=setting_sort_key)),
            "category": first["category"],
            "uniref90_frequency": first["uniref90_frequency"],
        }
        for setting in settings:
            key = (setting, feature_id)
            if key in indexed.index:
                record = indexed.loc[key]
                row[f"rank_{setting}"] = float(record["rank_mean"])
                row[f"dir_{setting}"] = "up" if record["effect_on_minus_off"] > 0 else "down"
            else:
                row[f"rank_{setting}"] = np.nan
                row[f"dir_{setting}"] = "not selected"
        row["summary"] = first["summary"]
        rows.append(row)
    core = pd.DataFrame(rows).sort_values(
        ["n_settings_in_top_k", "feature_id"], ascending=[False, True]
    )
    return core.reset_index(drop=True)


def seed_stability(per_run: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Per-setting seed agreement: selection identity, rank correlation, top-K overlap.

    cross_species is the one setting whose train split is itself seeded (val carve
    + subsample), so its selected set genuinely differs by seed; the others fix
    the split and reseed only the EBM's bagging.
    """
    rows = []
    for setting, group in per_run.groupby("setting"):
        seeds = sorted(group["seed"].unique())
        by_seed = {s: group[group["seed"] == s] for s in seeds}
        selections = {s: set(by_seed[s]["feature_id"]) for s in seeds}
        identical = len({frozenset(v) for v in selections.values()}) == 1
        jaccards, rhos, overlaps = [], [], []
        for i, a in enumerate(seeds):
            for b in seeds[i + 1 :]:
                union = selections[a] | selections[b]
                shared = sorted(selections[a] & selections[b])
                jaccards.append(len(shared) / len(union) if union else np.nan)
                ia = by_seed[a].set_index("feature_id")["importance"].loc[shared]
                ib = by_seed[b].set_index("feature_id")["importance"].loc[shared]
                rhos.append(float(spearmanr(ia, ib).statistic) if len(shared) > 2 else np.nan)
                ta = set(by_seed[a].nlargest(top_k, "importance")["feature_id"])
                tb = set(by_seed[b].nlargest(top_k, "importance")["feature_id"])
                overlaps.append(len(ta & tb))
        rows.append(
            {
                "setting": setting,
                "n_seeds": len(seeds),
                "identical_selection": identical,
                "mean_selection_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
                "mean_importance_spearman": float(np.nanmean(rhos)) if rhos else np.nan,
                f"mean_top{top_k}_overlap": float(np.mean(overlaps)) if overlaps else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("setting", key=lambda s: s.map(setting_sort_key)).reset_index(drop=True)


def _setting_payload(
    aggregated: pd.DataFrame,
    aurocs: dict[str, dict[str, float]],
    setting: str,
    is_null: bool,
    top_k: int,
) -> dict:
    """JSON block for one setting: its AUROC, null flag, and top-K feature rows."""
    top = aggregated[aggregated["setting"] == setting].nlargest(top_k, "importance_mean")
    return {
        **aurocs[setting],
        "is_null_control": is_null,
        "n_top_partial_vs_marginal_flips": int((~top["dir_matches_corr"]).sum()),
        "top_features": [
            {
                "feature_id": int(row["feature_id"]),
                "importance_mean": float(row["importance_mean"]),
                "importance_share_mean": float(row["importance_share_mean"]),
                "direction": row["direction"],
                "direction_seed_agreement": float(row["direction_seed_agreement"]),
                "shape_monotonicity_rho": float(row["shape_monotonicity_rho"]),
                "endpoint_label_corr": float(row["endpoint_label_corr"]),
                "dir_matches_corr": bool(row["dir_matches_corr"]),
                "category": row["category"],
            }
            for _, row in top.iterrows()
        ],
    }


def _truncate(text, width: int = 96) -> str:
    if not isinstance(text, str):
        return "-"
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def print_report(
    aggregated: pd.DataFrame,
    aurocs: dict[str, dict[str, float]],
    null_settings: set[str],
    top_k: int,
) -> None:
    for setting in sorted(aggregated["setting"].unique(), key=setting_sort_key):
        score = aurocs[setting]
        tag = "  [NULL CONTROL -- no shortcut recovered, features below are noise]" if setting in null_settings else ""
        print(f"\n{'=' * 110}")
        print(
            f"{setting}   test AUROC {score['auroc_mean']:.4f} +/- {score['auroc_std']:.4f} "
            f"({score['n_seeds']} seeds){tag}"
        )
        print("=" * 110)
        group = aggregated[aggregated["setting"] == setting].nlargest(top_k, "importance_mean")
        print(
            f"{'rank':>4} {'feat':>6} {'imp':>8} {'share':>7} {'dir':>6} {'mono':>6} {'corr':>7} "
            f"{'seeds':>5}  category"
        )
        print("-" * 110)
        for i, (_, row) in enumerate(group.iterrows(), start=1):
            direction = "up" if row["effect_on_minus_off"] > 0 else "down"
            if row["direction_seed_agreement"] < 1.0:
                direction += "?"
            # '*' marks a partial effect pointing against the marginal correlation.
            if not row["dir_matches_corr"]:
                direction += "*"
            print(
                f"{i:>4} {int(row['feature_id']):>6} {row['importance_mean']:>8.4f} "
                f"{row['importance_share_mean'] * 100:>6.2f}% {direction:>6} "
                f"{row['shape_monotonicity_rho']:>6.2f} "
                f"{row['endpoint_label_corr']:>7.3f} {int(row['n_seeds_selected']):>5}  "
                f"{row['category']}"
            )
            print(f"{'':>41}{_truncate(row['summary'], 88)}")
        flips = group.loc[~group["dir_matches_corr"]]
        print(
            f"\n  dir = partial effect of this feature's own EBM term (low->high activation bin); "
            f"mono = Spearman(bin, effect).\n"
            f"  corr = the MARGINAL endpoint-label correlation the top-300 were selected by."
        )
        if flips.empty:
            print("  * = partial/marginal disagreement: none in this top-K.")
        else:
            median_mono = float(flips["shape_monotonicity_rho"].abs().median())
            # A flip on a monotone shape is a real suppression effect; a flip on a
            # flat shape is just a term with no usable direction. Say which this is
            # instead of asserting the flattering reading.
            verdict = (
                "monotone shapes -- suppression flips, not shape artifacts"
                if median_mono >= 0.5
                else "shapes are near-flat, so these directions are weakly identified"
            )
            print(
                f"  * = the two disagree: {len(flips)}/{len(group)} of this setting's top "
                f"features, median |mono| {median_mono:.2f} -- {verdict}."
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-pair", type=Path, default=RESULTS_PAIR)
    ap.add_argument("--setting", action="append", default=None,
                    help="Restrict to these settings (repeatable). Default: all found on disk.")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Features per setting to print and to use for overlap / shared core.")
    ap.add_argument("--min-settings", type=int, default=2,
                    help="Shared core: appear in the top-K of at least this many non-null settings.")
    ap.add_argument("--null-auroc", type=float, default=NULL_AUROC,
                    help="Settings at or below this mean test AUROC are flagged NULL and "
                    "excluded from shared-core voting.")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_ANALYSIS / "endpoint_ebm_features")
    args = ap.parse_args()

    runs = discover_runs(args.results_pair)
    if args.setting:
        keep = set(args.setting)
        runs = [r for r in runs if r.setting in keep]
    if not runs:
        raise SystemExit("no endpoint-additive EBM runs found")
    missing = [r for r in runs if not r.summary_path.exists() or not r.effects_path.exists()]
    if missing:
        raise SystemExit(
            "runs without feature tables: " + ", ".join(f"{r.setting}/seed_{r.seed}" for r in missing)
        )
    print(f"[load] {len(runs)} runs across {len({r.setting for r in runs})} settings", flush=True)

    per_run = pd.concat([load_run(run) for run in runs], ignore_index=True)
    aggregated = aggregate_over_seeds(per_run)
    aurocs = setting_auroc(runs)
    null_settings = {s for s, v in aurocs.items() if v["auroc_mean"] <= args.null_auroc}

    tops = top_k_sets(aggregated, args.top_k)
    matrix = overlap_matrix(tops)
    live = sorted(set(aggregated["setting"]) - null_settings, key=setting_sort_key)
    core = shared_core(aggregated, tops, live, args.min_settings)
    stability = seed_stability(per_run, args.top_k)

    print_report(aggregated, aurocs, null_settings, args.top_k)

    print(f"\n{'=' * 110}\nTop-{args.top_k} overlap between settings\n{'=' * 110}")
    print(matrix.to_string())
    print(f"\n{'=' * 110}\nShared core: in the top-{args.top_k} of >= {args.min_settings} "
          f"non-null settings ({', '.join(live)})\n{'=' * 110}")
    if core.empty:
        print("(none)")
    else:
        rank_cols = [f"rank_{s}" for s in live]
        view = core[["feature_id", "n_settings_in_top_k", "category", *rank_cols]]
        print(view.to_string(index=False, na_rep="--", float_format=lambda v: f"{v:.1f}"))
        print()
        for _, row in core.iterrows():
            print(f"  sae_{int(row['feature_id']):05d} [{row['category']}] "
                  f"x{int(row['n_settings_in_top_k'])} ({row['settings_in_top_k']})")
            print(f"      {_truncate(row['summary'], 100)}")
    print(f"\n{'=' * 110}\nCross-seed stability\n{'=' * 110}")
    print(stability.to_string(index=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.out_dir / "feature_ranking_by_setting.tsv", sep="\t", index=False)
    top_rows = pd.concat(
        [
            aggregated[aggregated["setting"] == setting].nlargest(args.top_k, "importance_mean")
            for setting in sorted(aggregated["setting"].unique(), key=setting_sort_key)
        ],
        ignore_index=True,
    )
    top_rows.to_csv(args.out_dir / f"top{args.top_k}_features.tsv", sep="\t", index=False)
    matrix.to_csv(args.out_dir / f"top{args.top_k}_overlap_matrix.tsv", sep="\t")
    stability.to_csv(args.out_dir / "seed_stability.tsv", sep="\t", index=False)
    if not core.empty:
        core.to_csv(args.out_dir / "shared_core_features.tsv", sep="\t", index=False)

    payload = {
        "analysis": "endpoint_additive_ebm_feature_interpretation",
        "n_runs": len(runs),
        "settings": {
            setting: _setting_payload(aggregated, aurocs, setting, setting in null_settings, args.top_k)
            for setting in sorted(aggregated["setting"].unique(), key=setting_sort_key)
        },
        "top_k": args.top_k,
        "null_auroc_threshold": args.null_auroc,
        "shared_core_feature_ids": [] if core.empty else [int(v) for v in core["feature_id"]],
        "seed_stability": stability.to_dict(orient="records"),
        "caveats": {
            "shape_direction_is_ordinal": "effect_off/effect_on come from bin order; the EBM bin "
            "cut points are not exported, so no activation threshold can be quoted.",
            "partial_vs_marginal_direction": "'direction' is the feature's own EBM term (partial, "
            "other features held fixed); 'endpoint_label_corr' is the marginal association the "
            "top-300 were selected by. They disagree for a large share of top features while the "
            "shapes stay monotone, i.e. suppression among co-activating SAE features. Quote the "
            "partial direction when describing the fitted model, the marginal one when describing "
            "what the selection filter saw.",
            "per_setting_feature_selection": "each setting picks its own top-300 by endpoint-label "
            "correlation, so a feature missing from a setting may simply never have been offered.",
            "null_controls": "settings at or below the AUROC threshold recovered no shortcut; their "
            "feature tables rank noise.",
        },
    }
    out_json = args.out_dir / "endpoint_ebm_feature_interpretation.json"
    dump_experiment(
        out_json,
        task="analysis.endpoint_ebm_features",
        dataset="+".join(sorted(aggregated["setting"].unique(), key=setting_sort_key)),
        features="sae_max",
        split="test",
        model="endpoint_additive_ebm",
        seed=0,
        payload=payload,
        metrics={s: aurocs[s]["auroc_mean"] for s in aurocs},
    )
    print(f"\n[done] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
