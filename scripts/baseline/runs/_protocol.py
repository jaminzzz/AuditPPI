"""Shared evaluation protocol for the in-project PPI baselines.

The four language-model baselines (DeepNano / FlashPPI / MINT / PPLM) are judged
against the gated AuditPPI pair model, so they must expand families, pick their
train/val splits, group train-once-multi-eval, and score AUROC/AUPRC **exactly**
like :mod:`scripts.audit_pair.run_ppi_fingerprint_baseline` /
:mod:`src.ppi_fingerprint.baseline`. That family/train/val logic lives as private
functions there; to keep the comparison provably fair without importing the audit
runner's model machinery, the small amount of routing logic is duplicated here
verbatim (a baseline-only copy), and the two data-source shapes are wrapped in one
runner so each baseline script is a thin loader + a call.

Two cache shapes feed the runner:

* **per-protein** (``deepnano/`` ``flashppi/``): ``auditppi_protein_features_v1``
  payloads holding every endpoint sequence for a family (one file per family, and
  one per PRING species), keyed by sequence via ``seq2idx``. Pairs are gathered on
  the fly exactly like :func:`src.features.protein_cache.pair_feature_rows`.
* **per-pair** (``mint/`` ``pplm/``): ``auditppi_baseline_pair_features_v1``
  payloads with inline labels + already-assembled per-endpoint feature tensors,
  one file per benchmark name (``{stem}.pt`` where ``stem`` is
  :func:`benchmark_stem`).

Results land under ``results/main/baselines/{baseline}/{family}/seed_{S}/`` in the
same cell/summary layout and :func:`dump_experiment` spine as the audit model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from conf.model import DEFAULT_SEED
from conf.paths import PRING_CROSS_SPECIES, RESULTS_MAIN, SAE
from src.data.pring_graph import METHODS as PRING_METHODS

# --- protocol routing (verbatim copy of the audit runner's family logic) -------

FAMILIES = ("c1", "c2", "c3", "cross_species", "bernett", "pring")

# PRING human-graph sampling method used for the train side when the eval graph
# does not itself pin one (the zero-shot cross-species test graphs).
PRING_DEFAULT_METHOD = "BFS"

# Benchmark family -> its own native train split (mirrors config.NATIVE_TRAIN;
# PRING is resolved specially in train_name_for).
NATIVE_TRAIN = {
    "c1": "c1:train",
    "c2": "c2:train",
    "c3": "c3:train",
    "cross_species": "cross_species:human_train",
    "bernett": "bernett:train",
}


def family_of(name: str) -> str:
    """Top-level family token of a benchmark name (``c3:test`` -> ``c3``)."""
    return name.split(":")[0]


def _list_cross_species() -> list[str]:
    from src.data.pairs import list_cross_species

    return list_cross_species()


def family_evals(family: str) -> List[str]:
    """Expand a benchmark family into its eval benchmark name(s).

    Byte-for-byte the audit runner's ``family_evals`` so the baselines are scored
    on precisely the same held-out benchmarks.
    """
    if family in {"c1", "c2", "c3"}:
        return [f"{family}:test"]
    if family == "cross_species":
        species = [s for s in _list_cross_species() if s != "human_train"]
        return [f"cross_species:{s}" for s in species]
    if family == "bernett":
        return ["bernett:test"]
    if family == "pring":
        evals = [f"pring:human:test:{m}" for m in PRING_METHODS]
        evals += [f"pring:{sp}:test" for sp in PRING_CROSS_SPECIES]
        return evals
    raise ValueError(f"unknown family {family!r}")


def train_name_for(eval_name: str) -> str:
    """The native-train benchmark name for an eval benchmark."""
    family = family_of(eval_name)
    if family == "pring":
        parts = eval_name.split(":")
        method = parts[3] if len(parts) > 3 and parts[3] else PRING_DEFAULT_METHOD
        return f"pring:human:train:{method}"
    return NATIVE_TRAIN[family]


def val_name_for(eval_name: str) -> Optional[str]:
    """The official validation benchmark for early stopping, or ``None``.

    ``cross_species`` has no official val on disk -> the runner carves a
    stratified val from train (matching the audit convention).
    """
    family = family_of(eval_name)
    if family in {"c1", "c2", "c3"}:
        return f"{family}:val"
    if family == "bernett":
        return "bernett:val"
    if family == "pring":
        parts = eval_name.split(":")
        method = parts[3] if len(parts) > 3 and parts[3] else PRING_DEFAULT_METHOD
        return f"pring:human:val:{method}"
    return None


def eval_suffix(eval_name: str) -> str:
    """Path-safe eval stem with the family prefix stripped (audit convention)."""
    if ":" not in eval_name:
        return eval_name.replace("/", "_")
    family, rest = eval_name.split(":", 1)
    return rest.replace(":", "_") if rest else family


def benchmark_stem(name: str) -> str:
    """Path-safe stem for a ``load_benchmark`` key (matches the extractors)."""
    return name.replace(":", "_").replace("/", "_")


# --- output layout -------------------------------------------------------------

BASELINE_FEATURES = SAE / "baseline_features"
BASELINE_OUT_ROOT = RESULTS_MAIN / "baselines"


def _seed_dir(baseline: str, family: str, seed: int) -> Path:
    return BASELINE_OUT_ROOT / baseline / family / f"seed_{int(seed)}"


def cell_path(baseline: str, family: str, feat_tag: str, pair_mode: str,
              eval_name: str, seed: int) -> Path:
    stem = f"{feat_tag}_{pair_mode}_{eval_suffix(eval_name)}"
    return _seed_dir(baseline, family, seed) / "cells" / f"{stem}.json"


def summary_path(baseline: str, family: str, feat_tag: str, pair_mode: str,
                 seed: int) -> Path:
    stem = f"{feat_tag}_{pair_mode}"
    return _seed_dir(baseline, family, seed) / "summaries" / f"{stem}.json"


# --- endpoint assembly ---------------------------------------------------------

@dataclass
class EndpointBatch:
    """Per-endpoint feature tensors for a benchmark's pairs, plus coverage."""

    A: "object"          # torch.Tensor (n_scored, dim), float32
    B: "object"          # torch.Tensor (n_scored, dim), float32
    y: np.ndarray        # (n_scored,) int64
    n_total: int         # pairs in the benchmark
    n_scored: int        # pairs whose endpoints were both available


def _endpoint_matrix(cache: Dict, keys: Sequence[str]):
    """Concatenate one or more per-protein feature keys into one endpoint matrix."""
    import torch

    feats = cache["features"]
    missing = [k for k in keys if k not in feats]
    if missing:
        raise KeyError(f"cache missing feature keys {missing}; has {list(feats)}")
    mats = [feats[k] for k in keys]
    return mats[0] if len(mats) == 1 else torch.cat(mats, dim=1)


def gather_pair_endpoints(bench, cache: Dict, keys: Sequence[str]):
    """Map benchmark pairs to per-endpoint rows via ``seq2idx``.

    Clone of :func:`src.features.protein_cache.pair_feature_rows` that (a) reads
    the nested ``cache["features"]`` layout of the baseline caches and (b) can
    concatenate several pooled views into one endpoint vector. Returns
    ``(A, B, y, kept)`` or ``None`` when no pair had both endpoints cached.
    """
    import torch

    seq2idx = cache["seq2idx"]
    matrix = _endpoint_matrix(cache, keys)
    rows_a: List[int] = []
    rows_b: List[int] = []
    ys: List[int] = []
    kept: List[int] = []
    for i, ((a, b), y) in enumerate(zip(bench.pairs, bench.labels)):
        sa = bench.seqs.get(a)
        sb = bench.seqs.get(b)
        ja = seq2idx.get(sa) if sa is not None else None
        jb = seq2idx.get(sb) if sb is not None else None
        if ja is None or jb is None:
            continue
        rows_a.append(ja)
        rows_b.append(jb)
        ys.append(int(y))
        kept.append(i)
    if not kept:
        return None
    ia = torch.as_tensor(rows_a, dtype=torch.long)
    ib = torch.as_tensor(rows_b, dtype=torch.long)
    A = matrix.index_select(0, ia).float()
    B = matrix.index_select(0, ib).float()
    return A, B, np.asarray(ys, dtype=np.int64), kept


def protein_cache_path(subdir: str, name: str, suffix: str) -> Path:
    """Resolve a benchmark name to its per-protein baseline cache path.

    One file per family; PRING is per-species (``pring_{species}{suffix}.pt``),
    mirroring :func:`src.ppi_fingerprint.baseline._cache_path_for`.
    """
    family = family_of(name)
    if family == "pring":
        parts = name.split(":")
        species = parts[1] if len(parts) > 1 and parts[1] else "human"
        stem = f"pring_{species}"
    else:
        stem = family
    return BASELINE_FEATURES / subdir / f"{stem}{suffix}.pt"


def make_protein_loader(subdir: str, keys: Sequence[str], *, suffix: str) -> Callable[[str], EndpointBatch]:
    """A ``loader(name) -> EndpointBatch`` over per-protein baseline caches."""
    import torch

    from src.data.pairs import load_benchmark

    loaded: Dict[Path, Dict] = {}

    def loader(name: str) -> EndpointBatch:
        path = protein_cache_path(subdir, name, suffix)
        if not path.exists():
            raise FileNotFoundError(f"missing per-protein cache {path}")
        if path not in loaded:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format") != "auditppi_protein_features_v1":
                raise ValueError(f"{path} is not an auditppi_protein_features_v1 cache")
            loaded[path] = payload
        bench = load_benchmark(name, attach_seqs=True)
        out = gather_pair_endpoints(bench, loaded[path], keys)
        if out is None:
            raise RuntimeError(f"no cached endpoints for benchmark {name!r} in {path}")
        A, B, y, kept = out
        return EndpointBatch(A, B, y, len(bench.pairs), len(kept))

    return loader


def make_pair_loader(subdir: str, key_a: str, key_b: str) -> Callable[[str], EndpointBatch]:
    """A ``loader(name) -> EndpointBatch`` over per-pair baseline caches.

    The per-pair caches already carry assembled per-endpoint tensors and inline
    labels row-aligned to the full benchmark order, so every row is "scored".
    """
    import torch

    def loader(name: str) -> EndpointBatch:
        path = BASELINE_FEATURES / subdir / f"{benchmark_stem(name)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing per-pair cache {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "auditppi_baseline_pair_features_v1":
            raise ValueError(f"{path} is not an auditppi_baseline_pair_features_v1 cache")
        feats = payload["features"]
        for key in (key_a, key_b):
            if key not in feats:
                raise KeyError(f"{path} missing feature {key!r}; has {list(feats)}")
        A = feats[key_a].float()
        B = feats[key_b].float()
        y = np.asarray(payload["labels"]).astype(np.int64).reshape(-1)
        n = int(y.shape[0])
        return EndpointBatch(A, B, y, n, n)

    return loader


# --- runners -------------------------------------------------------------------

def _spine_kwargs(baseline: str, family: str, feat_tag: str, pair_mode: str,
                  seed: int, evals: Sequence[str]) -> dict:
    return dict(
        task="pair.baseline",
        dataset=family,
        features=feat_tag,
        split="multi" if len(evals) > 1 else evals[0],
        model=baseline,
        seed=seed,
    )


def _dump_summary(baseline: str, family: str, feat_tag: str, pair_mode: str,
                  seed: int, evals: Sequence[str], summary: dict, hparams: dict) -> Path:
    from src.experiments.results import dump_experiment

    out = summary_path(baseline, family, feat_tag, pair_mode, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_experiment(
        out,
        payload=summary,
        metrics=summary,
        hyperparameters=hparams,
        **_spine_kwargs(baseline, family, feat_tag, pair_mode, seed, evals),
    )
    return out


def _score_and_record(baseline, family, feat_tag, pair_mode, seed, eval_name,
                      train_name, scores, y, n_total, n_scored, summary, extra):
    import json

    from src.eval.classification import safe_auprc, safe_auroc

    auroc = safe_auroc(y, scores)
    auprc = safe_auprc(y, scores)
    n_skip = int(n_total - n_scored)
    res = {
        "baseline": baseline,
        "feat_tag": feat_tag,
        "eval": eval_name,
        "train": train_name,
        "pair_mode": pair_mode,
        "seed": int(seed),
        "n_train": int(extra.get("n_train", 0)),
        "n_scored": int(n_scored),
        "n_skipped_eval": n_skip,
        "pos_rate": round(float(np.mean(y)), 4) if len(y) else None,
        "auroc": round(auroc, 4) if auroc is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
    }
    res.update(extra)
    print(
        f"[{baseline}·{feat_tag}·{pair_mode}·seed{seed}·{eval_name}] "
        f"AUROC={res['auroc']} AUPRC={res['auprc']} "
        f"(n_train={res['n_train']}, skip={n_skip})",
        flush=True,
    )
    path = cell_path(baseline, family, feat_tag, pair_mode, eval_name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    key = f"{baseline}/{feat_tag}/{pair_mode}/seed{seed}/{eval_name}"
    summary[key] = {
        "auroc": res["auroc"], "auprc": res["auprc"],
        "n_skip": n_skip, "train": train_name, "pair_mode": pair_mode,
        "seed": int(seed),
    }


def run_trainable_baseline(
    *,
    baseline: str,
    family: str,
    loader: Callable[[str], EndpointBatch],
    feat_tag: str,
    pair_mode: str,
    seed: int,
    train_subsample: Optional[int],
    val_frac: float,
    device: Optional[str],
    hparams: dict,
) -> dict:
    """Train-once-multi-eval MLP-pair baseline over a family (audit protocol)."""
    import torch

    from src.features.sampling import stratified_subsample
    from src.models.architectures.mlp_pair import train_mlp_pair

    evals = family_evals(family)
    groups: Dict[str, List[str]] = defaultdict(list)
    for ev in evals:
        groups[train_name_for(ev)].append(ev)

    summary: dict = {}
    for train_name, group_evals in groups.items():
        val_name = val_name_for(group_evals[0])
        try:
            tr = loader(train_name)
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            for ev in group_evals:
                key = f"{baseline}/{feat_tag}/{pair_mode}/seed{seed}/{ev}"
                summary[key] = {"error": str(exc)}
                print(f"[skip] {key}: {exc}", flush=True)
            continue

        Atr, Btr, ytr = tr.A, tr.B, tr.y
        sub = stratified_subsample(ytr, train_subsample, seed)
        if sub is not None:
            ti = torch.as_tensor(sub, dtype=torch.long)
            Atr, Btr, ytr = Atr.index_select(0, ti), Btr.index_select(0, ti), ytr[sub]

        if val_name is not None:
            va = loader(val_name)
            Ava, Bva, yva = va.A, va.B, va.y
        else:
            n_val = max(1, int(len(ytr) * val_frac))
            val_idx = stratified_subsample(ytr, n_val, seed + 1)
            if val_idx is None:
                if len(ytr) <= 1:
                    raise ValueError(f"need >=2 train pairs for a val split; got {len(ytr)}")
                val_idx = np.array([0], dtype=np.int64)
            mask = np.ones(len(ytr), dtype=bool)
            mask[val_idx] = False
            mt = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
            mv = torch.as_tensor(val_idx, dtype=torch.long)
            Ava, Bva, yva = Atr.index_select(0, mv), Btr.index_select(0, mv), ytr[val_idx]
            Atr, Btr, ytr = Atr.index_select(0, mt), Btr.index_select(0, mt), ytr[mask]

        n_train = int(len(ytr))
        if len(group_evals) > 1:
            print(f"[fit] {baseline}/{feat_tag}/seed{seed} train={train_name} "
                  f"-> {len(group_evals)} evals (train-once)", flush=True)
        predictor = train_mlp_pair(
            Atr, Btr, ytr, Ava, Bva, yva,
            pair_mode=pair_mode, seed=seed, device=device,
        )
        for ev in group_evals:
            try:
                eb = loader(ev)
            except (FileNotFoundError, RuntimeError, KeyError) as exc:
                key = f"{baseline}/{feat_tag}/{pair_mode}/seed{seed}/{ev}"
                summary[key] = {"error": str(exc)}
                print(f"[skip] {key}: {exc}", flush=True)
                continue
            scores = predictor.predict_proba_pairs(eb.A, eb.B)
            _score_and_record(
                baseline, family, feat_tag, pair_mode, seed, ev, train_name,
                scores, eb.y, eb.n_total, eb.n_scored, summary,
                extra={"n_train": n_train},
            )

    out = _dump_summary(baseline, family, feat_tag, pair_mode, seed, evals, summary, hparams)
    _print_table(baseline, family, feat_tag, seed, summary, out)
    return summary


def run_ensemble_baseline(
    *,
    baseline: str,
    family: str,
    loaders: Dict[str, Callable[[str], EndpointBatch]],
    feat_tag: str,
    pair_mode: str,
    seed: int,
    train_subsample: Optional[int],
    val_frac: float,
    device: Optional[str],
    hparams: dict,
) -> dict:
    """Multi-branch ensemble MLP-pair baseline (faithful DeepNano-seq protocol).

    DeepNano-seq keeps one prediction head **per residue pool** (mean/min/max),
    each head consuming the pool's own ``[A‖B]`` concat, and reports the mean of
    the three heads' probabilities; the joint model trains all three under a
    summed (averaged) BCE loss. With frozen features the three heads share no
    parameters or gradients, so training them jointly with averaged loss is
    exactly equivalent to training each independently -- so this trains one
    :func:`train_mlp_pair` head per branch and averages their probabilities.

    Every branch reads the same cache (same ``seq2idx``, pairs, labels) so all
    branches are row-aligned by construction; the train subsample and the carved
    val split are therefore computed once (from the shared labels) and applied to
    every branch identically.
    """
    import torch

    from src.features.sampling import stratified_subsample
    from src.models.architectures.mlp_pair import train_mlp_pair

    branches = list(loaders.keys())
    evals = family_evals(family)
    groups: Dict[str, List[str]] = defaultdict(list)
    for ev in evals:
        groups[train_name_for(ev)].append(ev)

    summary: dict = {}
    for train_name, group_evals in groups.items():
        val_name = val_name_for(group_evals[0])
        try:
            trs = {br: loaders[br](train_name) for br in branches}
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            for ev in group_evals:
                key = f"{baseline}/{feat_tag}/{pair_mode}/seed{seed}/{ev}"
                summary[key] = {"error": str(exc)}
                print(f"[skip] {key}: {exc}", flush=True)
            continue

        # Labels are identical across branches (same cache/pairs) -> derive the
        # subsample and val split once from the first branch and reuse the exact
        # row indices for every branch.
        ref = trs[branches[0]]
        ytr = ref.y
        sub = stratified_subsample(ytr, train_subsample, seed)
        if sub is not None:
            ti = torch.as_tensor(sub, dtype=torch.long)
            trs = {br: EndpointBatch(
                b.A.index_select(0, ti), b.B.index_select(0, ti), b.y[sub],
                b.n_total, b.n_scored) for br, b in trs.items()}
            ytr = ytr[sub]

        if val_name is not None:
            try:
                vas = {br: loaders[br](val_name) for br in branches}
            except (FileNotFoundError, RuntimeError, KeyError) as exc:
                for ev in group_evals:
                    key = f"{baseline}/{feat_tag}/{pair_mode}/seed{seed}/{ev}"
                    summary[key] = {"error": str(exc)}
                    print(f"[skip] {key}: {exc}", flush=True)
                continue
            fit = {br: (trs[br].A, trs[br].B, trs[br].y) for br in branches}
            val = {br: (vas[br].A, vas[br].B, vas[br].y) for br in branches}
        else:
            n_val = max(1, int(len(ytr) * val_frac))
            val_idx = stratified_subsample(ytr, n_val, seed + 1)
            if val_idx is None:
                if len(ytr) <= 1:
                    raise ValueError(f"need >=2 train pairs for a val split; got {len(ytr)}")
                val_idx = np.array([0], dtype=np.int64)
            mask = np.ones(len(ytr), dtype=bool)
            mask[val_idx] = False
            mt = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
            mv = torch.as_tensor(val_idx, dtype=torch.long)
            fit, val = {}, {}
            for br in branches:
                A, B = trs[br].A, trs[br].B
                fit[br] = (A.index_select(0, mt), B.index_select(0, mt), ytr[mask])
                val[br] = (A.index_select(0, mv), B.index_select(0, mv), ytr[val_idx])

        n_train = int(len(fit[branches[0]][2]))
        if len(group_evals) > 1:
            print(f"[fit] {baseline}/{feat_tag}/seed{seed} train={train_name} "
                  f"-> {len(group_evals)} evals x {len(branches)} branches "
                  f"(train-once)", flush=True)
        predictors = {}
        for br in branches:
            Af, Bf, yf = fit[br]
            Av, Bv, yv = val[br]
            predictors[br] = train_mlp_pair(
                Af, Bf, yf, Av, Bv, yv,
                pair_mode=pair_mode, seed=seed, device=device,
            )

        for ev in group_evals:
            try:
                ebs = {br: loaders[br](ev) for br in branches}
            except (FileNotFoundError, RuntimeError, KeyError) as exc:
                key = f"{baseline}/{feat_tag}/{pair_mode}/seed{seed}/{ev}"
                summary[key] = {"error": str(exc)}
                print(f"[skip] {key}: {exc}", flush=True)
                continue
            probs = [predictors[br].predict_proba_pairs(ebs[br].A, ebs[br].B)
                     for br in branches]
            scores = np.mean(probs, axis=0)
            ref_eb = ebs[branches[0]]
            _score_and_record(
                baseline, family, feat_tag, pair_mode, seed, ev, train_name,
                scores, ref_eb.y, ref_eb.n_total, ref_eb.n_scored, summary,
                extra={"n_train": n_train, "ensemble": branches},
            )

    out = _dump_summary(baseline, family, feat_tag, pair_mode, seed, evals, summary, hparams)
    _print_table(baseline, family, feat_tag, seed, summary, out)
    return summary


def run_scoring_baseline(
    *,
    baseline: str,
    family: str,
    score_eval: Callable[[str], "tuple"],
    feat_tag: str,
    seed: int,
    hparams: dict,
    extra_per_eval: Optional[Callable[[str], dict]] = None,
) -> dict:
    """Zero-training scorer over a family (FlashPPI retrieval CLIP protocol).

    ``score_eval(eval_name)`` returns ``(scores, y, n_total, n_scored)``.
    """
    evals = family_evals(family)
    summary: dict = {}
    for ev in evals:
        try:
            scores, y, n_total, n_scored = score_eval(ev)
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            key = f"{baseline}/{feat_tag}/none/seed{seed}/{ev}"
            summary[key] = {"error": str(exc)}
            print(f"[skip] {key}: {exc}", flush=True)
            continue
        extra = extra_per_eval(ev) if extra_per_eval else {}
        _score_and_record(
            baseline, family, feat_tag, "none", seed, ev,
            train_name="(zero-training)",
            scores=scores, y=y, n_total=n_total, n_scored=n_scored,
            summary=summary, extra=extra,
        )
    out = _dump_summary(baseline, family, feat_tag, "none", seed, evals, summary, hparams)
    _print_table(baseline, family, feat_tag, seed, summary, out)
    return summary


def _print_table(baseline, family, feat_tag, seed, summary, out_path):
    print(f"\n=== {baseline} {family}/{feat_tag}/seed{seed}: AUROC / AUPRC ===", flush=True)
    for key, value in summary.items():
        print(f"  {key:56s} {value}", flush=True)
    print(f"\n[done] {out_path}", flush=True)


__all__ = [
    "FAMILIES",
    "EndpointBatch",
    "family_evals",
    "train_name_for",
    "val_name_for",
    "family_of",
    "eval_suffix",
    "benchmark_stem",
    "gather_pair_endpoints",
    "protein_cache_path",
    "make_protein_loader",
    "make_pair_loader",
    "run_trainable_baseline",
    "run_ensemble_baseline",
    "run_scoring_baseline",
    "cell_path",
    "summary_path",
]
