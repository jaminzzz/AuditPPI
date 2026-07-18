#!/usr/bin/env python3
"""Audit whether frozen ESM-C + SAE fingerprints predict human protein
essentiality, as a lightweight appendix comparison against PIC.

PIC (Kang et al.) fine-tunes ESM2-650M with attention pooling to predict human
essential proteins (HEPs). Here we ask a narrower, interpretability-motivated
question that fits the AuditPPI narrative: how much of that essentiality signal
is already linearly/tree-recoverable from a *frozen* pLM through a sparse,
interpretable SAE fingerprint, without any fine-tuning?

Design
------
- Labels + split are reproduced *exactly* from PIC's ``get_index`` on the full
  65,057-protein human table (random_seed=42, test_ratio=0.1, val_ratio=0.1),
  so our test proteins are a subset of PIC's official test proteins.
- Because we only cached the SAE/ESM-C features for the ~14.7k proteins that
  matched an existing ESM-C cache by exact sequence, every PIC split is
  intersected with the cached subset. This subset is NOT random (it is the set
  of proteins that happened to appear in prior PPI benchmarks), so absolute
  numbers are only loosely comparable to PIC's full-set numbers; the
  informative quantity is the *relative* ordering of feature kinds and the fact
  that a frozen SAE fingerprint recovers essentiality at all.
- Features: ``sae_max`` (16384), ``binary`` (sae_max>0), ``esmc_mean`` (2560),
  and a ``sequence_basic`` composition baseline. Classifier: XGBoost with
  scale_pos_weight, mirroring run_pring_high_participation_classifier.py.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from conf.model import DEFAULT_SEED
from conf.paths import RESULTS_PROTEIN, PIC_DATA, PIC_HUMAN_SAE_CACHE
from src.experiments.results import dump_experiment
from src.runtime import seed_all
from src.eval.classification import binary_classification_metrics
from src.models.estimators.xgboost import fit_xgb_classifier
from src.participation.features import sequence_features

DEFAULT_CACHE = PIC_HUMAN_SAE_CACHE
OUT_DIR = RESULTS_PROTEIN / "pic_essentiality"

FEATURE_KINDS = ("sae_max", "binary", "esmc_mean", "sequence_basic")


# --------------------------------------------------------------------------- #
# PIC split reproduction (verbatim logic from PIC/code/module/load_dataset.py) #
# --------------------------------------------------------------------------- #
def pic_get_index(labels: Sequence[int], *, test_ratio: float, val_ratio: float, random_seed: int):
    """Reproduce PIC's get_index on positional indices 0..N-1.

    PIC keys everything off the DataFrame's RangeIndex, which equals the row
    position, so operating on positions reproduces the exact same partition.
    """
    random.seed(random_seed)
    all_indexes = list(range(len(labels)))
    ess = [i for i, e in enumerate(labels) if int(e) == 1]
    non = [i for i, e in enumerate(labels) if int(e) == 0]
    test_indexes = random.sample(ess, int(test_ratio * len(ess))) + random.sample(
        non, int(test_ratio * len(non))
    )
    random.shuffle(test_indexes)
    train_indexes = list(set(all_indexes) - set(test_indexes))
    val_indexes = random.sample(train_indexes, int(val_ratio * len(train_indexes)))
    real_train = list(set(train_indexes) - set(val_indexes))
    return set(real_train), set(val_indexes), set(test_indexes)


def node_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return binary_classification_metrics(y, p)


def torch_load_cache(path: Path) -> dict:
    import inspect

    import torch

    kwargs = {"map_location": "cpu", "weights_only": False}
    if "mmap" in inspect.signature(torch.load).parameters:
        kwargs["mmap"] = True
    return torch.load(str(path), **kwargs)


def feature_matrix(cache: dict, rows: Sequence[int], kind: str) -> np.ndarray:
    import torch

    idx = torch.as_tensor(list(rows), dtype=torch.long)
    if kind in ("sae_max", "binary"):
        mat = cache["esmc_sae_max"]
    elif kind == "esmc_mean":
        mat = cache["esmc_mean"]
    else:
        raise ValueError(kind)
    X = mat.index_select(0, idx)
    if kind == "binary":
        X = X > 0
    return X.float().numpy().astype(np.float32, copy=False)


def sequence_matrix(seqs: Sequence[str], *, kmer: int) -> np.ndarray:
    return np.vstack([sequence_features(s, kmer=kmer) for s in seqs]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=PIC_DATA)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--label-col", default="human")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR / "human_xgboost")
    p.add_argument("--feature-kind", choices=[*FEATURE_KINDS, "all"], default="all")
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--kmer", type=int, default=2)
    p.add_argument("--n-estimators", type=int, default=3000)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--early-stopping-rounds", type=int, default=200)
    args = p.parse_args()

    # Global RNG seed for the whole run; the deterministic PIC split below reseeds
    # random.seed(args.seed) locally to reproduce its exact row shuffle.
    seed_all(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full PIC table -> labels in row order, reproduce PIC's split.
    with args.data.open("rb") as f:
        df = pickle.load(f)
    full_ids = [str(x) for x in df["ID"].tolist()]
    full_labels = [int(v) for v in df[args.label_col].tolist()]
    tr_pos, va_pos, te_pos = pic_get_index(
        full_labels,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )
    split_of = {}
    for i in tr_pos:
        split_of[full_ids[i]] = "train"
    for i in va_pos:
        split_of[full_ids[i]] = "val"
    for i in te_pos:
        split_of[full_ids[i]] = "test"
    print(
        f"[pic-split] full N={len(full_ids)} train={len(tr_pos)} val={len(va_pos)} "
        f"test={len(te_pos)} (seed={args.seed})",
        flush=True,
    )

    # 2. Load cached-feature subset; intersect each split with it by ENSP ID.
    cache = torch_load_cache(args.cache_path)
    cache_ids = [str(x) for x in cache["protein_ids"]]
    cache_labels = np.asarray(cache["labels"], dtype=np.int8)
    id2row = {pid: i for i, pid in enumerate(cache_ids)}
    cache_seqs = [str(s) for s in cache["sequences"]]

    split_rows = {"train": [], "val": [], "test": []}
    for pid, row in id2row.items():
        s = split_of.get(pid)
        if s is not None:
            split_rows[s].append(row)
    for s in split_rows:
        split_rows[s] = sorted(split_rows[s])

    def rows_labels(rows):
        return np.asarray([int(cache_labels[r]) for r in rows], dtype=np.int8)

    ytr = rows_labels(split_rows["train"])
    yva = rows_labels(split_rows["val"])
    yte = rows_labels(split_rows["test"])
    n_cache_in_split = len(split_rows["train"]) + len(split_rows["val"]) + len(split_rows["test"])
    print(
        f"[cache-subset] cached={len(cache_ids)} in-split={n_cache_in_split} "
        f"train={ytr.size}({ytr.mean():.4f}) val={yva.size}({yva.mean():.4f}) "
        f"test={yte.size}({yte.mean():.4f})",
        flush=True,
    )

    subset_summary = {
        "task": "pic_human_essentiality_frozen_sae_fingerprint",
        "label_col": args.label_col,
        "pic_split": {
            "source": "reproduced PIC get_index",
            "random_seed": args.seed,
            "test_ratio": args.test_ratio,
            "val_ratio": args.val_ratio,
            "full_n": len(full_ids),
            "full_train": len(tr_pos),
            "full_val": len(va_pos),
            "full_test": len(te_pos),
            "full_pos_rate": round(float(np.mean(full_labels)), 6),
        },
        "cached_subset": {
            "note": "features only cached for proteins matched to prior ESM-C caches by exact sequence; NOT a random subset",
            "n_cached": len(cache_ids),
            "n_in_split": n_cache_in_split,
            "n_train": int(ytr.size),
            "n_val": int(yva.size),
            "n_test": int(yte.size),
            "train_pos_rate": round(float(ytr.mean()), 6) if ytr.size else None,
            "val_pos_rate": round(float(yva.mean()), 6) if yva.size else None,
            "test_pos_rate": round(float(yte.mean()), 6) if yte.size else None,
        },
    }

    features = FEATURE_KINDS if args.feature_kind == "all" else (args.feature_kind,)
    results: dict[str, dict] = {}
    for kind in features:
        if kind == "sequence_basic":
            Xtr = sequence_matrix([cache_seqs[r] for r in split_rows["train"]], kmer=args.kmer)
            Xva = sequence_matrix([cache_seqs[r] for r in split_rows["val"]], kmer=args.kmer)
            Xte = sequence_matrix([cache_seqs[r] for r in split_rows["test"]], kmer=args.kmer)
        else:
            Xtr = feature_matrix(cache, split_rows["train"], kind)
            Xva = feature_matrix(cache, split_rows["val"], kind)
            Xte = feature_matrix(cache, split_rows["test"], kind)

        clf = fit_xgb_classifier(
            Xtr, ytr, Xva, yva,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            device=args.device,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        pte = clf.predict_proba(Xte)[:, 1]
        pva = clf.predict_proba(Xva)[:, 1]
        m_test = node_metrics(yte, pte)
        m_val = node_metrics(yva, pva)
        results[kind] = {
            "dim": int(Xtr.shape[1]),
            "best_iteration": int(getattr(clf, "best_iteration", -1)),
            "val": m_val,
            "test": m_test,
        }
        print(
            f"[{kind}] dim={Xtr.shape[1]} test AUROC={m_test['auroc']:.4f} "
            f"AUPRC={m_test['auprc']:.4f} (base={m_test['baseline_auprc']:.4f}) "
            f"F1={m_test['best_f1']:.4f}",
            flush=True,
        )

    out = {**subset_summary, "results": results}
    out_path = args.out_dir / f"pic_{args.label_col}_frozen_sae_xgboost.json"
    dump_experiment(
        out_path,
        task="protein.pic_essentiality",
        dataset=f"pic_{args.label_col}",
        features="multi",
        split="test",
        model="xgboost_classifier",
        seed=args.seed,
        payload=out,
        metrics={
            kind: {
                "auroc": res["test"]["auroc"],
                "auprc": res["test"]["auprc"],
            }
            for kind, res in results.items()
        },
    )
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
