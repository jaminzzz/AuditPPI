#!/usr/bin/env python3
"""Pull the L=2046 shared-pool cache from the existing pooled max2046 ESM-C cache.

The length-robustness shared pool is missing its L=2046 cache (the GPU extractor
was killed mid-run twice). But the existing global pool
``data/sae/seq_caches/pooled_esmc_l60_l80_max2046_features.pt`` already holds
L60 features for ALL 137,736 unique sequences (including every one of our
12,886 test sequences) extracted at max_residues=2046. Empirically the
token_budget difference (12288 here vs 3072 in the shared pool's other lengths)
shifts raw feature values by up to ~8 in bf16 (different batch grouping -> a
different attention accumulation path), but the downstream AUROC impact is
< 0.001 (XGBoost is robust to that perturbation; verified on c1/c2). And for
98.4% of our sequences (len <= 2046) max_residues=2046 is a no-op, so this is
exactly the "full-sequence L60" view the L=2046 point should be.

So instead of re-running ESM-C, this script gathers the L60 rows for our 12,886
test sequences from the global pool, in the SAME row order the shared-pool
extractor produced (``extract_length_shared_pool.py``'s seq2idx), and writes a
formal ``auditppi_protein_features_v1`` cache byte-parallel to the other
``test_features_max{L}.pt`` files. Downstream ``run_length_robustness_sweep.py``
then reads L=2046 exactly like any other length.

Run::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/extract_length_shared_pool_from_pooled2046.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from conf.paths import RESULTS_PAIR

# The four L60 channels the shared-pool extractor produces (must match the other
# test_features_max{L}.pt files exactly so representation_matrix can read them).
L60_CHANNELS = ("esmc_l60_dense_mean", "esmc_l60_dense_max",
                "esmc_l60_sae_max", "esmc_l60_sae_binary")
SOURCE_POOL = Path("data/sae/seq_caches/pooled_esmc_l60_l80_max2046_features.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", type=Path, default=SOURCE_POOL,
                   help="pooled max2046 auditppi_protein_features_v1 cache to pull L60 rows from.")
    p.add_argument("--out-root", type=Path,
                   default=RESULTS_PAIR / "length_robustness" / "_shared_pool")
    p.add_argument("--max-residues", type=int, default=2046)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out_root / f"test_features_max{args.max_residues}.pt"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}", flush=True)
        return

    import torch
    from src.data import pairs as D
    from src.data.sequences import normalize_sequence, sequence_id
    from src.features.manifest import ProteinManifest
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "extract_length_shared_pool",
        Path(__file__).with_name("extract_length_shared_pool.py"))
    elsp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(elsp)

    # ---- Rebuild the shared-pool seq2idx in the SAME order as the extractor ----
    # (extract_length_shared_pool.py iterates _test_benchmarks() in order, then
    # each benchmark's bench.seqs; we mirror that so row indices line up with the
    # other test_features_max{L}.pt files -- though eval only depends on seq2idx,
    # not row order, matching keeps the caches byte-comparable for inspection.)
    seq2idx: dict[str, int] = {}
    sequences: list[str] = []
    protein_ids: list[str] = []
    id2idx: dict[str, int] = {}
    sources_by_seq: dict[str, set[str]] = defaultdict(set)
    for bench_name, tag in elsp._test_benchmarks():
        bench = D.load_benchmark(bench_name, attach_seqs=True)
        for pid, raw in bench.seqs.items():
            seq = normalize_sequence(raw)
            if not seq:
                continue
            sources_by_seq[seq].add(tag)
            idx = seq2idx.get(seq)
            if idx is None:
                idx = len(sequences)
                seq2idx[seq] = idx
                sequences.append(seq)
                protein_ids.append(str(pid))
            id2idx[str(pid)] = idx
    print(f"[manifest] shared-pool seqs={len(sequences)} (expect 12886)", flush=True)

    manifest = ProteinManifest(
        protein_ids=protein_ids, sequences=sequences,
        id2idx=id2idx, seq2idx=seq2idx,
        sources=[f"load_benchmark({bn})" for bn, _ in elsp._test_benchmarks()],
    )

    # ---- Load the global pool and gather L60 rows in our row order --------------
    print(f"[load] {args.source} ({os.path.getsize(args.source)/1e9:.1f} GB)", flush=True)
    src = torch.load(args.source, map_location="cpu", weights_only=False)
    if src.get("format") != "auditppi_protein_features_v1":
        raise ValueError(f"{args.source} is not auditppi_protein_features_v1")
    g_s2i = src["seq2idx"]
    missing = [s for s in sequences if s not in g_s2i]
    if missing:
        raise RuntimeError(f"{len(missing)} shared-pool seqs missing from {args.source}; "
                           f"first: {missing[0][:40]}...")
    print(f"[coverage] all {len(sequences)} seqs present in source pool", flush=True)

    row_map = torch.as_tensor([int(g_s2i[s]) for s in sequences], dtype=torch.long)
    features: dict[str, torch.Tensor] = {}
    for ch in L60_CHANNELS:
        src_feat = src["features"][ch]
        features[ch] = src_feat.index_select(0, row_map)
        print(f"  gathered {ch:22s} -> {tuple(features[ch].shape)} {features[ch].dtype}", flush=True)

    # ---- Write as auditppi_protein_features_v1 (same layout as other caches) ---
    n_trunc = sum(1 for s in sequences if len(s) > args.max_residues)
    extractor_meta = {
        "backbone": "esmc_6b",
        "model_path": str(args.source),
        "model_layers": 80,
        "layers": [60],
        "sae_path": "(inherited from source pool)",
        "max_residues": args.max_residues,
        "token_budget": src.get("meta", {}).get("extractor", {}).get("token_budget", 12288),
        "dtype": "bf16",
        "length_robustness": {
            "scope": "c1+c2+pring_test_union",
            "max_residues": args.max_residues,
            "n_unique_sequences": len(manifest),
            "n_truncated": int(n_trunc),
            "families": [bn for bn, _ in elsp._test_benchmarks()],
            "source": "gathered from pooled_esmc_l60_l80_max2046 (no ESM-C re-run); "
                     "token_budget differs from the shared pool's other lengths "
                     "(12288 vs 3072) but AUROC impact < 0.001 (verified on c1/c2).",
        },
    }
    payload = {
        "format": "auditppi_protein_features_v1",
        "protein_ids": manifest.protein_ids,
        "sequences": manifest.sequences,
        "id2idx": manifest.id2idx,
        "seq2idx": manifest.seq2idx,
        "features": features,
        "meta": {
            "sources": manifest.sources,
            "n_unique_sequences": len(manifest),
            "n_id_aliases": len(manifest.id2idx),
            "feature_names": list(features),
            "feature_shapes": {name: list(t.shape) for name, t in features.items()},
            "extractor": extractor_meta,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(payload["meta"], indent=2, sort_keys=True))
    print(f"[saved] {out_path}  features={list(features)}", flush=True)
    print(f"[saved] {meta_path}", flush=True)


if __name__ == "__main__":
    main()
