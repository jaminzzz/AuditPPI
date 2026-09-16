#!/usr/bin/env python3
"""Re-extract the SHARED test endpoint feature pool at one truncation length.

Extension of ``extract_length_test_features.py``: instead of one family's test
split, this builds ONE cache over the **deduplicated union of test endpoint
sequences across c1 / c2 / pring (all species)**, so a single forward sweep over
~12.9k unique sequences serves every family's length-robustness evaluation. The
test side of the sweep is the expensive part (GPU + ESM-C), and many of these
families share sequences (e.g. pring human BFS/DFS/RW use the same proteins), so
deduplicating once and reusing the cache across families avoids re-extracting
the same sequence many times.

Sequences are pulled straight from the benchmark loader
(``load_benchmark(...).seqs``) -- the SAME source the per-family extractor and
the downstream ``pair_feature_rows`` eval use -- so the manifest ``seq2idx``
keys (full normalized sequence) are byte-identical to what eval looks up. This
keeps the kept-pair set identical across lengths and across families.

Cache layout is the formal ``auditppi_protein_features_v1`` (identical to the
per-family ``test_features_max{L}.pt``), so ``pair_feature_rows`` reads it with
no changes -- it only relies on ``seq2idx`` + ``representation_matrix``, which
are dataset-agnostic. Each family's evaluation simply points at this shared
pool instead of its own per-family cache.

Only ESM-C L60 is extracted (``--layers 60``); the extractor early-exits after
block 60 and never runs the upper 20 blocks (less compute, lower GPU memory
floor -- the 6B backbone is ~12 GB in bf16 and the cards are shared).

Product::

    results/audit_pair/length_robustness/_shared_pool/
      test_features_max{L}.pt        (auditppi_protein_features_v1)
      test_features_max{L}.pt.meta.json
      pool_manifest.json             (per-sequence source provenance; written once)

Run (pins one GPU; skips an existing cache unless --overwrite)::

    PY=/data/wmzhu/anaconda3/envs/E1/bin/python
    $PY scripts/audit_pair/extract_length_shared_pool.py \
        --max-residues 400 --layers 60 --device-id 4
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from conf.paths import ESMC_MODEL, ESMC_SAE, RESULTS_PAIR

# Test-side families covered by the shared pool. c3/bernett already have their
# own per-family length caches and are NOT re-extracted here; cross_species is
# excluded (93% of test unique seqs, and the manuscript covers it in one line).
PRING_METHODS = ("BFS", "DFS", "RANDOM_WALK")
PRING_SPECIES = ("human", "yeast", "ecoli", "arath")  # human uses all 3 methods
C_FAMILIES = ("c1", "c2")


def _test_benchmarks() -> list[tuple[str, str]]:
    """Return (benchmark_name, provenance_tag) for every test split in the pool.

    benchmark_name is a valid ``load_benchmark`` key; provenance_tag records the
    (family, split[, method]) origin written to pool_manifest.json.
    """
    items: list[tuple[str, str]] = []
    for fam in C_FAMILIES:
        items.append((f"{fam}:test", f"{fam}:test"))
    # pring human: all three sampling methods share the same test proteins but
    # carry different pair topologies; collect sequences from each so the union
    # is guaranteed to cover every method's endpoints.
    for method in PRING_METHODS:
        items.append((f"pring:human:test:{method}", f"pring:human:{method}"))
    # pring cross-species: BFS test graph per species
    for sp in PRING_SPECIES:
        if sp == "human":
            continue
        items.append((f"pring:{sp}:test:BFS", f"pring:{sp}:BFS"))
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--max-residues", type=int, required=True)
    p.add_argument("--layers", default="60", help="ESM-C layers, comma-separated (default just 60).")
    p.add_argument("--device-id", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--token-budget", type=int, default=3072)
    p.add_argument("--sae-token-chunk", type=int, default=256)
    p.add_argument("--out-root", type=Path, default=RESULTS_PAIR / "length_robustness" / "_shared_pool")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_features_max{args.max_residues}.pt"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}", flush=True)
        return

    if args.device.startswith("cuda") and args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        print(f"[device] CUDA_VISIBLE_DEVICES={args.device_id}", flush=True)

    # Import after CUDA_VISIBLE_DEVICES is set so the right GPU is bound.
    from src.data import pairs as D
    from src.data.sequences import normalize_sequence, sequence_id
    from src.features.extractors import extract_esmc_features, save_feature_cache
    from src.features.manifest import ProteinManifest

    # ---- Collect the deduplicated test endpoint sequence union ----------------
    seq2idx: dict[str, int] = {}
    sequences: list[str] = []
    protein_ids: list[str] = []
    id2idx: dict[str, int] = {}
    sources_by_seq: dict[str, set[str]] = defaultdict(set)
    id_seen: dict[str, set[str]] = defaultdict(set)

    benchmarks = _test_benchmarks()
    for bench_name, tag in benchmarks:
        bench = D.load_benchmark(bench_name, attach_seqs=True)
        n_added = 0
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
            tok = str(pid).strip()
            if tok and tok not in id_seen[seq]:
                id_seen[seq].add(tok)
            n_added += 1
        print(f"[collect] {bench_name:32s} seqs_in_split={n_added:>6} "
              f"pool_unique={len(sequences)}", flush=True)

    manifest = ProteinManifest(
        protein_ids=protein_ids,
        sequences=sequences,
        id2idx=id2idx,
        seq2idx=seq2idx,
        sources=[f"load_benchmark({bench_name})" for bench_name, _ in benchmarks],
    )
    n_trunc = sum(1 for s in sequences if len(s) > args.max_residues)
    print(f"[manifest] SHARED POOL unique_seqs={len(manifest)} "
          f"id_aliases={len(id2idx)} truncated@{args.max_residues}={n_trunc} "
          f"({100 * n_trunc / max(len(sequences), 1):.1f}%)", flush=True)

    # ---- Per-sequence provenance manifest (written once, independent of L) ------
    provenance_path = out_dir / "pool_manifest.json"
    provenance = {
        "scope": "c1+c2+pring_test_union",
        "n_unique_sequences": len(sequences),
        "families": [
            {"benchmark": bn, "tag": tg} for bn, tg in benchmarks
        ],
        "per_source_counts": {tag: sum(1 for s in sequences if tag in sources_by_seq[s])
                              for _, tag in benchmarks},
        "sequences": [
            {"seq_id": sequence_id(s), "length": len(s),
             "sources": sorted(sources_by_seq[s])}
            for s in sequences
        ],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(f"[saved] {provenance_path}", flush=True)

    # ---- ESM-C L60 forward at this truncation length ----------------------------
    layers = [int(v) for v in args.layers.split(",") if v.strip()]
    features, meta = extract_esmc_features(
        manifest,
        model_path=ESMC_MODEL,
        sae_path=ESMC_SAE,
        layers=layers,
        max_residues=args.max_residues,
        token_budget=args.token_budget,
        sae_token_chunk=args.sae_token_chunk,
        device=args.device,
        dtype=args.dtype,
    )
    meta["length_robustness"] = {
        "scope": "c1+c2+pring_test_union",
        "max_residues": args.max_residues,
        "n_unique_sequences": len(manifest),
        "n_truncated": int(n_trunc),
        "families": [bn for bn, _ in benchmarks],
    }
    save_feature_cache(
        out_path, manifest=manifest, features=features,
        extractor_meta=meta, overwrite=args.overwrite,
    )
    print(f"[saved] {out_path} features={list(features)}", flush=True)


if __name__ == "__main__":
    main()
