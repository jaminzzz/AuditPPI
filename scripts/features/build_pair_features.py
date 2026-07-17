#!/usr/bin/env python3
"""Build product/absdiff/symmetric/concat protein-pair feature matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = next(path for path in Path(__file__).resolve().parents if (path / ".project-root").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pairs import (  # noqa: E402
    PAIR_MODES,
    build_pair_payload,
    load_feature_cache,
    load_pair_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--a-col", default="query")
    parser.add_argument("--b-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--pair-key", choices=["auto", "id", "sequence"], default="auto")
    parser.add_argument("--mode", choices=PAIR_MODES, default="sym")
    parser.add_argument(
        "--concat-protocol",
        choices=["train", "eval"],
        default="train",
        help="train: duplicate AB/BA; eval: save X_ab/X_ba for prediction averaging",
    )
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite to replace it")
    cache = load_feature_cache(args.cache)
    rows_a, rows_b, labels, kept, total = load_pair_indices(
        args.pairs,
        cache=cache,
        a_col=args.a_col,
        b_col=args.b_col,
        label_col=args.label_col,
        pair_key=args.pair_key,
        skip_missing=args.skip_missing,
    )
    dtype = torch.float16 if args.output_dtype == "float16" else torch.float32
    payload = build_pair_payload(
        cache=cache,
        feature_name=args.feature,
        rows_a=rows_a,
        rows_b=rows_b,
        labels=labels,
        kept_pair_indices=kept,
        n_total=total,
        mode=args.mode,
        concat_protocol=args.concat_protocol,
        output_dtype=dtype,
        chunk_size=args.chunk_size,
    )
    payload["source_cache"] = str(args.cache)
    payload["source_pairs"] = str(args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    shapes = {key: list(value.shape) for key, value in payload.items() if isinstance(value, torch.Tensor)}
    print(f"[saved] {args.output} mode={args.mode} kept={len(kept)}/{total} shapes={shapes}", flush=True)


if __name__ == "__main__":
    main()
