#!/usr/bin/env python3
"""Extract per-protein FlashPPI gLM2 retrieval features.

Outputs the masked-mean gLM2 representation and the official normalized query
and key vectors. For an undirected pair, downstream scoring should average
``q(A)·k(B)`` and ``q(B)·k(A)``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / ".project-root").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conf.paths import BASELINES  # noqa: E402
from src.features.manifest import load_protein_manifest  # noqa: E402


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def pick_gpu() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
        ).decode()
        return sorted(
            ((int(line.split(",")[1]), line.split(",")[0].strip()) for line in output.splitlines()),
            reverse=True,
        )[0][1]
    except Exception:  # noqa: BLE001
        return "0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--input-format", choices=["auto", "fasta", "table"], default="auto")
    parser.add_argument("--sequence-cols", default="sequence")
    parser.add_argument("--id-cols", default="")
    parser.add_argument(
        "--model",
        type=Path,
        default=BASELINES / "FlashPPI" / "FlashPPI-weights",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        device_id = str(args.device_id) if args.device_id is not None else pick_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"[device] CUDA_VISIBLE_DEVICES={device_id}", flush=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModel, AutoTokenizer

    from src.features.extractors import save_feature_cache

    manifest = load_protein_manifest(
        args.input,
        input_format=args.input_format,
        sequence_cols=comma_list(args.sequence_cols),
        id_cols=comma_list(args.id_cols) or None,
    )
    if args.limit:
        keep = min(args.limit, len(manifest))
        manifest.protein_ids = manifest.protein_ids[:keep]
        manifest.sequences = manifest.sequences[:keep]
        manifest.seq2idx = {seq: i for i, seq in enumerate(manifest.sequences)}
        manifest.id2idx = {pid: idx for pid, idx in manifest.id2idx.items() if idx < keep}

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch_dtype,
    ).to(args.device).eval()
    plm_dim = int(model.config.plm_dim)
    clip_dim = int(model.config.clip_embed_dim)
    features = {
        "flashppi_glm2_mean": torch.empty((len(manifest), plm_dim), dtype=torch.float16),
        "flashppi_query": torch.empty((len(manifest), clip_dim), dtype=torch.float16),
        "flashppi_key": torch.empty((len(manifest), clip_dim), dtype=torch.float16),
    }
    device_type = torch.device(args.device).type
    use_amp = device_type == "cuda" and args.dtype != "fp32"
    start_time = time.time()

    with torch.inference_mode():
        for start in range(0, len(manifest), args.batch_size):
            end = min(start + args.batch_size, len(manifest))
            sequences = [seq[: args.max_length] for seq in manifest.sequences[start:end]]
            inputs = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            with torch.autocast(device_type=device_type, dtype=torch_dtype, enabled=use_amp):
                residue = model.encode_protein(inputs["input_ids"], inputs["attention_mask"])
                query = model.head_q(residue, inputs["attention_mask"])
                key = model.head_k(residue, inputs["attention_mask"])
                mask = inputs["attention_mask"].unsqueeze(-1).to(residue.dtype)
                mean = (residue * mask).sum(1) / mask.sum(1).clamp_min(1)
            features["flashppi_glm2_mean"][start:end].copy_(mean.half().cpu())
            features["flashppi_query"][start:end].copy_(query.half().cpu())
            features["flashppi_key"][start:end].copy_(key.half().cpu())
            if end == len(manifest) or (start // args.batch_size + 1) % 20 == 0:
                rate = end / max(time.time() - start_time, 1e-6)
                print(f"[FlashPPI] {end}/{len(manifest)} proteins {rate:.2f}/s", flush=True)

    save_feature_cache(
        args.output,
        manifest=manifest,
        features=features,
        extractor_meta={
            "baseline": "FlashPPI",
            "model": str(args.model),
            "backbone": "gLM2-650M",
            "plm_dim": plm_dim,
            "clip_dim": clip_dim,
            "max_length": args.max_length,
            "pair_score": "0.5 * (q(A) dot k(B) + q(B) dot k(A))",
        },
        overwrite=args.overwrite,
    )
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
