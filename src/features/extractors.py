"""Unified frozen-backbone protein feature extractors.

Supported formal feature families:

* ESM-C 6B layers 60 and 80: dense mean/max and SAE mean/max/binary.
* ESM-2 650M final layer: dense mean/max and InterPLM SAE mean/max/binary.

Model imports are deliberately lazy so the CLI can be inspected from any of
the project's environments without loading GPU dependencies.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

from conf.model import ESMC_LAYERS, MAX_RESIDUES
from src.runtime import ensure_on_sys_path

import torch

from .manifest import ProteinManifest
from .pooling import dense_mean_max, pool_esmc_topk_sae, pool_relu_sae


def _torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _length_batches(sequences: Sequence[str], max_residues: int, token_budget: int) -> list[list[int]]:
    order = sorted(range(len(sequences)), key=lambda i: min(len(sequences[i]), max_residues))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        length = min(len(sequences[idx]), max_residues) + 2
        proposed_max = max(current_max, length)
        if current and proposed_max * (len(current) + 1) > token_budget:
            batches.append(current)
            current = []
            current_max = 0
        current.append(idx)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def _residue_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.bool().nonzero(as_tuple=True)[0]
    if positions.numel() <= 2:
        raise ValueError("tokenized protein has no residue tokens after removing BOS/EOS")
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    mask[positions[1:-1]] = True
    return mask


def _find_transformer_blocks(model: torch.nn.Module, n_layers: int):
    candidates = []
    for name, module in model.named_modules():
        blocks = getattr(module, "blocks", None)
        if isinstance(blocks, torch.nn.ModuleList) and len(blocks) >= n_layers:
            candidates.append((name, blocks))
    if not candidates:
        raise AttributeError("could not locate the ESM-C transformer block stack")
    return sorted(candidates, key=lambda item: len(item[1]))[-1][1]


def _available_sae_layers(sae_path: Path) -> list[int]:
    out = []
    for path in sae_path.glob("layer_*.safetensors"):
        match = re.fullmatch(r"layer_(\d+)\.safetensors", path.name)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def _configure_esmc_attention_backend(device: str) -> None:
    """Force PyTorch SDPA when ESM-C runs on CPU.

    The Transformers ESM-C implementation prefers xFormers whenever the
    package is installed, but xFormers' memory-efficient attention has no CPU
    kernel. Disabling the optional fused backends for CPU leaves the upstream
    PyTorch ``scaled_dot_product_attention`` fallback intact without changing
    the installed Transformers package.
    """
    if torch.device(device).type != "cpu":
        return
    from transformers.models.esmc import modeling_esmc

    modeling_esmc._xformers_available = False
    modeling_esmc._flash_attn_available = False


def _load_esmc_sae(sae_path: Path, layers: Sequence[int], device: str):
    from transformers import AutoModel

    sae = AutoModel.from_pretrained(str(sae_path), trust_remote_code=True)
    available = _available_sae_layers(sae_path)
    missing = sorted(set(layers) - set(available))
    if missing:
        raise FileNotFoundError(f"missing ESM-C SAE layer files for {missing} under {sae_path}")

    # Some local snapshots contain layer_80.safetensors while their older
    # config.json still advertises only layer 60. Extend the in-memory config;
    # the external checkpoint directory remains untouched.
    sae.config.available_layers = available
    sae.initialize_layers(list(layers), device=device, dtype=torch.float32)
    sae.to(device=device, dtype=torch.float32).eval()
    return sae, available


def _allocate_feature_matrix(n: int, dim: int, *, binary: bool = False) -> torch.Tensor:
    return torch.empty((n, dim), dtype=torch.bool if binary else torch.float16)


@torch.inference_mode()
def extract_esmc_features(
    manifest: ProteinManifest,
    *,
    model_path: Path,
    sae_path: Path,
    layers: Sequence[int] = ESMC_LAYERS,
    max_residues: int = MAX_RESIDUES,
    token_budget: int = 3072,
    sae_token_chunk: int = 256,
    device: str = "cuda",
    dtype: str = "bf16",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from transformers import AutoModel, AutoTokenizer

    _configure_esmc_attention_backend(device)
    requested_layers = sorted(set(int(layer) for layer in layers))
    model_dtype = _torch_dtype(dtype)
    load_kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    try:
        model = AutoModel.from_pretrained(str(model_path), **load_kwargs)
    except (TypeError, ValueError):
        load_kwargs.pop("attn_implementation")
        model = AutoModel.from_pretrained(str(model_path), **load_kwargs)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    n_layers = int(model.config.n_layers)
    if any(layer < 1 or layer > n_layers for layer in requested_layers):
        raise ValueError(f"ESM-C layers must be within 1..{n_layers}, got {requested_layers}")

    sae, checkpoint_layers = _load_esmc_sae(sae_path, requested_layers, device)
    features: dict[str, torch.Tensor] = {}
    for layer in requested_layers:
        dense_dim = int(model.config.d_model)
        sae_dim = int(sae.layers[str(layer)].W_enc.shape[1])
        prefix = f"esmc_l{layer}_"
        features[prefix + "dense_mean"] = _allocate_feature_matrix(len(manifest), dense_dim)
        features[prefix + "dense_max"] = _allocate_feature_matrix(len(manifest), dense_dim)
        features[prefix + "sae_mean"] = _allocate_feature_matrix(len(manifest), sae_dim)
        features[prefix + "sae_max"] = _allocate_feature_matrix(len(manifest), sae_dim)
        features[prefix + "sae_binary"] = _allocate_feature_matrix(
            len(manifest), sae_dim, binary=True
        )

    captured: dict[int, torch.Tensor] = {}
    handles = []
    blocks = _find_transformer_blocks(model, n_layers)
    for layer in requested_layers:
        if layer == n_layers:
            continue

        def capture(_module, _inputs, output, *, layer_idx=layer):
            captured[layer_idx] = output[0] if isinstance(output, tuple) else output

        handles.append(blocks[layer - 1].register_forward_hook(capture))

    batches = _length_batches(manifest.sequences, max_residues, token_budget)
    start_time = time.time()
    done = 0
    try:
        for batch_no, indices in enumerate(batches, start=1):
            captured.clear()
            sequences = [manifest.sequences[i][:max_residues] for i in indices]
            encoded = tokenizer(sequences, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=False)
            layer_states = dict(captured)
            if n_layers in requested_layers:
                layer_states[n_layers] = output.last_hidden_state

            for batch_row, manifest_idx in enumerate(indices):
                mask = _residue_mask(encoded["attention_mask"][batch_row])
                for layer in requested_layers:
                    hidden = layer_states[layer]
                    if hidden.ndim != 3:
                        raise RuntimeError(
                            "ESM-C intermediate states were flattened; load with the SDPA backend"
                        )
                    residue = hidden[batch_row][mask].float()
                    dense_mean, dense_max = dense_mean_max(residue)
                    sae_layer = sae.layers[str(layer)]
                    sae_max, sae_mean, sae_binary = pool_esmc_topk_sae(
                        residue,
                        w_enc=sae_layer.W_enc,
                        b_dec=sae_layer.b_dec,
                        k=int(sae_layer.params.k),
                        chunk_size=sae_token_chunk,
                    )
                    prefix = f"esmc_l{layer}_"
                    features[prefix + "dense_mean"][manifest_idx].copy_(dense_mean.half().cpu())
                    features[prefix + "dense_max"][manifest_idx].copy_(dense_max.half().cpu())
                    features[prefix + "sae_mean"][manifest_idx].copy_(sae_mean.half().cpu())
                    features[prefix + "sae_max"][manifest_idx].copy_(sae_max.half().cpu())
                    features[prefix + "sae_binary"][manifest_idx].copy_(sae_binary.cpu())
            done += len(indices)
            if batch_no == len(batches) or batch_no % 20 == 0:
                rate = done / max(time.time() - start_time, 1e-6)
                print(f"[ESM-C] batch={batch_no}/{len(batches)} proteins={done}/{len(manifest)} {rate:.2f}/s", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    meta = {
        "backbone": "esmc_6b",
        "model_path": str(model_path),
        "model_layers": n_layers,
        "layers": requested_layers,
        "sae_path": str(sae_path),
        "sae_checkpoint_layers": checkpoint_layers,
        "sae_normalization": "raw TopK magnitudes after per-residue z-score",
        "sae_k": {str(layer): int(sae.layers[str(layer)].params.k) for layer in requested_layers},
        "sae_dim": {str(layer): int(sae.layers[str(layer)].W_enc.shape[1]) for layer in requested_layers},
        "dense_dim": int(model.config.d_model),
        "max_residues": max_residues,
        "token_budget": token_budget,
        "sae_token_chunk": sae_token_chunk,
        "dtype": dtype,
    }
    return features, meta


@torch.inference_mode()
def extract_esm2_features(
    manifest: ProteinManifest,
    *,
    model_name: str,
    interplm_root: Path,
    sae_checkpoint: Path,
    max_residues: int = MAX_RESIDUES,
    token_budget: int = 8192,
    sae_token_chunk: int = 256,
    normalize_sae_features: bool = True,
    device: str = "cuda",
    dtype: str = "bf16",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from transformers import AutoTokenizer, EsmModel

    # InterPLM is a vendored upstream tree, not an installed package.
    ensure_on_sys_path(interplm_root)
    from interplm.sae.dictionary import ReLUSAE

    model_dtype = _torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(device).eval()
    sae = ReLUSAE.from_pretrained(str(sae_checkpoint), device=device).to(device).eval()
    layer = int(model.config.num_hidden_layers)
    prefix = f"esm2_l{layer}_"
    features = {
        prefix + "dense_mean": _allocate_feature_matrix(
            len(manifest), int(model.config.hidden_size)
        ),
        prefix + "dense_max": _allocate_feature_matrix(
            len(manifest), int(model.config.hidden_size)
        ),
        prefix + "sae_mean": _allocate_feature_matrix(len(manifest), int(sae.dict_size)),
        prefix + "sae_max": _allocate_feature_matrix(len(manifest), int(sae.dict_size)),
        prefix + "sae_binary": _allocate_feature_matrix(
            len(manifest), int(sae.dict_size), binary=True
        ),
    }
    batches = _length_batches(manifest.sequences, max_residues, token_budget)
    start_time = time.time()
    done = 0

    device_type = torch.device(device).type
    use_amp = device_type == "cuda" and dtype != "fp32"
    for batch_no, indices in enumerate(batches, start=1):
        sequences = [manifest.sequences[i][:max_residues] for i in indices]
        encoded = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_residues + 2,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type=device_type, dtype=model_dtype, enabled=use_amp):
            hidden = model(**encoded).last_hidden_state
        for batch_row, manifest_idx in enumerate(indices):
            mask = _residue_mask(encoded["attention_mask"][batch_row])
            residue = hidden[batch_row][mask].float()
            dense_mean, dense_max = dense_mean_max(residue)

            def encode(chunk: torch.Tensor) -> torch.Tensor:
                return sae.encode(chunk, normalize_features=normalize_sae_features)

            sae_max, sae_mean, sae_binary = pool_relu_sae(
                residue,
                encode=encode,
                feature_dim=int(sae.dict_size),
                chunk_size=sae_token_chunk,
            )
            features[prefix + "dense_mean"][manifest_idx].copy_(dense_mean.half().cpu())
            features[prefix + "dense_max"][manifest_idx].copy_(dense_max.half().cpu())
            features[prefix + "sae_mean"][manifest_idx].copy_(sae_mean.half().cpu())
            features[prefix + "sae_max"][manifest_idx].copy_(sae_max.half().cpu())
            features[prefix + "sae_binary"][manifest_idx].copy_(sae_binary.cpu())
        done += len(indices)
        if batch_no == len(batches) or batch_no % 20 == 0:
            rate = done / max(time.time() - start_time, 1e-6)
            print(f"[ESM-2] batch={batch_no}/{len(batches)} proteins={done}/{len(manifest)} {rate:.2f}/s", flush=True)

    meta = {
        "backbone": "esm2_650m",
        "model_name": model_name,
        "layer": layer,
        "sae_type": "InterPLM ReLUSAE",
        "sae_checkpoint": str(sae_checkpoint),
        "sae_normalize_features": normalize_sae_features,
        "sae_has_normalization_factors": bool(sae.has_normalization_factors),
        "sae_dim": int(sae.dict_size),
        "dense_dim": int(model.config.hidden_size),
        "max_residues": max_residues,
        "token_budget": token_budget,
        "sae_token_chunk": sae_token_chunk,
        "dtype": dtype,
    }
    return features, meta


def save_feature_cache(
    path: Path,
    *,
    manifest: ProteinManifest,
    features: dict[str, torch.Tensor],
    extractor_meta: dict[str, Any],
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "feature_shapes": {name: list(tensor.shape) for name, tensor in features.items()},
            "extractor": extractor_meta,
        },
    }
    torch.save(payload, path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(payload["meta"], indent=2, sort_keys=True))
