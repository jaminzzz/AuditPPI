"""GPU selection and offline model-loading setup."""

from __future__ import annotations

import os
import subprocess


def pick_free_gpu(min_free_mb: int = 6000) -> int:
    """Return the physical GPU index with the most reported free memory."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode()
        best_index, best_free = 0, -1
        for line in output.strip().splitlines():
            index_text, free_text = line.split(",")
            index, free = int(index_text), int(free_text)
            if free > best_free:
                best_index, best_free = index, free
        if best_free < min_free_mb:
            print(
                f"[warn] freest GPU {best_index} has only {best_free} MiB free "
                f"(< {min_free_mb}); using it anyway.",
                flush=True,
            )
        return best_index
    except Exception as exc:  # pragma: no cover - depends on host GPU tooling
        print(
            f"[warn] pick_free_gpu failed ({exc}); defaulting to GPU 0",
            flush=True,
        )
        return 0


def setup_device(device_id: int | None = None) -> str:
    """Pin one physical GPU before torch import and force offline model loading."""
    if device_id is None:
        device_id = pick_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print(f"[device] CUDA_VISIBLE_DEVICES={device_id} (physical)", flush=True)
    return "cuda"


__all__ = ["pick_free_gpu", "setup_device"]
