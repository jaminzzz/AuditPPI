"""Shared codec + reader for the per-residue ESM-C SAE feature cache (DESIGN: SAE stage).

The cache stores, per protein (keyed by ``seq_id``), the SAE's top-``k`` sparse code for every residue:
``idx`` (int16, the active feature indices into the 16384 codebook) and ``val`` (fp16, their
activations). This is **lossless** vs the SAE output (``relu(topk)`` keeps exactly k nonzero) and ~130x
smaller than the dense 16384-d (256 B/residue vs ~33 KB).

The values are **plain numpy bytes** (no torch / pickle) so the writer (``E1`` env, runs ESM-C-6B) and
the reader (``genmol`` env, training) interoperate regardless of torch version. Storage is one or more
LMDB envs (sharded for parallel multi-GPU extraction); the reader opens all shards in a directory.
"""

import glob
import os
import struct

import numpy as np

_MAGIC = b"S1"   # format tag (k is fixed per cache; stored in the __meta__ record)

# Per-process registry of opened LMDB envs, keyed by realpath. LMDB refuses to open the same env
# twice within one process, so train + val datasets (and any other readers) over the same cache dir
# must share one handle. Readers are opened lazily inside __getitem__ (post-fork in DataLoader
# workers), so each worker process builds its own registry — no env handle is shared across a fork.
_ENV_CACHE: dict = {}


def _open_env(path: str):
    import lmdb
    rp = os.path.realpath(path)
    env = _ENV_CACHE.get(rp)
    if env is None:
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=2048, subdir=True)
        _ENV_CACHE[rp] = env
    return env


def pack_sparse(idx: np.ndarray, val: np.ndarray) -> bytes:
    """(L,k) int16 indices + (L,k) fp16 values -> compact bytes. L is recovered on read."""
    assert idx.shape == val.shape and idx.ndim == 2
    L, k = idx.shape
    return (_MAGIC + struct.pack("<II", L, k)
            + idx.astype("<i2").tobytes() + val.astype("<f2").tobytes())


def unpack_sparse(buf: bytes):
    """bytes -> (idx (L,k) int16, val (L,k) fp16)."""
    assert buf[:2] == _MAGIC, "bad SAE cache record"
    L, k = struct.unpack("<II", buf[2:10])
    off = 10
    idx = np.frombuffer(buf, dtype="<i2", count=L * k, offset=off).reshape(L, k)
    off += L * k * 2
    val = np.frombuffer(buf, dtype="<f2", count=L * k, offset=off).reshape(L, k)
    return idx, val


class SaeCacheReader:
    """Read-only multi-shard LMDB reader, keyed by ``seq_id`` (genmol/training side).

    Opens every ``*.lmdb`` under ``cache_dir`` with lock=False (safe for many DataLoader workers).
    ``get(seq_id)`` returns ``(idx, val)`` numpy arrays or ``None`` if the protein isn't cached."""

    def __init__(self, cache_dir: str):
        paths = sorted(glob.glob(os.path.join(cache_dir, "*.lmdb")))
        if not paths:
            raise FileNotFoundError(f"no *.lmdb shards under {cache_dir}")
        self._envs = [_open_env(p) for p in paths]
        self.paths = paths
        self.meta = self._read_meta()

    def _read_meta(self):
        for env in self._envs:
            with env.begin(buffers=True) as txn:
                buf = txn.get(b"__meta__")
                if buf is not None:
                    k, layer, dim = struct.unpack("<III", bytes(buf))
                    return {"k": k, "layer": layer, "dim": dim}
        return {}

    def get(self, seq_id: str):
        key = seq_id.encode()
        for env in self._envs:
            with env.begin(buffers=True) as txn:
                buf = txn.get(key)
                if buf is not None:
                    return unpack_sparse(bytes(buf))
        return None

    def __contains__(self, seq_id: str) -> bool:
        return self.get(seq_id) is not None
