#!/usr/bin/env python3
"""Cache ESM-C SAE features for PDB_PPI chains and AFDB_DDI domains.

Run the cache stage in the E1 conda env because it loads the Biohub ESM-C/SAE models:

    /data/wmzhu/anaconda3/envs/E1/bin/python scripts/cache/cache_pdb_afdb_sae.py --device-id 7

Design:
  * PDB_PPI is cached at extracted-chain level: ``ppi:<pdb_chain_id>``.
  * AFDB_DDI is cached at extracted-domain level: ``ddi:<domain_id>``. This intentionally uses the
    domain sequence directly, matching the current DDI training granularity.

The script first builds local manifests from the raw metadata under ``PPI_data``. Use
``--metadata-only`` to only build/check manifests without loading the GPU models.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import struct
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.data.sae_cache import pack_sparse
from conf.paths import ESMC_MODEL, ESMC_SAE, PPI_DATA, RESULTS_MISC

MODEL = ESMC_MODEL
SAE = ESMC_SAE
DEFAULT_DATA_ROOT = PPI_DATA
LAYER = 60
MAX_RESIDUES = 2046

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--sources", default="ppi,ddi", help="comma list: ppi,ddi")
    p.add_argument(
        "--ppi-polarities",
        default="posi,nega",
        help="comma list for PDB_PPI: posi,nega; use posi for interface-grounding positive chains",
    )
    p.add_argument("--out-dir", type=Path, default=RESULTS_MISC / "sae_pdb_ddi_cache")
    p.add_argument(
        "--sequence-tsv",
        type=Path,
        default=None,
        help="optional TSV/TSV.GZ with cache_key and sequence columns; bypass raw metadata collection",
    )
    p.add_argument("--metadata-only", action="store_true", help="write manifests, do not run ESM-C")
    p.add_argument("--limit", type=int, default=0, help="cap rows per raw CSV for smoke tests")
    p.add_argument("--max-seqs", type=int, default=0, help="cap cached sequence count after manifest")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--token-budget", type=int, default=1024)
    p.add_argument("--device-id", type=int, default=None, help="physical GPU; default = most free")
    p.add_argument("--map-size-gb", type=int, default=120)
    p.add_argument("--commit-every", type=int, default=2000)
    p.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    p.add_argument("--overwrite", action="store_true", help="recompute keys already present in LMDB")
    p.add_argument("--strict-conflicts", action="store_true", help="fail on duplicated keys with different sequences")
    return p.parse_args()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def in_shard(seq_id: str, shard: int, num_shards: int) -> bool:
    if num_shards == 1:
        return True
    h = int(hashlib.md5(seq_id.encode()).hexdigest(), 16)
    return h % num_shards == shard


def pick_gpu():
    import subprocess

    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    ).decode()
    return sorted(
        ((int(line.split(",")[1]), line.split(",")[0].strip()) for line in out.strip().splitlines()),
        reverse=True,
    )[0][1]


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else path.open()


def parse_pdb_sequence(path: Path) -> str | None:
    """Return the one-letter residue sequence from a PDB/PDB.gz file."""
    if not path.is_file():
        return None
    residues = OrderedDict()
    try:
        with open_text(path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                if len(line) < 54:
                    continue
                resn = line[17:20].strip()
                key = (line[21], line[22:26].strip(), line[26])
                if key not in residues:
                    residues[key] = THREE2ONE.get(resn, "X")
    except (OSError, gzip.BadGzipFile):
        return None
    if not residues:
        return None
    return "".join(residues.values())


def read_csv_rows(path: Path, limit: int):
    with path.open(newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh)):
            if limit and n >= limit:
                break
            yield row


def pdb_chain_path(root: Path, polarity: str, chain1: str, chain2: str, which: str) -> Path:
    pdbid = chain1.split("_")[0]
    sub = pdbid[1:3]
    base = root / "PDB_PPI" / f"{polarity}_pdbs_extracted" / sub
    path = base / f"{chain1}-{chain2}__{which}.pdb"
    if path.is_file():
        return path
    hits = list((root / "PDB_PPI" / f"{polarity}_pdbs_extracted").glob(f"*/{chain1}-{chain2}__{which}.pdb"))
    return hits[0] if hits else path


def ddi_domain_path(root: Path, domain_id: str) -> Path:
    acc = domain_id.rsplit("_", 1)[0]
    return root / "AFDB_DDI" / "dompdbs_extracted" / acc[-2:] / f"{domain_id}.pdb"


def add_sequence(seqs, meta, key: str, seq: str | None, rec: dict, problems: list[dict]):
    if not seq:
        problems.append({**rec, "cache_key": key, "reason": "missing_or_empty_sequence"})
        return
    if key in seqs:
        if seqs[key] != seq:
            problems.append({
                **rec,
                "cache_key": key,
                "reason": "sequence_conflict",
                "old_len": len(seqs[key]),
                "new_len": len(seq),
                "old_sha1": sha1(seqs[key]),
                "new_sha1": sha1(seq),
            })
        return
    seqs[key] = seq
    meta[key] = {
        **rec,
        "cache_key": key,
        "length": len(seq),
        "sequence_sha1": sha1(seq),
    }


def collect_ppi(root: Path, limit: int, seqs, meta, polarities=("posi", "nega")):
    problems = []
    path_seq_cache: dict[Path, str | None] = {}
    for polarity in polarities:
        csv_path = root / "PDB_PPI" / f"{polarity}_nohomo.csv"
        for row in read_csv_rows(csv_path, limit):
            chain1, chain2 = row["CHAIN1:CHAIN2"].split(":")
            for side, chain in (("a", chain1), ("b", chain2)):
                path = pdb_chain_path(root, polarity, chain1, chain2, chain)
                if path not in path_seq_cache:
                    path_seq_cache[path] = parse_pdb_sequence(path)
                add_sequence(
                    seqs,
                    meta,
                    f"ppi:{chain}",
                    path_seq_cache[path],
                    {
                        "source": "ppi",
                        "id": chain,
                        "polarity": polarity,
                        "pair_id": row["CHAIN1:CHAIN2"],
                        "side": side,
                        "pdb_path": str(path),
                    },
                    problems,
                )
    return problems


def collect_ddi(root: Path, limit: int, seqs, meta):
    problems = []
    domain_records = OrderedDict()
    for polarity in ("posi", "nega"):
        csv_path = root / "AFDB_DDI" / f"{polarity}_nohomo.csv"
        for row in read_csv_rows(csv_path, limit):
            dom_a, dom_b = row["PAIRID"].split(":")
            len_a, len_b = (int(x) for x in row["LEN"].split(":"))
            for side, dom, expected_len in (("a", dom_a, len_a), ("b", dom_b, len_b)):
                acc = dom.rsplit("_", 1)[0]
                if dom not in domain_records:
                    domain_records[dom] = {
                        "source": "ddi",
                        "id": dom,
                        "domain_id": dom,
                        "protein_acc": acc,
                        "expected_domain_len": expected_len,
                        "first_polarity": polarity,
                        "first_pair_id": row["PAIRID"],
                        "first_side": side,
                    }
                elif domain_records[dom]["expected_domain_len"] != expected_len:
                    domain_records[dom]["expected_domain_len_conflict"] = sorted(
                        {domain_records[dom]["expected_domain_len"], expected_len}
                    )

    for dom, rec in domain_records.items():
        dom_path = ddi_domain_path(root, dom)
        dom_seq = parse_pdb_sequence(dom_path)
        if rec.get("expected_domain_len_conflict"):
            problems.append({
                **rec,
                "cache_key": f"ddi:{dom}",
                "domain_pdb_path": str(dom_path),
                "reason": "domain_length_metadata_conflict",
            })
        if dom_seq and rec["expected_domain_len"] != len(dom_seq):
            problems.append({
                **rec,
                "cache_key": f"ddi:{dom}",
                "domain_pdb_path": str(dom_path),
                "reason": "domain_length_mismatch",
                "domain_len": len(dom_seq),
            })
        add_sequence(
            seqs,
            meta,
            f"ddi:{dom}",
            dom_seq,
            {
                **rec,
                "domain_pdb_path": str(dom_path),
            },
            problems,
        )
    return problems


def write_jsonl(path: Path, rows):
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_sequence_tsv(path: Path, seqs, meta):
    problems = []
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "cache_key" not in (reader.fieldnames or []) or "sequence" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain cache_key and sequence columns")
        for row in reader:
            key = row["cache_key"]
            seq = row["sequence"]
            rec = {k: v for k, v in row.items() if k != "sequence"}
            rec.setdefault("source", key.split(":", 1)[0] if ":" in key else "sequence_tsv")
            rec.setdefault("id", key.split(":", 1)[1] if ":" in key else key)
            add_sequence(seqs, meta, key, seq, rec, problems)
    return problems


def build_manifests(args):
    seqs: dict[str, str] = {}
    meta: dict[str, dict] = {}
    problems = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.sequence_tsv is not None:
        problems.extend(read_sequence_tsv(args.sequence_tsv, seqs, meta))
        manifest_rows = [meta[k] for k in sorted(meta)]
        problem_rows = sorted(problems, key=lambda r: (r.get("reason", ""), r.get("cache_key", ""), r.get("id", "")))
        write_jsonl(args.out_dir / "sequence_manifest.jsonl", manifest_rows)
        write_jsonl(args.out_dir / "sequence_problems.jsonl", problem_rows)
        summary = {
            "sequence_tsv": str(args.sequence_tsv),
            "sources": ["sequence_tsv"],
            "limit_per_csv": args.limit,
            "max_residues": args.max_residues,
            "n_sequences": len(seqs),
            "n_ppi_chains": sum(1 for k in seqs if k.startswith("ppi:")),
            "n_ddi_domains": sum(1 for k in seqs if k.startswith("ddi:")),
            "n_sequence_problems": len(problem_rows),
            "problem_reason_counts": {},
            "outputs": {
                "sequence_manifest": str(args.out_dir / "sequence_manifest.jsonl"),
                "sequence_problems": str(args.out_dir / "sequence_problems.jsonl"),
            },
        }
        with (args.out_dir / "summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        return seqs, summary

    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    if "afdb" in sources:
        sources.remove("afdb")
        sources.add("ddi")
    unknown = sources - {"ppi", "ddi"}
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; expected ppi,ddi")
    ppi_polarities = tuple(p.strip().lower() for p in args.ppi_polarities.split(",") if p.strip())
    unknown_polarities = set(ppi_polarities) - {"posi", "nega"}
    if unknown_polarities:
        raise ValueError(f"unknown PDB_PPI polarities {sorted(unknown_polarities)}; expected posi,nega")
    if "ppi" in sources:
        problems.extend(collect_ppi(args.data_root, args.limit, seqs, meta, ppi_polarities))
    if "ddi" in sources:
        problems.extend(collect_ddi(args.data_root, args.limit, seqs, meta))

    manifest_rows = [meta[k] for k in sorted(meta)]
    problem_rows = sorted(problems, key=lambda r: (r.get("reason", ""), r.get("cache_key", ""), r.get("id", "")))
    problem_reason_counts = {}
    for row in problem_rows:
        reason = row.get("reason", "unknown")
        problem_reason_counts[reason] = problem_reason_counts.get(reason, 0) + 1

    write_jsonl(args.out_dir / "sequence_manifest.jsonl", manifest_rows)
    write_jsonl(args.out_dir / "sequence_problems.jsonl", problem_rows)

    summary = {
        "data_root": str(args.data_root),
        "sources": sorted(sources),
        "limit_per_csv": args.limit,
        "max_residues": args.max_residues,
        "n_sequences": len(seqs),
        "n_ppi_chains": sum(1 for k in seqs if k.startswith("ppi:")),
        "n_ddi_domains": sum(1 for k in seqs if k.startswith("ddi:")),
        "n_sequence_problems": len(problem_rows),
        "problem_reason_counts": problem_reason_counts,
        "outputs": {
            "sequence_manifest": str(args.out_dir / "sequence_manifest.jsonl"),
            "sequence_problems": str(args.out_dir / "sequence_problems.jsonl"),
        },
    }
    with (args.out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    conflict_count = sum(1 for r in problem_rows if r.get("reason") == "sequence_conflict")
    if args.strict_conflicts and conflict_count:
        raise RuntimeError(f"{conflict_count} sequence conflicts; see {args.out_dir / 'sequence_problems.jsonl'}")
    return seqs, summary


def cache_sae(args, seqs: dict[str, str]):
    dev = str(args.device_id) if args.device_id is not None else pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    print(f"[device] CUDA_VISIBLE_DEVICES={dev}", flush=True)

    import lmdb
    import torch
    from transformers import AutoModel, AutoTokenizer

    ids = [k for k in sorted(seqs) if in_shard(k, args.shard, args.num_shards)]
    if args.max_seqs:
        ids = ids[:args.max_seqs]
    print(f"[data] shard {args.shard}/{args.num_shards}: {len(ids)} sequences", flush=True)

    print("[load] ESMC-6B + SAE ...", flush=True)
    model = AutoModel.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    sae = AutoModel.from_pretrained(str(SAE), trust_remote_code=True)
    sae.initialize_layers([LAYER])
    layer = sae.layers[str(LAYER)]
    w_enc = layer.W_enc.detach().float().cuda()
    b_dec = layer.b_dec.detach().float().cuda()
    k = int(layer.params.k)
    print(f"[sae] W_enc={tuple(w_enc.shape)} k={k}", flush=True)

    @torch.inference_mode()
    def encode_sparse(h):
        x = h.float()
        x = x - x.mean(-1, keepdim=True)
        x = x / (x.std(-1, keepdim=True) + 1e-5)
        pre = torch.relu((x - b_dec) @ w_enc)
        vals, idx = pre.topk(k, dim=-1)
        return idx.to(torch.int16).cpu().numpy(), vals.to(torch.float16).cpu().numpy()

    lmdb_path = args.out_dir / f"sae_pdb_ddi.shard{args.shard}of{args.num_shards}.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        map_size=args.map_size_gb * (1024**3),
        subdir=True,
        writemap=True,
        map_async=True,
    )
    txn = env.begin(write=True)

    order = sorted(range(len(ids)), key=lambda j: len(seqs[ids[j]]))
    i = done = skipped = 0
    t0 = time.time()
    while i < len(order):
        L0 = min(len(seqs[ids[order[i]]]), args.max_residues) + 2
        bs = max(1, args.token_budget // max(L0, 1))
        batch = order[i:i + bs]
        i += bs
        batch_ids = [ids[j] for j in batch]
        if not args.overwrite:
            keep = []
            for j, seq_id in zip(batch, batch_ids):
                if txn.get(seq_id.encode()) is None:
                    keep.append(j)
                else:
                    skipped += 1
            batch = keep
            batch_ids = [ids[j] for j in batch]
            if not batch:
                continue
        seqs_b = [seqs[seq_id][:args.max_residues] for seq_id in batch_ids]
        enc = tok(seqs_b, return_tensors="pt", padding=True)
        enc = {kk: vv.to("cuda") for kk, vv in enc.items()}
        with torch.inference_mode():
            h = model(**enc, output_hidden_states=True).hidden_states[LAYER]
        am = enc["attention_mask"].bool()
        for bi, seq_id in enumerate(batch_ids):
            mask = am[bi].clone()
            nz = mask.nonzero(as_tuple=True)[0]
            if nz.numel() > 2:
                mask[nz[0]] = False
                mask[nz[-1]] = False
            idx, val = encode_sparse(h[bi][mask])
            txn.put(seq_id.encode(), pack_sparse(idx, val))
            done += 1
            if done % args.commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)
        if done and done % 1024 < max(1, len(batch_ids)):
            rate = done / max(time.time() - t0, 1e-6)
            print(
                f"  done={done}/{len(ids)} skipped={skipped} {rate:.1f} seq/s "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )

    txn.put(b"__meta__", struct.pack("<III", k, int(LAYER), 16384))
    txn.commit()
    env.sync()
    env.close()
    print(
        f"[done] shard {args.shard}: encoded={done} skipped={skipped} -> {lmdb_path} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )


def main():
    args = parse_args()
    seqs, summary = build_manifests(args)
    print(
        f"[manifest] sequences={summary['n_sequences']} "
        f"ppi={summary['n_ppi_chains']} ddi={summary['n_ddi_domains']} "
        f"problems={summary['n_sequence_problems']}",
        flush=True,
    )
    print(f"[manifest] wrote {args.out_dir / 'summary.json'}", flush=True)
    if args.metadata_only:
        return
    cache_sae(args, seqs)


if __name__ == "__main__":
    main()
