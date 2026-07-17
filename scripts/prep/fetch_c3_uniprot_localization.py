#!/usr/bin/env python3
"""Fetch subcellular-localization annotation for all C3 proteins from UniProt.

The C3 protein ids ARE UniProt accessions (verified against the RAPPPID h5 sequence
table), so we can query UniProt directly with no id mapping.

We pull two fields per accession:
  - cc_subcellular_location : free-text SUBCELLULAR LOCATION comment
  - go_c                    : GO cellular-component terms (structured, with GO ids)

The GO CC field is the primary parse source downstream (structured), the free-text
comment is kept as an audit trail. Output is a single parquet keyed by accession,
cached so re-runs skip already-fetched ids. Only public accession ids leave the
machine; no project code or secrets are transmitted.

Run:
    /data/wmzhu/anaconda3/envs/primenet/bin/python scripts/fetch_c3_uniprot_localization.py
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from conf.paths import AUDIT

ALIGN_DIR = AUDIT / "negative_sampling_audit"
OUT = ALIGN_DIR / "c3_uniprot_localization.parquet"

ENDPOINT = "https://rest.uniprot.org/uniprotkb/accessions"
FIELDS = "accession,cc_subcellular_location,go_c"
UA = {"User-Agent": "AuditPPI/1.0 (subcellular-localization audit; research)"}


def all_c3_accessions() -> list[str]:
    ids: set[str] = set()
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ALIGN_DIR / f"c3_{split}_pair_ids.parquet")
        ids.update(df["id_a"].astype(str))
        ids.update(df["id_b"].astype(str))
    return sorted(ids)


def fetch_batch(accs: list[str], retries: int = 4) -> dict[str, tuple[str, str]]:
    """Return {accession: (subcellular_cc, go_c)} for one batch."""
    params = {"accessions": ",".join(accs), "format": "tsv", "fields": FIELDS}
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode()
            out: dict[str, tuple[str, str]] = {}
            lines = txt.splitlines()
            for line in lines[1:]:  # skip header
                cols = line.split("\t")
                acc = cols[0]
                sub = cols[1] if len(cols) > 1 else ""
                goc = cols[2] if len(cols) > 2 else ""
                out[acc] = (sub, goc)
            return out
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} after {wait}s ({type(e).__name__})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"batch failed after {retries} retries: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.3, help="polite delay between batches (s)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    accs = all_c3_accessions()
    print(f"[fetch] {len(accs)} unique C3 accessions")

    # resume from cache
    done: dict[str, tuple[str, str]] = {}
    if args.out.exists():
        prev = pd.read_parquet(args.out)
        for _, row in prev.iterrows():
            done[row["accession"]] = (row["subcellular_cc"], row["go_cc"])
        print(f"[fetch] resuming: {len(done)} already cached")

    todo = [a for a in accs if a not in done]
    print(f"[fetch] {len(todo)} to fetch in batches of {args.batch_size}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        got = fetch_batch(batch)
        for acc in batch:
            done[acc] = got.get(acc, ("", ""))  # missing acc => empty (obsolete/demerged)
        if (i // args.batch_size) % 10 == 0:
            _flush(done, args.out)
        print(f"  [{min(i+args.batch_size, len(todo))}/{len(todo)}] fetched", end="\r")
        time.sleep(args.sleep)

    _flush(done, args.out)
    n_empty = sum(1 for v in done.values() if not v[0] and not v[1])
    print(f"\n[fetch] done: {len(done)} accessions, {n_empty} with no localization annotation")


def _flush(done: dict[str, tuple[str, str]], out: Path) -> None:
    df = pd.DataFrame(
        [{"accession": k, "subcellular_cc": v[0], "go_cc": v[1]} for k, v in sorted(done.items())]
    )
    df.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
