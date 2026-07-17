#!/usr/bin/env python3
"""Collect the RF2-PPI benchmark protein sequences into AuditPPI/data (leakage-defense query set).

The AuditPPI STRING pre-training corpus must NOT contain any protein homologous to a benchmark
protein, or the downstream PPI evaluation is contaminated. This script gathers the canonical set of
RF2-PPI benchmark proteins and writes a single, de-duplicated, fully-verified FASTA that will be the
*query* for the 3-pass homology removal (mmseqs cluster + mmseqs search + psiblast) over STRING.

Benchmark protein universe (verified):
  - ``positives_and_negatives.tsv`` (the "33k" benchmark: 3,000 pos + 30,000 neg pairs) -> 17,341 unique
  - ``benchmark_accessions.txt`` (project canonical: pos_and_neg UNION interface-size) -> 17,680 unique
  - 17,341 is a strict subset of 17,680; the 339 extras are the interface-size benchmark proteins.
We collect the **full 17,680** (a strict superset) so leakage is defended for every benchmark table.

Sequence source (verified 0 missing for all 17,680):
  ``SAE_PPI/ppi_fingerprint/outputs/rosetta_benchmark/all_sequences_100pct.fasta``
Headers there come in three shapes — ``>ACC``, ``>sp|ACC|NAME ...``, ``>tr|ACC|NAME ...`` — all parsed.

Outputs (under ``AuditPPI/data/rf2ppi_benchmark/``):
  - ``benchmark_proteins.fasta``        one record per unique accession, bare-accession header, sorted
  - ``benchmark_accessions.txt``        the 17,680 accessions (sorted)
  - ``benchmark_accessions_33k.txt``    the 17,341 pos_and_neg subset (sorted)
  - ``manifest.json``                   provenance + counts + sha256 (so we can prove what was used)

Hard-fails (non-zero exit) if any benchmark accession lacks a sequence, any sequence is empty, or the
round-trip re-read of the written FASTA does not reproduce the exact accession set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from conf.paths import BENCHMARK_TSV, RF2PPI_BENCHMARK_DIR, RF2PPI_SEQ_SOURCE  # noqa: E402

# ---- sources (read-only) -------------------------------------------------------------------------
BENCH_TSV = BENCHMARK_TSV
ACC_FULL = RF2PPI_BENCHMARK_DIR / "benchmark_accessions.txt"   # 17,680 canonical
SEQ_FASTA = RF2PPI_SEQ_SOURCE                                  # verified 0-missing source

# ---- destination ---------------------------------------------------------------------------------
OUT_DIR = RF2PPI_BENCHMARK_DIR

_AA = set("ACDEFGHIKLMNPQRSTVWY")            # 20 standard
_AA_EXTRA = set("XBZJUO")                    # ambiguous / non-canonical (allowed, just reported)


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def parse_acc(header: str) -> str:
    """'>sp|P12345|NAME ...' | '>tr|P12345|...' | '>P12345 ...' | '>P12345' -> 'P12345'."""
    h = header[1:].strip() if header.startswith(">") else header.strip()
    if "|" in h:
        return h.split("|")[1]
    return h.split()[0]


def read_pos_neg_accessions(tsv: Path) -> set[str]:
    accs: set[str] = set()
    with tsv.open() as f:
        next(f)  # header: "Protein pairs\tCategory"
        for line in f:
            pair = line.split("\t")[0].strip()
            if "_" in pair:
                a, b = pair.rsplit("_", 1)
                accs.add(a)
                accs.add(b)
    return accs


def load_fasta(path: Path) -> dict[str, str]:
    """acc -> sequence (uppercase, no whitespace). On duplicate acc, keep the longest sequence and
    report conflicts; this is conservative for homology search (longer query catches more)."""
    seqs: dict[str, str] = {}
    conflicts = 0
    acc = None
    buf: list[str] = []

    def flush():
        nonlocal conflicts, acc, buf
        if acc is None:
            return
        s = "".join(buf).strip().upper().replace("*", "")
        if s:
            if acc in seqs and seqs[acc] != s:
                conflicts += 1
                if len(s) <= len(seqs[acc]):
                    return  # keep existing (longer/equal)
            seqs[acc] = s
        buf = []

    with path.open() as f:
        for line in f:
            if line.startswith(">"):
                flush()
                acc = parse_acc(line)
                buf = []
            else:
                buf.append(line.strip())
        flush()
    if conflicts:
        print(f"[warn] {conflicts} accessions had >1 differing sequence; kept the longest")
    return seqs


def main() -> None:
    for p in (BENCH_TSV, ACC_FULL, SEQ_FASTA):
        if not p.exists():
            die(f"missing source: {p}")

    pos_neg = read_pos_neg_accessions(BENCH_TSV)
    full = {l.strip() for l in ACC_FULL.read_text().splitlines() if l.strip()}
    print(f"[bench] pos_and_neg (33k) unique accessions: {len(pos_neg)}")
    print(f"[bench] canonical benchmark_accessions.txt : {len(full)}")
    if not pos_neg <= full:
        die(f"pos_neg NOT subset of canonical list (extra {len(pos_neg - full)}) — unexpected")
    print(f"[bench] pos_neg is a strict subset; interface-size extras: {len(full - pos_neg)}")

    seqs = load_fasta(SEQ_FASTA)
    print(f"[src]  all_sequences_100pct.fasta unique accessions: {len(seqs)}")

    missing = sorted(a for a in full if a not in seqs)
    if missing:
        die(f"{len(missing)} benchmark accessions have NO sequence, e.g. {missing[:10]} — refuse to write")
    print(f"[ok]   every one of {len(full)} benchmark accessions has a sequence (0 missing)")

    # collect, validate
    collected = {a: seqs[a] for a in full}
    empties = [a for a, s in collected.items() if not s]
    if empties:
        die(f"{len(empties)} empty sequences after collection, e.g. {empties[:10]}")
    weird_chars: dict[str, int] = {}
    for s in collected.values():
        for ch in set(s) - _AA - _AA_EXTRA:
            weird_chars[ch] = weird_chars.get(ch, 0) + 1
    if weird_chars:
        print(f"[warn] non-standard residue chars present (kept as-is): {weird_chars}")
    lengths = sorted(len(s) for s in collected.values())
    print(f"[stat] lengths: min {lengths[0]}  median {lengths[len(lengths)//2]}  max {lengths[-1]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fasta_path = OUT_DIR / "benchmark_proteins.fasta"
    with fasta_path.open("w") as f:
        for acc in sorted(collected):
            f.write(f">{acc}\n")
            s = collected[acc]
            for i in range(0, len(s), 60):
                f.write(s[i : i + 60] + "\n")

    (OUT_DIR / "benchmark_accessions.txt").write_text("\n".join(sorted(full)) + "\n")
    (OUT_DIR / "benchmark_accessions_33k.txt").write_text("\n".join(sorted(pos_neg)) + "\n")

    # round-trip verification: re-read what we just wrote
    reread = load_fasta(fasta_path)
    if set(reread) != full:
        die(f"round-trip mismatch: wrote {len(reread)} vs expected {len(full)}")
    if any(reread[a] != collected[a] for a in full):
        die("round-trip sequence mismatch")
    sha = hashlib.sha256(fasta_path.read_bytes()).hexdigest()
    print(f"[ok]   round-trip verified: {len(reread)} records reproduce the exact benchmark set")

    manifest = {
        "purpose": "RF2-PPI benchmark proteins = query set for STRING homology removal (leakage defense)",
        "sources": {
            "positives_and_negatives_tsv": str(BENCH_TSV),
            "canonical_accessions": str(ACC_FULL),
            "sequence_fasta": str(SEQ_FASTA),
        },
        "counts": {
            "pos_and_neg_33k_unique": len(pos_neg),
            "canonical_total_unique": len(full),
            "interface_size_extras": len(full - pos_neg),
            "written_records": len(collected),
            "missing_sequences": 0,
        },
        "length": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
        "non_standard_residue_chars": weird_chars,
        "outputs": {
            "fasta": str(fasta_path),
            "fasta_sha256": sha,
            "accessions_full": str(OUT_DIR / "benchmark_accessions.txt"),
            "accessions_33k": str(OUT_DIR / "benchmark_accessions_33k.txt"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] wrote {len(collected)} benchmark proteins -> {fasta_path}")
    print(f"       sha256={sha}")
    print(f"       manifest -> {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
