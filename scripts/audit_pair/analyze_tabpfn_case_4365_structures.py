#!/usr/bin/env python3
"""Download and compare AlphaFold structures for TabPFN C3 case 4365.

The comparison is side-aware:
  query long side  P35580 is compared to each retrieved pair's longer side;
  query short side P62736 is compared to each retrieved pair's shorter side.

Outputs:
  data/structure_comparison/tabpfn_case_4365/
    structures/*.pdb
    structure_similarity_to_query.tsv
    protein_structure_stats.tsv
    tabpfn_case_4365_structure_similarity.svg
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from conf.paths import (
    AUDIT,
    TABPFN_RETRIEVAL,
    USALIGN,
)

RETRIEVAL_DIR = TABPFN_RETRIEVAL
OUT_DIR = AUDIT / "structure_comparison/tabpfn_case_4365"
STRUCT_DIR = OUT_DIR / "structures"
MATRIX_DIR = OUT_DIR / "alignment_matrices"
SVG_OUT = OUT_DIR / "tabpfn_case_4365_structure_similarity.svg"


IDS = {
    4365: ("P35580", "P62736"),
    22721: ("P35579", "P63261"),
    12450: ("P35579", "P60709"),
    1340: ("P63267", "P35749"),
    10088: ("P35749", "P63261"),
    20631: ("P60709", "P35749"),
    3622: ("P35579", "P63267"),
    23821: ("P13533", "P68032"),
    29364: ("P21333", "P68133"),
    18200: ("P60763", "Q9UBC5"),
    4842: ("P60709", "O00159"),
}

NAME = {
    "P35580": "MYH10-like myosin heavy chain",
    "P35579": "MYH9-like myosin heavy chain",
    "P35749": "MYH11-like myosin heavy chain",
    "P13533": "MYH6-like myosin heavy chain",
    "O00159": "unconventional myosin-like motor",
    "P21333": "large actin-binding cytoskeletal protein",
    "Q9UBC5": "mid-size cytoskeletal/signaling protein",
    "P62736": "ACTA2-like actin",
    "P63261": "ACTG1-like actin",
    "P60709": "ACTB-like actin",
    "P63267": "ACTG2-like actin",
    "P68032": "ACTC1-like actin",
    "P68133": "ACTA1-like actin",
    "P60763": "small GTPase-like protein",
}

QUERY_LONG = "P35580"
QUERY_SHORT = "P62736"


THREE2ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass
class Alignment:
    aligned_len: int | None
    rmsd: float | None
    seq_id: float | None
    tm_chain1: float | None
    tm_chain2: float | None
    len_chain1: int | None
    len_chain2: int | None
    matrix: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def coverage_chain1(self) -> float | None:
        if not self.aligned_len or not self.len_chain1:
            return None
        return self.aligned_len / self.len_chain1

    @property
    def tm_mean(self) -> float | None:
        if self.tm_chain1 is None or self.tm_chain2 is None:
            return None
        return (self.tm_chain1 + self.tm_chain2) / 2.0


def esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x: float, y: float, s: str, size: int = 13, fill: str = "#1f2933", weight: str = "400") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>'


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 5) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0, opacity: float = 1.0) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" stroke-linecap="round"/>'
    )


def alpha_color(value: float | None, mode: str) -> str:
    if value is None:
        return "#e5e7eb"
    if mode == "tm":
        if value >= 0.90:
            return "#15803d"
        if value >= 0.70:
            return "#65a30d"
        if value >= 0.50:
            return "#d97706"
        return "#b91c1c"
    if value >= 0.85:
        return "#15803d"
    if value >= 0.70:
        return "#65a30d"
    if value >= 0.50:
        return "#d97706"
    return "#b91c1c"


def locate_or_download(acc: str, download: bool) -> Path:
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    out = STRUCT_DIR / f"{acc}.pdb"
    if out.is_file() and out.stat().st_size > 5000:
        return out

    if not download:
        raise FileNotFoundError(f"missing local AlphaFold PDB for {acc}")

    urls = [
        f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v5.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v3.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v2.pdb",
    ]
    errors = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AuditPPI-structure-case"})
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
            if len(data) < 5000 or not data.startswith(b"HEADER"):
                errors.append(f"{url}: unexpected content length={len(data)}")
                continue
            out.write_bytes(data)
            return out
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"could not fetch {acc}: {'; '.join(errors)}")


def parse_pdb_stats(path: Path) -> dict[str, float | int | str]:
    residues = {}
    ca_coords = []
    ca_b = []
    with path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM") or len(line) < 66:
                continue
            atom = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21]
            resid = line[22:26].strip()
            icode = line[26]
            key = (chain, resid, icode)
            residues.setdefault(key, THREE2ONE.get(resn, "X"))
            if atom == "CA":
                try:
                    ca_coords.append(
                        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    )
                    ca_b.append(float(line[60:66]))
                except ValueError:
                    pass
    arr = np.array(ca_b, dtype=float)
    if arr.size:
        mean_plddt = float(arr.mean())
        plddt70 = float((arr >= 70.0).mean())
        plddt90 = float((arr >= 90.0).mean())
    else:
        mean_plddt = float("nan")
        plddt70 = float("nan")
        plddt90 = float("nan")
    return {
        "residues": len(residues),
        "ca_atoms": len(ca_coords),
        "mean_plddt": mean_plddt,
        "frac_plddt_ge70": plddt70,
        "frac_plddt_ge90": plddt90,
    }


def parse_matrix(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    rows: list[list[float]] = []
    shifts: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in {"0", "1", "2", "3"}:
            try:
                shifts.append(float(parts[1]))
                rows.append([float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
    if len(rows) != 3 or len(shifts) != 3:
        return None
    return np.array(shifts, dtype=float), np.array(rows, dtype=float)


def run_usalign(pdb1: Path, pdb2: Path, matrix_name: str | None = None) -> Alignment:
    if not USALIGN.is_file():
        raise FileNotFoundError(f"USalign not found at {USALIGN}")
    cmd = [str(USALIGN), str(pdb1), str(pdb2), "-mol", "prot", "-ter", "2"]
    matrix = None
    matrix_path = None
    if matrix_name:
        MATRIX_DIR.mkdir(parents=True, exist_ok=True)
        matrix_path = MATRIX_DIR / f"{matrix_name}.txt"
        cmd.extend(["-m", str(matrix_path)])

    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=180)
    aligned_len = rmsd = seq_id = None
    len_chain1 = len_chain2 = None
    tm_chain1 = tm_chain2 = None

    m = re.search(
        r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+),\s*Seq_ID=n_identical/n_aligned=\s*([0-9.]+)",
        out,
    )
    if m:
        aligned_len = int(m.group(1))
        rmsd = float(m.group(2))
        seq_id = float(m.group(3))

    for chain, length in re.findall(r"Length of (?:Chain|Structure)_(\d):\s*(\d+)", out):
        if chain == "1":
            len_chain1 = int(length)
        elif chain == "2":
            len_chain2 = int(length)

    tms = re.findall(
        r"TM-score=\s*([0-9.]+)\s+\(normalized by length of (?:Chain|Structure)_(\d):\s*L=\s*(\d+)",
        out,
    )
    for tm, chain, length in tms:
        if chain == "1":
            tm_chain1 = float(tm)
            len_chain1 = int(length)
        elif chain == "2":
            tm_chain2 = float(tm)
            len_chain2 = int(length)

    if matrix_path:
        matrix = parse_matrix(matrix_path)
    return Alignment(aligned_len, rmsd, seq_id, tm_chain1, tm_chain2, len_chain1, len_chain2, matrix)


def read_ca(path: Path) -> np.ndarray:
    coords = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    pass
    return np.array(coords, dtype=float)


def transformed_ca(path: Path, matrix: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    coords = read_ca(path)
    if coords.size == 0 or matrix is None:
        return coords
    shift, rotation = matrix
    return coords @ rotation.T + shift


def project_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return points[:, :2] if points.size else np.zeros((0, 2))
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def polyline_from_points(points: np.ndarray, x: float, y: float, w: float, h: float, max_points: int = 700) -> str:
    if points.size == 0:
        return ""
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).round().astype(int)
        points = points[idx]
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scaled = (points - mn) / span
    px = x + scaled[:, 0] * w
    py = y + h - scaled[:, 1] * h
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))


def overlay_svg(
    query_path: Path,
    neighbor_path: Path,
    aln: Alignment,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    query_color: str,
    neighbor_color: str,
) -> list[str]:
    q = transformed_ca(query_path, aln.matrix)
    n = read_ca(neighbor_path)
    if q.size == 0 or n.size == 0:
        return [svg_text(x, y + 24, f"{title}: no CA coordinates", 12, "#6b7280")]

    combined = np.vstack([q, n])
    projected = project_points(combined)
    q2 = projected[: q.shape[0]]
    n2 = projected[q.shape[0] :]
    q_poly = polyline_from_points(q2, x + 20, y + 44, w - 40, h - 72)
    n_poly = polyline_from_points(n2, x + 20, y + 44, w - 40, h - 72)
    lines = [
        svg_rect(x, y, w, h, "#ffffff", "#d9e2ec", 8),
        svg_text(x + 18, y + 26, title, 15, "#111827", "700"),
        f'<polyline points="{q_poly}" fill="none" stroke="{query_color}" stroke-width="1.7" opacity="0.76"/>',
        f'<polyline points="{n_poly}" fill="none" stroke="{neighbor_color}" stroke-width="1.7" opacity="0.72"/>',
        svg_rect(x + 18, y + h - 29, 12, 12, query_color, "none", 2),
        svg_text(x + 36, y + h - 18, "query", 11, "#4b5563"),
        svg_rect(x + 92, y + h - 29, 12, 12, neighbor_color, "none", 2),
        svg_text(x + 110, y + h - 18, "neighbor", 11, "#4b5563"),
    ]
    if aln.tm_chain1 is not None and aln.rmsd is not None and aln.aligned_len is not None:
        lines.append(
            svg_text(
                x + 220,
                y + h - 18,
                f"TMq {aln.tm_chain1:.3f} | RMSD {aln.rmsd:.2f} A | aligned {aln.aligned_len}",
                11,
                "#4b5563",
            )
        )
    return lines


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def read_case_rows() -> tuple[dict[str, str], list[dict[str, str]]]:
    query = next(csv.DictReader((RETRIEVAL_DIR / "query_summary.csv").open()))
    rows = [
        r
        for r in csv.DictReader((RETRIEVAL_DIR / "neighbor_details.csv").open())
        if r["query_rank"] == "1"
    ]
    rows.sort(key=lambda r: int(r["neighbor_rank"]))
    return query, rows


def normalized_pair(a: str, b: str, la: int, lb: int) -> tuple[str, str, int, int]:
    if la >= lb:
        return a, b, la, lb
    return b, a, lb, la


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def draw_figure(
    query: dict[str, str],
    pair_rows: list[dict[str, object]],
    stats: dict[str, dict[str, object]],
    overlays: dict[str, tuple[Path, Path, Alignment]],
) -> None:
    svg: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1580" height="1220" viewBox="0 0 1580 1220">',
        '<rect width="1580" height="1220" fill="#fbfcfd"/>',
        '<style>text{font-family:Inter,Arial,Helvetica,sans-serif}.mono{font-family:Menlo,Consolas,monospace}</style>',
        svg_text(54, 58, "Structure Comparison for TabPFN C3 Query 4365", 28, "#111827", "700"),
        svg_text(
            54,
            88,
            "Side-aware AFDB/USalign view: query myosin-like side vs neighbor long side; query actin side vs neighbor short side.",
            15,
            "#4b5563",
        ),
    ]

    svg.append(svg_rect(54, 120, 1472, 118, "#ffffff", "#d9e2ec", 9))
    svg.append(svg_text(82, 154, "Query pair", 17, "#111827", "700"))
    svg.append(
        svg_text(
            82,
            182,
            f"{QUERY_LONG} ({NAME[QUERY_LONG]}) paired with {QUERY_SHORT} ({NAME[QUERY_SHORT]})",
            14,
            "#374151",
        )
    )
    svg.append(
        svg_text(
            82,
            208,
            f"TabPFN probability {float(query['tabpfn_proba']):.6f}; top-10 retrieved train pairs are all positive.",
            14,
            "#166534",
            "700",
        )
    )

    for idx, acc in enumerate([QUERY_LONG, QUERY_SHORT]):
        x = 700 + idx * 360
        st = stats[acc]
        svg.append(svg_text(x, 154, acc, 17, "#111827", "700"))
        svg.append(svg_text(x, 180, f"residues {st['residues']} | mean pLDDT {float(st['mean_plddt']):.1f}", 13, "#374151"))
        svg.append(svg_text(x, 204, f"pLDDT >=70: {100 * float(st['frac_plddt_ge70']):.1f}%", 13, "#374151"))

    table_x, table_y = 54, 270
    svg.append(svg_rect(table_x, table_y, 1472, 550, "#ffffff", "#d9e2ec", 9))
    svg.append(svg_text(table_x + 28, table_y + 34, "Retrieved pairs: structural similarity to the query sides", 18, "#111827", "700"))
    headers = [
        ("rank", 82),
        ("retrieved train pair", 145),
        ("attention", 495),
        ("input Jaccard", 610),
        ("long side vs P35580", 765),
        ("short side vs P62736", 1040),
        ("readout", 1300),
    ]
    for h, x in headers:
        svg.append(svg_text(x, table_y + 70, h, 12, "#6b7280", "700"))
    svg.append(svg_line(82, table_y + 82, 1496, table_y + 82, "#e5e7eb", 1))

    max_att = max(float(r["attention"]) for r in pair_rows)
    for i, r in enumerate(pair_rows):
        y = table_y + 118 + i * 41
        jac = float(r["input_jaccard"])
        att = float(r["attention"])
        long_tm = r["long_tm_query"]
        short_tm = r["short_tm_query"]
        bg = "#eaf7ef" if i < 7 else "#f8fafc"
        svg.append(svg_rect(74, y - 25, 1422, 34, bg, "none", 5))
        svg.append(svg_text(94, y - 4, str(r["rank"]), 13, "#111827", "700"))
        svg.append(svg_text(145, y - 4, f"{r['long_acc']} - {r['short_acc']}", 13, "#111827", "700"))
        svg.append(svg_text(145, y + 13, f"{NAME.get(str(r['long_acc']), 'protein')} / {NAME.get(str(r['short_acc']), 'protein')}", 10, "#64748b"))

        svg.append(svg_rect(495, y - 17, 82, 11, "#e5e7eb", "none", 4))
        svg.append(svg_rect(495, y - 17, 82 * att / max_att, 11, "#2563eb", "none", 4))
        svg.append(svg_text(495, y + 13, f"{att:.4f}", 10, "#475569"))

        svg.append(svg_rect(610, y - 17, 100, 11, "#e5e7eb", "none", 4))
        svg.append(svg_rect(610, y - 17, 100 * jac, 11, "#16a34a", "none", 4))
        svg.append(svg_text(610, y + 13, f"{jac:.3f}", 10, "#475569"))

        svg.append(svg_rect(765, y - 18, 155, 12, "#e5e7eb", "none", 4))
        svg.append(svg_rect(765, y - 18, 155 * float(long_tm or 0.0), 12, alpha_color(long_tm, "tm"), "none", 4))
        svg.append(svg_text(930, y - 7, f"TMq {fmt(long_tm)}", 11, "#374151", "700"))
        svg.append(svg_text(765, y + 13, f"cov {fmt(r['long_cov'])}, RMSD {fmt(r['long_rmsd'], 2)} A", 10, "#64748b"))

        svg.append(svg_rect(1040, y - 18, 155, 12, "#e5e7eb", "none", 4))
        svg.append(svg_rect(1040, y - 18, 155 * float(short_tm or 0.0), 12, alpha_color(short_tm, "tm"), "none", 4))
        svg.append(svg_text(1205, y - 7, f"TMq {fmt(short_tm)}", 11, "#374151", "700"))
        svg.append(svg_text(1040, y + 13, f"cov {fmt(r['short_cov'])}, RMSD {fmt(r['short_rmsd'], 2)} A", 10, "#64748b"))

        if (long_tm or 0.0) >= 0.70 and (short_tm or 0.0) >= 0.90:
            readout = "same structural pattern"
            color = "#166534"
        elif (short_tm or 0.0) >= 0.90:
            readout = "actin side matches"
            color = "#92400e"
        else:
            readout = "weaker structural match"
            color = "#991b1b"
        svg.append(svg_text(1300, y - 4, readout, 12, color, "700"))

    svg.append(svg_rect(54, 854, 710, 295, "#ffffff", "#d9e2ec", 9))
    svg.append(svg_text(82, 888, "Interpretation", 18, "#111827", "700"))
    notes = [
        "1. High-attention neighbors are not arbitrary: their long side often aligns to P35580 and their short side almost perfectly aligns to P62736-like actin.",
        "2. Actin isoforms are near-duplicates structurally; the query actin side has TM-score near 1.0 against ACTB/ACTG/ACTC/ACTA isoforms.",
        "3. Myosin-heavy-chain neighbors share the motor/head and extended helical/coiled-coil architecture, but full-length AF2 coiled-coil orientation should be read cautiously.",
        "4. The lower-ranked outliers keep the positive label, but their structural match to the query pair is weaker; this agrees with their lower input-feature Jaccard.",
    ]
    yy = 918
    for note in notes:
        svg.append(svg_text(82, yy, note, 13, "#374151"))
        yy += 42

    if "myosin" in overlays:
        q_path, n_path, aln = overlays["myosin"]
        svg.extend(
            overlay_svg(
                q_path,
                n_path,
                aln,
                794,
                854,
                350,
                295,
                "Representative overlay: P35580 vs P35579",
                "#2563eb",
                "#16a34a",
            )
        )
    if "actin" in overlays:
        q_path, n_path, aln = overlays["actin"]
        svg.extend(
            overlay_svg(
                q_path,
                n_path,
                aln,
                1176,
                854,
                350,
                295,
                "Representative overlay: P62736 vs P63261",
                "#7c3aed",
                "#ea580c",
            )
        )

    svg.append(
        svg_text(
            54,
            1190,
            "TMq = TM-score normalized by query-side length. Coverage = aligned residues / query-side length. Structures are AlphaFold DB monomer models.",
            12,
            "#6b7280",
        )
    )
    svg.append("</svg>")
    SVG_OUT.write_text("\n".join(svg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="fail if a structure is not already local")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    query, rows = read_case_rows()
    accessions = {QUERY_LONG, QUERY_SHORT}
    for row in rows:
        train_idx = int(row["train_idx"])
        accessions.update(IDS[train_idx])

    pdbs: dict[str, Path] = {}
    missing: dict[str, str] = {}
    for acc in sorted(accessions):
        try:
            pdbs[acc] = locate_or_download(acc, download=not args.no_download)
        except Exception as exc:  # keep the case analysis running if a low-rank structure is absent.
            missing[acc] = str(exc)

    for required in (QUERY_LONG, QUERY_SHORT):
        if required not in pdbs:
            raise RuntimeError(f"required query-side structure is missing for {required}: {missing.get(required)}")

    if missing:
        write_tsv(
            OUT_DIR / "missing_structures.tsv",
            [{"accession": acc, "name": NAME.get(acc, ""), "reason": reason} for acc, reason in sorted(missing.items())],
            ["accession", "name", "reason"],
        )

    stats = {acc: parse_pdb_stats(path) for acc, path in pdbs.items()}

    stat_rows = []
    for acc in sorted(stats):
        stat_rows.append({"accession": acc, "name": NAME.get(acc, ""), **stats[acc], "pdb_path": str(pdbs[acc])})
    write_tsv(
        OUT_DIR / "protein_structure_stats.tsv",
        stat_rows,
        ["accession", "name", "residues", "ca_atoms", "mean_plddt", "frac_plddt_ge70", "frac_plddt_ge90", "pdb_path"],
    )

    alignment_cache: dict[tuple[str, str], Alignment] = {}

    def align(query_acc: str, neighbor_acc: str, matrix_name: str | None = None) -> Alignment:
        if query_acc not in pdbs or neighbor_acc not in pdbs:
            q_len = int(stats[query_acc]["residues"]) if query_acc in stats else None
            n_len = int(stats[neighbor_acc]["residues"]) if neighbor_acc in stats else None
            return Alignment(None, None, None, None, None, q_len, n_len)
        key = (query_acc, neighbor_acc)
        if key not in alignment_cache or matrix_name:
            alignment_cache[key] = run_usalign(pdbs[query_acc], pdbs[neighbor_acc], matrix_name=matrix_name)
        return alignment_cache[key]

    pair_rows: list[dict[str, object]] = []
    for row in rows:
        rank = int(row["neighbor_rank"])
        train_idx = int(row["train_idx"])
        a, b = IDS[train_idx]
        la, lb = int(row["train_seq_a_len"]), int(row["train_seq_b_len"])
        long_acc, short_acc, long_len, short_len = normalized_pair(a, b, la, lb)
        long_aln = align(QUERY_LONG, long_acc)
        short_aln = align(QUERY_SHORT, short_acc)
        pair_rows.append(
            {
                "rank": rank,
                "train_idx": train_idx,
                "long_acc": long_acc,
                "short_acc": short_acc,
                "long_train_len": long_len,
                "short_train_len": short_len,
                "attention": float(row["decoder_attention"]),
                "input_jaccard": float(row["input_jaccard"]),
                "long_aligned_len": long_aln.aligned_len,
                "long_rmsd": long_aln.rmsd,
                "long_seq_id": long_aln.seq_id,
                "long_tm_query": long_aln.tm_chain1,
                "long_tm_neighbor": long_aln.tm_chain2,
                "long_cov": long_aln.coverage_chain1,
                "short_aligned_len": short_aln.aligned_len,
                "short_rmsd": short_aln.rmsd,
                "short_seq_id": short_aln.seq_id,
                "short_tm_query": short_aln.tm_chain1,
                "short_tm_neighbor": short_aln.tm_chain2,
                "short_cov": short_aln.coverage_chain1,
            }
        )

    write_tsv(
        OUT_DIR / "structure_similarity_to_query.tsv",
        pair_rows,
        [
            "rank",
            "train_idx",
            "long_acc",
            "short_acc",
            "long_train_len",
            "short_train_len",
            "attention",
            "input_jaccard",
            "long_aligned_len",
            "long_rmsd",
            "long_seq_id",
            "long_tm_query",
            "long_tm_neighbor",
            "long_cov",
            "short_aligned_len",
            "short_rmsd",
            "short_seq_id",
            "short_tm_query",
            "short_tm_neighbor",
            "short_cov",
        ],
    )

    overlays = {}
    if "P35579" in pdbs:
        overlays["myosin"] = (
            pdbs[QUERY_LONG],
            pdbs["P35579"],
            run_usalign(pdbs[QUERY_LONG], pdbs["P35579"], "P35580_to_P35579"),
        )
    if "P63261" in pdbs:
        overlays["actin"] = (
            pdbs[QUERY_SHORT],
            pdbs["P63261"],
            run_usalign(pdbs[QUERY_SHORT], pdbs["P63261"], "P62736_to_P63261"),
        )
    draw_figure(query, pair_rows, stats, overlays)
    print(SVG_OUT)
    print(OUT_DIR / "structure_similarity_to_query.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
