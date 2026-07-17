#!/usr/bin/env python3
"""Draw a retrieval-style explanation figure for TabPFN C3 query 4365.

The figure is generated as SVG so it remains editable in vector editors.
It uses existing retrieval CSVs; it does not train or rerun TabPFN.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".project-root").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from conf.paths import TABPFN_RETRIEVAL as RETRIEVAL_DIR, AUDIT  # noqa: E402

OUT = AUDIT / "ppi_fingerprint" / "tabpfn_case_4365_retrieval.svg"


NAME = {
    "P35580": "MYH10-like myosin heavy chain",
    "P35579": "MYH9-like myosin heavy chain",
    "P35749": "MYH11-like myosin heavy chain",
    "P13533": "MYH6-like myosin heavy chain",
    "P21333": "large actin-binding cytoskeletal protein",
    "O00159": "unconventional myosin-like motor",
    "P62736": "ACTA2-like actin",
    "P63261": "ACTG1-like actin",
    "P60709": "ACTB-like actin",
    "P63267": "ACTG2-like actin",
    "P68032": "ACTC1-like actin",
    "P68133": "ACTA1-like actin",
    "P60763": "small GTPase-like protein",
    "Q9UBC5": "mid-size cytoskeletal/signaling protein",
}


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


FEATURE_GROUPS = [
    ("Interface-adjacent", ["10302", "5500"], "#c06b2c"),
    ("Helix/scaffold/coiled-coil", ["14728", "4091", "4197", "4705", "7999", "11037"], "#3467a7"),
    ("Low-complexity/disorder", ["2257", "11753"], "#7a4fa3"),
    ("Domain/context pockets", ["5895", "2749"], "#5c7f33"),
]


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: float, y: float, s: str, size: int = 14, fill: str = "#1f2933", weight: str = "400") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 6) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 2, opacity: float = 1.0) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" stroke-linecap="round"/>'
    )


def protein_chip(x: float, y: float, acc: str, length: int, kind: str) -> list[str]:
    color = "#2f6fb0" if kind == "long" else "#2f8f62"
    pale = "#dcecff" if kind == "long" else "#dcf4e7"
    width = 220 if kind == "long" else 150
    out = [rect(x, y, width, 42, pale, color, 7)]
    out.append(text(x + 12, y + 17, acc, 15, color, "700"))
    out.append(text(x + 12, y + 34, f"{length} aa", 12, "#4b5563"))
    return out


def normalize_pair(a: str, b: str, la: int, lb: int) -> tuple[str, str, int, int]:
    if la >= lb:
        return a, b, la, lb
    return b, a, lb, la


def main() -> None:
    query = next(csv.DictReader((RETRIEVAL_DIR / "query_summary.csv").open()))
    rows = [r for r in csv.DictReader((RETRIEVAL_DIR / "neighbor_details.csv").open()) if r["query_rank"] == "1"]

    svg: list[str] = []
    svg.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1080" viewBox="0 0 1500 1080">'
    )
    svg.append('<rect width="1500" height="1080" fill="#fbfcfd"/>')
    svg.append(
        '<style>text{font-family:Inter,Arial,Helvetica,sans-serif}.mono{font-family:Menlo,Consolas,monospace}</style>'
    )

    svg.append(text(54, 58, "TabPFN C3 Retrieval Case: High-Confidence Positive Prediction", 28, "#111827", "700"))
    svg.append(
        text(
            54,
            88,
            "Query test pair 4365 is classified by matching a cluster of near-identical positive train pairs.",
            16,
            "#4b5563",
        )
    )

    # Query panel.
    svg.append(rect(54, 120, 1392, 168, "#ffffff", "#d9e2ec", 10))
    svg.append(text(82, 154, "Query", 18, "#111827", "700"))
    svg.append(text(82, 181, f"C3 test idx {query['query_idx']} | label {query['query_label']} | TabPFN proba {float(query['tabpfn_proba']):.6f}", 15, "#374151"))
    svg.append(text(82, 212, "normalized pair view", 12, "#6b7280"))
    svg.extend(protein_chip(82, 228, "P35580", int(query["query_seq_a_len"]), "long"))
    svg.append(text(318, 254, "paired with", 13, "#6b7280"))
    svg.extend(protein_chip(396, 228, "P62736", int(query["query_seq_b_len"]), "short"))
    svg.append(text(580, 246, "MYH10-like myosin heavy chain", 14, "#2f6fb0", "700"))
    svg.append(text(580, 270, "ACTA2-like actin", 14, "#2f8f62", "700"))
    svg.append(text(920, 154, "Top-10 retrieval summary", 18, "#111827", "700"))
    svg.append(text(920, 181, "positive neighbors: 10 / 10", 15, "#166534", "700"))
    svg.append(text(920, 207, f"positive decoder-attention mass: {float(query['top_neighbor_positive_attention_mass']):.3f}", 15, "#166534", "700"))
    svg.append(text(920, 234, "Interpretation: the model sees a dense positive context, not an isolated rule.", 14, "#4b5563"))

    # Neighbor table.
    x0, y0 = 54, 322
    svg.append(rect(x0, y0, 1392, 520, "#ffffff", "#d9e2ec", 10))
    svg.append(text(x0 + 28, y0 + 34, "Decoder-attention neighbors from C3 train", 18, "#111827", "700"))
    headers = [("rank", 82), ("train pair normalized by length/family", 150), ("label", 610), ("attention", 690), ("input Jaccard", 840), ("shared active features", 1010)]
    for h, x in headers:
        svg.append(text(x, y0 + 70, h, 12, "#6b7280", "700"))
    svg.append(line(82, y0 + 80, 1418, y0 + 80, "#e5e7eb", 1))

    max_attention = max(float(r["decoder_attention"]) for r in rows)
    for i, r in enumerate(rows):
        yy = y0 + 110 + i * 39
        train_idx = int(r["train_idx"])
        a, b = IDS.get(train_idx, ("?", "?"))
        la, lb = int(r["train_seq_a_len"]), int(r["train_seq_b_len"])
        na, nb, nla, nlb = normalize_pair(a, b, la, lb)
        jac = float(r["input_jaccard"])
        att = float(r["decoder_attention"])
        row_fill = "#edf8f1" if i < 7 else "#f7f7f8"
        if i < 7:
            row_fill = "#eaf7ef" if jac > 0.85 else "#f1f8f4"
        svg.append(rect(74, yy - 24, 1344, 32, row_fill, "none", 5))
        svg.append(text(92, yy - 3, r["neighbor_rank"], 14, "#111827", "700"))
        svg.append(text(150, yy - 3, f"{na} ({nla} aa) - {nb} ({nlb} aa)", 14, "#111827", "700"))
        svg.append(text(150, yy + 15, f"{NAME.get(na, 'protein')} / {NAME.get(nb, 'protein')}", 11, "#64748b"))
        svg.append(text(626, yy - 3, r["train_label"], 14, "#166534", "700"))
        svg.append(rect(690, yy - 17, 125, 13, "#e5e7eb", "none", 4))
        svg.append(rect(690, yy - 17, 125 * att / max_attention, 13, "#2563eb", "none", 4))
        svg.append(text(690, yy + 15, f"{att:.4f}", 11, "#475569"))
        svg.append(rect(840, yy - 17, 130, 13, "#e5e7eb", "none", 4))
        svg.append(rect(840, yy - 17, 130 * jac, 13, "#16a34a", "none", 4))
        svg.append(text(840, yy + 15, f"{jac:.3f}", 11, "#475569"))
        svg.append(text(1010, yy - 3, f"{r['input_shared_active']} / {r['input_union_active']}", 14, "#111827", "700"))
        features = r["shared_sae_features"].replace("AND:", "").split(";")
        svg.append(text(1080, yy - 3, ", ".join(features[:8]) + (" ..." if len(features) > 8 else ""), 12, "#475569"))

    # Similarity interpretation panel.
    svg.append(rect(54, 876, 1392, 150, "#ffffff", "#d9e2ec", 10))
    svg.append(text(82, 910, "What is similar?", 18, "#111827", "700"))
    svg.append(text(82, 940, "1. Pair-level family pattern: most high-attention neighbors are long myosin-like proteins paired with actin-like isoforms.", 14, "#374151"))
    svg.append(text(82, 966, "2. Feature-level overlap: the first five neighbors share 77 active pair features with the query, Jaccard about 0.92.", 14, "#374151"))
    svg.append(text(82, 992, "3. Label context: all top-10 retrieved train pairs are positives, so the local context strongly supports a positive prediction.", 14, "#374151"))

    # Feature groups.
    fx = 850
    svg.append(text(fx, 910, "Repeated SAE feature themes", 18, "#111827", "700"))
    cx = fx
    cy = 932
    for title, fids, color in FEATURE_GROUPS:
        svg.append(rect(cx, cy, 250, 30, "#f8fafc", color, 6))
        svg.append(text(cx + 10, cy + 20, title, 12, color, "700"))
        svg.append(text(cx + 10, cy + 47, ", ".join(fids), 13, "#374151"))
        cx += 275
        if cx > 1320:
            cx = fx
            cy += 68

    svg.append(text(54, 1054, "Note: attention/retrieval is associative evidence, not a causal ablation. Generated from existing TabPFN retrieval CSVs.", 12, "#6b7280"))
    svg.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(svg))
    print(OUT)


if __name__ == "__main__":
    main()
