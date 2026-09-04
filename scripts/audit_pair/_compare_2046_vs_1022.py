#!/usr/bin/env python3
"""Compare 1022 vs 2046 sae_max esmcL60 xgb aggregate (3-seed mean±std)."""
import json
from pathlib import Path

ROOT = Path("results/main")
A1022 = ROOT / "ppi_fingerprint" / "{fam}" / "xgb" / "aggregate" / "esmcL60.json"
A2046 = ROOT / "ppi_fingerprint_max2046" / "{fam}" / "xgb" / "aggregate" / "esmcL60.json"

FAMS = ["c1", "c2", "c3", "cross_species", "bernett", "pring"]

def load(fam, template):
    p = Path(str(template).format(fam=fam))
    if not p.exists():
        return None
    d = json.load(open(p))
    return d.get("metrics", d.get("payload", {}))

print(f"{'family/eval':38s} {'1022 AUROC':>18s} {'2046 AUROC':>18s} {'ΔAUROC':>9s} | {'1022 AUPRC':>18s} {'2046 AUPRC':>18s} {'ΔAUPRC':>9s}")
print("-" * 140)
for fam in FAMS:
    m1022 = load(fam, A1022)
    m2046 = load(fam, A2046)
    if m1022 is None or m2046 is None:
        print(f"{fam}: 1022={'yes' if m1022 else 'NO'} 2046={'yes' if m2046 else 'NO'}")
        continue
    # sae_max keys: differ in path format (1022 uses bare, 2046 uses sym-tagged)
    def sae_keys(m):
        return [k for k in m if "sae_max" in k]
    k1022 = sorted(sae_keys(m1022))
    k2046 = sorted(sae_keys(m2046))
    # match by eval suffix (last token after /)
    def suffix(k): return k.split("/")[-1]
    by2046 = {suffix(k): m2046[k] for k in k2046}
    for k in k1022:
        s = suffix(k)
        a = m1022[k]; b = by2046.get(s)
        if b is None:
            print(f"{fam}/{s}: 2046 missing")
            continue
        da = b["auroc_mean"] - a["auroc_mean"]
        dp = b["auprc_mean"] - a["auprc_mean"]
        print(f"{fam:8s}/{s:28s} {a['auroc_mean']:.4f}±{a['auroc_std']:.4f} {b['auroc_mean']:.4f}±{b['auroc_std']:.4f} {da:+.4f} | {a['auprc_mean']:.4f}±{a['auprc_std']:.4f} {b['auprc_mean']:.4f}±{b['auprc_std']:.4f} {dp:+.4f}")
