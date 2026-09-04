#!/usr/bin/env python3
"""Throwaway probe: recompute the full TreeSHAP matrix for one frozen booster and
check it reproduces the stored feature_ranking CSV. Deleted after verification."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import xgboost as xgb

from conf.model import DEFAULT_BACKBONE, resolve_backbone_layer
from conf.paths import RESULTS_PAIR
from src.features.pairs import sym_features
from src.ppi_fingerprint.baseline import _assemble

family = sys.argv[1] if len(sys.argv) > 1 else "c3"
slug = family.replace(":", "_")
layer = resolve_backbone_layer(DEFAULT_BACKBONE, None)
d = RESULTS_PAIR / "length_robustness" / slug
model_path = d / f"model_sae_max_esmcL{layer}_sym.ubj"
rank_csv = d / "feature_ranking_sae_max_sym.csv"

val_name = f"{family}:val"
print(f"[probe] assembling {val_name}", flush=True)
_, Ava, Bva, yva, _ = _assemble(val_name, "sae_max", backbone=DEFAULT_BACKBONE, layer=layer)
Xva = sym_features(Ava, Bva)
print(f"[probe] Xva={Xva.shape} pos={float(yva.mean()):.4f}", flush=True)

booster = xgb.Booster()
booster.load_model(str(model_path))
print(f"[probe] loaded {model_path.name}", flush=True)

contribs = booster.predict(xgb.DMatrix(Xva), pred_contribs=True)
shap = contribs[:, : Xva.shape[1]]
bias = contribs[:, -1]
print(f"[probe] shap matrix {shap.shape}  bias[0]={bias[0]:.6f}", flush=True)
print(f"[probe] nnz cols (any nonzero shap) = {int((np.abs(shap) > 0).any(axis=0).sum())}")

mean_abs = np.abs(shap).mean(axis=0)
ref = pd.read_csv(rank_csv)
shap_col = [c for c in ref.columns if c.endswith("mean_abs_shap")][0]
top = ref.head(10)
mine = np.argsort(mean_abs)[::-1][:10]
print(f"\n[probe] stored top-10 flat vs recomputed top-10 flat")
print(f"  stored:      {list(top['flat_feature'])}")
print(f"  recomputed:  {[int(i) for i in mine]}")
print(f"  stored {shap_col}: {[round(float(v),6) for v in top[shap_col]]}")
print(f"  recomputed        : {[round(float(mean_abs[i]),6) for i in mine]}")

merged = ref.set_index("flat_feature")[shap_col]
recomp = pd.Series(mean_abs, name="recomp")
delta = (merged.sort_index().values - recomp.sort_index().values)
print(f"\n[probe] max |stored - recomputed| over all 32768 cols = {np.abs(delta).max():.3e}")
print(f"[probe] shap row-sum vs margin check: std of (shap.sum(1)+bias) = {float((shap.sum(1)+bias).std()):.4f}")
