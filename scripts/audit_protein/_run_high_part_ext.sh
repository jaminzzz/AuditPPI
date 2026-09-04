#!/usr/bin/env bash
# Fig2b extension: high-participation min-hub audit for bernett + pring + cross_species.
# Protocol matches existing c1/c2/c3 products: q90, mindeg5, all reps, esmc L60, CPU XGB.
# cross_species: val carved from human_train pairs (val_frac=0.1, seed+1); test=human_test.
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI
export CUDA_VISIBLE_DEVICES=""
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() { echo "[$(date '+%F %T')] $*"; }

log "=== bernett high-participation (all reps, mindeg5) ==="
$PY scripts/audit_protein/run_c3_high_participation_classifier.py \
    --family bernett --rep all --min-degree 5 --device cpu

log "=== pring high-participation (all methods x all reps, mindeg5) ==="
$PY scripts/audit_protein/run_c3_high_participation_classifier.py \
    --family pring --method all --rep all --min-degree 5 --device cpu

log "=== cross_species high-participation (human_test, all reps, mindeg5) ==="
$PY scripts/audit_protein/run_c3_high_participation_classifier.py \
    --family cross_species --rep all --min-degree 5 --val-frac 0.1 --device cpu

log "HIGH_PART_EXT_DONE"
