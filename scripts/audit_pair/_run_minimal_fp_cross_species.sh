#!/usr/bin/env bash
# Fig1e: cross_species minimal-fingerprint ranking + sweep.
# Protocol (matches run_cross_species_tabpfn_topk):
#   - train/val carved from human_train (val_frac=0.1, seed+1; then train_subsample)
#   - default test = human_test; then zero-shot species tests reuse the ranking
# XGB forced to CPU (GPUs occupied).
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI
export CUDA_VISIBLE_DEVICES=""
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() { echo "[$(date '+%F %T')] $*"; }

RANK_DIR=results/audit_pair/minimal_fingerprint/cross_species

log "=== [cross_species] ranking sae_max (carve val from human_train) ==="
$PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
    --family cross_species --rep sae_max --val-frac 0.1 --xgb-cpu

log "=== [cross_species] ranking binary ==="
$PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
    --family cross_species --rep binary --val-frac 0.1 --xgb-cpu

log "=== [cross_species] sweep human_test ==="
$PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
    --family cross_species --reps sae_max binary --val-frac 0.1 --xgb-cpu

# Zero-shot species: same human ranking + train/val, different test graph.
for sp in ecoli fly mouse worm yeast; do
  log "=== [cross_species_${sp}] species-test sweep ==="
  $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
      --family cross_species --val-frac 0.1 \
      --test-benchmark "cross_species:${sp}" \
      --reps sae_max binary --xgb-cpu \
      --ranking-dir "$RANK_DIR"
  log "=== [cross_species_${sp}] DONE ==="
done

log "CROSS_SPECIES_MINIMAL_FP_DONE"
