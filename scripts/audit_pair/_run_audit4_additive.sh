#!/usr/bin/env bash
# Sequential driver for the pending Fig2c endpoint-additive audit jobs (#4).
#
# SEQUENTIAL BY DESIGN: shared-memory budget (120 GB). Every job runs
# one-at-a-time, so peak RSS = single-job peak, never stacked. Do NOT
# parallelize. Rough per-job peaks (dense endpoint arrays, 16384-dim float32):
#   MLP  bernett       ~2 GB   (batch-gather: never materializes the pair graph)
#   MLP  cross_species ~2 GB   (batch-gather over the full 421k-pair graph)
#   EBM  bernett       ~36 GB  (a+b dense for train/val/test: 163k/59k/52k pairs)
#   EBM  cross_species ~27 GB  (train capped to 100k pairs + 52k human_test)
#   EBM  pring         ~25 GB  (DFS is the largest human graph at 113k pairs)
#
# DEVICE: GPU0-first with CPU fallback. The MLP honours CUDA_VISIBLE_DEVICES=0
# via --device cuda. The EBM is CPU-only by construction (InterpretML), so it
# ignores the GPU entirely and is bounded by --n-jobs.
#
# Covers what is still missing for Fig2c (verified against results/ 2026-08-01):
#   additive MLP: bernett, cross_species          (seeds 42/43/44 -> plot needs 3)
#   additive EBM: bernett, cross_species, pring   (seeds 42/43/44; pring per-method)
# c1/c2/c3 are already complete for both models and are NOT re-run here.
#
# NOTE: additive AUROC ~0.5 on bernett is the EXPECTED negative control, not a
# failure: bernett's negatives are degree-matched in expectation, so the
# endpoint-only shortcut this model probes has been sampled away by design.
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI
export CUDA_VISIBLE_DEVICES=0        # GPU0 first
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-auditppi

log() { echo "[$(date '+%F %T')] $*"; }

SEEDS="42 43 44"

# =========================================================================
# 1) additive MLP: bernett + cross_species (3 seeds each -> Fig2c main panel)
# =========================================================================
for fam in bernett cross_species; do
  for seed in $SEEDS; do
    log "=== [Fig2c additive-MLP $fam] seed=$seed ==="
    $PY scripts/audit_pair/run_clevel_sae_endpoint_additive_mlp.py \
        --family "$fam" --rep sae_max --backbone esmc --layer 60 \
        --seed "$seed" --device cuda
  done
done
log "ADDITIVE_MLP_DONE"

# =========================================================================
# 2) additive EBM: bernett + cross_species (3 seeds each -> Supplementary)
# =========================================================================
for fam in bernett cross_species; do
  for seed in $SEEDS; do
    log "=== [Fig2c additive-EBM $fam] seed=$seed ==="
    $PY scripts/audit_pair/run_clevel_sae_endpoint_additive_ebm.py \
        --family "$fam" --rep sae_max --backbone esmc --layer 60 \
        --seed "$seed"
  done
done
log "ADDITIVE_EBM_CLEVEL_DONE"

# =========================================================================
# 3) additive EBM: PRING (all 3 sampling methods + zero-shot species)
# =========================================================================
for seed in $SEEDS; do
  log "=== [Fig2c additive-EBM pring] seed=$seed (BFS/DFS/RANDOM_WALK) ==="
  $PY scripts/audit_pair/run_pring_sae_endpoint_additive_ebm.py \
      --method all --rep sae_max --backbone esmc --layer 60 \
      --seed "$seed"
done
log "ADDITIVE_EBM_PRING_DONE"

log "AUDIT4_ALL_DONE"
