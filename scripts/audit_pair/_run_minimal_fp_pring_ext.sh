#!/usr/bin/env bash
# Fig1e extension: PRING DFS / RANDOM_WALK human sweeps + cross-species tests.
# Rankings for non-BFS methods are rebuilt; species tests reuse the BFS human ranking.
# XGB forced to CPU (all GPUs occupied).
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI
export CUDA_VISIBLE_DEVICES=""
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() { echo "[$(date '+%F %T')] $*"; }

# --- DFS / RANDOM_WALK: ranking + human sweep ---
for method in DFS RANDOM_WALK; do
  tag="pring_${method,,}"
  log "=== [$tag] ranking sae_max ==="
  $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
      --family pring --method "$method" --rep sae_max --xgb-cpu
  log "=== [$tag] ranking binary ==="
  $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
      --family pring --method "$method" --rep binary --xgb-cpu
  log "=== [$tag] sweep ==="
  $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
      --family pring --method "$method" --reps sae_max binary --xgb-cpu
  log "=== [$tag] DONE ==="
done

# --- Cross-species transfer: train human BFS, test yeast/ecoli/arath ---
# Ranking reused from results/.../pring/ (BFS).
for sp in yeast ecoli arath; do
  log "=== [pring_${sp}] species-test sweep (train=human BFS) ==="
  $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
      --family pring --method BFS \
      --test-benchmark "pring:${sp}:test" \
      --reps sae_max binary --xgb-cpu \
      --ranking-dir results/audit_pair/minimal_fingerprint/pring
  log "=== [pring_${sp}] DONE ==="
done

log "PRING_MINIMAL_FP_EXT_DONE"
