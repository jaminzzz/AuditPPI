#!/usr/bin/env bash
# One sequential driver for all pending Fig1e / Fig2b extension jobs.
#
# SEQUENTIAL BY DESIGN: shared-memory budget (120 GB). Every job runs
# one-at-a-time, so peak RSS = single-job peak (~48 GB cross_species pair,
# ~30 GB PRING pair, ~5 GB protein), never stacked. Do NOT parallelize.
#
# DEVICE: GPU0-first with automatic CPU fallback. setup_device(0) pins GPU0 for
# the pair scripts (--device-id 0); the protein classifier honours the exported
# CUDA_VISIBLE_DEVICES=0 with --device cuda. Both XGB factories try CUDA and
# fall back to CPU on ANY GPU error (incl. OOM) via fit_with_cpu_fallback, so
# small-K cells that fit in GPU0's free VRAM use the GPU and the big full-dim /
# 100k-row fits transparently drop to CPU. No hard crash on GPU OOM.
#
# Covers what is still missing (verified against results/ on 2026-07-31):
#   Fig2b (protein high-participation):
#     - cross_species: sae_max + esmc_mean (binary already done)
#   Fig1e (pair minimal-fingerprint):
#     - cross_species: ranking(sae_max,binary) + human_test sweep + 5 species
#     - pring DFS / RANDOM_WALK: ranking(sae_max,binary) + human sweep + species
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI
export CUDA_VISIBLE_DEVICES=0        # GPU0 first; XGB factories fall back to CPU on OOM
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() { echo "[$(date '+%F %T')] $*"; }

XRANK=results/audit_pair/minimal_fingerprint/cross_species
PRANK=results/audit_pair/minimal_fingerprint/pring   # BFS human ranking (species reuse)

# =========================================================================
# 1) Fig2b: cross_species high-participation (complete the missing reps)
# =========================================================================
for rep in sae_max esmc_mean; do
  log "=== [Fig2b cross_species] high-participation rep=$rep ==="
  $PY scripts/audit_protein/run_c3_high_participation_classifier.py \
      --family cross_species --rep "$rep" --min-degree 5 --val-frac 0.1 --device cuda
done
log "FIG2B_CROSS_SPECIES_DONE"

# =========================================================================
# 2) Fig1e: cross_species ranking + sweeps (memory-safe index-first carve)
# =========================================================================
log "=== [Fig1e cross_species] ranking sae_max ==="
$PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
    --family cross_species --rep sae_max --val-frac 0.1 --device-id 0
log "=== [Fig1e cross_species] ranking binary ==="
$PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
    --family cross_species --rep binary --val-frac 0.1 --device-id 0

log "=== [Fig1e cross_species] sweep human_test ==="
$PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
    --family cross_species --reps sae_max binary --val-frac 0.1 --device-id 0

for sp in ecoli fly mouse worm yeast; do
  log "=== [Fig1e cross_species_${sp}] species-test sweep ==="
  $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
      --family cross_species --val-frac 0.1 \
      --test-benchmark "cross_species:${sp}" \
      --reps sae_max binary --device-id 0 \
      --ranking-dir "$XRANK"
done
log "FIG1E_CROSS_SPECIES_DONE"

# =========================================================================
# 3) Fig1e: PRING DFS / RANDOM_WALK ranking + human sweep
# =========================================================================
for method in DFS RANDOM_WALK; do
  tag="pring_${method,,}"
  log "=== [Fig1e $tag] ranking sae_max ==="
  $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
      --family pring --method "$method" --rep sae_max --device-id 0
  log "=== [Fig1e $tag] ranking binary ==="
  $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py \
      --family pring --method "$method" --rep binary --device-id 0
  log "=== [Fig1e $tag] human sweep ==="
  $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
      --family pring --method "$method" --reps sae_max binary --device-id 0
done
log "FIG1E_PRING_DFS_RW_DONE"

# =========================================================================
# 4) Fig1e: PRING cross-species transfer (train human BFS, test other species)
#    Ranking reused from the BFS human ranking dir.
# =========================================================================
for sp in yeast ecoli arath; do
  log "=== [Fig1e pring_${sp}] species-test sweep (train=human BFS) ==="
  $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py \
      --family pring --method BFS \
      --test-benchmark "pring:${sp}:test" \
      --reps sae_max binary --device-id 0 \
      --ranking-dir "$PRANK"
done
log "FIG1E_PRING_SPECIES_DONE"

log "ALL_PENDING_DONE"
