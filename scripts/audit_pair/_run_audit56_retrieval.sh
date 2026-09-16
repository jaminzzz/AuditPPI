#!/usr/bin/env bash
# Sequential driver for the pending TabPFN retrieval/attention audits (#5 + #6).
#
# WAITS for the #4 additive driver to exit before starting, so the two never
# overlap: shared-memory budget is 120 GB and both phases are heavy. Everything
# below then runs one-at-a-time, GPU0-first.
#
# Covers the families the attention audit had not been extended to. Already done
# and NOT re-run here:
#   c1 / c2 / c3            -> leakage_audit/tabpfn_{c}_attention_feature_label/
#   cross-species (6 graphs)-> leakage_audit/{human_test,ecoli,fly,mouse,worm,yeast}/
# Remaining:
#   bernett                 -> leakage_audit/tabpfn_bernett_attention_feature_label/
#   pring human BFS         -> leakage_audit/tabpfn_pring_human_bfs_attention_feature_label/
#
# Only BFS is run for PRING: it is the released main split (DFS/RANDOM_WALK are
# supplementary settings) and it is the only PRING graph with its own
# tabpfn_topk feature ranking, which the audit consumes as input.
#
# Each run writes query_audit.tsv (one row per query: attention mass, feature
# Jaccard, same-label share, risk score), neighbor_audit_topk.tsv, summary.json,
# AUDIT_SUMMARY.md and an SVG. Those TSVs are what the SI retrieval figure reads.
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI
export CUDA_VISIBLE_DEVICES=0
export DO_NOT_TRACK=1
export POSTHOG_DISABLED=1
export TABPFN_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-auditppi

log() { echo "[$(date '+%F %T')] $*"; }

# ---- wait for the #4 additive driver to finish (serial budget) --------------
if pgrep -f "_run_audit4_additive.sh" >/dev/null 2>&1; then
  log "waiting for the #4 additive driver to finish before starting..."
  while pgrep -f "_run_audit4_additive.sh" >/dev/null 2>&1; do sleep 60; done
  log "#4 driver finished; starting retrieval audits"
fi

# ---- bernett ---------------------------------------------------------------
log "=== [Fig2e/f retrieval] bernett ==="
$PY scripts/audit_pair/audit_tabpfn_clevel_attention_feature_label.py \
    --family bernett --rep binary --backbone esmc --layer 60 \
    --query-split test --device-id 0 --tabpfn-device cuda
log "RETRIEVAL_BERNETT_DONE"

# ---- pring (human BFS) -----------------------------------------------------
log "=== [Fig2e/f retrieval] pring human BFS ==="
$PY scripts/audit_pair/audit_tabpfn_clevel_attention_feature_label.py \
    --family pring --pring-method BFS --rep binary --backbone esmc --layer 60 \
    --query-split test --device-id 0 --tabpfn-device cuda
log "RETRIEVAL_PRING_DONE"

log "AUDIT56_ALL_DONE"
