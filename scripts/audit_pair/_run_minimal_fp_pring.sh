#!/usr/bin/env bash
# Fig1e: minimal-fingerprint ranking + sweep for pring (BFS), GPU 3 -> CPU fallback.
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
export CUDA_VISIBLE_DEVICES=3
cd /data/wmzhu/PPI/AuditPPI
fam=pring
$PY scripts/audit_pair/build_minimal_fingerprint_ranking.py --family "$fam" --rep sae_max --device-id 0
$PY scripts/audit_pair/build_minimal_fingerprint_ranking.py --family "$fam" --rep binary --device-id 0
$PY scripts/audit_pair/run_minimal_fingerprint_sweep.py --family "$fam" --device-id 0
echo "PRING_MINIMAL_FP_DONE"
