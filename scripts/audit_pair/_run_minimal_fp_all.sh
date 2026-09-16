#!/usr/bin/env bash
# Fig1e: minimal-fingerprint ranking + sweep for c1/c2/bernett (c3 already done).
# GPU 3 pinned; fit_xgb falls back to CPU automatically if CUDA OOMs.
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
export CUDA_VISIBLE_DEVICES=3
cd /data/wmzhu/PPI/AuditPPI

for fam in c1 c2 bernett; do
  echo "=== [$fam] build ranking sae_max ==="
  $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py --family "$fam" --rep sae_max --device-id 0
  echo "=== [$fam] build ranking binary ==="
  $PY scripts/audit_pair/build_minimal_fingerprint_ranking.py --family "$fam" --rep binary --device-id 0
  echo "=== [$fam] sweep (both reps) ==="
  $PY scripts/audit_pair/run_minimal_fingerprint_sweep.py --family "$fam" --device-id 0
  echo "=== [$fam] DONE ==="
done
echo "ALL_MINIMAL_FP_DONE"
