#!/usr/bin/env bash
# Shared-pool length-robustness EVALUATION: score each c1/c2/pring family's
# frozen booster (from train_length_baseline_model.py) against the SHARED
# test-length pool (from extract_length_shared_pool.py).
#
# CPU only (XGBoost inference). The shared pool serves all families; each
# family reads its rows by sequence key from the same per-length caches.
#
# Prerequisites:
#   - 8 shared-pool caches exist at results/audit_pair/length_robustness/_shared_pool/
#     (run _run_length_extract_shared.sh)
#   - per-family boosters exist at results/audit_pair/length_robustness/{family_slug}/
#     (run train_length_baseline_model.py --family <key>)
set -uo pipefail

PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI || exit 1

export CUDA_VISIBLE_DEVICES=""   # CPU inference; no GPU needed
export DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

POOL=results/audit_pair/length_robustness/_shared_pool
LOG_DIR=/tmp/length_sweep_shared
mkdir -p "$LOG_DIR"

# family keys: c1/c2 (native) + pring human 3 methods (own booster each) +
# pring cross-species 3 species (booster trained on human BFS, zero-shot test).
FAMILIES=(
  c1
  c2
  pring:human:BFS
  pring:human:DFS
  pring:human:RANDOM_WALK
  pring:yeast
  pring:ecoli
  pring:arath
)

total=${#FAMILIES[@]}
i=0 ok=0 fail=0
echo "[length-sweep-shared] $total families  (start $(date '+%F %T'))"

for fam in "${FAMILIES[@]}"; do
  i=$((i+1))
  tag="[$i/$total] $fam"
  log="$LOG_DIR/$(echo "$fam" | tr : _).log"
  echo "$tag  RUN  $(date '+%T')"
  $PY scripts/audit_pair/run_length_robustness_sweep.py \
      --family "$fam" --shared-pool "$POOL" --overwrite \
      >"$log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "$tag  OK"
    ok=$((ok+1))
  else
    echo "$tag  FAIL rc=$rc (see $log)"
    fail=$((fail+1))
  fi
done

echo "[length-sweep-shared] DONE $(date '+%F %T')  ok=$ok failed=$fail total=$total"
