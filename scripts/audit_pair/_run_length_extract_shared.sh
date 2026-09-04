#!/usr/bin/env bash
# Shared-pool test-length robustness: re-extract the DEDUPLICATED union of
# c1/c2/pring test endpoint sequences at every truncation length. One pool
# (12,886 unique seqs) serves every family's length-robustness evaluation.
#
# Grid: 1 sequence set x 8 lengths = 8 caches. ESM-C L60 only (--layers 60),
# so the extractor early-exits after block 60 and never runs the upper 20
# blocks (less compute + lower GPU memory floor -- the 6B backbone is ~12 GB
# in bf16 and the cards are shared).
#
# Pinned to one GPU (default 4; override: DEVICE=n bash _run_length_extract_shared.sh).
# Resumable: extract_length_shared_pool.py skips a cache that already exists.
set -u

PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI || exit 1

DEVICE=${DEVICE:-4}
LENGTHS=(100 200 400 600 800 1000 1500 2046)

OUT_ROOT=results/audit_pair/length_robustness/_shared_pool
LOG_DIR=/tmp/length_extract_shared
mkdir -p "$LOG_DIR"

total=${#LENGTHS[@]}
i=0 ok=0 skip=0 fail=0
echo "[length-extract-shared] $total caches on GPU$DEVICE  (start $(date '+%F %T'))"

for L in "${LENGTHS[@]}"; do
  i=$((i+1))
  out="$OUT_ROOT/test_features_max${L}.pt"
  log="$LOG_DIR/max${L}.log"
  tag="[$i/$total] L=$L"
  if [[ -f "$out" ]]; then
    echo "$tag  SKIP (exists)"
    skip=$((skip+1))
    continue
  fi
  echo "$tag  RUN  $(date '+%T')"
  $PY scripts/audit_pair/extract_length_shared_pool.py \
      --max-residues "$L" --layers 60 \
      --device-id "$DEVICE" \
      >"$log" 2>&1
  rc=$?
  if [[ $rc -eq 0 && -f "$out" ]]; then
    echo "$tag  OK"
    ok=$((ok+1))
  else
    echo "$tag  FAIL rc=$rc (see $log)"
    fail=$((fail+1))
  fi
done

echo "[length-extract-shared] DONE $(date '+%F %T')  ok=$ok skipped=$skip failed=$fail total=$total"
