#!/usr/bin/env bash
# Test-length robustness: re-extract each dataset's TEST endpoint features at
# every truncation length. This is the GPU side of the sweep; the fixed training
# side (train_length_baseline_model.py) and the CPU eval side
# (run_length_robustness_sweep.py) live in their own scripts.
#
# Grid: 2 datasets x 8 lengths = 16 caches. ESM-C L60 only (--layers 60), so the
# extractor early-exits after block 60 and never runs the upper 20 blocks
# (less compute + lower GPU memory floor -- important since the 6B backbone is
# ~12 GB in bf16 and the cards are shared).
#
# Pinned to one GPU (default 4; override: DEVICE=n bash _run_length_extract.sh).
# Resumable: extract_length_test_features.py skips a cache that already exists.
set -u

PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI || exit 1

DEVICE=${DEVICE:-4}
FAMILIES=(c3 bernett)
LENGTHS=(100 200 400 600 800 1000 1500 2046)

OUT_ROOT=results/audit_pair/length_robustness
LOG_DIR=/tmp/length_extract
mkdir -p "$LOG_DIR"

total=$(( ${#FAMILIES[@]} * ${#LENGTHS[@]} ))
i=0 ok=0 skip=0 fail=0
echo "[length-extract] $total caches on GPU$DEVICE  (start $(date '+%F %T'))"

for fam in "${FAMILIES[@]}"; do
  for L in "${LENGTHS[@]}"; do
    i=$((i+1))
    out="$OUT_ROOT/$fam/test_features_max${L}.pt"
    log="$LOG_DIR/${fam}_max${L}.log"
    tag="[$i/$total] $fam L=$L"
    if [[ -f "$out" ]]; then
      echo "$tag  SKIP (exists)"
      skip=$((skip+1))
      continue
    fi
    echo "$tag  RUN  $(date '+%T')"
    $PY scripts/audit_pair/extract_length_test_features.py \
        --family "$fam" --max-residues "$L" --layers 60 \
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
done

echo "[length-extract] DONE $(date '+%F %T')  ok=$ok skipped=$skip failed=$fail total=$total"
