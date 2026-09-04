#!/usr/bin/env bash
# Full baseline downstream runs for every family EXCEPT c3 (already done).
#
# Same protocol decision as _run_concat_c3.sh: the three trainable baselines
# (DeepNano / MINT / PPLM) all use the mlp_pair head and are best under
# pair_mode=concat, run across seeds 42/43/44. FlashPPI is a zero-training
# scorer (pair_mode=none); its seed is a label only, so its three "seeds" are
# identical -- we still emit them to match the c3 result tree.
#
# Each baseline run is self-contained; this driver just sequences them on one
# GPU. set -u (no -e): a single failing family/eval must not abort the queue.
#
# Usage:  bash scripts/baseline/runs/_run_concat_rest.sh [GPU_ID]
set -u

# CUDA enumerates by speed, so --device-id N does NOT match nvidia-smi index N
# by default (MINT --device-id 4 once landed on smi 0). Force PCI_BUS_ID order
# so --device-id 0 maps to smi index 0 (the empty card).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU="${1:-4}"
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
RUNS=scripts/baseline/runs
LOG=results/main/baselines/_concat_rest.log
SEEDS=(42 43 44)
FAMILIES=(c1 c2 cross_species bernett pring)

: >"$LOG"
echo "=== concat-rest driver start $(date '+%F %T') on GPU $GPU ===" | tee -a "$LOG"
echo "families: ${FAMILIES[*]} | seeds: ${SEEDS[*]}" | tee -a "$LOG"

run() {  # run <tag> <cmd...>
  local tag="$1"; shift
  echo "=== [$(date '+%F %T')] $tag ===" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "--- [$(date '+%F %T')] $tag rc=$rc ---" | tee -a "$LOG"
}

for f in "${FAMILIES[@]}"; do
  for s in "${SEEDS[@]}"; do
    run "deepnano_${f}_concat_s${s}" \
      "$PY" "$RUNS/train_deepnano_baseline.py" \
      --family "$f" --seed "$s" --device-id "$GPU"
    run "mint_${f}_concat_s${s}" \
      "$PY" "$RUNS/train_mint_baseline.py" \
      --family "$f" --pair-mode concat --seed "$s" --device-id "$GPU"
    run "pplm_${f}_concat_s${s}" \
      "$PY" "$RUNS/train_pplm_baseline.py" \
      --family "$f" --pair-mode concat --seed "$s" --device-id "$GPU"
    run "flashppi_${f}_s${s}" \
      "$PY" "$RUNS/eval_flashppi_baseline.py" \
      --family "$f" --seed "$s"
  done
done

echo "=== concat-rest driver done $(date '+%F %T') ===" | tee -a "$LOG"
