#!/usr/bin/env bash
# Re-run the trainable C3 baselines under the concat pair-mode across seeds.
#
# User decision: DeepNano / MINT / PPLM all use the mlp_pair head, and on SAE
# features the mlp head is best under pair_mode=concat (order-sensitive [A||B]
# with the AB/BA train/eval protocol). So the three trainable C3 baselines are
# rebuilt at pair-mode=concat for seeds 42/43/44. FlashPPI is a zero-training
# scorer (pair_mode=none) and is untouched.
#
# Each run skips nothing on its own; this driver just sequences them. Logs land
# next to the summary tree. Pinned to one GPU via --device-id.
#
# Usage:  bash scripts/baseline/runs/_run_concat_c3.sh [GPU_ID]
set -u

GPU="${1:-4}"
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
RUNS=scripts/baseline/runs
LOG=results/main/baselines/_concat_c3.log
SEEDS=(42 43 44)

: >"$LOG"
echo "=== concat driver start $(date '+%F %T') on GPU $GPU ===" | tee -a "$LOG"

run() {  # run <tag> <cmd...>
  local tag="$1"; shift
  echo "=== [$(date '+%F %T')] $tag ===" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "--- [$(date '+%F %T')] $tag rc=$rc ---" | tee -a "$LOG"
}

for s in "${SEEDS[@]}"; do
  run "deepnano_c3_concat_s${s}" \
    "$PY" "$RUNS/train_deepnano_baseline.py" \
    --family c3 --pair-mode concat --seed "$s" --device-id "$GPU"
  run "mint_c3_concat_s${s}" \
    "$PY" "$RUNS/train_mint_baseline.py" \
    --family c3 --pair-mode concat --seed "$s" --device-id "$GPU"
  run "pplm_c3_concat_s${s}" \
    "$PY" "$RUNS/train_pplm_baseline.py" \
    --family c3 --pair-mode concat --seed "$s" --device-id "$GPU"
done

echo "=== concat driver done $(date '+%F %T') ===" | tee -a "$LOG"
