#!/usr/bin/env bash
# PPLM-only extraction, hard-pinned to the physically empty card 4.
#
# Runs in parallel with the main _run_all_features.sh driver (which is still on
# MINT). Skip-on-exists means whichever driver reaches a PPLM output first wins;
# the other just skips it, so the two never duplicate work.
#
# Device pinning: CUDA's default enumeration is by speed, so --device-id N does
# NOT match nvidia-smi index N (that is why MINT --device-id 4 landed on smi
# index 0). CUDA_DEVICE_ORDER=PCI_BUS_ID forces CUDA numbering to match
# nvidia-smi's PCI order, so --device-id 4 lands on smi index 4 (the empty card).
set -u  # no -e: one failing job must not abort the queue.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU=4
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
FEAT=scripts/baseline/features
LOGDIR=data/sae/baseline_features/_logs
mkdir -p "$LOGDIR"

# Same pair-key matrix as the main driver, ordered small -> large so the quick
# ones land first and the 422k-pair cross_species:human_train is last.
PAIR_KEYS=(
  c3:val c3:test c1:val c1:test c2:val c2:test
  pring:yeast:test pring:ecoli:test pring:arath:test
  c3:train c2:train c1:train
  pring:human:val:BFS pring:human:test:BFS pring:human:train:BFS
  pring:human:val:DFS pring:human:test:DFS pring:human:train:DFS
  pring:human:val:RANDOM_WALK pring:human:test:RANDOM_WALK pring:human:train:RANDOM_WALK
  bernett:val bernett:test bernett:train
  cross_species:ecoli cross_species:fly cross_species:mouse
  cross_species:worm cross_species:yeast cross_species:human_test
  cross_species:human_train
)

run() {  # run <log-tag> <cmd...>
  local tag="$1"; shift
  local log="$LOGDIR/${tag}.log"
  echo "=== [$(date '+%F %T')] $tag ===" | tee -a "$log"
  "$@" >>"$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! [$(date '+%F %T')] $tag FAILED rc=$rc (see $log)" | tee -a "$LOGDIR/_pplm_driver.log"
  else
    echo "--- [$(date '+%F %T')] $tag done" | tee -a "$LOGDIR/_pplm_driver.log"
  fi
}

echo "=== pplm driver start $(date '+%F %T') on smi GPU $GPU (PCI_BUS_ID order) ===" \
  | tee -a "$LOGDIR/_pplm_driver.log"

for key in "${PAIR_KEYS[@]}"; do
  tag="pplm_$(echo "$key" | tr ':' '_')"
  run "$tag" \
    "$PY" "$FEAT/extract_pplm_features.py" --benchmark "$key" --device-id "$GPU"
done

echo "=== pplm driver done $(date '+%F %T') ===" | tee -a "$LOGDIR/_pplm_driver.log"
