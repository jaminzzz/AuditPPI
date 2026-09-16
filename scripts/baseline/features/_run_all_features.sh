#!/usr/bin/env bash
# Full baseline feature extraction on a single GPU, resumable.
#
# Ordered cheap -> expensive: per-protein (DeepNano ESM-2 / FlashPPI) first,
# then MINT pairs, then PPLM pairs (per-pair attention forward; slowest) last.
# Each extractor skips outputs that already exist, so re-running this script
# resumes wherever it stopped. All jobs are pinned to one GPU via --device-id.
#
# Usage:  bash scripts/baseline/features/_run_all_features.sh [GPU_ID]
set -u  # NB: no -e; one failing job must not abort the rest of the queue.

GPU="${1:-4}"
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
FEAT=scripts/baseline/features
LOGDIR=data/sae/baseline_features/_logs
mkdir -p "$LOGDIR"

PROTEIN_CACHES=data/sae/protein_caches
# 6 pair families' per-protein caches (pic_human excluded: essentiality, not PPI).
PROTEIN_KEYS=(bernett c1 c2 c3 cross_species pring_human pring_yeast pring_ecoli pring_arath)

# Pair benchmark keys, all splits, per the fingerprint split matrix.
PAIR_KEYS=(
  c1:train c1:val c1:test
  c2:train c2:val c2:test
  c3:train c3:val c3:test
  cross_species:human_train cross_species:human_test
  cross_species:ecoli cross_species:fly cross_species:mouse
  cross_species:worm cross_species:yeast
  bernett:train bernett:val bernett:test
  pring:human:train:BFS pring:human:val:BFS pring:human:test:BFS
  pring:human:train:DFS pring:human:val:DFS pring:human:test:DFS
  pring:human:train:RANDOM_WALK pring:human:val:RANDOM_WALK pring:human:test:RANDOM_WALK
  pring:yeast:test pring:ecoli:test pring:arath:test
)

run() {  # run <log-tag> <cmd...>
  local tag="$1"; shift
  local log="$LOGDIR/${tag}.log"
  echo "=== [$(date '+%F %T')] $tag ===" | tee -a "$log"
  "$@" >>"$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! [$(date '+%F %T')] $tag FAILED rc=$rc (see $log)" | tee -a "$LOGDIR/_driver.log"
  else
    echo "--- [$(date '+%F %T')] $tag done" | tee -a "$LOGDIR/_driver.log"
  fi
}

echo "=== driver start $(date '+%F %T') on GPU $GPU ===" | tee -a "$LOGDIR/_driver.log"

# 1) per-protein: DeepNano (ESM-2 only) + FlashPPI
for key in "${PROTEIN_KEYS[@]}"; do
  cache="$PROTEIN_CACHES/${key}_protein_features_max1022.pt"
  run "deepnano_${key}" \
    "$PY" "$FEAT/extract_deepnano_features.py" \
    --backbone esm2 --from-protein-cache "$cache" --device-id "$GPU"
  run "flashppi_${key}" \
    "$PY" "$FEAT/extract_flashppi_features.py" \
    --from-protein-cache "$cache" --device-id "$GPU"
done

# 2) MINT pairs (single joint forward per pair)
for key in "${PAIR_KEYS[@]}"; do
  tag="mint_$(echo "$key" | tr ':' '_')"
  run "$tag" \
    "$PY" "$FEAT/extract_mint_features.py" --benchmark "$key" --device-id "$GPU"
done

# 3) PPLM pairs (attention forward per pair; slowest -- large cross_species/pring last)
for key in "${PAIR_KEYS[@]}"; do
  tag="pplm_$(echo "$key" | tr ':' '_')"
  run "$tag" \
    "$PY" "$FEAT/extract_pplm_features.py" --benchmark "$key" --device-id "$GPU"
done

echo "=== driver done $(date '+%F %T') ===" | tee -a "$LOGDIR/_driver.log"
