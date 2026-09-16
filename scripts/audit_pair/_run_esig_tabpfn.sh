#!/usr/bin/env bash
# eSIG-Net physicochemical fingerprint (573-D) TabPFN comparison line.
# Full-data protocol (no train-row subsample; keep all 1146 eSIG pair dims):
#   tabpfn : 6 fam x 3 seed x {sym, concat} = 36 runs
#
# eSIG is backbone/layer-agnostic: every run passes --reps esig and lands under
# its own ``esig`` axis tag (esig.json / esig_concat.json), NEVER clobbering SAE.
#
# TabPFN flags for "full data + full eSIG features":
#   --train-subsample 1000000  → stratified_subsample keeps all rows (largest
#                                family train is cross_species ~422k)
#   --top-k 2000               → xgb_topk keeps min(k, n_feat)=1146 for
#                                sym/concat (no feature truncation)
#
# Pinned to GPU4. Resumable: skips a run whose summary JSON already exists.
# If GPU4 is still held by the SAE tabm sweep, this script waits until free.
set -u

PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI || exit 1

DEVICE=4
ROOT=results/main/ppi_fingerprint
REP=esig
BTAG=esig
MODEL=tabpfn

FAMILIES=(c1 c2 c3 cross_species bernett pring)
SEEDS=(42 43 44)
MODES=(sym concat)

# CLI still requires backbone/layer; eSIG ignores them (axis_tag → "esig").
BACKBONE=esmc
LAYER=60

# Full-data / full-feature overrides (defaults are 100k rows / top-500).
TRAIN_SUBSAMPLE=1000000
TOP_K=2000

# summary_stem: tabular sym is bare; non-sym tabular carries the mode token.
summary_file () {  # $1=fam $2=seed $3=mode
  local fam=$1 seed=$2 mode=$3 stem
  if [[ "$mode" != "sym" ]]; then
    stem="${BTAG}_${mode}"
  else
    stem="${BTAG}"
  fi
  echo "$ROOT/$fam/$MODEL/seed_${seed}/summaries/${stem}.json"
}

wait_for_gpu () {
  # Wait until the SAE tabm sweep (or any other heavy job) is off GPU$DEVICE,
  # so TabPFN gets a clean card. Poll every 60s.
  local free_mib_need=18000
  echo "[esig:tabpfn] waiting for GPU$DEVICE to free up (need ~${free_mib_need} MiB free) ..."
  while true; do
    # Still-running SAE tabm driver on this card?
    if pgrep -af "_run_tabm_pair_sae_max" | grep -v grep >/dev/null 2>&1; then
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE" 2>/dev/null | tr -d ' ')
      echo "[esig:tabpfn] SAE tabm still running on GPU$DEVICE (used ${used} MiB); sleep 60s  $(date '+%T')"
      sleep 60
      continue
    fi
    local total used free
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$DEVICE" 2>/dev/null | tr -d ' ')
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE" 2>/dev/null | tr -d ' ')
    free=$(( total - used ))
    if [[ "$free" -ge "$free_mib_need" ]]; then
      echo "[esig:tabpfn] GPU$DEVICE free enough: ${free} MiB free  $(date '+%T')"
      break
    fi
    echo "[esig:tabpfn] GPU$DEVICE used=${used}/${total} free=${free} < ${free_mib_need}; sleep 60s  $(date '+%T')"
    sleep 60
  done
}

wait_for_gpu

total=$(( ${#FAMILIES[@]} * ${#SEEDS[@]} * ${#MODES[@]} ))
i=0
done_cnt=0
skip_cnt=0
fail_cnt=0

echo "[esig:tabpfn] $total runs on GPU$DEVICE  full-data (subsample=$TRAIN_SUBSAMPLE top_k=$TOP_K)  (start $(date '+%F %T'))"

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      i=$((i+1))
      summary=$(summary_file "$fam" "$seed" "$mode")
      log="/tmp/esig_tabpfn_${fam}_seed${seed}_${mode}.log"
      tag="[tabpfn $i/$total] $fam seed$seed esig $mode"
      if [[ -f "$summary" ]]; then
        echo "$tag  SKIP (exists)"
        skip_cnt=$((skip_cnt+1))
        continue
      fi
      echo "$tag  RUN  $(date '+%T')"
      $PY scripts/audit_pair/run_ppi_fingerprint_baseline.py \
          --model "$MODEL" --family "$fam" \
          --backbone "$BACKBONE" --layer "$LAYER" \
          --reps "$REP" \
          --pair-mode "$mode" \
          --train-subsample "$TRAIN_SUBSAMPLE" \
          --top-k "$TOP_K" \
          --device-id "$DEVICE" --seed "$seed" \
          >>"$log" 2>&1
      rc=$?
      if [[ $rc -eq 0 && -f "$summary" ]]; then
        echo "$tag  OK"
        done_cnt=$((done_cnt+1))
      else
        echo "$tag  FAIL rc=$rc (see $log)"
        fail_cnt=$((fail_cnt+1))
      fi
    done
  done
done

echo "[esig:tabpfn] DONE $(date '+%F %T')  ok=$done_cnt skipped=$skip_cnt failed=$fail_cnt total=$total"
