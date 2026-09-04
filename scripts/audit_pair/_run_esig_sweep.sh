#!/usr/bin/env bash
# eSIG-Net physicochemical fingerprint (573-D) comparison line, aligned to the
# SAE fingerprint baseline's own pair_mode / family / seed grid. eSIG is
# backbone/layer-agnostic, so every run passes --reps esig and lands under its
# own ``esig`` axis tag (esig[_mode].json), NEVER clobbering the SAE summaries.
#
# Two stages, run in order (xgb first so results are inspectable before the
# heavier mlp_pair stage):
#   xgb      : 6 fam x 3 seed x {sym, product, absdiff}   = 54 runs
#   mlp_pair : 6 fam x 3 seed x {sym, concat, rich}       = 54 runs
# The mode sets mirror exactly what the SAE side ran for each model.
#
# Pinned to GPU0. Resumable: skips a run whose summary JSON already exists.
set -u

PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI || exit 1

DEVICE=0
ROOT=results/main/ppi_fingerprint
REP=esig
BTAG=esig   # eSIG collapses {backbone}L{layer} onto this fixed axis tag

FAMILIES=(c1 c2 c3 cross_species bernett pring)
SEEDS=(42 43 44)

# SAE-side pair_mode sets, per model (see summaries on disk):
#   xgb      -> sym + the two sym feature-block ablations (product / absdiff)
#   mlp_pair -> the endpoint-fold pair modes we compare eSIG on (sym/concat/rich)
XGB_MODES=(sym product absdiff)
MLP_MODES=(sym concat rich)

# Backbone/layer are ignored by eSIG but the CLI still requires them; pass the
# canonical strongest axis so the (unused) resolution succeeds.
BACKBONE=esmc
LAYER=60

# summary_stem rule (src/ppi_fingerprint/config.py): tabular sym is BARE, pair
# models and any non-sym tabular mode carry the mode token.
summary_file () {  # $1=fam $2=model $3=seed $4=mode
  local fam=$1 model=$2 seed=$3 mode=$4
  local stem
  if [[ "$model" == "mlp_pair" || "$model" == "tabm_pair" || "$mode" != "sym" ]]; then
    stem="${BTAG}_${mode}"
  else
    stem="${BTAG}"
  fi
  echo "$ROOT/$fam/$model/seed_${seed}/summaries/${stem}.json"
}

run_stage () {  # $1=model  $2..=modes
  local model=$1; shift
  local modes=("$@")
  local total=$(( ${#FAMILIES[@]} * ${#SEEDS[@]} * ${#modes[@]} ))
  local i=0 done_cnt=0 skip_cnt=0 fail_cnt=0
  echo "[esig:$model] $total runs on GPU$DEVICE  (start $(date '+%F %T'))"
  for fam in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for mode in "${modes[@]}"; do
        i=$((i+1))
        local summary; summary=$(summary_file "$fam" "$model" "$seed" "$mode")
        local log="/tmp/esig_${model}_${fam}_seed${seed}_${mode}.log"
        local tag="[$model $i/$total] $fam seed$seed esig $mode"
        if [[ -f "$summary" ]]; then
          echo "$tag  SKIP (exists)"
          skip_cnt=$((skip_cnt+1))
          continue
        fi
        echo "$tag  RUN  $(date '+%T')"
        $PY scripts/audit_pair/run_ppi_fingerprint_baseline.py \
            --model "$model" --family "$fam" \
            --backbone "$BACKBONE" --layer "$LAYER" \
            --reps "$REP" \
            --pair-mode "$mode" \
            --device-id "$DEVICE" --seed "$seed" \
            >>"$log" 2>&1
        local rc=$?
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
  echo "[esig:$model] DONE $(date '+%F %T')  ok=$done_cnt skipped=$skip_cnt failed=$fail_cnt total=$total"
}

# Stage 1: xgb (inspect before advancing). Stage 2: mlp_pair.
run_stage xgb     "${XGB_MODES[@]}"
run_stage mlp_pair "${MLP_MODES[@]}"

echo "[esig] ALL STAGES DONE $(date '+%F %T')"
