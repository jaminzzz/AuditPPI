#!/usr/bin/env bash
# eSIG-Net physicochemical fingerprint (573-D) TabM-pair comparison line, aligned
# to the SAE tabm_pair sweep (esmcL60/sae_max) on its own pair_mode grid so eSIG
# and SAE tabm results sit one-to-one in the same table. eSIG is
# backbone/layer-agnostic: every run passes --reps esig and lands under its own
# ``esig`` axis tag (esig_{mode}.json), NEVER clobbering the SAE summaries.
#
#   tabm_pair : 6 fam x 3 seed x {sym, concat, rich} = 54 runs
# The mode set mirrors the SAE tabm_pair sweep (_run_tabm_pair_sae_max.sh).
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
TABM_MODES=(sym concat rich)

# Backbone/layer are ignored by eSIG but the CLI still requires them; pass the
# canonical strongest axis so the (unused) resolution succeeds.
BACKBONE=esmc
LAYER=60

# summary_stem rule (src/ppi_fingerprint/config.py): pair models always carry the
# mode token, so tabm_pair summaries are esig_{mode}.json.
summary_file () {  # $1=fam $2=model $3=seed $4=mode
  local fam=$1 model=$2 seed=$3 mode=$4
  echo "$ROOT/$fam/$model/seed_${seed}/summaries/${BTAG}_${mode}.json"
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

run_stage tabm_pair "${TABM_MODES[@]}"

echo "[esig] TABM STAGE DONE $(date '+%F %T')"
