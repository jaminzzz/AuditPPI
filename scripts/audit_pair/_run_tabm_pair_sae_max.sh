#!/usr/bin/env bash
# Narrowed tabm_pair architecture ablation, aligned to mlp_pair on the strongest
# axis only: esmcL60 + sae_max, sweeping the 3 pair_modes across all 6 families
# and 3 seeds = 54 runs. Each run trains a SINGLE rep (--reps sae_max), so each
# summary holds only the sae_max key -- a one-to-one comparison against the
# sae_max row of the matching mlp_pair summary.
# Pinned to GPU4. Resumable: skips a run whose summary JSON already exists.
set -u

PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI || exit 1

DEVICE=4
MODEL=tabm_pair
ROOT=results/main/ppi_fingerprint

FAMILIES=(c1 c2 c3 cross_species bernett pring)
SEEDS=(42 43 44)
MODES=(sym concat rich)
BACKBONE=esmc
LAYER=60
REP=sae_max
BTAG="${BACKBONE}L${LAYER}"

total=$(( ${#FAMILIES[@]} * ${#SEEDS[@]} * ${#MODES[@]} ))
i=0
done_cnt=0
skip_cnt=0
fail_cnt=0

echo "[sweep] tabm_pair narrowed ($BTAG/$REP): $total runs on GPU$DEVICE  (start $(date '+%F %T'))"

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      i=$((i+1))
      summary="$ROOT/$fam/$MODEL/seed_${seed}/summaries/${BTAG}_${mode}.json"
      tag="[$i/$total] $fam seed$seed $BTAG $mode"
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
          --device-id "$DEVICE" --seed "$seed" \
          >>"/tmp/tabm_saemax_${fam}_seed${seed}_${BTAG}_${mode}.log" 2>&1
      rc=$?
      if [[ $rc -eq 0 && -f "$summary" ]]; then
        echo "$tag  OK"
        done_cnt=$((done_cnt+1))
      else
        echo "$tag  FAIL rc=$rc (see /tmp/tabm_saemax_${fam}_seed${seed}_${BTAG}_${mode}.log)"
        fail_cnt=$((fail_cnt+1))
      fi
    done
  done
done

echo "[sweep] DONE $(date '+%F %T')  ok=$done_cnt skipped=$skip_cnt failed=$fail_cnt total=$total"
