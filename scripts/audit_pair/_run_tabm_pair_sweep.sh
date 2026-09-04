#!/usr/bin/env bash
# Full tabm_pair sweep, symmetric to the completed mlp_pair matrix:
#   6 families x 3 seeds x 3 backbone-layer x 3 pair_mode  = 162 summaries,
#   each summary sweeps the 3 default reps (binary / sae_max / esmc_mean).
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
# backbone:layer pairs
AXES=("esmc:60" "esmc:80" "esm2:33")

total=$(( ${#FAMILIES[@]} * ${#SEEDS[@]} * ${#MODES[@]} * ${#AXES[@]} ))
i=0
done_cnt=0
skip_cnt=0
fail_cnt=0

echo "[sweep] tabm_pair full matrix: $total runs on GPU$DEVICE  (start $(date '+%F %T'))"

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for axis in "${AXES[@]}"; do
      backbone="${axis%%:*}"
      layer="${axis##*:}"
      btag="${backbone}L${layer}"
      for mode in "${MODES[@]}"; do
        i=$((i+1))
        summary="$ROOT/$fam/$MODEL/seed_${seed}/summaries/${btag}_${mode}.json"
        tag="[$i/$total] $fam seed$seed $btag $mode"
        if [[ -f "$summary" ]]; then
          echo "$tag  SKIP (exists)"
          skip_cnt=$((skip_cnt+1))
          continue
        fi
        echo "$tag  RUN  $(date '+%T')"
        $PY scripts/audit_pair/run_ppi_fingerprint_baseline.py \
            --model "$MODEL" --family "$fam" \
            --backbone "$backbone" --layer "$layer" \
            --pair-mode "$mode" \
            --device-id "$DEVICE" --seed "$seed" \
            >>"/tmp/tabm_sweep_${fam}_seed${seed}_${btag}_${mode}.log" 2>&1
        rc=$?
        if [[ $rc -eq 0 && -f "$summary" ]]; then
          echo "$tag  OK"
          done_cnt=$((done_cnt+1))
        else
          echo "$tag  FAIL rc=$rc (see /tmp/tabm_sweep_${fam}_seed${seed}_${btag}_${mode}.log)"
          fail_cnt=$((fail_cnt+1))
        fi
      done
    done
  done
done

echo "[sweep] DONE $(date '+%F %T')  ok=$done_cnt skipped=$skip_cnt failed=$fail_cnt total=$total"
