#!/usr/bin/env bash
# Slice all remaining max2046 per-dataset protein caches from the pooled 2046 cache.
# Pure CPU, serial (avoids 16GB pooled cache x N in RAM). c1/c2/c3 already done.
set -euo pipefail
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
cd /data/wmzhu/PPI/AuditPPI
POOLED=data/sae/seq_caches/pooled_esmc_l60_l80_max2046_features.pt
OUT=data/sae/protein_caches

echo "=== bernett ==="
$PY scripts/prep/slice_dataset_protein_cache.py \
  --input baselines/mint/downstream/GeneralPPI/ppi/Intra1_seqs.csv \
  --input baselines/mint/downstream/GeneralPPI/ppi/Intra0_seqs.csv \
  --input baselines/mint/downstream/GeneralPPI/ppi/Intra2_seqs.csv \
  --sequence-cols seq1,seq2 \
  --esmc-pooled "$POOLED" \
  --output "$OUT/bernett_protein_features_max2046.pt" --overwrite 2>&1 | tail -2

echo "=== cross_species ==="
$PY scripts/prep/slice_dataset_protein_cache.py \
  --input data/raw/cross_species/human.ppi.qrels.seq.train.csv \
  --input data/raw/cross_species/human.ppi.qrels.seq.test.csv \
  --input data/raw/cross_species/ecoli.ppi.qrels.seq.test.csv \
  --input data/raw/cross_species/fly.ppi.qrels.seq.test.csv \
  --input data/raw/cross_species/mouse.ppi.qrels.seq.test.csv \
  --input data/raw/cross_species/worm.ppi.qrels.seq.test.csv \
  --input data/raw/cross_species/yeast.ppi.qrels.seq.test.csv \
  --sequence-cols query,text \
  --esmc-pooled "$POOLED" \
  --output "$OUT/cross_species_protein_features_max2046.pt" --overwrite 2>&1 | tail -2

echo "=== pring_human ==="
$PY scripts/prep/slice_dataset_protein_cache.py \
  --input baselines/PRING/data_process/pring_dataset/human/human_simple.fasta \
  --esmc-pooled "$POOLED" \
  --output "$OUT/pring_human_protein_features_max2046.pt" --overwrite 2>&1 | tail -2

echo "=== pring_yeast ==="
$PY scripts/prep/slice_dataset_protein_cache.py \
  --input baselines/PRING/data_process/pring_dataset/yeast/yeast_simple.fasta \
  --esmc-pooled "$POOLED" \
  --output "$OUT/pring_yeast_protein_features_max2046.pt" --overwrite 2>&1 | tail -2

echo "=== pring_ecoli ==="
$PY scripts/prep/slice_dataset_protein_cache.py \
  --input baselines/PRING/data_process/pring_dataset/ecoli/ecoli_simple.fasta \
  --esmc-pooled "$POOLED" \
  --output "$OUT/pring_ecoli_protein_features_max2046.pt" --overwrite 2>&1 | tail -2

echo "=== pring_arath ==="
$PY scripts/prep/slice_dataset_protein_cache.py \
  --input baselines/PRING/data_process/pring_dataset/arath/arath_simple.fasta \
  --esmc-pooled "$POOLED" \
  --output "$OUT/pring_arath_protein_features_max2046.pt" --overwrite 2>&1 | tail -2

echo "ALL_SLICES_DONE"
