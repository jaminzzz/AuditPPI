# Unified protein feature extraction

This directory contains the new dataset-neutral feature pipeline. Existing
cache scripts remain available for provenance; new datasets should use this
pipeline so the ESM/SAE implementation has one source of truth.

## Formal per-protein features

The ESM-C run writes layers 60 and 80 in one cache:

```text
esmc_l{60,80}_dense_{mean,max}
esmc_l{60,80}_sae_{mean,max,binary}
```

The ESM-2 run writes its final layer (33 for ESM-2 650M):

```text
esm2_l33_dense_{mean,max}
esm2_l33_sae_{mean,max,binary}
```

Binary features are defined only for the non-negative SAE representation.

### RAPPPID C1/C2/C3 example

Run ESM-C in the `E1` environment:

```bash
/data/wmzhu/anaconda3/envs/E1/bin/python \
  scripts/features/extract_protein_features.py \
  --backbone esmc \
  --input data/raw/rapppid_c1/c1.train.csv \
  --input data/raw/rapppid_c1/c1.val.csv \
  --input data/raw/rapppid_c1/c1.test.csv \
  --input data/raw/rapppid_c2/c2.train.csv \
  --input data/raw/rapppid_c2/c2.val.csv \
  --input data/raw/rapppid_c2/c2.test.csv \
  --input data/raw/rapppid_c3/c3.train.csv \
  --input data/raw/rapppid_c3/c3.val.csv \
  --input data/raw/rapppid_c3/c3.test.csv \
  --sequence-cols query,text \
  --layers 60,80 \
  --output data/sae/seq_caches/rapppid_esmc_l60_l80_features.pt
```

Run ESM-2 + InterPLM SAE in the E1 environment:

```bash
/data/wmzhu/anaconda3/envs/E1/bin/python \
  scripts/features/extract_protein_features.py \
  --backbone esm2 \
  --input data/raw/rapppid_c3/c3.train.csv \
  --input data/raw/rapppid_c3/c3.val.csv \
  --input data/raw/rapppid_c3/c3.test.csv \
  --sequence-cols query,text \
  --output data/sae/seq_caches/rapppid_esm2_l33_features.pt
```

FASTA inputs require no column arguments. For ID/sequence tables use, for
example, `--sequence-cols sequence --id-cols uniprot_id`.

## Pair features

The primary order-invariant representation is `sym`:

```text
[A * B, abs(A - B)]
```

`product` and `absdiff` are the two ablations. `concat` is ordered:

* training protocol writes both `[A,B]` and `[B,A]` with duplicated labels;
* evaluation protocol writes `X_ab` and `X_ba` separately so the downstream
  model averages the two predictions for each original pair.

```bash
/data/wmzhu/anaconda3/envs/E1/bin/python \
  scripts/features/build_pair_features.py \
  --cache data/sae/seq_caches/rapppid_esmc_l60_l80_features.pt \
  --feature esmc_l60_sae_max \
  --pairs data/raw/rapppid_c3/c3.train.csv \
  --mode sym \
  --output data/sae/reps/unified/c3_train_esmc_l60_sae_max_sym.pt
```

For concat validation/test caches use `--mode concat --concat-protocol eval`.
