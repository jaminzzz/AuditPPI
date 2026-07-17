# Baseline-specific feature extraction

Each baseline has an independent entry point because the underlying feature
semantics differ. Existing upstream and historical scripts are retained.

| Script | Input granularity | Output contract |
|---|---|---|
| `extract_deepnano_features.py` | one protein | final-layer mean/min/max |
| `extract_flashppi_features.py` | one protein | gLM2 mean + normalized query/key vectors |
| `extract_pplm_features.py` | one protein pair | mean/max inter/intra attention + contextual endpoints |
| `extract_mint_features.py` | one protein pair | jointly contextualized mean-pooled endpoints |

## DeepNano

Only final-layer variants are produced:

- ESM-2-650M layer 33;
- ESM-C-6B layer 80.

The original `scripts/cache/cache_deepnano_embeddings.py` is unchanged and
remains available as historical provenance for the earlier layer-60 ESM-C run.

```bash
# ESM-C final layer; E1 environment
PYTHONPATH=. /data/wmzhu/anaconda3/envs/E1/bin/python \
  scripts/baseline/features/extract_deepnano_features.py \
  --backbone esmc \
  --input data/rapppid_c3/c3.train.csv \
  --input data/rapppid_c3/c3.val.csv \
  --input data/rapppid_c3/c3.test.csv \
  --sequence-cols query,text

# ESM-2 final layer; genmol environment
PYTHONPATH=. /data/wmzhu/anaconda3/envs/genmol/bin/python \
  scripts/baseline/features/extract_deepnano_features.py \
  --backbone esm2 \
  --input data/rapppid_c3/c3.train.csv \
  --input data/rapppid_c3/c3.val.csv \
  --input data/rapppid_c3/c3.test.csv \
  --sequence-cols query,text
```

The resulting per-protein cache can be passed to
`scripts/features/build_pair_features.py` for concat or symmetric downstream
features.

## PPLM

PPLM is pair-conditioned. It jointly encodes a pair and saves the exact
components consumed by the official PPLM-PPI head for both mean and max paths:

```text
inter-chain attention       660 = 33 layers x 20 heads
intra-chain attention A     660
intra-chain attention B     660
contextual embedding A     1280
contextual embedding B     1280
```

```bash
PYTHONPATH=. /data/wmzhu/anaconda3/envs/genmol/bin/python \
  scripts/baseline/features/extract_pplm_features.py \
  --pairs data/rapppid_c3/c3.test.csv \
  --output data/sae/baseline_features/pplm/c3_test.pt
```

## FlashPPI

FlashPPI produces independent per-protein retrieval features:

- `flashppi_glm2_mean`: 1,280-dimensional masked mean backbone feature;
- `flashppi_query`: 1,024-dimensional normalized query vector;
- `flashppi_key`: 1,024-dimensional normalized key vector.

For an undirected pair, average the two directional retrieval scores:

```text
0.5 * (q(A) dot k(B) + q(B) dot k(A))
```

```bash
PYTHONPATH=. /data/wmzhu/anaconda3/envs/E1/bin/python \
  scripts/baseline/features/extract_flashppi_features.py \
  --input data/rapppid_c3/c3.train.csv \
  --input data/rapppid_c3/c3.val.csv \
  --input data/rapppid_c3/c3.test.csv \
  --sequence-cols query,text \
  --output data/sae/baseline_features/flashppi/rapppid_proteins.pt
```

## MINT

MINT is also pair-conditioned. It concatenates the two tokenized chains,
assigns chain IDs, runs the multimer model, and mean-pools the final layer over
each chain separately. This matches the upstream GeneralPPI wrapper.

```bash
PYTHONPATH=. /data/wmzhu/anaconda3/envs/genmol/bin/python \
  scripts/baseline/features/extract_mint_features.py \
  --pairs data/rapppid_c3/c3.test.csv \
  --output data/sae/baseline_features/mint/c3_test.pt
```

PPLM and MINT outputs use `auditppi_baseline_pair_features_v1`; DeepNano and
FlashPPI use the shared per-protein `auditppi_protein_features_v1` schema.
