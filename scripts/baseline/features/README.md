# Baseline-specific feature extraction

Each baseline has an independent entry point because the underlying feature
semantics differ.

| Script | Input granularity | Output contract |
|---|---|---|
| `extract_deepnano_features.py` | one protein | final-layer mean/min/max |
| `extract_flashppi_features.py` | one protein | gLM2 mean + normalized query/key vectors |
| `extract_pplm_features.py` | one protein pair | mean/max inter/intra attention + contextual endpoints |
| `extract_mint_features.py` | one protein pair | jointly contextualized mean-pooled endpoints |

## Data sourcing

The two granularities pull their inputs differently:

- **Pair scripts (PPLM, MINT)** take `--benchmark NAME`, a
  `src.data.pairs.load_benchmark` key. One enumerator serves all six pair
  families, including bernett (MINT `Intra{1,0,2}_seqs.csv`, no protein ids) and
  PRING (`.txt` edge lists + per-species `protein_id.csv`), which have no
  `query,text,label` CSV. Row order and labels are exactly `load_benchmark`
  order.
- **Per-protein scripts (DeepNano, FlashPPI)** take `--from-protein-cache PATH`,
  reusing the unique-sequence manifest of an existing
  `auditppi_protein_features_v1` cache (`data/sae/protein_caches/*.pt`). This
  keeps their rows aligned to the pair-side protein caches by construction; the
  multi-GB feature tensors are memory-mapped and never materialized.

Both modes **skip** if the output already exists (print `[skip]` and exit 0), so
the loops below are resumable. Pass `--overwrite` to force a rebuild.

## Output layout

```text
data/sae/baseline_features/
  pplm/{benchmark_stem}.pt         # e.g. c1_train.pt, cross_species_ecoli.pt, pring_human_test_BFS.pt
  mint/{benchmark_stem}.pt
  flashppi/{cache_stem}.pt         # e.g. c1.pt, cross_species.pt, pring_human.pt
  deepnano/{cache_stem}_{backbone}.pt   # e.g. c1_esm2.pt
```

Pair outputs use `benchmark_stem` (colons → underscores). Per-protein outputs
are inherently per-family (one file holds all splits' unique proteins), so they
are keyed by the source cache stem rather than `{family}_{split}`.

PPLM and MINT outputs use `auditppi_baseline_pair_features_v1`; DeepNano and
FlashPPI use the shared per-protein `auditppi_protein_features_v1` schema.

---

## PPLM (pair)

PPLM is pair-conditioned. It jointly encodes a pair and saves the exact
components consumed by the official PPLM-PPI head for both mean and max paths:

```text
inter-chain attention       660 = 33 layers x 20 heads
intra-chain attention A     660
intra-chain attention B     660
contextual embedding A     1280
contextual embedding B     1280
```

> **Cost.** PPLM runs a per-pair attention forward with `need_head_weights`, so
> it is the slowest baseline. `cross_species:human_train` alone is ~422k pairs
> (multiple hours); each PRING human split runs once per sampling method.

```bash
PY=/data/wmzhu/anaconda3/envs/E1/bin/python

for BENCH in \
  c1:train c1:val c1:test \
  c2:train c2:val c2:test \
  c3:train c3:val c3:test \
  cross_species:human_train cross_species:human_test \
  cross_species:ecoli cross_species:fly cross_species:mouse \
  cross_species:worm cross_species:yeast \
  bernett:train bernett:val bernett:test \
  pring:human:train:BFS pring:human:val:BFS pring:human:test:BFS \
  pring:human:train:DFS pring:human:val:DFS pring:human:test:DFS \
  pring:human:train:RANDOM_WALK pring:human:val:RANDOM_WALK pring:human:test:RANDOM_WALK \
  pring:yeast:test pring:ecoli:test pring:arath:test ; do
  "$PY" scripts/baseline/features/extract_pplm_features.py --benchmark "$BENCH"
done
```

## MINT (pair)

MINT concatenates the two tokenized chains, assigns chain IDs, runs the
multimer model, and mean-pools the final layer over each chain separately. This
matches the upstream GeneralPPI wrapper. Same `--benchmark` list as PPLM:

```bash
PY=/data/wmzhu/anaconda3/envs/E1/bin/python

for BENCH in \
  c1:train c1:val c1:test \
  c2:train c2:val c2:test \
  c3:train c3:val c3:test \
  cross_species:human_train cross_species:human_test \
  cross_species:ecoli cross_species:fly cross_species:mouse \
  cross_species:worm cross_species:yeast \
  bernett:train bernett:val bernett:test \
  pring:human:train:BFS pring:human:val:BFS pring:human:test:BFS \
  pring:human:train:DFS pring:human:val:DFS pring:human:test:DFS \
  pring:human:train:RANDOM_WALK pring:human:val:RANDOM_WALK pring:human:test:RANDOM_WALK \
  pring:yeast:test pring:ecoli:test pring:arath:test ; do
  "$PY" scripts/baseline/features/extract_mint_features.py --benchmark "$BENCH"
done
```

## DeepNano (per-protein)

Only the ESM-2-650M layer-33 final layer is produced (mean/min/max). ESM-C is
intentionally omitted for this baseline. Runs over the six pair families' v1
protein caches:

```bash
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
CACHES=data/sae/protein_caches

for CACHE in \
  c1 c2 c3 cross_species bernett \
  pring_human pring_yeast pring_ecoli pring_arath ; do
  "$PY" scripts/baseline/features/extract_deepnano_features.py \
    --backbone esm2 \
    --from-protein-cache "$CACHES/${CACHE}_protein_features_max1022.pt"
done
```

## FlashPPI (per-protein)

FlashPPI produces independent per-protein gLM2 retrieval features:

- `flashppi_glm2_mean`: 1,280-D masked-mean backbone feature;
- `flashppi_query` / `flashppi_key`: 1,024-D normalized retrieval vectors.

For an undirected pair, average the two directional scores:
`0.5 * (q(A)·k(B) + q(B)·k(A))`.

```bash
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
CACHES=data/sae/protein_caches

for CACHE in \
  c1 c2 c3 cross_species bernett \
  pring_human pring_yeast pring_ecoli pring_arath ; do
  "$PY" scripts/baseline/features/extract_flashppi_features.py \
    --from-protein-cache "$CACHES/${CACHE}_protein_features_max1022.pt"
done
```

> `pic_human_protein_features_max1022.pt` is also available as a
> `--from-protein-cache` target, but it is the PIC essentiality set (not a PPI
> pair family), so it is left out of the loops above.

## Legacy `--input` mode

Both per-protein scripts still accept `--input FILE [--input FILE ...]`
(FASTA or CSV/TSV via `--sequence-cols`/`--id-cols`) with an explicit
`--output`, for ad-hoc extraction outside the cached families. `--input` and
`--from-protein-cache` are mutually exclusive.
