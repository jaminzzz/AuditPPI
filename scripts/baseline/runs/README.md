# Baseline downstream runs

Four language-model PPI baselines (DeepNano / FlashPPI / MINT / PPLM), each
scored through the **exact AuditPPI pair protocol** so their numbers are directly
comparable to the gated audit model:

* family → eval expansion, native-train / official-val selection, and
  train-once-multi-eval grouping are a verbatim copy of
  `scripts/audit_pair/run_ppi_fingerprint_baseline.py` /
  `src/ppi_fingerprint/baseline.py` (shared in `_protocol.py`);
* the trainable baselines use the same `mlp_pair` head + `--pair-mode` vocabulary
  as the audit model, so the only axis versus the audit model is the
  representation;
* AUROC / AUPRC come from the same `safe_auroc` / `safe_auprc`.

Results land under `results/main/baselines/{baseline}/{family}/seed_{S}/`
(`cells/` per-eval + `summaries/` per-run, `dump_experiment` spine).

Feature caches must already exist under `data/sae/baseline_features/` (built by
`scripts/baseline/features/`). Per-protein baselines (DeepNano/FlashPPI) have one
cache per family (PRING per-species); per-pair baselines (MINT/PPLM) have one
cache per benchmark split. Evals whose cache is missing are skipped per-eval with
a clear message, so partially-extracted families still produce what they can.

```bash
PY=/data/wmzhu/anaconda3/envs/E1/bin/python
FAMILIES="c1 c2 c3 cross_species bernett pring"

# DeepNano -- frozen ESM-2, mean/min/max pools concatenated -> mlp_pair head
for f in $FAMILIES; do
    $PY scripts/baseline/runs/train_deepnano_baseline.py --family $f --device-id 4
done

# FlashPPI -- zero-training retrieval CLIP score (no head, no --pair-mode)
for f in $FAMILIES; do
    $PY scripts/baseline/runs/eval_flashppi_baseline.py --family $f
done

# MINT -- joint-embedding endpoints -> mlp_pair head (per-pair caches)
for f in $FAMILIES; do
    $PY scripts/baseline/runs/train_mint_baseline.py --family $f --device-id 4
done

# PPLM -- endpoint-only embeddings (mean pool) -> mlp_pair head (per-pair caches)
for f in $FAMILIES; do
    $PY scripts/baseline/runs/train_pplm_baseline.py --family $f --device-id 4
done
```

Common flags: `--pair-mode {sym,concat,rich,product,absdiff,sum}` (trainable
baselines; default `sym`), `--seed`, `--device-id`. DeepNano adds `--pools`
(subset of mean/min/max); PPLM adds `--pool {mean,max}`; FlashPPI adds
`--normalize` (cosine CLIP).
