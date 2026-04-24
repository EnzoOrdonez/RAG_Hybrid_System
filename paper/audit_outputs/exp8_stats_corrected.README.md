# A6 — `d_z` mis-labeling in `exp8_stats_corrected.csv`

**Status**: documented, NOT patched. See *Replacement* section below.

---

## What this file is

`exp8_stats_corrected.csv` was produced by audit §16 ("Module 16 recompute") and claims to apply Benjamini-Hochberg FDR and Holm-Bonferroni corrections across 12 pair-wise tests on `exp8/retrieval_metrics.json`, reporting Cohen's `d_z` (paired), `d_av`, and `d_rm` per Lakens 2013.

It does apply the corrections correctly. Its p-value columns (`p_raw`, `p_bh`, `p_holm`) reproduce numerically in the Phase 2.5 recompute.

**Its Cohen's-d columns are mislabeled.**

## What went wrong

The column `d_z_reported` contains values the audit read literally from the pre-fix `retrieval_metrics.json:statistical_tests[pair].cohens_d`. That source field was written by `scripts/compute_retrieval_metrics.py:194-195` (pre-Phase 1) using this formula:

```python
# COMPUTE_RETRIEVAL_METRICS.PY PRE-PHASE 1 (bugged, audit Flag 21)
pooled_std = np.sqrt((np.std(vals1) ** 2 + np.std(vals2) ** 2) / 2)
cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std
```

That formula is Cohen's `d_s` for **independent samples**, using pooled SD. It is incorrect for paired data (same 200 queries evaluated by every system).

The audit §16 recompute consumed that value under the label `d_z_reported` and then back-inferred the paired correlation `r` from it. The inferred `r` values (~0.51 across pairs) are consequently contaminated by the mislabeling: they are the `r` that would be consistent *with the bugged formula*, not the real paired correlation.

## Arithmetic demonstration — BM25 vs Hibrido, NDCG@5

Per-query aggregates from `exp8/retrieval_metrics.json:systems`:

```
mean_a = 0.5545     std_a = 0.3203   (BM25,    ddof=0)
mean_b = 0.7362     std_b = 0.2569   (Hibrido, ddof=0)
```

### Pre-Phase-1 formula (d_pooled, mislabeled as d_z in m16 CSV)

```
pooled_sd = sqrt((0.3203^2 + 0.2569^2) / 2) = 0.2903
d_pooled  = (mean_a - mean_b) / pooled_sd
          = (0.5545 - 0.7362) / 0.2903
          = -0.6259          # matches m16 CSV d_z_reported
```

### Correct paired d_z (Phase 2.5 recompute)

Per-query paired diff `diff[i] = ndcg@5_hibrido[i] - ndcg@5_bm25[i]`, direct compute on 200 per-query values:

```
mean_diff = 0.1817
std_diff  = 0.3618            (ddof=1, direct)
d_z       = mean_diff / std_diff
          = 0.1817 / 0.3618
          = 0.5023            # Phase 2.5 CSV value
```

Sign is `b - a` convention in the Phase 2.5 output; magnitude differs from m16 by 0.12 (0.63 vs 0.50), the gap driven by the actual paired correlation:

```
r(bm25, hibrido)_observed = 0.2332     # direct corrcoef on per-query NDCG@5
r_inferred_by_m16         = 0.5122     # back-solved from mislabeled d_pooled
```

## Interpretation

- `d_pooled = 0.626` is **not an error-free answer to the wrong question** — it is the correct value of a formula that does not apply to paired data.
- `d_z = 0.502` is the right answer under the paired-samples convention (Lakens 2013). Still "medium" by Cohen's thresholds, but only 0.002 above the |d| ≥ 0.5 cutoff.
- The m16 CSV's `d_av` column (independent of `d_z` label) was computed from the same mislabeled source and inherits the same bias in magnitude.

## Replacement

The authoritative d_z / d_av / d_rm / BH / Holm table for `exp8` is now:

> **[`paper/audit_outputs/exp8_stats_phase2_5.csv`](exp8_stats_phase2_5.csv)**

Generator: `scripts/audit/_write_phase2_5_csv.py`
Source JSON: `experiments/results/exp8/retrieval_metrics.json` (regenerated 2026-04-23 by `scripts/recompute_retrieval_stats.py` against the deterministic retrieval output in `results.json` and the same cross-encoder oracle `ms-marco-MiniLM-L-12-v2`).

Validate numerically:

```bash
python scripts/audit/smoke_recompute_retrieval_stats.py
SMOKE_SKIP_RECOMPUTE=1 python scripts/audit/smoke_recompute_retrieval_stats.py  # quick re-verify
```

## Rule (ratified 2026-04-23)

- `paper/audit_findings.md` — **IMMUTABLE**. Archaeological record of the zero-trust audit.
- `paper/audit_outputs/exp8_stats_corrected.csv` — **IMMUTABLE**. Frozen m16 evidence. Do not edit; do not delete.
- This sidecar (`exp8_stats_corrected.README.md`) — pointer from the frozen file to the authoritative replacement.
- `paper/audit_outputs/exp8_stats_phase2_5.csv` — **authoritative**. Cite this file, not the m16 CSV, for any `d_z` / BH claim about exp8 from here on.

Downstream consumers (paper figures, tables, abstract) should read `exp8_stats_phase2_5.csv`. The old CSV remains in the repo only as historical evidence of how the mislabeling propagated.
