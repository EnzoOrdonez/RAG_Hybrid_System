# Code Fixes — 2026-04-24 (pre-annotation cycle)

Branch: `fase-3.5-nli-recompute-saved-answers` (no new branch cut; all edits
land on the current working tree).

All changes here are infrastructure for the 2026-05 human annotation cycle
and the Flag-17 / Flag-142 revalidation. No experiments were rerun.

## Summary

Two classes of bug block a scientifically valid re-evaluation:

1. **Flag 17 — circular retrieval oracle.** `scripts/compute_retrieval_metrics.py`
   used `cross-encoder/ms-marco-MiniLM-L-12-v2` as a relevance oracle, which is
   also the reranker in the PROPOSED_HYBRID pipeline. Any metric that filtered
   candidates through the oracle was biased toward the Hybrid system by
   construction. Fix: swap oracle to `BAAI/bge-reranker-large` (independent
   architecture: XLM-RoBERTa vs. BERT-mini), and flag all derived metrics as
   PENDING REVALIDATION.
2. **Flag 142 — broken faithfulness.** The widely-cited `+16.8%` faithfulness
   gain from exp7 rests on n=29–30 queries and a NLI detector that outputs
   0.19–0.25 where a DeBERTa-v3-small threshold of 0.7 was applied. Numerically
   nonsensical. Fix: tag as DISPUTED; hold recomputation until the calibrated
   NLI detector is wired up after the annotation cycle.

An unrelated data bug — `data/corpus_stats.json` reported only Azure (812 docs) —
was fixed while we were in the area, because several parts of the paper cite
corpus size.

## File-by-file changes

### New files

- `data/corpus_stats.json` (regenerated from `data/processed/` +
  `data/chunks/adaptive/`; old file preserved as `data/corpus_stats.json.bak`).
- `data/evaluation/gold_queries_50.json` (50-query stratified sample, seed=42).
- `data/evaluation/stratification_report.json` (per-stratum request /
  available / selected / redistribution decisions).
- `data/evaluation/annotation_pool_full.json` (retrieval pool with full
  metadata for audit).
- `data/evaluation/annotation_enzo.csv`,
  `data/evaluation/annotation_advisor.csv`,
  `data/evaluation/annotation_classmate.csv` (three byte-identical blind
  annotator CSVs, 1055 rows each across 50 queries, no `source_systems` column).
- `scripts/build_annotation_pool.py` (pool builder; BM25 + Dense + Hybrid
  top-10, no reranker, no query expansion, no terminology normalization;
  dedup + deterministic per-query shuffle).
- `paper/disputed_claims_2026-04-24.md` (inventory of every tagged claim).
- `paper/code_fixes_2026-04-24.md` (this file).

### Edited files

- `scripts/compute_retrieval_metrics.py` — line 44–45: swap oracle
  `cross-encoder/ms-marco-MiniLM-L-12-v2` → `BAAI/bge-reranker-large` with
  Flag-17 rationale in a top-of-constant comment. Added a TODO on
  `RELEVANCE_THRESHOLD = 0.0`: BGE-reranker outputs sigmoid-scaled [0,1]
  probabilities (smoke-test: 0.993 for a relevant pair, 9.2e-5 for an
  irrelevant one), while ms-marco gave logits in ±10. Threshold must be
  re-calibrated post-annotation.
- `README.md` — lines 40–48: Performance table annotated with
  `[PENDING REVALIDATION — Flag 17 ...]` block; Hybrid-outperforms sentence
  tagged. Lines 164–165: exp7 `+16.8%` claim replaced with `[DISPUTED — Flag 142
  ...]`; exp8b/exp8 row appended with `[PENDING REVALIDATION]`.
- `paper/overleaf_ready/main.tex` — abstract lines 65–71: inline red-bold
  `\textbf{[PENDING REVALIDATION ...]}` after the p-value / Cohen's-d clause;
  `16.8 percent` replaced with `\textbf{[DISPUTED — Flag 142 ...]}`. Line 200
  caveat strengthened from blue placeholder to red `[DISPUTED]` with explicit
  "numerical gain is withheld."

### Explicitly NOT changed

- No modifications to the Hybrid pipeline or to the reranker used inside it
  (`src/pipeline/pipeline_config.py`, `experiments/experiment_configs.py`,
  `src/reranking/*`). The oracle is swapped; the pipeline's reranker is untouched.
- No modifications to `src/generation/hallucination_detector.py`.
- No touching of `data/raw/`, `data/processed/`, `data/chunks/`, or
  `data/indices/`. The annotation pool reads from these but does not write.
- No new experiments run. `scripts/compute_retrieval_metrics.py` was not
  executed with the new oracle; BGE-reranker-large was only smoke-tested
  (`CrossEncoder('BAAI/bge-reranker-large').predict(...)`) to confirm
  HuggingFace load.
- No synthesized ground truth. `data/evaluation/ground_truth.json` still has
  `judgments = []` and will only be filled by human annotators.

## Invalidated experiments

Retrieval metrics (Precision@K, Recall@K, MRR, NDCG@K) from these runs were
computed with the Flag-17 circular oracle and are PENDING REVALIDATION:

- `experiments/results/exp3/*` — retrieval-strategy grid. All numerical
  comparisons between BM25 / Dense / Hybrid rely on the circular oracle.
- `experiments/results/exp4/*` — reranker comparison. Same oracle
  contamination; the BGE-reranker-large row inside exp4 is the only variant
  that would not degrade under the fix, but its position in the ranking
  changes.
- `experiments/results/exp6/*` — ablation waterfall. Component deltas use
  oracle-scored NDCG@5, so every "X contributes Y points" number needs
  recomputation.
- `experiments/results/exp7/*` — cross-cloud normalization. Retrieval
  portion PENDING REVALIDATION; faithfulness portion DISPUTED (Flag 142).
- `experiments/results/exp8/*` and `experiments/results/exp8b/*` — end-to-end.
  Directional ranking `BM25 < Dense < Hybrid` is plausibly directionally
  correct but magnitudes and p-values are invalid.

Faithfulness metrics from exp5 and exp7 are DISPUTED (Flag 142).

## What requires recomputation post-annotation

Once the 50-query gold set (`data/evaluation/ground_truth.json`, populated
from the three annotator CSVs + adjudication) lands:

1. Rerun `scripts/compute_retrieval_metrics.py` with oracle =
   `BAAI/bge-reranker-large` and with `RELEVANCE_THRESHOLD` recalibrated
   against the gold labels (pick the threshold that maximizes F1 against the
   human judgments, or use the gold labels directly and drop the oracle).
2. Report Precision@{1,3,5}, Recall@{1,3,5}, MRR, NDCG@5 for BM25 / Dense /
   Hybrid over the 50-query set. Add Wilcoxon signed-rank tests with
   Benjamini–Hochberg + Holm corrections (infrastructure already present in
   `scripts/compute_retrieval_metrics.py`, lines 182–255).
3. Recompute faithfulness with the calibrated NLI detector (Phase 4 work —
   out of scope for this fix bundle).
4. Regenerate the README Performance table, the abstract numbers in
   `paper/overleaf_ready/main.tex`, and Figures 3 (ablation) and 4
   (cross-cloud improvement) from the new numbers.

## What is NOT invalidated

- Pipeline code correctness: `src/retrieval/*`, `src/reranking/*`,
  `src/pipeline/*` behavior was not the bug — the bug was in evaluation.
- Corpus composition: `data/corpus_stats.json` now reflects the actual
  multi-cloud composition (aws=996, azure=1461, cncf=92, gcp=240, kubernetes=1162
  for documents; size_500 chunks: aws=8616, azure=18290, cncf=167, gcp=7003,
  k8s=12242). Old `.bak` preserved if audit needs the broken figure.
- Latency measurements (`experiments/results/*/results.json` → `latency_ms`
  fields) are independent of the retrieval oracle and remain valid.
- Query set `data/evaluation/test_queries.json` (200 queries, 5 clouds) is
  intact; the 50-query gold subset is a stratified sample of it.
  (**Superseded in the PM cycle — see below.**)

---

## 2026-04-24 PM — CNCF augmentation + Minimal Mode

After the morning pass, the stratified sample produced only **n=2 for CNCF**
(vs. target 6). Root cause: `data/evaluation/test_queries.json` had exactly
one pure-CNCF query, so even counting multi-cloud tuples the CNCF pool was
4, and two of those overlapped with the kubernetes picks. Two mitigations
this afternoon:

### 1. Five CNCF queries added manually

Appended to `data/evaluation/test_queries.json` (previous file preserved as
`data/evaluation/test_queries.json.bak_2026-04-24`). New total: 205 queries
(was 200). Schema unchanged; `cloud_providers=["cncf"]`,
`relevant_chunk_ids=[]`, `answer=""` for all five. IDs `q201`–`q205`.

| ID | Question | category / type / difficulty |
|---|---|---|
| q201 | What are OpenTelemetry Collectors and how do they differ from OpenTelemetry Agents? | observability / comparative / medium |
| q202 | How does Envoy proxy handle HTTP/2 multiplexing compared to traditional load balancers? | networking / comparative / hard |
| q203 | What are the main components of a Fluentd log aggregation architecture? | logging / factual / medium |
| q204 | How do I configure Argo Workflows to run parallel task execution? | workflow / procedural / medium |
| q205 | What security features does Falco provide for runtime threat detection in containerized environments? | security / factual / medium |

Pure-CNCF queries post-augmentation: 6. CNCF pool (including multi-cloud
tuples): 9.

### 2. Re-executed stratified sampling (seed=42)

Stale artifacts from the morning removed
(`gold_queries_50.json`, `stratification_report.json`,
`annotation_pool_full.json`, `annotation_enzo.csv`,
`annotation_advisor.csv`, `annotation_classmate.csv`) and regenerated.

New distribution:

- **First pass** (scarcity-first): aws=13, azure=13, gcp=10, kubernetes=4
  (deficit 4), cncf=6 (no deficit).
- **Redistribution of kubernetes deficit** (4 slots) across aws:azure:gcp
  at 13:13:10 via largest-remainder: aws +2, azure +1, gcp +1.
- **Final**: aws=15, azure=14, gcp=11, kubernetes=4, cncf=6. **Total = 50.**
- 13 multi-cloud queries in the selection. 4 of the 5 manually added CNCF
  queries were drawn.

`stratification_report.json` now includes a `manually_added_queries` field
listing `[q201, q202, q203, q204, q205]` and a `manually_added_note`
pointing here.

### 3. Transition to Minimal Mode for external annotation

Original plan (AM): three annotators each label all 50 queries (~1055 rows
each, an estimated 17 hours per annotator). Too expensive for external
reviewers. Revised plan (PM):

- **Enzo** labels all 50 queries → `annotation_enzo.csv` (1027 rows).
- **Advisor** and **classmate** each label the same 25-query shared subset
  → `annotation_advisor.csv` and `annotation_classmate.csv` (514 rows each,
  byte-identical).

Shared-subset selection is a stratified proportional sample at 50% of each
stratum (seed=123, largest-remainder rounding). Targets: aws=8, azure=7,
gcp=5, kubernetes=2, cncf=3 → total 25.

Overlap subset is persisted in `data/evaluation/shared_queries_25.json` so
that inter-annotator agreement (Cohen's κ, Krippendorff's α) can be
computed over exactly the 25 shared queries once labels come back.

### 4. Verification

```
test_queries len     : 205   (200 + 5 new)
gold_queries_50.json : 50    (stratified sample)
shared_queries_25.json: 25   (subset of gold)
annotation_enzo.csv  : 1027 rows
annotation_advisor   : 514 rows
annotation_classmate : 514 rows
advisor == classmate : byte-identical (md5 match, diff = empty)
advisor ⊂ enzo       : True
```

### 5. What is still PENDING / explicitly not done in the PM cycle

- No commits executed (user explicitly requested read-only status audit
  before deciding which uncommitted pre-existing changes to keep). See
  `paper/uncommitted_changes_report_2026-04-24.md`.
- No experiments run. No oracle recomputation. BGE-reranker-large threshold
  calibration still a TODO in `scripts/compute_retrieval_metrics.py`.
- `src/`, `data/raw/`, `data/processed/`, `data/chunks/`, `data/indices/`
  untouched this afternoon (same scope guard as AM).

