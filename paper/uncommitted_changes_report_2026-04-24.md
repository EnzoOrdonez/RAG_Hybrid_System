# Uncommitted Changes Report — 2026-04-24

Read-only audit of the working tree. No `git add`, `git commit`, `git stash`,
`git restore`, or `git checkout` was executed. Enzo decides next steps.

Branch: `fase-3.5-nli-recompute-saved-answers`
Most recent commit: `c119c99 docs(phase-3.5): correction_log Fase 3.5 entry`

## Statistics

- **Modified**: 27 files
- **Deleted**: 13 files (all inside `experiments/results/exp{6,7,8,8b}/`, checkpoint files)
- **Untracked**: 14 paths (includes this session's new artifacts plus two pre-existing: `rerun_phase2.log`, `titulacion/`)

## Full `git status --short -unormal`

```
 M README.md
 M benchmark.log
 M data/corpus_stats.json
 M data/evaluation/test_queries.json
 M data/llm_cache/llama3.1_8b-instruct-q4_K_M_cache.json
 M data/llm_cache/mistral_7b-instruct_cache.json
 M data/llm_cache/qwen2.5_7b-instruct_cache.json
 M experiments/results/exp5/aggregated_metrics.json
 M experiments/results/exp5/checkpoint_llm_llama3.1.json
 M experiments/results/exp5/checkpoint_llm_mistral.json
 M experiments/results/exp5/checkpoint_llm_qwen2.5.json
 M experiments/results/exp5/results.json
 D experiments/results/exp6/checkpoint_ablation_+dense.json
 D experiments/results/exp6/checkpoint_ablation_+expansion.json
 D experiments/results/exp6/checkpoint_ablation_+normalization.json
 D experiments/results/exp6/checkpoint_ablation_+reranker.json
 D experiments/results/exp6/checkpoint_ablation_bm25_only.json
 D experiments/results/exp7/checkpoint_cross_cloud_no_norm.json
 D experiments/results/exp7/checkpoint_cross_cloud_with_norm.json
 D experiments/results/exp8/checkpoint_RAG_Hibrido_Propuesto.json
 D experiments/results/exp8/checkpoint_RAG_Lexico_(BM25).json
 D experiments/results/exp8/checkpoint_RAG_Semantico_(Dense).json
 D experiments/results/exp8b/checkpoint_RAG_Hibrido_Propuesto.json
 D experiments/results/exp8b/checkpoint_RAG_Lexico_(BM25).json
 D experiments/results/exp8b/checkpoint_RAG_Semantico_(Dense).json
 M paper/overleaf_ready/main.tex
 M scripts/build_index.py
 M scripts/compute_retrieval_metrics.py
 M src/embedding/index/hybrid_index.py
 M src/pipeline/pipeline_config.py
 M src/pipeline/rag_pipeline.py
 M src/preprocessing/text_cleaner.py
 M src/reranking/multidimensional_scorer.py
 M src/retrieval/bm25_retriever.py
 M src/retrieval/dense_retriever.py
 M src/retrieval/hybrid_retriever.py
 M src/retrieval/query_processor.py
 M src/ui/components/index_loader.py
 M src/ui/pages/chat_page.py
 M src/ui/pages/explorer_page.py
?? data/corpus_stats.json.bak
?? data/evaluation/annotation_advisor.csv
?? data/evaluation/annotation_classmate.csv
?? data/evaluation/annotation_enzo.csv
?? data/evaluation/annotation_pool_full.json
?? data/evaluation/gold_queries_50.json
?? data/evaluation/shared_queries_25.json
?? data/evaluation/stratification_report.json
?? data/evaluation/test_queries.json.bak_2026-04-24
?? paper/code_fixes_2026-04-24.md
?? paper/disputed_claims_2026-04-24.md
?? rerun_phase2.log
?? scripts/build_annotation_pool.py
?? titulacion/
```

## Deleted files

All 13 deletions are inside `experiments/results/exp{6,7,8,8b}/` and are
per-config JSON checkpoints (one per system/ablation-stage, 1100–7800 lines
each). These look like mid-run snapshots written by the benchmark runner and
rewritten on rerun. Removing them loses resumability for those experiments
but does not remove the final `results.json` or `aggregated_metrics.json` in
any of those directories.

```
D experiments/results/exp6/checkpoint_ablation_+dense.json           (7809 lines)
D experiments/results/exp6/checkpoint_ablation_+expansion.json       (7809 lines)
D experiments/results/exp6/checkpoint_ablation_+normalization.json   (7809 lines)
D experiments/results/exp6/checkpoint_ablation_+reranker.json        (7809 lines)
D experiments/results/exp6/checkpoint_ablation_bm25_only.json        (7809 lines)
D experiments/results/exp7/checkpoint_cross_cloud_no_norm.json       (1140 lines)
D experiments/results/exp7/checkpoint_cross_cloud_with_norm.json     (1140 lines)
D experiments/results/exp8/checkpoint_RAG_Hibrido_Propuesto.json     (7809 lines)
D experiments/results/exp8/checkpoint_RAG_Lexico_(BM25).json         (7809 lines)
D experiments/results/exp8/checkpoint_RAG_Semantico_(Dense).json     (7809 lines)
D experiments/results/exp8b/checkpoint_RAG_Hibrido_Propuesto.json    (7809 lines)
D experiments/results/exp8b/checkpoint_RAG_Lexico_(BM25).json        (7809 lines)
D experiments/results/exp8b/checkpoint_RAG_Semantico_(Dense).json    (7809 lines)
```

## Grouped analysis

### Group 1 — `experiments/results/exp5/` (M, 5 files, ~46k line churn)

- `aggregated_metrics.json` (+132 / −81 lines). Diff sample: every
  `hall_*`, `lat_*` key is rewritten with new numbers; comments like
  `// ... 3588 lines ...` indicate the full block is replaced. Consistent
  with a Phase-3.5 NLI recompute: the prior `hall_faithfulness_mean=0.344` is
  replaced by a new value.
- `checkpoint_llm_llama3.1.json`, `checkpoint_llm_mistral.json`,
  `checkpoint_llm_qwen2.5.json`, `results.json`: all wholesale rewrites
  (~7800 lines each; results.json is 22k). Shape is identical; values
  differ — again the NLI recompute signature.

Last commit `24888f5 data(phase-3.5): recompute NLI aggregates for
exp6/exp8/exp8b over saved answers` covered exp6/exp8/exp8b but **not**
exp5. These exp5 edits look like the same recompute applied to exp5 and
not yet committed.

**Recommendation:** Probable Phase 3.5 legítimo — commitear antes de seguir
(mirror the `data(phase-3.5):` commit message format that was used for
exp6/exp8/exp8b).

### Group 2 — `experiments/results/exp{6,7,8,8b}/` (D, 13 checkpoint files)

Deletions only, of per-config checkpoint JSONs. The corresponding
`results.json` / `aggregated_metrics.json` in those exp dirs are not
modified, so aggregate outputs survive. These deletions look like cleanup
after the recompute pass finished.

**Recommendation:** Probable cleanup post-Phase 3.5 — commitear (message
suggestion: `chore(phase-3.5): drop mid-run checkpoints post-recompute`).
If the checkpoints need to be regenerable, verify the recompute script
still emits them; otherwise accept the loss of resumability.

### Group 3 — `src/` (M, 11 files, ~900 lines of churn total)

Sampled diffs show a mix of stylistic refactors and one functional API
change that matters for this session:

- `src/retrieval/bm25_retriever.py` — adds a `last_updated` field on
  `RetrievalResult`, adds `_normalize_last_updated()`, and extends
  `.search()` signature from `use_expansion: bool = True` to
  `use_expansion: Optional[bool] = None, enable_query_expansion: bool = True,
  enable_terminology_normalization: bool = True`. **`scripts/build_annotation_pool.py`
  (new this session) depends on the new `enable_query_expansion` /
  `enable_terminology_normalization` kwargs.**
- `src/retrieval/dense_retriever.py`, `src/retrieval/hybrid_retriever.py`,
  `src/retrieval/query_processor.py` — similar signature + `last_updated`
  propagation changes.
- `src/pipeline/pipeline_config.py`, `src/pipeline/rag_pipeline.py` —
  cosmetic: removed inline comments (`"bm25", "dense", "hybrid"`,
  `"linear", "rrf"`), removed banner comments, no behavioral change.
- `src/embedding/index/hybrid_index.py` — cosmetic: removed unused `time`
  import, stripped inline comment on `chunk_map`, docstring reshuffle.
- `src/preprocessing/text_cleaner.py` — added three new regexes (Google
  Cloud "Send feedback" / "Stay organized" / "Save and categorize" boiler),
  stripped inline comments.
- `src/reranking/multidimensional_scorer.py` — cosmetic.
- `src/ui/pages/chat_page.py`, `src/ui/pages/explorer_page.py`,
  `src/ui/components/index_loader.py` — import reordering, removed unused
  imports, light refactor.
- `scripts/build_index.py` — changed chunk loading from `glob` to `rglob`
  and added `isinstance(data, list)` handling (now tolerates both a
  single-dict file and a list-of-dicts file). Functional fix.

**Recommendation:** Probable Phase 3.5 legítimo — commitear antes de seguir.
This group is a prerequisite for today's annotation-pool work (the new
`enable_query_expansion`/`enable_terminology_normalization` kwargs are
load-bearing). Splitting into two commits is sensible: (a) the functional
retrieval API change + scripts/build_index.py rglob fix, (b) the cosmetic
refactors.

### Group 4 — `data/llm_cache/` (M, 3 files, ~9k line churn)

Three LLM response caches (`llama3.1_8b-instruct-q4_K_M_cache.json`,
`mistral_7b-instruct_cache.json`, `qwen2.5_7b-instruct_cache.json`) were
rewritten. These are derived artifacts from running queries against Ollama
and refresh on every benchmark run.

**Recommendation:** Cambios experimentales no productivos — considerar
gitignorar `data/llm_cache/*.json` (generated data). If they have been
committed before out of necessity, commit the thrashed caches with a
`chore:` prefix and open a follow-up issue to gitignore.

### Group 5 — `benchmark.log` (M, +3171 lines) + `rerun_phase2.log` (untracked, 440 KB)

Log files. `benchmark.log` is tracked; `rerun_phase2.log` is not.

**Recommendation:** Cambios experimentales no productivos — stash or
discard. Better: add `*.log` to `.gitignore` and remove `benchmark.log`
from tracking in a dedicated commit (`chore: stop tracking benchmark logs`).

### Group 6 — `data/corpus_stats.json`, `data/evaluation/test_queries.json`,
`README.md`, `paper/overleaf_ready/main.tex`, `scripts/compute_retrieval_metrics.py`

**This is today's work.** Four modified + nine untracked (see Group 8)
come from the 2026-04-24 AM + PM sessions (audit fixes + CNCF augmentation).

**Recommendation:** Commit these together when the Phase 3.5 base lands.
Suggested message prefix: `fix(eval): break oracle circularity + stratified
annotation pool (Flag 17 / Flag 142)`.

### Group 7 — Untracked backups (this session's audit trail)

- `data/corpus_stats.json.bak` — pre-regeneration snapshot of the
  Azure-only 812-doc file.
- `data/evaluation/test_queries.json.bak_2026-04-24` — pre-CNCF-augmentation
  snapshot of the 200-query set.

**Recommendation:** Commit alongside Group 6 for audit trail, or add
`*.bak`, `*.bak_*` to `.gitignore` and keep them local only.

### Group 8 — Untracked session artifacts (new annotation pipeline)

- `data/evaluation/annotation_pool_full.json` (50 queries, avg 20.5 chunks)
- `data/evaluation/annotation_enzo.csv` (1027 rows — full pool)
- `data/evaluation/annotation_advisor.csv` (514 rows — 25 shared queries)
- `data/evaluation/annotation_classmate.csv` (514 rows — byte-identical to advisor)
- `data/evaluation/gold_queries_50.json`
- `data/evaluation/shared_queries_25.json`
- `data/evaluation/stratification_report.json`
- `scripts/build_annotation_pool.py`
- `paper/code_fixes_2026-04-24.md`
- `paper/disputed_claims_2026-04-24.md`
- `paper/uncommitted_changes_report_2026-04-24.md` (this file)

**Recommendation:** Commit with Group 6 (`fix(eval): ...`). The CSVs may
balloon a commit; consider gitignoring `data/evaluation/annotation_*.csv`
if they are produced by a script and not hand-edited.

### Group 9 — `titulacion/` (untracked dir with `emails_lewis_y_gyt.md`)

Personal / administrative file (advisor correspondence). Unrelated to the
project.

**Recommendation:** Requiere revisión manual del usuario. If unrelated to
the repo, add `titulacion/` to `.gitignore` (or a `.git/info/exclude` entry
for local-only exclusion).

## Suggested commit sequence

If all of the above check out under manual review, the cleanest sequence is:

1. `data(phase-3.5): recompute NLI aggregates for exp5 over saved answers`
   — Group 1 only.
2. `chore(phase-3.5): drop mid-run checkpoints post-recompute`
   — Group 2 only.
3. `feat(retrieval): add enable_query_expansion / enable_terminology_normalization kwargs + last_updated field`
   — the functional slice of Group 3 (retrievers, query_processor,
   hybrid_index, scripts/build_index.py rglob fix).
4. `refactor(pipeline,ui): remove inline banner comments, reorder imports`
   — the cosmetic slice of Group 3.
5. `chore: gitignore llm_cache + benchmark.log` (optional but recommended).
6. `fix(eval): break oracle circularity + stratified annotation pool (Flag 17 / Flag 142)`
   — Groups 6, 7, 8 (today's work).

Everything up to (3) unblocks today's annotation pipeline; (6) is what the
paper revalidation depends on.
