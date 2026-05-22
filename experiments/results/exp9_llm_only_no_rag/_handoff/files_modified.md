# Files modified — exp9_llm_only_no_rag PR

Classification: **ADDITIVE PURE** (only adds new code paths; existing call sites unchanged) vs **MODIFIER** (changes the behaviour of an existing code path that exp1-exp8 may have exercised). Each entry specifies why exp1-exp8 are or are not affected.

## ADDITIVE PURE (7 files / sites)

### 1. `src/pipeline/pipeline_config.py` — ADDITIVE PURE
- New `LLM_ONLY_NO_RAG` PipelineConfig instance.
- New `"llm_only"` key in `PIPELINE_CONFIGS` dict.
- Docstring updated.
- The 3 existing configs (BASELINE_LEXICAL, BASELINE_SEMANTIC, PROPOSED_HYBRID) are byte-identical.
- **Why exp1-exp8 unaffected**: those experiments import the 3 existing configs by name from this module; the new addition is invisible to them.

### 2. `src/generation/prompt_templates.py` — ADDITIVE PURE
- New `NO_RAG_SYSTEM_PROMPT` and `NO_RAG_PROMPT` constants.
- `SYSTEM_PROMPT`, `RAG_PROMPT`, `CROSS_CLOUD_PROMPT`, `PROCEDURAL_PROMPT` and `TEMPLATE_MAP` are byte-identical.
- `get_template()` and `build_context()` unchanged.
- **Why exp1-exp8 unaffected**: those experiments use the existing templates via `get_template(query_type)`; the new prompts are reached only when `retrieval_method == "none"`, a value that exp1-exp8 configs never set.

### 3. `src/pipeline/rag_pipeline.py` — ADDITIVE PURE
- `_build_retriever`: added a leading `if retrieval_method == "none": return None` branch. The bm25/dense/hybrid branches and the trailing `raise ValueError` are unchanged.
- `query()`: added a leading `if retrieval_method == "none": return self._query_no_rag(...)` branch. The original 1→7 stage flow (query_processing → retrieval → reranking → generation → hallucination → format → response) is unchanged.
- New `_query_no_rag` method (64 LOC) — only reachable from the new branch.
- New imports `NO_RAG_PROMPT`, `NO_RAG_SYSTEM_PROMPT`.
- **Why exp1-exp8 unaffected**: the new branches gate on `retrieval_method == "none"`, which is false for all exp1-exp8 configs (they use `"bm25"`, `"dense"`, or `"hybrid"`). Control flow is byte-identical for those configs.

### 4. `experiments/experiment_configs.py` — ADDITIVE PURE
- New `EXP9_LLM_ONLY_NO_RAG` ExperimentConfig instance.
- New `"exp9_llm_only_no_rag"` key in `EXPERIMENT_CONFIGS` dict.
- New import of `LLM_ONLY_NO_RAG`.
- Docstring updated.
- The 8 existing experiment definitions (EXP1..EXP8B) are byte-identical.
- **Why exp1-exp8 unaffected**: those experiments are referenced by their existing dict keys; the new entry does not collide.

### 5. `scripts/run_llm_only_benchmark.py` — NEW FILE
- Standalone runner. Imports `EXP9_LLM_ONLY_NO_RAG`, instantiates `BenchmarkRunner`, calls `run_experiment` with `max_queries` argument.
- **Why exp1-exp8 unaffected**: this script only runs exp9. exp1-exp8 are launched via their own scripts (`scripts/run_all_experiments.py`, `scripts/recompute_retrieval_stats.py`, etc.) which are unchanged.

### 6. `README.md` — ADDITIVE PURE
- 1 new row + 1 footnote inserted into the Performance comparison table.
- All other content byte-identical.
- **Why exp1-exp8 unaffected**: documentation file; does not affect benchmark execution.

### 7. `scripts/_handoff_reclassify.py` — NEW FILE (audit response, no production behaviour)
- Ad-hoc reclassification helper that reads `experiments/results/exp9_llm_only_no_rag/results.json` and writes `_handoff/reclassification.json`.
- **Why exp1-exp8 unaffected**: read-only script; not imported by production code.

## MODIFIER (3 files / sites)

These three sites have **changed behaviour for code paths that exp1-exp8 could in principle have exercised**. For each, I trace whether the changed path was reachable from those experiments' saved-results call graph.

### A. `src/generation/hallucination_detector.py` :: `check()` empty-chunks branch — MODIFIER
- **Before**: `if not retrieved_chunks: return self._empty_report(time, hallucination_rate=1.0)` — returns method="none", total_claims=0.
- **After**: `if not retrieved_chunks: return self._no_evidence_report(response, time)` — returns method="no_evidence", segments claims, faith=0.0 or 1.0 depending on whether any claim survives extraction.
- **Reachability from exp1-exp8**: `check()` is called from `compute_hallucination_metrics` (`hallucination_metrics.py:47`), which is in turn called from `BenchmarkRunner._run_single_query` line 289. The OLD line-287 guard `and response.retrieved_chunks` prevented the call when retrieved_chunks was empty, so `check()`'s empty-chunks branch was effectively dead code from the benchmark runner's perspective.
- **Was it ever reached for exp1-exp8?** Looking at `RAGPipeline.query()` lines 225-231: when retrieval returns 0 candidates, the pipeline early-returns with `answer="No relevant documentation found..."` and `error="No chunks retrieved"`. That response carries `retrieved_chunks=[]`. The old guard at `_run_single_query:287` then suppressed the hallucination call entirely → `hall_metrics={}` and the row was excluded from aggregation. So this branch in `check()` was historically unreachable from exp1-exp8.
- **Why exp1-exp8 are byte-equivalent**: even after this MODIFIER, exp1-exp8 still go through the guard at `_run_single_query:287` ... see entry B below. The combined effect is that the old behaviour is preserved for exp1-exp8 paths.

### B. `src/evaluation/benchmark_runner.py` :: `_run_single_query` hallucination guard — MODIFIER
- **Before**: `if compute_hallucination and response.answer and response.retrieved_chunks:` → call `compute_hallucination_metrics`.
- **After**: `if compute_hallucination and response.answer:` → call `compute_hallucination_metrics` with `retrieved_chunks=response.retrieved_chunks or []`.
- **Reachability from exp1-exp8**: yes, this guard is on every per-query path for those experiments. The change matters when `response.retrieved_chunks == []`.
- **When does that happen for exp1-exp8?** Only when `RAGPipeline.query()` early-returns at line 225-231 because retrieval produced 0 candidates. For all 3 retrieval methods over the existing 200 test queries, this is *empirically zero occurrences* (verified in exp8's saved results: every per-query row has non-empty `retrieved_ids` for all 3 systems). So even though the guard was relaxed, in practice the relaxation did not change any exp1-exp8 row.
- **What changed for the new system (LLM-only)**: this is precisely the path now used.
- **Was this verified empirically?** Yes — Block 3 of this audit re-executed 3 exp8 Hybrid queries against the post-change code and compared byte-for-byte; see `_handoff/audit_report.md` / chat report.

### C. `src/evaluation/benchmark_runner.py` :: `_save_aggregated_metrics` — MODIFIER (additive field)
- **Before**: no `hall_honest_decline_rate_mean` key.
- **After**: new key computed across all valid queries with non-empty `hall_metrics`, defaulting `is_honest_decline` to `False` if the field is missing.
- **Reachability from exp1-exp8**: this method is called when those experiments are *re-aggregated* (e.g. by running `BenchmarkRunner.run_experiment` again). The saved `exp8/aggregated_metrics.json` from before this PR is byte-identical because aggregation has not been re-run.
- **If exp8 is re-aggregated post-change**: `hall_honest_decline_rate_mean` would be added with value 0.0 across all 3 systems (because exp8 per-query results predate the `is_honest_decline` field, so the default-to-False kicks in for every row). This is a documentation difference, not a behaviour regression — all existing fields keep their values.

### D. `src/evaluation/benchmark_runner.py` :: `_run_single_query` is_honest_decline surfacing — MODIFIER (additive field)
- **Before**: `hall_metrics` did not contain `is_honest_decline`.
- **After**: `hall_metrics["is_honest_decline"]` is set from `response.confidence == "HONEST_DECLINE"`.
- **Reachability from exp1-exp8**: any new run of those experiments would add this field per query. Saved files are unchanged.
- **Why this is safe**: pure field addition. No existing field renamed or removed.

### E. `src/evaluation/hallucination_metrics.py` :: empty-chunks guard — MODIFIER
- **Before**: `if not response_text or not retrieved_chunks: return {method:"none", ...}`.
- **After**: `if not response_text: return {method:"none", ...}` — passes through to `detector.check()` when only `retrieved_chunks` is empty.
- **Reachability from exp1-exp8**: only via the guard at entry B above. Same analysis applies — empirically zero exp1-exp8 rows ever hit this path.

### F. `src/evaluation/benchmark_runner.py` :: `run_experiment` index-load skip — ADDITIVE
- **Before**: always called `_get_hybrid_index(...)` and skipped the config on None.
- **After**: skips `_get_hybrid_index` when `retrieval_method == "none"`, else falls through to original behaviour.
- **Reachability from exp1-exp8**: condition is false for those configs; original behaviour preserved.

## Summary

- 7 files are ADDITIVE PURE: exp1-exp8 byte-equivalent by construction.
- 3 files have MODIFIER sites with traced reachability: in every case, the modified path was either unreachable from exp1-exp8 (A, E), reachable but empirically not exercised (B), or pure additive fields (C, D, F).
- Block 3 of this audit's chat report contains the empirical 3-query byte-equivalence check against exp8's Hybrid system.
