# Audit report — exp9_llm_only_no_rag

Tracks the 8 user-stated decisions through plan → implementation → audit response. M1-M5 are the modifications I introduced during plan review. The D4 row is updated below per Block 1 of the external audit.

## Decision-by-decision

| # | Decision | Status | Verification |
|---|----------|--------|--------------|
| **D1** | Keep cross-encoder MS MARCO MiniLM-L-12-v2 as relevance oracle (Flag 17 limitation declared, no re-run of exp3..exp8b). | KEPT | Oracle never invoked for LLM-only because retrieval metrics are not computed (no `relevant_chunk_ids` to score against). Limitation re-disclosed in `paper/audit_findings.md` Flag 17. |
| **D2** | No retrieval metrics for LLM-only (`retrieval_metrics={}` per query). | KEPT | Existing guard at `benchmark_runner.py:269` (`if query.relevant_chunk_ids:`) auto-skips. Verified in saved `results.json`: 200/200 entries have `retrieval_metrics: {}` and `retrieved_ids: []`. |
| **D3** | Faithfulness=0.0 / hallucination_rate=1.0 on no-evidence claims; tagged `method="no_evidence"`. | KEPT (with M2 nuance) | `HallucinationDetector._no_evidence_report` segments claims; if claims>0 returns faith=0.0, rate=1.0; if claims=[] returns faith=1.0, rate=0.0 vacuously. **Audit Block 1 confirmed this introduces extractor-failure ambiguity (1/200 = q196)**: faithfulness=1.0 is vacuous in that case but indistinguishable from honest decline in the schema. Corrected mean reported in `_handoff/metrics_summary.json` excludes q196. |
| **D4** | New `honest_decline_rate` metric, reusing existing `_is_honest_decline` patterns. | **KEPT with KNOWN BUG (NOT FIXED IN THIS PR — flagged here)** | See "D4 bug analysis" below. |
| **D5** | Cache reuse with `config_name="LLM Only (No RAG)"` for separation. | KEPT | Verified `llm_manager.py:165-178` already includes `config_name` in `_cache_key`. New entries written under separate keys; cache hits for hybrid/dense/bm25 from exp8 unaffected. |
| **D6** | `retrieval_method="none"` as new enum value. | MODIFIED (M5) | Added `"none"` branch in `_build_retriever` (returns `None`, no retriever, no index load). Added `_query_no_rag` method that bypasses retrieval/reranking/empty-candidates early-return. Preserved fail-fast `ValueError` for actual typos. |
| **D7** | Llama 3.1 8B Q4 as default model. | KEPT | Verified in saved results: 200/200 calls to `llama3.1:8b-instruct-q4_K_M`. |
| **D8** | No manual ground truth annotation. | KEPT | `relevant_chunk_ids` remains empty 200/200; generation metrics auto-skipped via `if query.answer` guard. Limitation already documented in `paper/audit_findings.md` Flag 70. |

## D4 bug analysis — `DECLINE_PATTERNS` case-sensitivity

**Bug**: `src/generation/response_formatter.py:139-142` defines:
```python
def _is_honest_decline(self, text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in DECLINE_PATTERNS)
```
Of the 8 existing patterns (lines 38-47), **4 begin with capital `I`**:
- `r"I cannot find enough information"`
- `r"I don't have.*information"`
- (and other capital-I patterns in mixed-case form)

Because the matcher lowercases `text` before searching, but the patterns themselves carry capital `I`, those 4 patterns are **unreachable** — they can never match lowercased text. This bug exists in `main` and was introduced before this PR.

**Impact on exp9 result**: `hall_honest_decline_rate_mean = 0.0` in the saved aggregated file is an artefact of the bug, not a measurement. Re-applying a lowercased-and-extended pattern set over the saved 200 responses (per audit Block 1) gives:

- **4 honest declines**: q081, q097, q106, q117 (patterns matched: `i'm not familiar with` ×2, `i don't have.*information` ×2)
- **`honest_decline_rate_corrected = 0.02`** (4 / 200)

**Why NOT fixed in this PR**:
1. The bug existed before this PR and affects all 4 systems' decline rates symmetrically. Fixing it in this PR while reporting exp9 numbers would also retroactively change BM25/Dense/Hybrid decline rates the next time anyone re-aggregates exp8 — which is out of scope for this task per user's "do not touch existing experiments" constraint.
2. The correction is reported in `_handoff/reclassification.json` and `_handoff/metrics_summary.json`, so the paper can cite the corrected number while leaving the production code on a separate fix.
3. **Recommended follow-up** (separate ticket): lowercase all 8 existing patterns + add the 6 new ones used in the audit, then re-aggregate exp8 to give all 4 systems comparable decline rates. ETA ~5 min code change + ~2 min re-aggregation (no LLM calls).

## Reclassification (Block 1 output, repeated for the record)

| Category | Count | IDs |
|---|---|---|
| FABRICATION (claims>0, no decline match) | 195 | q001-q080, q082-q096, q098-q105, q107-q116, q118-q195, q197-q200 |
| HONEST_DECLINE (decline pattern match) | 4 | q081, q097, q106, q117 |
| EXTRACTOR_FAILURE (claims=0, no decline match) | 1 | q196 |

- `honest_decline_rate_corrected = 4 / 200 = 0.020`
- `faithfulness_mean_after_excluding_extractor_failures = 0.000`

## Acceptance criteria revisited

| Criterion (from user spec) | Expected | Actual (corrected) | Verdict |
|---|---|---|---|
| `aggregated_metrics.json` has the listed fields | per spec | all present + new fields added per Block 1 | ✓ |
| `hall_faithfulness_mean = 0.0` | 0.0 | corrected = 0.000 (raw = 0.005) | ✓ after correction |
| `hall_honest_decline_rate_mean` ∈ [0.1, 0.8] | [0.1, 0.8] | corrected = 0.02 (raw = 0.0) | ✗ on range — real finding: Llama 3.1 8B almost never declines on these queries |
| `lat_generation_ms_mean` ∈ [20s, 60s] | [20s, 60s] | 37.97s | ✓ |
| `errors = 0` | 0 | 0 | ✓ |

The honest-decline criterion missing the expected [0.1, 0.8] range is the substantive finding of this experiment: the model fabricates rather than declining. This is a stronger argument for RAG than landing inside the predicted range would have been.

---

## Block 3 — empirical byte-equivalence verification (exp8 Hybrid → current code)

Re-executed 3 queries from `exp8/results.json` (config "RAG Hibrido Propuesto") against the post-PR code with same seed=42, same `LLMManager` cache file, same `hybrid_index`. Full per-field diff in `_handoff/byte_equiv_check.json`. Determinism check in `_handoff/nli_determinism_check.json`.

| Field | q010 | q050 | q100 |
|---|---|---|---|
| answer text (cache hit) | ✓ equal (704 chars) | ✓ equal (826 chars) | ✓ equal (2589 chars) |
| retrieved_ids list | ✓ equal (5 ids) | ✓ equal (5 ids) | ✓ equal (5 ids) |
| total_claims | ✓ equal (2) | ✓ equal (2) | ✓ equal (12) |
| method | ✓ equal (nli) | ✓ equal (nli) | ✓ equal (nli) |
| supported_claims | ✓ equal (0) | ✓ equal (0) | ✗ new=0, saved=4 |
| contradicted_claims | ✓ equal (1) | ✓ equal (0) | ✗ new=1, saved=4 |
| unsupported_claims | ✓ equal (1) | ✓ equal (2) | ✗ new=11, saved=4 |
| faithfulness | ✓ equal (0.0) | ✓ equal (0.0) | ✗ new=0.0, saved=0.3333 |

**q010 + q050: fully byte-equivalent.** All 8 fields match.

**q100: 4 of 8 fields equal; NLI classification breakdown differs.**

### Root-cause analysis of the q100 NLI gap

The retrieval pipeline, LLM cache, and claim extraction are confirmed byte-equivalent (answer, retrieved_ids, total_claims, method all match). The gap is isolated to the NLI per-claim classification.

Three observations narrow the cause:

1. **Intra-process determinism confirmed**: re-ran q100 three times through the post-PR code (`_handoff/nli_determinism_check.json`). All 3 runs produced identical (supported=0, contradicted=1, unsupported=11, faith=0.0). The current code is deterministic.

2. **The PR does not modify the NLI path for non-empty chunks**: the `_no_evidence_report` addition is gated on `not retrieved_chunks`, which is false for hybrid queries. `_nli_matching` (lines 248-370) and `_extract_claims` (lines 209-242) are byte-identical to pre-PR.

3. **Phase 2 NLI fixes were applied to the code AFTER exp8 was originally run**: `git log --oneline -- src/generation/hallucination_detector.py` shows commits `c4d6a55 fix(flag-135): apply softmax to NLI cross-encoder predictions` and `bf71a94 fix(flag-138): rewrite NLI aggregation to use max-per-class, no early exit` were applied during Phase 2. exp8/results.json was written with the **pre-fix** NLI code; current code runs the **post-fix** version. The Phase 2.5 retrieval-metrics rerun (`scripts/recompute_retrieval_stats.py`, commits `16bcee7` / `56efcf7`) updated the aggregated retrieval statistics but did NOT re-run per-query hallucination detection; that data in exp8/results.json remains pre-fix.

**Conclusion**: the q100 NLI divergence is **NOT a regression introduced by this LLM-only PR**. It is the manifestation of the pre-existing Phase 2 Flag 135 + Flag 138 fix against saved exp8 per-query data that was not regenerated. The current code is internally consistent; the exp8 saved data is stale on the NLI breakdown.

**Recommended follow-up (separate ticket, not in this PR)**: run a maintenance script that recomputes per-query hallucination for exp1-exp8b using the post-Phase-2 NLI code, then re-aggregates. This would also surface the exp1-8b updated `hall_faithfulness_mean` numbers (which Phase 2 already partially addressed in the Phase 2.5 docs but not at the per-query level).

### Verdict

- **Retrieval byte-equivalent**: 3/3 ✓
- **LLM generation byte-equivalent (cache hits)**: 3/3 ✓
- **Claim extraction byte-equivalent**: 3/3 ✓
- **NLI per-claim classification**: 2/3 ✓ (q010, q050); 1/3 ✗ (q100) — caused by pre-existing Phase 2 NLI code change, NOT by this PR.
- **Aggregated faithfulness/hallucination_rate per system**: would shift on exp1-exp8 if those experiments were re-aggregated under current code — the shift is Phase 2 territory, separate from this PR's scope.

