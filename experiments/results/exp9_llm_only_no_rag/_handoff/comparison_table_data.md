# 4-system comparison table — exp8 (BM25/Dense/Hybrid) + exp9 (LLM-only)

All values from saved aggregated files:
- `experiments/results/exp8/retrieval_metrics.json` (Phase 2.5 — BH/Holm/d_z corrected)
- `experiments/results/exp8/aggregated_metrics.json` (hallucination + latency)
- `experiments/results/exp9_llm_only_no_rag/aggregated_metrics.json` + `_handoff/reclassification.json` (corrected)

| Sistema | P@5 | R@5 | MRR | NDCG@5† | Faithfulness | Hallucination Rate | Honest Decline Rate | Generation Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| LLM-only (no RAG, control) | N/A‡ | N/A‡ | N/A‡ | N/A‡ | 0.000§ | 1.000§ | 0.020¶ | 37,591 |
| BM25 (lexical) | 0.718 | 0.368 | 0.828 | 0.554 | 0.374 | 0.626 | not measured\| | 26,695 |
| Dense (semantic) | 0.782 | 0.424 | 0.894 | 0.661 | 0.398 | 0.602 | not measured\| | 31,841 |
| **Hybrid (ours)** | **0.851** | **0.472** | **0.942** | **0.736** | 0.369 | 0.631 | not measured\| | 83,651 |

### Footnotes

† **NDCG@5 instead of NDCG@10**: the saved `exp8/retrieval_metrics.json` only carries NDCG@5. Adding NDCG@10 to the existing exp8 results requires re-scoring the cross-encoder oracle over the full top-10 lists. Marked here as NDCG@5 to be transparent; if NDCG@10 is required for the paper, request and we re-score (no LLM calls; ~30 min compute).

‡ **N/A for retrieval metrics on LLM-only**: there is no retrieval stage in the LLM-only system, so Precision/Recall/MRR/NDCG cannot be computed against `relevant_chunk_ids`. (And `relevant_chunk_ids` is empty 200/200 anyway, per Flag 70 — see Phase 2 audit.)

§ **Faithfulness=0.000 / Hallucination Rate=1.000 for LLM-only is the CORRECTED value** from `_handoff/reclassification.json`, excluding the one extractor-failure row (q196). The originally-saved `hall_faithfulness_mean=0.005` and `hall_hallucination_rate_mean=0.995` included q196's vacuous faithfulness=1.0 (see q196 explanation in `qualitative_examples.md`). The corrected value reflects only queries that actually produced verifiable claims (195 fabrications) plus the 4 honest declines whose post-disclaimer claims also scored 0.0.

¶ **Honest Decline Rate=0.020 for LLM-only is the CORRECTED value** (4/200), measured by re-applying the extended decline pattern set listed in `audit_report.md` over the saved 200 responses (NOT re-running the LLM, NOT mutating production code). The originally-aggregated `hall_honest_decline_rate_mean=0.0` reflects the latent case-bug in `response_formatter.DECLINE_PATTERNS` (4 of 8 patterns start with capital `I` and are unreachable because the matcher lowercases input before regex). See `audit_report.md` for full bug analysis.

\| **Not measured for BM25/Dense/Hybrid**: the `is_honest_decline` field was not present in exp8's saved per-query results (added by this PR's `_run_single_query` change). Computing it for BM25/Dense/Hybrid retroactively from the saved exp8 `answer` field is feasible (same regex scan over saved text) but not done in this PR — would also be invalidated by the case-bug in the existing 8 patterns. Recommended follow-up: re-aggregate exp8 with the extended pattern set so all 4 systems have comparable `honest_decline_rate`.

### Headline reading

- **Hybrid leads on every retrieval metric** (P@5, R@5, MRR, NDCG@5) and on no-retrieval metrics (Faithfulness, Hallucination Rate) it sits within ±3 pp of the two baselines — consistent with the exp8 finding that retrieval-stage gains do not translate proportionally into generation-stage gains (Flag 142 / paper §5).
- **LLM-only collapses on faithfulness**: 0.000 vs 0.369–0.398 across the three RAG systems. This is the absolute value RAG adds, measured under a consistent operational definition (a claim is faithful iff there is retrieved evidence that NLI-entails it; absent evidence, the claim is unsupported by definition).
- **Generation latency: LLM-only sits between BM25 and Hybrid** (37.6 s p50 vs BM25 26.7 s, Hybrid 83.7 s). Without retrieved context to ground a focused answer, Llama 3.1 8B produces longer, more discursive answers — driving generation time up over the no-context baseline expectation.
- **Honest decline disparity remains unmeasured for RAG systems** — flagged as the highest-priority follow-up for completeness of the comparison.
