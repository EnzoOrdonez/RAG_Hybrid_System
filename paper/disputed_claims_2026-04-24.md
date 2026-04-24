# Disputed Claims Inventory — 2026-04-24

Snapshot of every quantitative claim in public-facing artifacts that is now
flagged as DISPUTED or PENDING REVALIDATION, pending the 2026-05 human
annotation cycle and the Phase-4 NLI recalibration.

Two flag classes are used:

- **DISPUTED** — the underlying measurement is known broken (e.g., n≈29 and a
  mis-calibrated NLI detector). Number and claim should not be cited.
- **PENDING REVALIDATION** — the measurement may hold directionally but was
  computed under a methodological flaw (e.g., circular oracle). Keep for now,
  recompute on the gold set.

## Inventory

### README.md

| Line (pre-edit) | Original text | Flag | Replacement / annotation |
|---|---|---|---|
| 40–48 (Performance table) | `P@1 / Recall@5 / MRR / NDCG@5 / Faithfulness` row for BM25, Dense, Hybrid plus "Hybrid outperforms both baselines with statistical significance (p < 0.0001, Cohen's d = 0.626)." | PENDING REVALIDATION (Flag 17) for P@1/Recall@5/MRR/NDCG@5; DISPUTED (Flag 142) for Faithfulness column | Added a `[PENDING REVALIDATION — Flag 17 ...]` block immediately below the table and appended `[PENDING REVALIDATION ...]` to the significance sentence. Numbers left intact to document the prior state. |
| 164 (Experiments table, exp7 row) | `+16.8% faithfulness with normalization` | DISPUTED (Flag 142) | Replaced with `[DISPUTED — Flag 142 audit: n=29–30, NLI model broken (DeBERTa-v3-small, threshold 0.7, outputs 0.19–0.25). Pending recomputation with gold truth May 2026]`. |
| 165 (Experiments table, exp8b / exp8 row) | `Hybrid > Dense > BM25 consistently` | PENDING REVALIDATION (Flag 17) | Appended `[PENDING REVALIDATION — Flag 17: circular retrieval oracle; see Performance section]`. |

### paper/overleaf_ready/main.tex

| Line (pre-edit) | Original text | Flag | Replacement / annotation |
|---|---|---|---|
| 65–68 (abstract, results clause) | `hybrid pipeline achieves P@1 = 0.930, MRR = 0.942, ... Cohen's d = 0.626).` | PENDING REVALIDATION (Flag 17) | Inline `\textcolor{red}{\textbf{[PENDING REVALIDATION — Flag 17 audit: retrieval oracle was identical to the pipeline reranker ...]}}` appended after the significance clause. |
| 68–70 (abstract, +16.8% claim) | `Dictionary-based cross-provider query expansion contributes a 16.8 percent gain in answer faithfulness over an unexpanded baseline.` | DISPUTED (Flag 142) | `16.8 percent` replaced with a red-bold `\textbf{[DISPUTED — Flag 142 audit: n=29–30, broken NLI detector ...]}` block; sentence still ends with `gain ... over an unexpanded baseline.` to preserve grammar. |
| 200 (ablation subsection caveat) | `\textcolor{blue}{[PLACEHOLDER: ... +16.8\% faithfulness gain (subject to re-evaluation in Phase 4 ...)]}` | DISPUTED (Flag 142) | Strengthened to red-bold `[DISPUTED — Flag 142 audit ...]` with explicit "figure and numerical gain are withheld until recomputation." Placeholder for figure insertion kept but moved behind the DISPUTED marker. |

### paper/audit_findings.md and paper/correction_log.md

No edits. Both already describe the +16.8 % claim as broken (Flag 135/142)
and the circular oracle as invalid (Flag 17). Preserving them as-is keeps
the audit trail immutable.

## Summary

- 4 READM E entries touched.
- 3 main.tex entries touched.
- 0 edits to audit/correction logs (historical record preserved).
- All numerical values retained on disk — nothing was deleted, so the audit
  can reconstruct the pre-revalidation state.
