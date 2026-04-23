# CloudRAG — Overleaf Submission Bundle (LACCI 2026)

This folder is ready to drag-and-drop into a new Overleaf project based on the
**IEEE Conference Template**.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Master LaTeX file. Title, authors, abstract, all 7 sections with placeholders, AI-disclosure, acknowledgments. |
| `references.bib` | Triaged bibliography. ~35 entries. Foundational refs added; weak venues flagged with `note = {TODO: ...}`. |
| `figures/` | All 9 PNG figures from `output/figures/` + 9 LaTeX table snippets from `output/tables/`. |

## How to use in Overleaf

1. Create a new project on Overleaf using the **IEEE Conference Template** (NOT Bare Demo, NOT Computer Society).
2. **Delete** the template's example files (`conference_101719.tex`, `fig1.png`, `IEEEtran_HOWTO.pdf`). Keep `IEEEtran.cls`.
3. Upload `main.tex` and `references.bib` to the project root.
4. Upload the entire `figures/` folder (drag the folder, Overleaf preserves structure).
5. Set the main document to `main.tex` (Menu → Settings → Main document).
6. Click **Recompile**. First compile may take longer due to BibTeX.

## Critical TODOs before submission

These are marked inline with `% TODO:` or `\textcolor{blue}{[PLACEHOLDER:]}` in `main.tex`.

### Blocking (must resolve before any review)

1. **Confirm Lewis as co-author.** Email him explicitly. If he declines, remove the second `\IEEEauthorblock` and move him to Acknowledgments.
2. **Verify Lewis' real institutional email** — `lfuentes@ulima.edu.pe` is a placeholder.
3. **Confirm Submodalidad A.3 applies in Ing. Sistemas.** The PDF the user provided is from Administración. Get the equivalent for Facultad de Ingeniería from Oficina de Grados y Títulos.
4. **Verify the 5 references flagged** in `references.bib` with `note = {TODO: ...}`. These had metadata errors in the original list:
   - `yu2024ragsurvey` (was arXiv → actually Springer chapter)
   - `li2025attributefilter` (was Proc. ACM Manag. Data → actually arXiv)
   - `krishna2024factfetchreason` (was NAACL → confirm acceptance)
   - `peng2025latesplit` (DOI was malformed)
   - `aljohani2026hybrid` (MDPI venue → verify which journal)

### High-priority content writing (next 2 weeks)

Replace each `\textcolor{blue}{[PLACEHOLDER: ...]}` block with real prose. Order suggested:

- Methodology subsections (Section III) — fastest because content is in the codebase.
- Experimental Setup (Section IV) — straightforward documentation.
- Results (Section V) — paste numbers from `output/csv/` into existing tables; figures already linked.
- Related Work (Section II) — last because it requires polishing the bibliography.
- Introduction (Section I) — last; needs to land the narrative based on results.

### Tables

The `figures/` folder contains 9 `table_*.tex` files exported from the experiments. To use them in `main.tex`:

```latex
\input{figures/table_retrieval_metrics.tex}
```

Inspect each `.tex` first — they may need column-width tweaks for the IEEE 2-column layout.

## Page-budget warning

LACCI 2026 limit = **6 pages including references**. Current skeleton with 7 figures will likely exceed the budget. Plan to drop figures during writing:

- Mandatory: `fig_end_to_end`, `fig_retrieval_comparison`, `fig_ablation_waterfall`.
- Recommended: `fig_cross_cloud_improvement`, `fig_llm_comparison`.
- Cuttable: `fig_latency_breakdown`, `fig_reranker_impact`, `fig_retrieval_metrics*` (redundant if main retrieval table is included).

## Submission

1. Final PDF must be generated from Overleaf (`Submit → Download PDF`).
2. Upload to EasyChair (LACCI 2026 submission system).
3. Deadline: **26 May 2026**.
4. Notification: 26 July 2026.
5. Camera-ready: 24 August 2026.
6. Conference: 3–6 November 2026 (Lima, Peru).
