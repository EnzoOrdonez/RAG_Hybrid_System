# CloudRAG Tier 0 correction log

Ejecución por fases del plan Tier 0 (ver `paper/audit_findings.md` §21.5). Zero-trust: cada flag verificado contra el código antes del fix, cada fix con smoke pre-commit, discrepancias documentadas en `paper/audit_findings_cc_addenda.md`.

---

## Fase 1 — Fixes sin re-run

- **Branch**: `fix/phase-1-no-rerun`
- **Base**: `main` @ `42b95ba` (codex/plan-a-thesis-safe apuntaba al mismo commit)
- **Commits**:
  - `e1b6386` chore: track paper/ baseline and scripts/audit/smoke_test_nli.py
  - `b62d683` fix(flag-76): raise on missing keys in results_exporter aggregation
  - `5418299` fix(flag-98,165): replace 300-query claim with 200 in main.tex
  - `444fe62` fix(flag-160,124): rename 'normalization' to 'query expansion' in main.tex prose
  - `b05d7a5` feat(flag-103,108): add BH-FDR correction to pairwise statistical comparisons
  - `efd0dc7` fix(flag-159): remove unsupported 'Docker artifacts' claim from abstract

### Flags procesados

| Flag | Fuente | Archivo | Status |
|------|--------|---------|--------|
| 76 (≡91) | audit §21.5 T0.7 (cuerpo M13 lo etiqueta 91) | `src/evaluation/results_exporter.py:540,655` | FIJADO: raise KeyError diagnóstico |
| 98 | audit §14.1 / §12.1 | `paper/overleaf_ready/main.tex` L64,97,164,250 | FIJADO: 300 → 200 |
| 165 | audit §20.14 | `paper/overleaf_ready/main.tex` (abstract/contribs alineado con README L40) | FIJADO |
| 103 (≡152 en otro contexto) | audit §21.5 T0.8 | `src/evaluation/statistical_analysis.py` | FIJADO: `apply_multiple_comparison_correction` + `apply_corrections_to_results` |
| 108 | audit §15.2 / §21.5 T0.8 | `src/evaluation/statistical_analysis.py`; `run_all_comparisons` emite p_bh/p_holm etiquetados | PARCIAL: ver Addenda A4 (script paralelo `compute_retrieval_metrics.py` sin patch) |
| 124 | audit §18.1 | `paper/overleaf_ready/main.tex` L68-69, L98, L145-146 | FIJADO (prosa) |
| 160 | audit §20.9 | `paper/overleaf_ready/main.tex` L68-69, L98, L145-146, L199, L204 | FIJADO (prosa) |
| 159 | audit §21.5 T0.10 (condición (d) usuario) | `paper/overleaf_ready/main.tex` L70-71 | FIJADO: abstract; Addenda A5 para L101/L173 |

### Equivalencia de numeración

**Flag 91 (cuerpo audit §13.2, `.get(col, 0)` línea 655 de `results_exporter.py`) == Flag 76 (tabla §21.5 Tier 0 T0.7 referencia el mismo archivo/línea)**. El plan/mission usa la etiqueta §21.5. Los fixes se aplican al mismo call-site regardless de la etiqueta.

### Síntomas confirmados (zero-trust)

| Síntoma esperado | Comando verificación | Resultado |
|------------------|----------------------|-----------|
| `results_exporter.py:540` con `r.get("retrieval_metrics", {}).get(metric_key, 0)` | `grep -n "r\.get.*retrieval_metrics" src/evaluation/results_exporter.py` | ✓ línea 540 confirmada pre-fix |
| `results_exporter.py:655` con `row[col] = data[config].get(col, 0)` | `grep -n "row\[col\] = data\[config\]\.get" ...` | ✓ línea 655 confirmada pre-fix |
| main.tex 4 menciones "300" | `grep -n "300" paper/overleaf_ready/main.tex` | ✓ L64, 97, 164, 250 pre-fix |
| README L40 dice "200 queries" | `grep -n "200 queries" README.md` | ✓ L40 confirmado |
| `experiment_configs.py` sin `max_queries=300` | `grep "max_queries" experiments/experiment_configs.py` | ✓ 8×200 + 1×30 (exp7); nada con 300 |
| `statistical_analysis.py` sin BH/Holm/FDR | `Grep "bonferroni\|holm\|benjamini\|fdr\|multipletests"` | ✓ 0 matches pre-fix |
| `terminology_normalizer.py` docstring dice "Does NOT replace text" | `Read:1-20` | ✓ L2-3 confirmado |
| `query_processor.py` L131-132 expansión BM25-only | `Read:115-135` | ✓ semantic_query = query (raw) confirmado |
| `hallucination_detector.py:275` sin `apply_softmax` | `Read:265-310` | ✓ (no se modifica en Fase 1 — Fase 2) |
| `paper/audit_outputs/exp8_stats_corrected.csv` 13 filas | `Read:1-20` | ✓ 12 comparaciones + header |
| `scripts_audit/m16_recompute_stats.py` | `Glob **/m16_recompute_stats.py` | ❌ NO EXISTE (ver Discrepancias) |

### Contadores "300" en main.tex

- **ANTES** (commit `e1b6386` tip, pre-1.2):
  ```
  $ grep -c "300" paper/overleaf_ready/main.tex
  4
  $ grep -n "300" paper/overleaf_ready/main.tex
  64:Inference based faithfulness filter. On a 300-query expert-curated test
  97:  \item A reproducible multi-cloud QA benchmark covering AWS, Azure, and GCP (300 expert queries with normalized gold evidence).
  164:\textcolor{blue}{[PLACEHOLDER: 300 expert-curated queries across AWS/Azure/GCP. Query-type distribution (config / troubleshooting / conceptual / cross-cloud mapping). Gold evidence: document + span. Annotation protocol.]}
  250:\textcolor{blue}{[PLACEHOLDER: Summary: CloudRAG hybrid pipeline + normalization + NLI filter improves P@1 by 14.5 pp over BM25 and 7.0 pp over dense-only on a 300-query multi-cloud benchmark; full replication package released.]}
  ```
- **DESPUÉS** (commit `5418299` tip, post-1.2):
  ```
  $ grep -c "300" paper/overleaf_ready/main.tex
  0
  $ grep -n "300" paper/overleaf_ready/main.tex
  (no output)
  ```
- **Diff**: 4 - 0 = 4 reemplazos aplicados. No hay residuos "300, 500, 700" en main.tex (la tabla de chunk-size ablation vive en README, no en main.tex).

### Pre-check de label/ref para "norm" (Fase 1.3)

Antes del rename normalization → query expansion:
```
$ grep -n '\label\|\ref' paper/overleaf_ready/main.tex | grep -i 'norm'
134:\textcolor{blue}{[PLACEHOLDER: Insert Fig.~\ref{fig:pipeline} (fig\_end_to_end.png). Describe 5 stages: (1) ingestion + normalization, (2) hybrid retrieval BM25+dense, (3) RRF fusion, (4) cross-encoder reranking, (5) generation + NLI faithfulness filter.]}
```
Único match: `\ref{fig:pipeline}` cuyo **identificador no contiene "norm"** (el grep matchea por la palabra "normalization" que aparece en la prosa del placeholder, no en el label). **No hay labels ni refs que requieran rename**. El match de L134 queda como prose para reescritura en Fase 5 (ver Addenda A2).

### Discrepancias vs audit/mission

1. **`scripts_audit/m16_recompute_stats.py` NO EXISTE en el repo.** Audit §21.5 T0.8 y §Cierre afirman que existe. Glob `**/m16_recompute_stats.py` → 0 matches. La carpeta `scripts_audit/` tampoco existe (sí existe `scripts/audit/`).
   - **Decisión**: re-implementar BH/Holm directamente en `src/evaluation/statistical_analysis.py` y validar numéricamente contra `paper/audit_outputs/exp8_stats_corrected.csv` (que sí existe). Test pasó rtol=1e-9 para BH y Holm, 12/12 comparaciones.
2. **`main.tex` L199 no tenía la cadena completa "cross-cloud terminology normalization"** — solo "Cross-Cloud Normalization Impact". El rename se aplicó a "Cross-Provider Query Expansion Impact" consistente con L145 y L204.
3. **Números de línea menores desfasados** (vs mission spec): README L42 en spec → real L40; main.tex L69 prose atraviesa L68-69. Impacto nulo.
4. **Inconsistencia de numeración interna al audit**: cuerpo §13.2 Flag 91 == tabla §21.5 T0.7 "Flag 76" (mismo síntoma, misma línea, etiquetas distintas). Se usó la etiqueta §21.5.

### Addenda (síntomas nuevos no documentados, NO corregidos sin aprobación)

Archivo: `paper/audit_findings_cc_addenda.md`.

- **A1** — 9 instancias adicionales de `.get(..., 0)` silent zero en `results_exporter.py` (líneas 241, 287, 313, 314, 353, 410, 425, 463, 495) no citadas por el audit pero con misma patología que Flag 76.
- **A2** — 3 menciones residuales de "normalization" en placeholders de main.tex (L93, L122, L134) que belong a la prosa de Fase 5.
- **A3** — L63 del abstract también decía "cross-cloud terminology normalization module"; se renombró junto con L68-69 por pertenecer a la misma oración (out of explicit plan scope pero necesario para coherencia del abstract).
- **A4** — `scripts/compute_retrieval_metrics.py:182-215` tiene un pipeline estadístico paralelo sin BH/Holm y con Cohen's d incorrecto (pooled SD independiente vs datos pareados, Flag 21). Flag 108 queda parcialmente cerrado hasta que se deprecate ese path.
- **A5** — 2 menciones adicionales de "Docker" en main.tex (L101 contribución, L173 placeholder); plan Paso 1.5 limitado al abstract por instrucción explícita.

### Tests post-fix

| Test | Script | Resultado |
|------|--------|-----------|
| `ResultsExporter._export_experiment_latex` raises `KeyError` con dict sin metric | `scripts/audit/smoke_fix_flag76.py` | ✅ PASS (mensaje incluye metric name + config + experiment id) |
| `ResultsExporter.fig_end_to_end` raises `KeyError` con results sin `retrieval_metrics` | idem | ✅ PASS |
| BH-FDR `apply_multiple_comparison_correction` reproduce `p_bh` del CSV m16 | `scripts/audit/smoke_bh_fdr.py` | ✅ PASS rtol=1e-9, 12/12 comparaciones |
| Holm `apply_multiple_comparison_correction` reproduce `p_holm` del CSV m16 | idem | ✅ PASS rtol=1e-9, 12/12 |
| `from src.evaluation.statistical_analysis import *` sin errores de import tras añadir `Optional` fields | `python -c "..."` | ✅ PASS |
| main.tex `grep -c "300"` después de Paso 1.2 | bash | ✅ 4 → 0 |
| main.tex `grep -i 'normaliz'` después de Paso 1.3 | bash | 3 residuos solo en placeholders de Fase 5 (documentado A2) |
| main.tex abstract sin "Docker artifacts" después de Paso 1.5 | bash | ✅ (L101/L173 residuales documentados A5) |

### Guard de alcance — sin violaciones

Archivos que el plan marca "NO tocar en Fase 1" (reservados a Fase 2):
- `src/generation/hallucination_detector.py` — intacto
- `src/generation/llm_manager.py` — intacto
- `src/evaluation/benchmark_runner.py` — intacto
- `src/utils/reproducibility.py` — no creado

Verificado con `git diff main..HEAD --stat`: ninguno de esos archivos aparece.

### Dependencias agregadas

- `statsmodels>=0.14.0` añadido a `requirements.txt` (era missing; audit §20.5 Flag 156). Instalado en el env del usuario para correr el smoke de BH.

### Comandos pendientes para el usuario

1. **Revisión por auditor externo** del branch `fix/phase-1-no-rerun` antes de merge a `main`.
2. **Decisiones a tomar** sobre las 5 addenda (A1-A5) antes de Fase 2:
   - ¿Extender fix de `results_exporter.py` a las 9 líneas adicionales?
   - ¿Cerrar Flag 108 también sobre `compute_retrieval_metrics.py`?
   - ¿Renombrar L101 y L173 en Fase 5 o antes?
3. **No hacer merge a `main` sin revisión externa.** Plan no autoriza merge automatizado.
4. **Para Fase 2**: el usuario ejecutará manualmente el re-run de experimentos en su GPU (24-48h post-fix) cuando Fase 2 esté lista.

---
