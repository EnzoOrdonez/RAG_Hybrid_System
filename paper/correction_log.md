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

## Addenda resueltos en Fase 1 (revisión intermedia)

Tras la revisión inicial del branch, el usuario aprobó cerrar 3 de las 5 addenda antes del merge. Commits adicionales sobre el mismo branch `fix/phase-1-no-rerun`:

| # | Addendum | Commit | Flags cerrados |
|---|----------|--------|----------------|
| 8 | **A1** — 9 instancias adicionales de `.get(..., 0)` silent zero en `results_exporter.py` | `06ab376` fix(flag-76-extended): raise on missing keys across all exporter figures | Flag 76 completo (cero `.get(..., 0)` restantes), Flag 94 (radar colapsando por silent skip) |
| 9 | **A4** — pipeline estadístico paralelo en `scripts/compute_retrieval_metrics.py:182` sin BH ni d_av | `44af679` refactor(flag-108-full): delegate compute_retrieval_metrics stats to statistical_analysis | Flag 108 en artefacto público (`experiments/results/exp*/retrieval_metrics.json:statistical_tests`), Flag 21 (Cohen's d con pooled SD independiente vs datos pareados), Flag 22 (cutoff 10 arbitrario para Wilcoxon — delegación elimina el camino) |
| 10 | **A5** — Docker claim fuera del abstract: L101 (contribución) | `d202f5f` fix(flag-159-extended): remove Docker claim from contributions bullet | Flag 159 en L101. L173 diferido a Fase 5 (placeholder). |

### Addenda aprobados retroactivamente

- **A3** — el rename de L63 "cross-cloud terminology normalization module" en el mismo commit que L68-69 es aceptado como scope correcto para coherencia del abstract.

### Addenda diferidos a Fase 5

- **A2** — 3 menciones residuales de "normalization" en placeholders `[PLACEHOLDER: ...]` de main.tex (L93 Related Work, L122 Domain RAG, L134 pipeline overview). Esos placeholders se reescriben como prosa técnica en Fase 5; los cambios pertenecen a esa fase.

### Tests post-fix adicionales

| Test | Script | Resultado |
|------|--------|-----------|
| Las 7 figuras restantes de ResultsExporter rechazan métricas faltantes con KeyError diagnóstico | `scripts/audit/smoke_fix_flag76.py` (9 cases) | ✅ 9/9 PASS |
| `run_pairwise_tests_with_corrections` emite shape nested + p_bh/sig_bh/p_holm/sig_holm + labels por comparación | `scripts/audit/smoke_compute_retrieval_stats.py` | ✅ 6/6 PASS (shape 12 labeled; leaf keys completos; labels consistentes con nested path; family_size=12 global; BH monotónico; sig flags booleanos) |
| Grep final: cero `.get(..., 0)` en `src/evaluation/results_exporter.py` | bash | ✅ 0 matches |
| Grep final: una sola mención "Docker" en main.tex (L173 placeholder) | bash | ✅ |

### Guard de Fase 2 — sin violaciones (re-verificado)

- `src/generation/hallucination_detector.py` — intacto
- `src/generation/llm_manager.py` — intacto
- `src/evaluation/benchmark_runner.py` — intacto
- `src/utils/reproducibility.py` — no creado

### Estado final de addenda

| # | Estado | Fase de resolución |
|---|--------|---------------------|
| A1 | CERRADO | Fase 1 commit 06ab376 |
| A2 | DIFERIDO | Fase 5 (placeholders) |
| A3 | APROBADO retroactivamente | Fase 1 commit 444fe62 |
| A4 | CERRADO | Fase 1 commit 44af679 |
| A5 | PARCIAL (L101 cerrado, L173 diferido) | Fase 1 commit d202f5f + Fase 5 para L173 |

### Total de commits en `fix/phase-1-no-rerun`

10 (1 chore + 9 fixes/features/refactors + 1 docs). Aún pendiente de merge. Esperar revisión externa completa antes de mergear a `main`.

---

## Fase 2 — Fix NLI + seeds + re-run script (camino crítico)

- **Branch**: `fix/phase-2-nli-and-seeds`
- **Base**: `main` @ `270ee58` (tras merge de Fase 1 preservando commits individuales)
- **Commits**:
  - `c4d6a55` fix(flag-135): apply softmax to NLI cross-encoder predictions
  - `bf71a94` fix(flag-138): rewrite NLI aggregation to use max-per-class, no early exit
  - `b29c3a2` fix(flag-137,140): exclude method="none" from faithfulness aggregates, add n_effective
  - `793e7b0` feat(flag-152,153,155): global seed propagation + Ollama seed option
  - `b1aff17` chore: add rerun_post_fixes.sh script for Phase 3 preparation

### Flags procesados

| Flag | Fuente | Archivo | Status |
|------|--------|---------|--------|
| 135 | audit §19.1 | `src/generation/hallucination_detector.py:275` | FIJADO: `apply_softmax=True` |
| 138 | audit §19.4 | `src/generation/hallucination_detector.py:_nli_matching` | FIJADO: max-per-class Honovich 2022 rule; early-exit bug removido |
| 137 | audit §19.3 | `src/evaluation/benchmark_runner.py:_save_aggregated_metrics` | FIJADO: filter `method in {none,error}` + `hall_n_effective` field |
| 140 | audit §19.6 | idem | FIJADO (mismo fix que 137) |
| 136 | audit §19.2 (lateral) | idem | INCLUIDO: claim counts absolutos (total/supported/contradicted/unsupported_claims) emitidos en aggregated_metrics.json |
| 152 | audit §20.1 | `src/utils/reproducibility.py` + `benchmark_runner.py` | FIJADO: `set_all_seeds` aplicado en `__init__` y `run_experiment` |
| 153 | audit §20.2 | `scripts/rerun_post_fixes.sh` | PARCIAL: script exporta `PYTHONHASHSEED=42`; in-process hash() randomization solo se puede fijar al startup del intérprete (documentado en `ensure_hashseed_at_startup`) |
| 155 | audit §20.4 | `src/generation/llm_manager.py:_generate_ollama` | FIJADO: `"seed": self.seed` en options dict |
| 156 | audit §20.5 (lateral) | `requirements.txt` | NO tocado en Fase 2 (Tier 2/Fase 6) |

### Síntomas confirmados (zero-trust)

| Síntoma esperado | Comando verificación | Resultado |
|------------------|----------------------|-----------|
| `hallucination_detector.py:275` sin `apply_softmax` | Read | ✓ pre-fix: `predict(pairs, batch_size=32, show_progress_bar=False,)` |
| `hallucination_detector.py:291-299` tie-break con guard `!= "supported"` | Read | ✓ pre-fix: early exit confirmado |
| `benchmark_runner.py:569-579` aggregate faith/hall_rate sobre todos los valid, sin filter method | Read | ✓ pre-fix: flat mean sin filter |
| `benchmark_runner.py:120` sólo almacena `self.seed=seed`, sin `set_all_seeds` | Grep | ✓ pre-fix: 0 llamadas a random/numpy/torch seeding |
| `llm_manager.py:306-313` options dict sin `"seed"` | Read | ✓ pre-fix: solo temperature + num_predict |
| `sentence-transformers` versión soporta `apply_softmax` | Python inspect | ✓ 5.2.3, `predict` params incluye `apply_softmax` |

### Discrepancias vs plan

1. **Plan especificaba `sentence-transformers==5.4.1`** pero el env tiene `5.2.3`. `apply_softmax` parameter ya existe en 5.2.3 — el fix funciona. El pin exacto se resuelve en Fase 6.
2. **Plan §2.5 mencionaba `.sh` o `.ps1`**; escrito `.sh` (shell disponible en el env del usuario). Si se necesita `.ps1` para PowerShell nativo, hacer traducción en Fase 6.
3. **Plan paso 2.4 pedía verificar `bit-identidad del score` en smoke NLI**. Movido a smoke_test_nli.py (requiere red para bajar el modelo — correr manual). Smoke autónomo `smoke_nli_aggregation.py` cubre la lógica de agregación sin red.

### Addenda (síntomas nuevos no documentados, NO corregidos sin aprobación)

Archivo: `paper/audit_findings_cc_addenda.md`.

_Sin addenda nuevas en Fase 2 — las 5 anteriores (A1-A5) ya cerradas/diferidas._

### Tests post-fix

| Test | Script | Resultado |
|------|--------|-----------|
| `set_all_seeds(42)` reproducible across random/numpy/torch, rechaza seed<0, escribe `PYTHONHASHSEED` | `scripts/audit/smoke_seeds.py` | ✅ 3/3 PASS |
| `LLMManager(seed=N)` persiste `self.seed`; `_generate_ollama` pasa `seed` al dict de options | idem | ✅ 2/2 PASS |
| `BenchmarkRunner(seed=99)` llama `set_all_seeds(99)` al `__init__` | idem | ✅ PASS |
| `_nli_matching` agrega por max-per-class: 5 casos (mixed signal contradice, consistent support, ambos bajo threshold, ambos cross con contr>ent, ambos cross con ent>contr) | `scripts/audit/smoke_nli_aggregation.py` (mock NLI) | ✅ 5/5 PASS |
| `_save_aggregated_metrics` filtra method=none/error, emite `hall_n_effective`, reproduce tabla audit §19.3 a 1e-3 | `scripts/audit/smoke_hall_n_effective.py` (200 QueryResult sintéticos por sistema) | ✅ 6/6 PASS (BM25/Semantic/Hibrido match targets 0.331/0.352/0.325; filter delta +4.35 a +4.54 pts per system) |
| `smoke_test_nli.py` extendido con assertions post-softmax (entailment > 0.7 / contradiction > 0.7 / softmax sum ≈ 1.0) | `scripts/audit/smoke_test_nli.py` | ⏳ requiere red huggingface.co — correr manual |
| Script `rerun_post_fixes.sh` sintaxis válida + incluye pre-flight checks de marcadores de fix | `bash -n scripts/rerun_post_fixes.sh` | ✅ PASS |

### Guard de Fase 2 respetado (re-verificado contra scope explícito)

- `src/evaluation/results_exporter.py` — intacto (Fase 1 work; no tocar)
- `src/evaluation/statistical_analysis.py` — intacto (Fase 1 work)
- `scripts/compute_retrieval_metrics.py` — intacto
- `paper/overleaf_ready/main.tex` — intacto

### Dependencias

Sin nuevas dependencias en Fase 2. `statsmodels` ya agregado en Fase 1.

### Comandos pendientes para el usuario

1. **Revisión externa del branch `fix/phase-2-nli-and-seeds`** antes de merge a `main`.
2. **NO ejecutar `scripts/rerun_post_fixes.sh` desde Claude Code** — el usuario lo corre manualmente en su GPU (RTX 3060 Laptop, 24-48h wall time).
3. **Antes del re-run**: (a) merge de Fase 2 a `main`, (b) `export PYTHONHASHSEED=42` en la shell del re-run, (c) confirmar que `data/llm_cache/` y `experiments/results/exp{5,6,7,8,8b}/checkpoint_*.json` se pueden borrar sin perder trabajo único.
4. **Correr `python scripts/audit/smoke_test_nli.py` manualmente** con red a huggingface.co disponible — valida apply_softmax + label order + obvious-pair thresholds post-fix antes del re-run largo.
5. **Notificar a Claude Code** cuando los re-runs terminen para empezar Fase 3 (regenerar figuras y tablas desde los JSONs nuevos).

### Total de commits en `fix/phase-2-nli-and-seeds`

5 (4 fixes/features + 1 chore script). Pendiente de merge. NO mergear sin revisión externa.

---


