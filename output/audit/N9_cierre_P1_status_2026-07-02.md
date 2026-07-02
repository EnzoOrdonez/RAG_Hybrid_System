# N9 — Estado ítem-por-ítem de los hallazgos P1 (gate de cierre de auditoría)

**Fecha:** 2026-07-02 · **Para:** Enzo (y revisión por IA externa)
**Contexto:** Fase 1 reportó 8 P1 principales + P1 secundarios en `N9_auditoria_integral_fase1_2026-07-01.md`.
Fase 2 ejecutó fixes en 4 commits (`e671f67`, `3d1e2a6`, `61ad8e8`, `5d4ff2c`, pusheados con tag
`nota3-N9-v4-2026-07-02`). Este documento da el estado REAL por ítem, con evidencia verificada
HOY contra disco/commits — no lo que los commits dicen que hicieron. Réplicas H5 corriendo aparte.

## Convenciones de estado

- **FIXED** — corregido en código/datos y verificado (comando o archivo citado).
- **FIXED-BY-DOC** — no se cambió código; el riesgo quedó documentado donde un usuario lo verá (decisión deliberada).
- **PARCIAL** — parte corregida, parte pendiente (se detalla qué).
- **DIFERIDO** — deliberadamente pospuesto con destino conocido (p. ej. reescritura A.3/paper).
- **NO-FIXED** — sin corregir y sin documentar; queda expuesto.

## Tabla principal (los 8 P1 del reporte de Fase 1)

| # | Hallazgo P1 | Estado | Evidencia |
|---|---|---|---|
| 1 | UI Experiment Runner podía sobrescribir evidencia firmada y correr generación LLM con un click | **FIXED** | `experiments_page.py:13-15` → `RESULTS_DIR = experiments/results_ui` (sandbox) + comentario de freeze; commit `61ad8e8`; py_compile OK |
| 2 | `smoke_nli_aggregation.py` FALLABA (esperaba v0; default runtime vb_agree desde N8); correction_log decía "5/5 PASS" | **FIXED** | `nli_variant="v0"` fijado + comentario de contrato Flag 138; ejecución verificada **exit 1→0**; fila N9 en `correction_log.md`; commit `61ad8e8` |
| 3 | Metadata errónea en artefactos v3 (`generated=2026-06-11` hardcodeado; v3_small etiquetado "ledger N5") | **PARCIAL** | Script FIXED: `compute_faithfulness_metrics.py` ahora `date.today()` + `_ledger_for()` (v3→N8, v4→N9); v4 en disco verifica `ledger N9 / 2026-07-02` ✓. **PERO** `faithfulness_metrics_v3_small.json` EN DISCO conserva `"ledger N5" / 2026-06-11` (verificado hoy) — no se regeneró para no reescribir un artefacto ya committeado. Impacto en cifras: NINGUNO (superseded por v4, que es el citable). Opción: regenerar v3 con el script corregido (números idénticos, solo metadata) o dejarlo anotado |
| 4 | README apuntaba al tag v2 como "the paper's numbers exactly"; única figura de fidelidad era v2; tablas v1/v2 auto-etiquetadas "primaria" sin nota superseded | **PARCIAL** | README FIXED (párrafo del tag reescrito: v3/v4 offline, tag post-N9 anunciado; sección "Reproducing the Nota 3 report") + `f2_fidelidad_v4.png` generada (commit `5d4ff2c`). **PENDIENTE**: headers de `tabla6_fidelidad_v2__*.md` y `tabla5_*_v2` siguen diciendo "Métrica primaria v2 (N5)" sin nota superseded (verificado hoy); la nota superseded vive solo en la tabla v4 y el ledger. Cosmético, sin cifra afectada |
| 5 | Ledger histórico afirma falsamente que MultidimensionalScorer estaba cableado; placeholder "FAISS IVF-PQ" en main.tex; 3 flags de config muertos | **PARCIAL** | Corrección de registro FIXED: entrada N9 del addenda documenta que Módulo 9/Flags 61-68 auditaron código nunca ejecutado (`audit_findings.md` es inmutable — la corrección vive en el addenda, por diseño). **DIFERIDO**: placeholder IVF-PQ en `main.tex:152` (prosa del paper, regla frase-por-frase — anotado en N9 para el rewrite: el índice real es FlatIP exacto). **NO-FIXED**: flags muertos en `pipeline_config.py` (`multidimensional_scoring`, `terminology_normalization`, `reranker_top_k=20`) — documentados en ledger, código sin limpiar (decisión de limpieza pendiente) |
| 6 | Ruteo de templates acoplado a `query_expansion`: el pipeline con config evaluada NO reproduce los prompts de exp12 | **NO-FIXED** | Ni desacoplado ni documentado en TRACEABILITY (las notas v4 que añadí cubren tabla-4-overwrite/exp13-v2/modelos-locales, NO esto). Impacto en cifras citadas: NINGUNO — exp12 (fuente de Tabla 6) usó el runner canónico que sí ruteó 115/194; solo afecta demo/UI y a quien intente reproducir prompts vía `RAGPipeline.query()` |
| 7 | Estudio B.4: break auto-continuaba a los 2:00; `reading_time_ms` medía latencia del sistema; checkpoint corrupto se sobrescribía mudo | **FIXED** | `evaluation_page.py` (click obligatorio + warning max_break), `session_manager.py` (`system_latency_ms`/`read_and_rate_ms` + checkpoint corrupto → `.corrupt.json` + log), `analyze_user_sessions.py` (claves nuevas con fallback); commit `61ad8e8`; py_compile 7/7. Estudio aún no corrido → cero datos afectados |
| 8 | Dashboard presentaba fidelidad v1 superada como "Official thesis baseline" (exp8b); exp10-13 invisibles | **PARCIAL** | Captions FIXED (exp8b/exp8 marcados históricos + puntero a Tabla 6 v4; commit `61ad8e8`). **NO-FIXED**: exp10-13 siguen sin aparecer en el selector (carecen de `aggregated_metrics.json`) — el caption ahora lo declara, pero el dashboard no muestra la evidencia real. Cosmético para el repo; relevante solo si se capturan pantallas para la demo |

## P1/pendientes secundarios (fuera de la tabla de 8)

| # | Hallazgo | Estado | Evidencia |
|---|---|---|---|
| 9 | Corpus: "2 697 documentos" citado en A.3/README vs **2 644** doc_id únicos en el índice (−53 Azure, subsample) | **DIFERIDO (prosa)** | Ledger N9 registra la redacción sugerida ("2 697 procesados, 2 644 representados en el índice"). `README.md:95` y `corpus_stats.json` conservan 2 697 (verificado hoy). **ÚNICO pendiente que toca una cifra YA CITADA** — ver sección siguiente |
| 10 | H6 residual: 62/14 469 claims con boilerplate sobreviven el filtro N8 (cota ≤0,8 pp) | **FIXED-BY-DOC** | Decisión de Enzo (limitación, sin recálculo); párrafo listo en reporte Fase 1; ledger N9 |
| 11 | `smoke_test_nli` hardcodeaba nombre del hub (falla offline) | **FIXED** | Resolver local-first contra `data/models/`; ejecución verificada **exit 1→0**; commit `61ad8e8` |
| 12 | `smoke_recompute_retrieval_stats` requiere ms-marco no cacheado | **DIFERIDO** | Requisito ya documentado en su docstring + escape `SMOKE_SKIP_RECOMPUTE=1`; triage en ledger N9. Se destraba descargando el modelo (~120 MB) |
| 13 | Receta de Tabla 4 sobrescribe `retrieval_metrics__*.json` in-place en exp11 | **FIXED-BY-DOC** | Nota (a) de la sección v4 en `docs/TRACEABILITY_nota3.md`; flag `--out-dir` NO implementado |
| 14 | exp13 queda en métrica v2 (sin rescore v3/v4) | **FIXED-BY-DOC** | Nota (b) en TRACEABILITY: intra-modelo, mismo confound en ambos brazos; añadir nota de instrumento si el paper lo mezcla con v4 |
| 15 | `rescore_nli_v2.py` sin marca de histórico (re-ejecutarlo pisaría evidencia N5) | **FIXED** | Docstring "HISTORICO (N5) — superseded... NO re-ejecutar"; commit `5d4ff2c` |
| 16 | Asesor #9: capturas de demo | **NO-FIXED (deliberado)** | Decisión tuya pendiente (cuáles/cuándo); dashboard además muestra datos históricos (#8) — capturar DESPUÉS de decidir si se actualiza |
| 17 | Asesor #10-salvedad: tag de evidencia | **FIXED** | Tag `nota3-N9-v4-2026-07-02` creado y pusheado; README lo anuncia |

## (2) NO-FIXED/DIFERIDO que tocan cifras o figuras YA CITADAS en paper/A.3

Revisado ítem por ítem — **exactamente UNO toca una cifra citada**:

- **#9 (corpus 2 697 vs 2 644)**: el A.3 declara "2 697 documentos y 24 481 fragmentos" (citado
  textual por el asesor, item 4 de su matriz). Los 24 481 chunks cuadran EXACTOS; los "2 697
  documentos" son correctos como *documentos procesados* pero el índice experimental representa
  2 644. No invalida ninguna métrica (P@k/MRR/nDCG/fidelidad operan sobre chunks); es precisión
  de descripción del corpus. Destino ya decidido: matizar en la reescritura del A.3/paper con la
  redacción del ledger N9. **Si prefieres cerrarlo ANTES del rewrite, el fix mínimo es 1 línea en
  README/corpus_stats con el desglose — dime y lo hago.**

El resto de NO-FIXED (ruteo de templates #6, flags muertos #5c, headers v2 #4b, dashboard #8b,
metadata v3 en disco #3b, smoke #12, capturas #16): **ninguno altera cifra o figura citada** —
verificado contra el origen real de cada número publicado (Tabla 4 ← compute_retrieval_metrics
sobre exp11; Tabla 6 ← runner exp12 + rescores v3 → v4; latencias ← checkpoints firmados).

## (3) Framing del hallazgo entre-modelos v4 — DOS OPCIONES, decide Enzo

Hecho bruto (verificado): bajo v4, el verificador **small** (primario, el de la Tabla 6) da
**1/18** significativo (denso granite-vs-mistral, d_z=+0,42, p_bh=0,014, n=75); el verificador
**base** da **1/18** pero en un par DISTINTO (léxico gemma-vs-granite, d_z=−0,53, p_bh=0,038,
n=42). Ningún par es significativo bajo AMBOS.

**Framing A — "1/18 bajo el verificador primario":**
> "Bajo el instrumento corregido (v4), 1 de 18 contrastes entre-modelos resulta significativo con
> el verificador primario (granite > mistral en el escenario denso, d_z=+0,42, p_BH=0,014); el
> verificador secundario no replica ese par (marca uno distinto), por lo que se reporta como
> evidencia débil y dependiente del instrumento."
- Pro: conserva la corrección de N8 ("el 'todo n.s.' de N5 era artefacto del instrumento — hay
  algo de señal"); fiel al instrumento primario declarado.
- Contra: asimétrico con el estándar del hallazgo central (que exige robustez bajo AMBOS
  verificadores para declararse); un revisor puede preguntar por qué el positivo no pasa el mismo
  filtro que el negativo.

**Framing B — "0/18 robusto bajo ambos verificadores":**
> "Aplicando el mismo criterio de robustez que el hallazgo principal (sostenerse bajo ambos
> verificadores NLI), ninguna diferencia entre-modelos sobrevive: cada verificador marca 1/18
> par significativo, pero no coinciden — consistente con diferencias marginales al borde de la
> potencia del instrumento, no con un efecto robusto."
- Pro: metodológicamente simétrico (mismo estándar dual-verificador para positivos y negativos);
  el más conservador; inmune a la crítica de instrument-shopping.
- Contra: renuncia al matiz "N8 corrigió el 'todo n.s.'" (vuelve, en la práctica, a un n.s.
  entre-modelos, ahora con mejor justificación); pierde el único resultado positivo entre-modelos.

**Dato adicional en camino:** las réplicas H5 (corriendo, ~2 h restantes) miden si el par del
primario (denso granite-vs-mistral) es siquiera estable run-a-run. Si sale SENSIBLE, el framing B
se refuerza solo; si sale ESTABLE, el framing A gana defensa. **Puedes decidir ahora o esperar
las réplicas.**

---
*Generado en la sesión N9 Fase 2/3. Fuentes verificables: commits `e671f67..5d4ff2c`, tag
`nota3-N9-v4-2026-07-02`, `paper/audit_findings_cc_addenda.md` (N9),
`output/audit/N9_auditoria_integral_fase1_2026-07-01.md`.*
