# N9 — Auditoría integral pre-documentación — Reporte Fase 1

**Fecha:** 2026-07-01 · **Alcance:** todo el repositorio (áreas A–H del encargo) · **Modo:** solo lectura
**Estado del repo auditado:** rama `main`, tag `nota3-N8-correccion-2026-06-30`, working tree limpio.
**Método:** 15 agentes de auditoría en paralelo (8 completados; 7 interrumpidos por límite de gasto y
cerrados/parcialmente cerrados inline) + verificación adversarial de hallazgos que afectan cifras +
verificación directa propia de los números críticos. Este documento NO está commiteado; es el insumo
para la Fase 2 y la entrada N9 del ledger.

---

## VEREDICTO

**NO limpio para avanzar directo a documentación.** Un hallazgo P0 confirmado (fidelidad 1,0 "vacua")
mueve cifras entregadas de la Tabla 6 v3 y el claim "2/18 pares significativos"; hay además ~8 P1
operativos que conviene cerrar antes o durante la reescritura. El hallazgo central de la tesis
("mejor retrieval no mejora significativamente la fidelidad") **se sostiene bajo todos los análisis
de esta ronda** — nada de lo encontrado lo toca.

### Decisiones de Enzo (2026-07-01, registradas)

| Tema | Decisión |
|---|---|
| P0 vacuas | **Excluir del denominador** (consistente con Flag 137). Implica regenerar métricas/Tabla 6 (artefactos nuevos, propuesta: sufijo `_v4`), corregir "2/18"→"1/18" y ledger. |
| H5 réplicas | **Correr variante dirigida** (~300 generaciones, ≈3,3 h) — se encola con su propio gate de ETA, DESPUÉS de aplicar la exclusión P0 (las réplicas deben medir la estabilidad de los pares bajo la convención final). |
| H6 residual | **Solo párrafo de limitación** (0,43 % residual con cota documentada; sin recálculo v4 por boilerplate). |
| Cierre N9 | Continuar A3/A6/F1/H1 inline en esta sesión; reporte completo en este md. |

---

## P0 — Fidelidad 1,0 "vacua" en el denominador primario v3

**Qué es:** 59 de 1 798 respuestas elegibles tienen `genuine == 0` (el extractor no produjo ningún
claim verificable: todos los claims son artefactos de formato) y reciben `faithfulness = 1.0` que
ENTRA al denominador primario de la Tabla 6 v3.

**Evidencia (triple confirmación — 2 agentes independientes convergieron y lo reconté directamente):**
- Código: `scripts/rescore_nli_v3.py:104-107` (`if not genuine: ... "faithfulness": 1.0`) y
  `src/generation/hallucination_detector.py:314` (misma convención runtime).
- Conteo directo contra `experiments/results/exp12_matrix/faithfulness_rescore_v3__{small,base}__vb_agree.json`:
  1 798 respuestas, **59 con `genuine==0`, las 59 con `faithfulness==1.0`**.
  Distribución: **gemma 29, qwen 24, granite 5, mistral 1**.
  Ejemplo: `q015` léxico|granite → `total_claims=2, not_a_claim=2, genuine=0, faithfulness=1.0`.
- Bajo el v2 publicado (results.json firmado) esas mismas filas promediaban 0,136 (mediana 0, 44/59 ceros).

**Impacto cuantificado (por los dos agentes, cifras coincidentes):**
- Tabla 6 v3: hasta −0,106 en celdas gemma/qwen (p. ej. qwen léxico 0,361 → 0,255 sin vacuas).
- "2/18 pares significativos (small)" → **"1/18"** (cae léxico gemma-vs-mistral, p_bh=0,0442;
  sobrevive denso granite-vs-mistral, p_bh=0,0136).
- **"Retrieval n.s." ROBUSTO en ambos casos: 0/12 pares RAG-vs-RAG con y sin vacuas.**

**Por qué es inconsistencia y no convención defendible a secas:** Flag 137 ya excluye `method='none'`
(respuestas sin claims) del promedio por "fidelidad sintética"; una respuesta con 0 claims *genuinos*
tampoco aporta señal NLI, pero cuenta como perfectamente fiel. El sesgo favorece a los modelos
artefacto-pesados (gemma/qwen) — la MISMA dirección del confound H1 que N8 corrigió. El ledger N8
atribuye la subida de gemma/qwen ("+0,05..+0,11") solo a limpiar el denominador, sin revelar que parte
sustancial viene de 59 flips 0→1,0.

**Decisión tomada: excluir.** Fase 2: recalcular métricas con `genuine==0` excluido (mismo trato que
`method='none'`), exportar Tabla 6 nueva (propuesta de nombre: `faithfulness_metrics_v4*.json`,
`tabla6_fidelidad_v4`), rehacer Wilcoxon+BH, corregir RESULTADOS_RESUMEN/TRACEABILITY, entrada N9.
Nada de esto toca los artefactos firmados: archivos nuevos.

---

## Hallazgos P1 (para Fase 2, requieren OK por ítem)

1. **UI Experiment Runner puede sobrescribir evidencia firmada.** `src/ui/pages/experiments_page.py:13,135,159-171`
   apunta `RESULTS_DIR` a `experiments/results/` y el botón *Run Experiment* instancia `BenchmarkRunner`
   sobre los exp-ids congelados: un click regeneraría checkpoints/results.json y correría generación LLM
   nueva — ambas restricciones inviolables, sin guard ni confirmación. Propuesta: deshabilitar el botón o
   redirigir a `experiments/results_ui/`.
2. **`smoke_nli_aggregation.py` FALLA hoy** (AssertionError caso 1, exit 1): espera semántica v0 y el
   default runtime cambió a `vb_agree` (`hallucination_detector.py:203`, decisión N8). El correction_log
   aún lo registra como "5/5 PASS". Propuesta: fijar `nli_variant='v0'` en el smoke (documenta Flag 138)
   o duplicar expectativas para vb_agree + actualizar correction_log.
3. **Metadata errónea en el artefacto citable:** `faithfulness_metrics_v3_small.json` dice
   `generated="2026-06-11"` (hardcodeado) y `metric="... ledger N5"` — debería ser N8 y fecha real
   (`compute_faithfulness_metrics.py:469-471`). Se corrige de paso al generar los v4.
4. **README desactualizado post-N8:** README:134-136 afirma que el tag `nota3-evidencia-2026-06-11`
   (v2, ledger N1-N7) reproduce "the paper's numbers exactly"; las cifras citables son v3 (pronto v4) y
   los commits N8 no están pusheados. Cero menciones a v3 en README. Además la única figura de fidelidad
   (`output/figures/nota3/f2_fidelidad_v2.png`) sigue siendo v2 y las tablas v1/v2 se auto-etiquetan
   "Métrica primaria" sin nota de superseded → riesgo de mezclar v2 con la tabla nueva en el A.3.
5. **Ledger histórico afirma algo falso sobre MultidimensionalScorer:** `paper/audit_findings.md:370`
   dice que estaba cableado en `rag_pipeline.py:159`; git history completo demuestra que NUNCA tuvo
   call-sites (los Flags 61-68 auditan un módulo que jamás corrió). Ninguna prosa entregable lo menciona
   (verificado). Como `audit_findings.md` es inmutable, la corrección va como entrada N9 en el addenda.
   Mismo grupo: `pipeline_config` tiene 3 flags que ningún código lee (`multidimensional_scoring`,
   `terminology_normalization`, `reranker_top_k=20` — el funnel real es 50→5 directo).
6. **Placeholder del paper describe un índice que no existe:** `main.tex:152` "[PLACEHOLDER: ... FAISS
   IVF-PQ ...]" — el índice real es `IndexFlatIP` **exacto** (`faiss_index.py:219-222`, mapping
   `index_type: FlatIP`). Corregir al redactar §III.E (además juega A FAVOR: búsqueda densa exacta).
7. **Ruteo de templates acoplado a `query_expansion`:** con la config evaluada (expansion OFF post-N4),
   `RAGPipeline.query()` usa template default para TODO; exp12 (Tabla 6) ruteó 115/194 queries a
   templates no-default (64 procedural + 51 cross_cloud) vía el runner (`rag_pipeline.py:120-123,204-210`
   vs `run_generation_matrix.py:156`). El camino canónico de prompts es el runner. Propuesta: desacoplar
   la clasificación del flag o documentarlo en TRACEABILITY.
8. **Estudio B.4 (pendiente de correr) — 3 defectos pre-estudio:**
   (a) el break auto-continúa a los 2:00 exactos sin click del participante y arranca el reloj de la
   siguiente pregunta (`evaluation_page.py:253-258`; `max_break=600` es variable muerta);
   (b) `reading_time_ms` es en realidad LATENCIA del sistema (search_clicked→response_shown) y
   `rating_time_ms` engloba lectura+calificación (`session_manager.py:242-243`) — y
   `analyze_user_sessions.py:181-182` los agrega literalmente;
   (c) `load_checkpoint` con except mudo: checkpoint corrupto ⇒ sesión nueva que SOBRESCRIBE los ratings
   acumulados sin aviso (`session_manager.py:355-356` + `evaluation_page.py:71-78`).
9. **Dashboard muestra métricas v1 superadas como "Official thesis baseline"** (exp8b, caption hardcodeado
   `dashboard_page.py:53-56`); exp10-13 no aparecen (no tienen `aggregated_metrics.json`). No usar esa
   página en demos/capturas hasta actualizarla.
10. **Corpus: 2 697 vs 2 644.** `corpus_stats.json`/README/TRACEABILITY dicen 2 697 documentos, pero el
    índice solo representa **2 644 doc_id únicos** (−53, todos Azure: el subsample Fase 2.5 eliminó todos
    los chunks de 53 docs de Azure Functions/Blob Storage; los conteos de chunks cuadran exactos:
    35 003 − 10 522 = 24 481). Ninguna métrica cambia. Para A.3/paper: precisar "2 697 documentos
    procesados, de los cuales 2 644 quedan representados en el índice tras el subsample estratificado".

## H6 — cerrado con cifra exacta (párrafo de limitación listo)

**Números finales:** boilerplate "Stay organized with collections…" en **8 144 chunks (24 428
ocurrencias), 100 % GCP = 95,7 % de los 8 509 chunks GCP**. Por campo: `text` 8 144, `heading_path`
8 142, `section_hierarchy` 8 142 (8 142 en los tres; 2 solo en text; 8 142×3+2 = 24 428 ✓).
**Índice contaminado:** BM25 y denso se construyen SOLO desde `text` (`build_index.py:147-149,188-193`;
`hybrid_index.py:52,62-69`); 200/200 chunks afectados muestreados llevan tokens del boilerplate en el
corpus BM25 vivo (nota: el scraping concatenó sin espacio → tokens fusionados tipo `resourcesstay`/
`collectionssave`; cualquier filtro futuro debe usar regex de subcadena cruda, no stopwords).
**Respuestas contaminadas:** exp12 160/3 104 RAG (5,2 %: granite 80, mistral 52, gemma 22, qwen 6;
sin_rag 0/776; 85 % dentro de la cadena `[Source: …]`), exp13 15/50.
**Veredicto de impacto:** SIN sesgo direccional — los afectados son 33,3 % del corpus pero solo
14,9-16,7 % de los slots top-5 según escenario (sub-representados), 0/194 queries contienen tokens del
boilerplate, y el footprint es casi idéntico entre léxico/denso/híbrido (±1,8 pp) → las comparaciones
entregadas son simétricas y se sostienen; los valores ABSOLUTOS incluyen el ruido (esa es la limitación).
**Residual nuevo:** el filtro N8 no captura claims-cita con boilerplate: 62/14 469 = 0,43 % del
denominador NLI, cota dura ≤0,8 pp por config, distribución balanceada entre escenarios → no voltea
ningún veredicto. **Decisión: documentar como limitación de curación (sin recálculo).**

Borrador de limitación (ES, para adaptar al A.3/paper):
> Durante el scraping de la documentación de GCP, un elemento de interfaz («Stay organized with
> collections / Save and categorize content based on your preferences») quedó incrustado en el texto de
> 8 144 de los 8 509 fragmentos de ese proveedor (95,7 %; 33,3 % del corpus total), afectando por igual
> los campos de título y cuerpo indexados. Al ser una propiedad uniforme del corpus, el ruido es
> simétrico entre los tres sistemas comparados (su presencia en los 5 pasajes recuperados varía menos de
> 2 puntos porcentuales entre escenarios y ninguna consulta de evaluación contiene sus términos), por lo
> que no altera las comparaciones reportadas; sí contamina el 5,2 % de las respuestas generadas (que lo
> reproducen en citas textuales) y un 0,43 % de los claims evaluados por NLI (cota ≤0,8 pp por
> configuración). Se documenta como limitación de curación; el corpus no se reconstruyó para no invalidar
> la evidencia firmada de los experimentos 10-13.

## H5 — memo (decisión: correr variante dirigida, con gate)

- **Dependencia de las cifras entregadas:** los 2/18 pares significativos (small) involucran AMBOS a
  mistral: denso granite-vs-mistral (d_z=+0,415, p_bh=0,0136, n=75) y léxico gemma-vs-mistral
  (d_z=−0,445, p_bh=0,0442, n=49 — este cae con la exclusión P0). 6/12 celdas RAG de la Tabla 6 son de
  gemma/mistral. El 100 % del claim entre-modelos descansa en modelos no deterministas a temp=0.
- **Nota metodológica:** a temp=0 el seed de Ollama es inoperante (greedy); lo que amenaza las cifras es
  la varianza run-a-run por no-determinismo de kernels → el diseño correcto son **réplicas**, no seeds.
- **Diseño aprobado (pendiente de gate de lanzamiento):** variante dirigida a las celdas implicadas,
  ~300 generaciones ≈ **3,3 h** (latencias reales exp12: gemma p50 38,8 s / p95 40,3 s; mistral p50
  34,7 s / p95 93,0 s). Peor caso p95: ~7 h. **Secuencia:** primero aplicar exclusión P0 y recomputar
  pares (v4); las réplicas miden la estabilidad de los pares QUE SOBREVIVAN bajo la convención final.
  Es generación nueva → se lanza solo con OK explícito sobre el ETA y el diseño final.

## Datos (B1) — LIMPIO

194 queries (ids únicos) ✓ · cross-cloud 25 = subset exacto ✓ · 24 481 chunks, per-provider exacto
(aws 6 366 / azure 9 606 / gcp 8 509) ✓ · `chunk_id`==filename, únicos ✓ · backup 200 = 194 actuales +
6 removidas (sets exactos) ✓ · chunk_map y FAISS mapping cubren exactamente los 24 481 ✓ (FlatIP, 1024
dims) · `bm25.pkl` escaneado con pickletools: 0 clases externas ✓ · `relevant_chunk_ids` vacíos en las
194 (consistente con oráculo dinámico, N2 — no es hallazgo) · `ground_truth.json` = placeholder huérfano
de febrero (`judgments=[]`, nadie lo lee) — P2 de higiene.

## Tests (E1) y higiene (G1) — cerrados inline

- **pytest: 7 passed, 1 SKIPPED** (`test_nli_output_is_softmax_probabilities`) — ojo: la suite se
  describe como "8 verde"; hay que decir "7+1 skip" o des-skipear.
- Propuesta priorizada de tests (no escritos, para tu OK):
  1. `statistical_analysis.py` (Wilcoxon/d_z/BH) — CERO tests hoy; protege "2/18→1/18" y "retrieval n.s.".
  2. Regresión del detector v3: caso vacuas (`genuine==0`) + artefactos — protege la Tabla 6 v4.
  3. Arreglar `smoke_nli_aggregation.py` (P1 #2) para que la capa smoke vuelva a ser verde honesta.
- Higiene: `.env` NO trackeado ✓ · único trackeado >5 MB = exp12 `results.json` (8,7 MB, evidencia
  aceptada) ✓ · 0 TODO/FIXME/XXX/HACK en src+scripts ✓ · sin patrones de secretos en tracked ✓ ·
  GPL-3.0 consistente LICENSE↔README ✓.

## Documentación cruzada (D1) — números entregados CONSISTENTES

Tabla 4 (retrieval bge) idéntica README ↔ tabla exportada ↔ RESULTADOS_RESUMEN ✓ · Tabla 6 v3 idéntica
en las 3 fuentes ✓ · "0/12 n.s." y "2/18" reproducidos por conteo directo de los JSON ✓ · corpus
general 3 951/46 318 vs experimental 2 697/24 481 correctamente diferenciado ✓ · "0,930" y "+16,8 %"
solo en contextos retirados/etiquetados ✓ (main.tex sigue con cifras viejas = rewrite pendiente
conocido, no hallazgo nuevo).
P2 de staleness: referencias "exp1..exp13" (en disco: exp3-13+exp8b; afecta CLAUDE.md, README:241,
MODELS.md:89) · exp9 corrió con 200 queries y README lo agrupa bajo "194-query set" (la cifra 195/200
es correcta; el paraguas no) · README Key Features L32 con modelos viejos (4º lugar que N8 no cubrió) ·
NOTA3_NEXT_STEPS.md:39 "30 q cross-cloud" (son 25; el ledger N6 ya lo tiene para A.3) y header obsoleto ·
footer de RESULTADOS_RESUMEN dice "(N1–N6)" y no lista artefactos v3 · ledger N8 enumera 5 de los 6
pares sig base (falta denso mistral-vs-qwen, d_z=+0,50, p_bh=0,021, n=46; el conteo 6/18 sí es correcto).

## 10 oportunidades del asesor (20/06) — verificación punto por punto

| # | Item (repositorio) | Veredicto |
|---|---|---|
| 1 | Sección README "Evidencia/Reproducción del informe" | **PARCIAL** — contenido disperso (Experiments + Environment & reproducibility); sin sección titulada así; recetas en TRACEABILITY |
| 2 | Separar sistema completo / demo / benchmark | RESUELTO (README Corpus/Architecture + MODELS.md) |
| 3 | Alinear modelos del README | **PARCIAL** — N8 corrigió 3 lugares; Key Features L32 pendiente |
| 4 | Tabla Markdown de trazabilidad (resultado→script→salida) | RESUELTO (`docs/TRACEABILITY_nota3.md`, matriz con esas columnas) |
| 5 | README en data/evaluation explicando 200→194 | **NO RESUELTO** — existe `test_queries_removed_log.json` pero ningún README ahí |
| 6 | Identificar carpetas históricas vs evidencia final | RESUELTO (TRACEABILITY:18, README:242; con el caveat exp1/exp2) |
| 7 | Comandos mínimos para reproducir tablas | RESUELTO ("Recetas mínimas" en TRACEABILITY) |
| 8 | Tiempos estimados + hardware + CPU/GPU | **PARCIAL** — hardware/GPU sí (README:112,199-202); tiempos de ejecución no |
| 9 | Evidencia visual/capturas de demo separadas | **NO RESUELTO** — 0 capturas trackeadas (solo `demo_acceptance` textual) |
| 10 | Mantener y destacar el tag de evidencia | RESUELTO con salvedad (README apunta al tag v2 como "las cifras del paper"; N8 sin pushear — P1 #4) |

Fuente: `Feedback_Asesor_20_06_2026.docx` (extraído completo; los "10 del documento" aplican al A.3, no
al repo). Cierre propuesto en Fase 2: sección README + README de data/evaluation + tiempos estimados +
2-4 capturas de demo + Key Features.

## Retrieval/reranking, generación, pipeline (A1/A2/A4) — sano en lo que importa

RRF correcto (rangos 1-based, dedup, empates deterministas, k=60 código=config=README) ✓ · embudo
50→5 consistente y verificado contra exp11 (194×4 con exactamente 5 chunks) ✓ · FlatIP exacto +
L2-norm ✓ · fixes D11/D12 intactos ✓ · fixes N8 del detector coherentes (exclusión de artefactos,
agregación max-entailment, denominadores decline-aware; el bug DECLINE_PATTERNS de exp9 está cerrado
y no afecta cifras — exp12 corrió post-fix) ✓ · rescore v3 importa las MISMAS funciones del detector
(sin divergencia offline/runtime) ✓ · fix RFC4180 del CSV presente y correcto ✓ · cobertura re-score
1 798/1 798 ✓.
P2 latentes (no afectan cifras): filtro k8s/cncf devolvería 0 resultados mudos (0/194 afectadas;
README:19,24 aún anuncia "Kubernetes and CNCF"), `HybridIndex.load()` silencioso si falta chunk_map
(patrón Flag 76 latente), `grid_search_alpha` muerto, `rescore_nli_v2.py` sin marca "HISTÓRICO"
(re-ejecutarlo sobrescribiría evidencia N5), `run.py --compare` dice "3 systems" y corre 4, familia BH
between-scenario es de 24 pares (el "0/12" es el subconjunto RAG-vs-RAG — conservador-seguro; solo
cuidar la redacción en el paper), exp13 queda en métrica v2 (robustez intra-modelo argumentable; si el
paper mezcla exp13-v2 con Tabla-v4, añadir nota de instrumento).

## Cobertura de esta Fase 1

| Área | Estado |
|---|---|
| A1 retrieval/reranking | ✅ workflow |
| A2 generación/detector | ✅ workflow |
| A3 evaluación/estadística | ✅ inline (ver addendum) |
| A4 pipeline/UI/run.py | ✅ workflow |
| A5 scripts fidelidad | ✅ workflow |
| A6 scripts stats/matriz | ✅ inline (ver addendum) |
| B1 integridad datos | ✅ workflow |
| B2 H6 boilerplate | ✅ workflow (cerrado con cifra) |
| C1 memo H5 | ✅ workflow |
| D1 coherencia docs | ✅ workflow |
| D2 asesor 10 puntos | ✅ inline |
| E1 tests | ✅ inline (pytest + propuesta) |
| F1 reproducibilidad | ✅ inline (ver addendum) |
| G1 higiene | ✅ inline |
| H1 barrido abierto | ✅ inline (ver addendum) |

Los resultados de A3/A6/F1/H1 están en el addendum al final de este documento (cierre inline 01-02/07).

## ADDENDUM — cierre inline de A3 / A6 / F1 / H1 (01-02/07)

### A3 — Evaluación/estadística: VERIFICADO SANO
- `statistical_analysis.py` (565 L, leído completo): Wilcoxon con manejo de all-zero-diffs (p=1,0,
  `wilcoxon_zero_diff`), d_z pareado = mean(diff)/std(diff, ddof=1) correcto, bootstrap sembrado
  (`RandomState(42)`, remuestreo pareado), BH/Holm vía `statsmodels.multipletests` aplicados sobre la
  FAMILIA completa (Flag 108 cerrado de verdad), t-test solo si diff+ambas marginales normales.
- **Pareo verificado empíricamente**: `run_all_comparisons:513-516` trunca a `min_len` y parea por
  índice (no por query_id) — latente; comprobado con Python que exp10 (3×194) y exp11 (4×194) guardan
  las queries en orden IDÉNTICO entre configs → los Wilcoxon de la Tabla 4 publicada están bien
  pareados. Los v3 de fidelidad parean por intersección de query_id (camino distinto, ya verificado).
  P2: convertir el truncado silencioso en error o parear por query_id.
- `benchmark_runner.py`: patrón `.get(clave, 0)` en la agregación (L592-775, estilo Flag 76) — solo
  alcanza exp3-9/UI (exp10-13 usan runners dedicados); P2 latente.
- **Capa smoke ejecutada: 6/9 PASS** — `smoke_fix_flag76` PASA (el fix Flag 76 sigue: raise en métrica
  ausente), `smoke_bh_fdr` PASA (BH/Holm validados contra el CSV inmutable), extractor N8/seeds/
  hall_n_effective/compute_retrieval_stats PASAN. Fallan: `smoke_nli_aggregation` (P1 #2, ya
  diagnosticado) y 2 por modelo no cacheado (ver F1).

### A6 — Scripts retrieval/estadística/matriz: VERIFICADO SANO
- `compute_retrieval_metrics.py` (736 L, leído completo): P@k/R@k/MRR/nDCG matemáticamente correctos
  (nDCG graduado con `max(0,·)` e IDCG sobre el pool), oráculo PARAMETRIZADO con flag explícito
  `oracle_is_circular`, umbrales binarios por percentil del pool (p50 primario — fix del 0.0 fijo que
  inflaba precisión, documentado), stats delegadas al módulo compartido (sin duplicación divergente).
  Menores: header de consola imprime el umbral legacy 0.0 (solo display); `_std` con ddof=0 (display).
- `recompute_retrieval_stats.py`: scope bloqueado a exp8/exp8b (histórico) con backup `_pre_fix` — OK.
- `run_generation_matrix.py` (runner de exp12): `except` de L77 es solo telemetría GPU (benigno);
  `contexts.get(scenario, {}).get(qid, [])` L207 = patrón Flag 76 latente pero HOY inalcanzable
  (contexts se construye de exp11 con `SystemExit` si falta config, y la evidencia muestra 5/5
  retrieved_ids en todos los RAG). P2: cambiar a acceso directo que falle ruidoso.
- Exportadores leen los archivos correctos: `_export_tabla6_v3.py` ← `faithfulness_metrics_v3_small.json`
  ✓; `_export_paper_tables_nota3.py` ← v1/v2 (por diseño, era pre-N8); `_make_figures_nota3.py:73` lee
  `faithfulness_metrics_v2.json` → **f2/f3 son v2 POR CONSTRUCCIÓN** (cierra la causa del P1 #4: para
  la métrica final hay que extender el script de figuras, no solo regenerar).

### F1 — Reproducibilidad: recetas verificadas estáticamente; 2 bloqueos reales
- Todos los comandos de "Recetas mínimas" (TRACEABILITY) y del Quick Start referencian scripts
  existentes con flags que COINCIDEN con su argparse (verificado uno por uno: `--experiment`,
  `--oracle-model/--oracle-label`, `--write-v1`, `--faithfulness-source`, `--out-tag`, posicionales de
  `_export_*`/`_latency_p50p95`, `--verifier/--variant/--margin`). pytest 7+1skip ✓.
- **Bloqueo 1 (P1-repro):** los oráculos de la receta de Tabla 4 (`BAAI/bge-reranker-large` y
  `cross-encoder/ms-marco-MiniLM-L-12-v2`) NO están en ninguna caché local hoy (el hub cache solo
  contiene una entrada INCOMPLETA de `nli-deberta-v3-small`, sin snapshots) → esas recetas fallan
  offline en esta máquina y requerirían re-descarga. Los rescores NLI sí corren: usan
  `data/models/nli-deberta-v3-{base,small}` locales (714 MB + 2,6 GB, gitignored) — cadena de evidencia
  v3 intacta y verificada.
- **Bloqueo 2 (P1-repro, fix trivial):** `smoke_test_nli.py:29` y `smoke_recompute_retrieval_stats.py`
  hardcodean nombres del hub en vez del resolver local-first de `rescore_nli_v3.py:44-51` → 2 smokes
  fallan offline aunque el modelo esté en `data/models/`. Fix: reutilizar el resolver.
- **Cuidado adicional:** la receta de Tabla 4 escribe `retrieval_metrics__<label>.json` EN el
  directorio de evidencia de exp11 (mismo nombre → sobrescritura in-place del artefacto publicado;
  determinista en teoría, pero contradice el principio "nada firmado se sobrescribe" de la ronda v3).
  Propuesta: nota en TRACEABILITY o flag `--out-dir`.

### H1 — Barrido abierto: sin hallazgos nuevos materiales
- `paper/overleaf_ready/figures/`: las 9 PNGs + 9 .tex son TODAS de la era exp3-8b (incluida
  `fig_cross_cloud_improvement.png` del +16,8 % retirado y `table_exp7.tex`) — estado pre-rewrite ya
  conocido (N8 Parte 3); al reescribir el paper hay que reemplazar el directorio completo, no editar.
- `notebooks/` (3), `paper/annotation_guidelines_es.md`, `annotation_package_internal.md`,
  `config/*.yaml`: grep de cifras huérfanas (0,93 / 16,8 / 46 318 / 3 951 / llama3 / qwen2.5 / 200
  queries) → CERO hits.
- `config/config.yaml`: claves de la era de ingesta que el pipeline evaluado NO lee (reranker
  `ms-marco-mini-6` L118 vs L-12-v2 evaluado; `query_expansion.enabled: true` L111 vs OFF evaluado;
  pesos `multidimensional` L120-126 del scorer muerto) — P2, mismo grupo que los flags muertos de
  `pipeline_config`; el runtime evaluado se configura en `pipeline_config.py` (verificado consistente).
  L58 `all-MiniLM-L6-v2` es el embedder del chunking semántico (rol distinto, no es inconsistencia).
- `titulacion/emails_lewis_y_gyt.md:18`: correspondencia histórica con cifras retiradas (P@1 0,93,
  +16,8 %, p<0,0001) presentadas afirmativamente — no editable retroactivamente (email enviado);
  P2-nota: visible para un jurado que navegue el repo; decidir si se anota como histórico.

## Plan Fase 2 (borrador, cada ítem con su OK)

1. **P0**: recalcular fidelidad excluyendo `genuine==0` → `faithfulness_metrics_v4*.json` +
   `tabla6_fidelidad_v4` + stats (esperado: "1/18") + corrección RESULTADOS_RESUMEN/TRACEABILITY +
   entrada N9. Artefactos v3 se conservan como superseded.
2. Guard del Experiment Runner de la UI (freeze).
3. Fix `smoke_nli_aggregation.py` + correction_log.
4. Metadata de `compute_faithfulness_metrics.py` (fecha real + etiqueta N8/N9).
5. README: mención v3/v4, Key Features, sección "Reproducción del informe", tiempos estimados,
   README de data/evaluation (asesor #1/#3/#5/#8), quitar k8s/cncf o nota.
6. Figura f2 regenerada con métrica final (o nota superseded explícita).
7. Fixes B.4 (break, naming de timings, checkpoint) antes de la ventana de estudio.
8. Marcar `rescore_nli_v2.py` como histórico; docstrings menores; `exp1..exp13`→`exp3..exp13+exp8b`.
9. **H5**: lanzar variante dirigida (~3,3 h) tras el recálculo v4, con gate de ETA.
10. pytest + smokes verdes, diff de `experiments/results/` probando intacto, entrada N9, gate de push.
