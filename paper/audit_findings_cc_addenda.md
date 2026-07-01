# Audit findings — Claude Code addenda

Hallazgos relacionados encontrados durante ejecución Tier 0 que NO están en `paper/audit_findings.md`. **No se corrigen sin aprobación explícita del usuario** (disciplina zero-trust bidireccional, mission spec §"Regla de oro").

---

## Fase 1 addenda

### A1 — `.get(..., 0)` silent zeros adicionales en `results_exporter.py`

El audit §21.5 Tier 0 (T0.7 / Flag 76) cita explícitamente las líneas **540** y **655** de `src/evaluation/results_exporter.py`. Un grep exhaustivo durante la verificación zero-trust muestra el mismo patrón en **9 líneas adicionales**:

```
241:                    matrix[i][j] = data[key].get("ret_ndcg@5_mean", 0)
287:            values = [data[method].get(m, 0) for m in metrics]
313:        ndcg5 = [data[c].get("ret_ndcg@5_mean", 0) for c in configs]
314:        recall5 = [data[c].get("ret_recall@5_mean", 0) for c in configs]
353:        values = [data.get(s, {}).get("ret_ndcg@5_mean", 0) for s in stages]
410:            if any(data[m].get(key, 0) > 0 for m in models):
425:            values = [data[model].get(m[0], 0) for m in available_metrics]
463:            values = np.array([data[c].get(key, 0) for c in configs])
495:            values = [data[config].get(m, 0) for m in metrics]
```

Cada una está en una figura distinta (`fig_statistical_heatmap`, `fig_method_comparison_radar`, distintos charts de exp7/exp5/etc.). Misma patología que Flag 76 / Flag 91: si la métrica falta, se fabrica `0` silenciosamente, produciendo barras/heatmaps en cero en lugar de crashear.

**Consecuencia si NO se corrigen**: tras aplicar el fix a 540/655, las regeneraciones en Fase 3 podrían aún producir figuras vacías en otros charts sin diagnóstico. El fix parcial pospone el problema.

**Recomendación (pendiente de aprobación)**: extender el fix a las 9 líneas adicionales, con mismo formato de `KeyError` diagnóstico. Effort estimado: 30-45 min. Ideal de aplicar en Fase 1 (mismo commit o commit separado `fix(flag-76-extended): raise on missing keys across all exporter figures`). **No lo aplico hasta que el usuario apruebe.**

### A2 — Menciones residuales de "normalization" en placeholders de main.tex

El plan Fase 1 paso 1.3 lista estos 6 puntos para renombrar a "query expansion": L68-69, L98, L145, L146, L199, L204. Todos aplicados en el commit `fix(flag-160,124)`.

Tras aplicar, un `grep -n -i 'normaliz' paper/overleaf_ready/main.tex` muestra 3 menciones residuales, todas dentro de `[PLACEHOLDER: ...]` que serán reescritos en Fase 5:

```
93 : ...benchmarks... do not evaluate cross-cloud terminology normalization.
122: ...Position the gap: none address cross-cloud terminology normalization.
134: Describe 5 stages: (1) ingestion + normalization, (2) hybrid retrieval...
```

- L93 y L122 son **afirmaciones de gap de literatura** (Related Work). Si el framing del paper cambia de "terminology normalization" a "cross-provider query expansion", el gap-statement debería reformularse en Fase 5 o bien declararse que el gap es sobre ambos conceptos.
- L134 es el **label del primer stage del pipeline** en el §III.A overview. Con la nueva narrativa (metadata tagging a nivel de índice, BM25 expansion a nivel query), el stage 1 debería llamarse "ingestion + dictionary-based tagging" o similar.

No se modifican en esta fase — son prose nueva que corresponde a Fase 5. **Pendiente de redacción por el usuario en Fase 5; yo solo flagging.**

### A3 — Abstract L63 también mencionaba "cross-cloud terminology normalization module"

Durante 1.3, al renombrar la prosa "Cross-cloud terminology normalization contributes..." (L68-69) observé que la misma oración en el abstract (L62-64) describía el módulo con el mismo compound: *"a cross-cloud terminology normalization module"*. Esa línea específica no estaba en la enumeración de 6 puntos del plan. La renombré igualmente a *"a dictionary-based cross-provider query expansion module"* porque es el mismo claim arquitectónico en la misma oración del abstract; dejar solo 5 renombres y uno coherente hubiera dejado el abstract internamente inconsistente.

Decisión documentada aquí por transparencia; si el usuario considera que fue fuera de alcance, revertir con `git revert` el commit `fix(flag-160,124)` o hacer patch puntual.

### A4 — `scripts/compute_retrieval_metrics.py` tiene un pipeline estadístico paralelo sin BH-FDR

Audit §3 Módulo 3 ya documentó la duplicación de sistemas de métricas entre `src/evaluation/retrieval_metrics.py` y `scripts/compute_retrieval_metrics.py`. Al aplicar Flag 103/108 noté que el impacto es más profundo:

`scripts/compute_retrieval_metrics.py:182-215` implementa su propio `run_statistical_tests` que:
- Calcula Cohen's d con pooled SD para muestras INDEPENDIENTES (fórmula de audit Flag 21), no d_z ni d_av. Los datos son pareados (mismas queries, distintos sistemas).
- No aplica BH/Holm.
- Emite la salida a `experiments/results/exp8/retrieval_metrics.json:statistical_tests` — que es exactamente el campo que el paper termina citando.

Mi patch en `src/evaluation/statistical_analysis.py` agrega BH-FDR + Holm correctamente, pero la siguiente corrida de `compute_retrieval_metrics.py` **seguiría** produciendo `statistical_tests` sin corrección y con d-incorrecto.

**Efecto práctico**: para cerrar Flag 108 a nivel de artefacto público del paper, hay que:
(a) Deprecar `scripts/compute_retrieval_metrics.py:run_statistical_tests` y hacer que invoque `run_all_comparisons` + `apply_corrections_to_results` de `statistical_analysis.py`, o
(b) Re-implementar BH/Holm + d_av dentro de `scripts/compute_retrieval_metrics.py`.

Opción (a) es mejor arquitectura (elimina duplicación Módulo 3). Effort ~1-2h.

**No lo aplico en Fase 1** porque el plan explícito del paso 1.4 indica únicamente `src/evaluation/statistical_analysis.py` como archivo objetivo. Flag 108 queda parcialmente cerrado: la infraestructura está lista, pero el script productor del JSON reportado no la usa todavía.

### A5 — Menciones residuales de "Docker" en main.tex fuera del abstract

El plan Fase 1 paso 1.5 especificó renombrar únicamente la mención del abstract L70-71. Tras el fix, un `grep -n -i 'docker' paper/overleaf_ready/main.tex` muestra 2 menciones restantes:

```
101: \item An open-source reference implementation (code, data, Docker) for one-command replication.
173: [PLACEHOLDER: Hardware (CPU/GPU/RAM). Software: Python 3.11, sentence-transformers, faiss-cpu, Ollama. Docker image + Zenodo DOI + GitHub URL. Fixed random seeds.]
```

- **L101**: contribution bullet con el mismo claim falso que el abstract. Flag 159 aplica igual — no existe Dockerfile en el repo.
- **L173**: placeholder del §IV.D "Implementation and Reproducibility" que se reescribe en Fase 5.

**No se corrigen en Fase 1** porque la instrucción explícita del usuario fue "remueve el claim 'Docker artifacts' del abstract en Fase 1" (condición (d) de la aprobación). L101 es contribución, no abstract; L173 es placeholder para Fase 5.

Pendiente de decisión: aplicar el mismo rename en L101 como parte de Fase 5 (reescritura de contribuciones), o como patch puntual si el usuario lo pide antes.

---

## Nota 3 addenda (2026-06-07)

### N1 — exp7 "+16,8 %" NO es atribuible a la expansión: ambos brazos corrieron retrieval idéntico

Veredicto definitivo (supera la hipótesis de Flag 5 / "BM25 recall" en `audit_findings.md` L2030). Prueba por código:

- exp7 tiene 2 configs, ambos `fusion_method="rrf"` (`experiments/experiment_configs.py`).
- En RRF, `HybridRetriever.search` pasaba la query **cruda** a ambas piernas: `hybrid_retriever.py:52` (`query=processed.bm25_query if fusion=="linear" else query`). La expansión (`processed.bm25_query`) **nunca** llegaba a BM25 en el camino RRF (bug D11).
- `terminology_normalization` no se lee en ningún punto del retrieval (código muerto en runtime, consistente con Flag 40 / §6.4).
- ⇒ Los dos brazos (`cross_cloud_no_norm`, `cross_cloud_with_norm`) recuperaron **los mismos chunks en el mismo orden**. El delta 0,2916 − 0,2498 = +16,75 % es ruido de generación + NLI descalibrado (Flag 135) sobre n=29; **no** es una ganancia de retrieval ni de "normalization".
- **Corrección de L2030**: la explicación "lo que mejora es el recall de BM25 por la expansión" es **incorrecta** para el híbrido RRF — la expansión no alcanzaba BM25.

Resolución: D11 ahora aplica la expansión real en RRF (flag de config, default OFF); se evalúa limpiamente en **exp13** (expansión ON vs OFF, 30 queries cross-cloud, generación determinista). Hasta exp13, el claim +16,8 % queda **retirado** (refuerza Flag 142: candidato #1 a remover del abstract).

### N2 — Circularidad del oráculo cuantificada (exp11, 194 q, reranker D12)

El reranker del pipeline (`ms-marco-MiniLM-L-12-v2`) era también el oráculo de relevancia de `compute_retrieval_metrics.py`. Tras D12 (reranker ve texto completo, igual que el oráculo) el híbrido post-rerank alcanza **NDCG@5 = 0,995 por construcción** (tautológico). Medido además con oráculo **independiente** (`BAAI/bge-reranker-large`, ya cacheado — sin descarga):

| Sistema | NDCG@5 ms-marco (circular) | NDCG@5 bge (indep) |
|---|---|---|
| Léxico (BM25) | 0,552 | 0,442 |
| Denso (BGE) | 0,649 | 0,624 |
| Híbrido pre-rerank (RRF) | 0,668 | 0,603 |
| Híbrido post-rerank | **0,995** | **0,740** |

- Híbrido(post) vs Denso: d_z circular +1,38 → **indep +0,45, p_BH < 0,001** (real y significativo, pero la magnitud estaba inflada por la circularidad).
- Híbrido(pre, solo fusión RRF) vs Denso: **n.s.** en ambos oráculos (d = +0,07 / −0,08). La ventaja del híbrido proviene de la etapa de **reranking**, no de la fusión RRF.
- El orden Híbrido > Denso > Léxico se mantiene entre oráculos (mitigación de circularidad satisfecha).

Tablas: `output/tables/nota3/oracle_stability__exp11_retrieval194_fullrerank.{md,csv}`.

### N3 — "Sin RAG": la fidelidad NLI es 0 por construcción (tratamiento para la Tabla 6)

En el escenario sin recuperación (`retrieval_method="none"`), `HallucinationDetector.check(retrieved_chunks=[])` toma el path `_no_evidence_report`: sin contexto, ningún claim puede verificarse, así que todo claim se marca `unsupported_no_evidence` y `faithfulness=0` (salvo declinación honesta → 1,0 vacuo). Por tanto la fidelidad "sin RAG" ≈ tasa de declinación; **no** mide corrección de la respuesta.

Consecuencia: en la Tabla 6 (4 escenarios × 4 modelos) la columna "sin RAG" **no es comparable** con los escenarios RAG como si fuera una fidelidad equivalente. Mide "¿se abstuvo de afirmar sin contexto?", no "¿acertó?". Sin respuestas gold (`answer` vacío en `test_queries.json`) no hay medida de corrección para sin-RAG.

Decisión (a implementar en el esquema exp12 y el export Nota 3):
- Reportar "sin RAG" con `answered_rate` y `decline_rate`, no solo `faithfulness`.
- Anotar explícitamente en la Tabla 6 que la fidelidad de "sin RAG" es 0-por-construcción.
- El valor del control sin-RAG es evidenciar que RAG **reduce** la afirmación sin evidencia, no una comparación numérica directa de fidelidad.

---

## Nota 3 addenda — exp13 (2026-06-11)

### N4 — exp13 cierra el caso de la expansión cross-cloud: no aporta

Con el fix D11 (expansión real en el camino RRF, default OFF) se corrió exp13: 25 q cross-cloud (>1 provider), brazos OFF vs ON, reranker corregido (D12), generación granite4.1 determinista (temp=0).

- **D11 funciona:** la expansión ahora llega a BM25 → el retrieval cambió en **7/25** consultas (exp7: 0; sus brazos eran idénticos). El mecanismo que el paper afirmaba ahora sí opera.
- **Pero no ayuda:** bajo oráculo independiente (bge-reranker-large), ON es direccionalmente **peor** (NDCG@5 0,852→0,820; recall@5 0,863→0,789; avg_score 0,287→0,269), **n.s. tras BH** (d_z≈−0,4, p_BH=0,13). Fidelidad OFF 0,175 ≈ ON 0,174 (n.s.).
- **Conclusión:** el claim "+16,8 % por normalización/expansión" queda retirado **definitivamente** — era un no-op (N1) y, bien implementado, la expansión cross-cloud no mejora retrieval ni fidelidad en el corpus reconstruido. Consistente con la literatura (la expansión ayuda al BM25 aislado, pero el híbrido RRF+rerank ya recupera lo relevante; añadir términos solo introduce ruido).

Artefactos: `experiments/results/exp13_expansion/{results.json, faithfulness_metrics.json, retrieval_metrics__bge-indep.json}`; `data/evaluation/cross_cloud_subset.json`.

### N5 — Fidelidad: denominador contaminado por declinaciones + instrumento NLI cuestionado (2026-06-11)

Auditoría externa del 06-11 (verificada íntegramente desde checkpoints/results.json antes de
tocar código; las 11 celdas reproducidas exactas) + hallazgo propio adicional.

**(a) Denominador contaminado (A1, confirmado).** `compute_faithfulness_metrics.py` promediaba
toda query con claims sin excluir declinaciones; los textos de declinación generan claims que
el NLI evalúa (granite-híbrido: 116/120 declines con claims, fidelidad media de declinadas
0,129). Sesgo asimétrico por modelo — antes → después (excluyendo el flag v1):

| celda | publicada | answered-only (flag v1) |
|---|---|---|
| granite léxico/denso/híbrido | 0,170 / 0,193 / 0,202 | 0,243 / 0,267 / 0,316 |
| gemma léxico/denso/híbrido | 0,331 / 0,322 / 0,268 | 0,400 / 0,331 / 0,285 |
| mistral léxico/denso/híbrido | 0,222 / 0,258 / 0,256 | 0,231 / 0,266 / 0,274 |
| qwen léxico/denso/híbrido | 0,306 / 0,254 / 0,278 | 0,268 / 0,238 / 0,257 |

Hunde a granite y INFLA a qwen → indefendible como métrica única.

**(b) Flag de decline roto en ambas direcciones (hallazgo propio).** FN: rechazos no marcados
en apertura — mistral ~18 % de sus "contestadas", gemma ~10-13 %, qwen ~11-15 %, granite ~3-5 %
(variantes "there is no information regarding X" fuera de los 14 patrones). FP: ~60 % de los
"declines" de granite/qwen son textos >150 tokens con citas y ≥8 claims (respuestas parciales
con hedge). Por eso el fix v2 NO reutiliza el flag: clasificador de análisis a 3 clases
(`pure_decline` = marcador en primeros 300c / `hedged_partial` / `answered`), métrica PRIMARIA
= excluir solo pure_decline (decisión de Enzo 06-11), con 3 sensibilidades (flag v1; estricta
marcador-en-cualquier-parte; publicada v1). Artefactos: `faithfulness_metrics_v2.json`,
`tabla6_fidelidad_v2`, `tabla6_sensibilidad_denominador`, `tabla6c_clasificacion_v2`.

**(c) Re-stats (A2, confirmado y corregido).** Las familias B/C corrían sobre la métrica
contaminada (p. ej. granite denso-vs-híbrido n=187 incluía ~120 declinaciones). v2 re-testea
sobre la intersección de no-excluidas por par (n documentado, nota de potencia si n<60).
Veredicto: **"método de retrieval n.s. en fidelidad" SE SOSTIENE**; granite/mistral monótonos
hacia el híbrido pero p_BH≥0,11; la excepción v1 (denso>léxico en qwen) no sobrevive; entre
modelos todo n.s. bajo v2 (los "sig" v1 eran artefacto del denominador); RAG≫sin-RAG se
sostiene salvo qwen (no testeable, N6.4).

**(d) Instrumento NLI (A3, confirmado y agravado).** Banda de % contradicted 27–39 % casi
insensible a modelo Y escenario; donde varía, sube con MEJOR contexto (gemma léxico→híbrido
26,8→34,5 %). q085 granite-híbrido reproducido bit-exacto: 28/28 claims procedurales
fundamentados = "contradicted" a prob ~0,99 ("Open your web browser and go to the Azure
portal" = 0,986). Mecánica: lado contradicted toma max sobre 5 chunks, umbral 0,7, sin guarda
simétrica (`hallucination_detector.py:413-424`) — la inflación que Flag 138 corrigió del lado
supported sigue activa del lado contradicted, y el modelo small emite contradicción ~0,99 en
hipótesis imperativas (los claims sintetizados "Header: contenido." pueden amplificar).
Mitigación ejecutada (3b, 2026-06-11): muestra de 50 claims para juicio humano
(`output/audit/claim_audit_sample`, pendiente de revisión de Enzo) + re-score íntegro con
segundo verificador (nli-deberta-v3-base, fp16, mismo procedimiento; 0 mismatches de claims)
+ ablación de formato (small sin prefijo "Header:"). Resultados:

- **Ablación de formato: negativa limpia.** kappa 0,87, Spearman 0,944, órdenes por modelo
  idénticos, q085 idéntico → el formato sintetizado de claims no es la causa.
- **Acuerdo entre verificadores: débil a nivel claim, inestable a nivel orden fino.**
  kappa 0,411 en la muestra de 50 (las contradicciones de small migran a unsupported bajo
  base); Spearman de medias por config 0,825 (métrica publicada) pero **0,559 n.s. en la
  primaria v2**; el orden léxico/denso/híbrido cambia en 3/4 modelos. Niveles absolutos
  sistemáticamente más bajos bajo base.
- **q085: 28/28 contradicted TAMBIÉN bajo el verificador base** → el artefacto es del
  PROCEDIMIENTO (max sobre 5 chunks largos + umbral 0,7 sin guarda simétrica + dominio
  técnico), no del tamaño del verificador ni del formato del claim.
- **Espejo estadístico: familia B (RAG-vs-RAG) bajo verificador base, denominador primario,
  pareado por intersección: TODO n.s. tras BH** (mejor |d_z|=0,38). Junto con v1 y v2-small:
  el hallazgo "el método de retrieval no mueve la fidelidad" se sostiene bajo 2 verificadores
  × 4 denominadores.

**Implicación para el paper:** la fidelidad NLI absoluta NO es interpretable como tasa real
de alucinación (los dos instrumentos discrepan en nivel y en orden fino); solo se reportan
contrastes que sobreviven a ambos verificadores, con la auditoría del instrumento declarada.
Artefactos: `faithfulness_rescore__{nli-base,small-noheader}.json`,
`output/audit/rescore_v2_summary.md`, `output/audit/claim_audit_sample_scores_v2.json`.

### N6 — Consistencia documental para la reescritura del A.3 (2026-06-11)

Cuatro correcciones/precisiones que la reescritura de §4.5/§6.3 debe incorporar:

1. **"30 consultas entre nubes" → 25.** El A.3 v6.1 dice 30; exp13 usó las **25**
   consultas cross-provider (>1 proveedor) del set depurado de 194
   (`data/evaluation/cross_cloud_subset.json`, len=25, verificado). La cifra "30"
   aparece también en la resolución de N1 (línea "se evalúa… 30 queries cross-cloud",
   escrita antes de construir el subset) — léase 25 ahí también. Procedencia del
   subset: filtrado automático de las 194 por `providers>1`, no una selección manual.
2. **Latencias de qwen3.5 = cota superior.** Sus corridas de exp12 (06-09 → 06-11)
   compartieron GPU con un segundo proceso en parte de la ventana. Decisión: NO
   re-muestrear (qwen3.5 es el más lento con o sin contención; mayor precisión no
   cambia ninguna conclusión); la tabla `latency__exp12_matrix.md` lleva la nota.
3. **D8 confirmado: índice denso = búsqueda exacta.** El índice construido
   (`data/indices/faiss_bge-large_adaptive_500.index`) es `faiss.IndexFlatIP`
   (inner product, 24 481 vectores), verificado por inspección del artefacto;
   `faiss_index.py:59` es el camino por defecto. **NO** es IVF-PQ: el texto del A.3
   que mencione índices aproximados debe corregirse — no hay error de cuantización
   ni parámetro nprobe que reportar.
4. **qwen3.5 sin_rag: 177/194 respuestas vacías** (len=0, sin error marcado) →
   `method="none"`, n_eff=17. La celda "Sin RAG" de qwen en la Tabla 6 descansa en
   17 consultas y el contraste "RAG ≫ sin RAG" es **no testeable** para qwen bajo la
   métrica v2 (n pareado 5-7). Documentar como limitación del control para ese modelo
   (probable interacción del template sin contexto con qwen3.5; no se regenera).

### N7 — Cierre operativo de la ronda (2026-06-11, segunda pasada de auditoría)

Auditoría externa de cierre (7 ítems) verificada íntegramente + auditoría integral propia.
Acciones ejecutadas (commits atómicos de esta fecha):

1. **Reproducibilidad reparada (lo grave):** la implementación D11 (`hybrid_retriever.py`
   routing bm25_query + `hybrid_index.py` plumbing) estaba SIN versionar — exp13 corrió con
   código que el repo no contenía. Y `test_queries.json` trackeado seguía en 200: el set
   canónico 194 + la bitácora de remoción (`test_queries_removed_log.json`) + backup quedan
   versionados. El claim del A.3 ("bitácora versionada junto al código") ahora es verdadero.
2. **Sweep del commit F6 resuelto:** `6f4819b` arrastró 1 línea pre-existente que cableaba
   `config.query_expansion` al branch híbrido de `query()`; con `PROPOSED_HYBRID.
   query_expansion=True` la config canónica expandía — incoherente con N4 y con cómo corrió
   exp12. Decisión (Enzo): **`query_expansion=False`** en PROPOSED_HYBRID (comentario citando
   N4); el mecanismo D11 queda disponible vía flag/runner. Demo checkbox default off.
3. **exp13 bajo métrica v2:** `faithfulness_metrics_v2.json` nuevo — "OFF≈ON" SE SOSTIENE
   (primaria 0,285 vs 0,324, n pareado=10, d_z=+0,21, p_bh=0,53, n.s.). Nota: el runner de
   exp13 no persistió el flag v1 → sens_a==publicada; el clasificador v2 por marcadores
   detecta 56-60 % de declinaciones puras.
4. **README público corregido:** mostraba P@1 0,930 (oráculo circular, corpus pre-rebuild,
   200 queries) y el "+16,8 %" retirado como hallazgo vigente. Reescrito sobre Tabla 4
   (oráculo independiente, NDCG@5 0,740 titular / 0,995 anotado circular, 194 q) y la tabla
   de experimentos exp9-13 con el retiro explícito. Históricos (titulacion/, exp9/_handoff)
   intactos a propósito.
5. **Tabla 4 + figuras finales** generadas por script (`tabla4_retrieval__exp11`,
   `_make_figures_nota3.py` → f1-f4 en `output/figures/nota3/`).
6. **Higiene:** index.lock huérfano; .gitignore con política (data/models/ 700 MB,
   *.log raíz, settings.local de Claude); benchmark.log (3,7 MB) y rerun_phase2.log
   destrackeados; checkpoints crudos exp12 (16) + exp10 versionados (decisión: evidencia);
   caches LLM del set final versionados; script phase3 movido fuera de pytest
   (suite verde: 5+3 passed, 0 errors).
7. **Demo (I7):** namespace de caché `demo::` en el path streaming + caché desactivado en el
   LLMManager de la demo (consistencia perceptual para SUS); tests de paridad extendidos.
8. **Publicación:** push de `pre-corpus-rebuild-2026-05-21` + tag anotado
   **`nota3-evidencia-2026-06-11`** a origin (pre-aprobado por Enzo tras checklist de
   secretos/.env/tamaños). Pendiente de decisión de Enzo (reportado, no ejecutado):
   renombrar la rama (el nombre ya no describe su contenido) o fusionar a main.

**Queda FUERA del repo (handoff humano):** revisión de los 50 claims
(`output/audit/claim_audit_sample`), reescritura A.3 v7 / A.1, Figuras 1-2 draw.io,
instrumento SUS/Likert + guía B.4, video de sustentación, actas/rúbricas.

---

## N8 — Auditoría crítica del instrumento de fidelidad (NLI) + trazabilidad (2026-06-30)

Auditoría de hipótesis (Fase 1 read-only, evidencia propia reproducida) sobre si el verificador
NLI es lo bastante confiable para defender los contrastes finos de fidelidad. **El veredicto
central de la ronda — "el método de retrieval no mueve la fidelidad" — NO cambia:** los artefactos
del instrumento son planos entre escenarios y el contraste entre-modelos ya era n.s. en el v2
publicado (0 pares sig_bh). Lo que el instrumento mueve son los **niveles absolutos** (Tabla 6 v2
0,23–0,38) y la **banda contradicted**, que es lo que A.3 V7.2 / LACCI citan.

### Restricción de entorno (importante)
El entorno de Claude Code que ejecutó esta ronda **no tiene stack científico** (sin
torch/sentence-transformers/numpy/scipy). Por tanto: el fix de extracción (puro Python) y todas las
correcciones de doc/scripts se hicieron y verificaron aquí; **el re-score NLI v3 y las re-stats
deben correrse en la máquina de Enzo** (GPU/Ollama). Esta ronda entrega código corregido + scripts
`_v3` + recetas; las cifras `_v3` las produce Enzo.

### H1 — el extractor deja pasar artefactos de formato (CONFIRMADO, cuantificado)
Sobre los 16k claims de exp12: **10,8% son artefactos** (headers ATX, filas de tabla, spans rotos
por `**` sin balancear, meta-comentario de cobertura). Asimétrico por modelo: **mistral 2,3% ·
granite 8,0% · qwen3.5 20,3% · gemma 23,0%** → **confound no documentado** para la comparación
ENTRE modelos (gemma/qwen formatean más en markdown). Plano entre escenarios (lex 11,1 / den 10,5 /
hib 10,7) → NO amenaza el hallazgo central. En la muestra etiquetada, **6/8 artefactos → contradicted**
(inflan alucinación, deflan fidelidad). **Fix (Q2=ambos):** `classify_artifact()` etiqueta los
artefactos `not_a_claim`; quedan en `claim_details`/`total_claims` para trazabilidad pero **fuera del
denominador** (`supported/(total−not_a_claim)`). Runtime corregido para el futuro + re-score `_v3`
para los datos firmados. Mecanismo: Wanner et al. 2024 (filtrar subclaims incoherentes sube la
fidelidad medida).

### H2 — agregación NLI sin guarda simétrica (CONFIRMADO en código)
El lado contradicted carece de la guarda que Flag 138 dio al supported (max sobre 5 chunks, umbral
0,7, sin margen ni acuerdo). **Decisión Q3=añadir corrección + re-score v3.** Extraída la lógica a
`decide_nli_status()` (compartida runtime/offline) con variantes: `v0` (legado), `va_margin`
(contr>ent+δ), `vb_agree` (≥2 chunks). **SUB-GATE:** `scripts/_h2_variant_eval.py` (base-rate
sintético + re-score de muestra + q085) produce el trade-off; **Enzo elige la variante** antes del
re-score definitivo. Base-rate de falso-contradicted aún por medir (su máquina).

### H2b — q085 es también un fallo de RELEVANCIA (CONFIRMADO, matiza H2)
Pregunta "configurar networking de Azure VNet" → mejor chunk recuperado = "Azure **Functions**
private site access". Las 28 "contradicciones" son en parte retrieval irrelevante para preguntas
procedimentales multi-paso (modelo respondió de memoria; chunk no cubre los pasos) + etiqueta
inflada (debió ser `unsupported`). Corpus congelado → limitación documentada; la corrección H2
(unsupported vs contradicted) la mitiga parcialmente.

### H3 — auditoría humana sin completar (CONFIRMADO)
`output/audit/claim_audit_sample.csv`: `juicio_humano` 0/50. Pendiente de Enzo. Se regenera limpia
con `build_claim_audit_sample.py --out-suffix _v3` (la corrección excluye artefactos de los estratos)
+ columna `pre_clasificacion`.

### H4 — discrepancia entre verificadores (CONFIRMADO, re-chequear post-fix)
`rescore_v2_summary.md`: kappa 0,411 (base vs small), orden léx/den/híb cambia 3/4 modelos,
Spearman primaria 0,559 n.s. El re-score v3 corre AMBOS verificadores para ver si el fix H1/H2
reduce la discrepancia. Migración a verificador de premisa larga (MiniCheck/AlignScore;
Tang 2024 / Zha 2023; cf. Schuster 2022 sobre premisas largas) = trabajo futuro.

### H5 — varianza multi-seed de gemma/mistral NO cuantificada (CONFIRMADO → limitación)
Solo existe `nli_determinism_check.json` (determinismo del NLI, no del LLM). **Decisión Q4=documentar**
como limitación esta ronda; headline = granite (determinista). Sin generación nueva.

### H6 — boilerplate de scraping en el texto indexado (CONFIRMADO, más grave de lo creído → limitación)
`"Stay organized with collections…"` (UI de docs de GCP) en **8.144/24.481 chunks (33%)**, todo GCP,
y **en el campo `text`** (no solo metadata) → contamina BM25 + denso + el contexto que ve el LLM
(ej. claim citando `[Source: gcp/GKE/GKE and Cloud RunStay…]`). Corpus congelado (rebuild invalidaría
exp10-13) → **limitación de curación documentada**; filtro propuesto para trabajo futuro (no ejecutado).

### Parte 2 (asesor) — trazabilidad
README cableaba modelos viejos (llama3.1/mistral/qwen2.5) sin nota demo-vs-evaluado en 3 lugares
(Architecture, Configuration, Quick Start); MODELS.md ya era correcto. Corregido + nota de corpus
experimental (2.697/24.481) + nuevo `docs/TRACEABILITY_nota3.md` (matriz verificada + recetas).

### Parte 3 — orfandad / silent-zero
- `compute_retrieval_metrics.py` hardcodeaba el caption "exp8, 200 queries" → parametrizado.
- `_latency_p50p95.py` inyectaba 0,0 ms ante claves faltantes (deflaba el p50/p95) → solo claves presentes.
- `experiments_page.py` "Full (200 queries)" → 194.
- **D11/D12/expansion-OFF verificados intactos** (no se revirtieron desde el 06-11).
- **`paper/overleaf_ready/main.tex` SIGUE con cifras retiradas** (P@1 0,930, "200-query", "16,8%" en
  L64-69,200,250) + tablas `0.930` viejas en `output/tables/` y `paper/overleaf_ready/figures/`.
  **FLAGEADO, NO reescrito** — es prosa del paper (se corrige aparte, fuera de esta ronda de repo).

### Impacto en cifras (Fase 3 REAL ejecutada, 2026-06-30 GPU)

Re-score v3 corrido en `py -V:3.14` (torch+cu126, RTX 3060). **Verificador = base**
(`nli-deberta-v3-base` local); el small del runtime NO está disponible offline (TLS bloqueado,
N5) → las cifras absolutas v3 NO son directamente comparables a la Tabla 6 publicada (que usó
small). Comparación controlada de 3 vías (`faithfulness_answered` primaria, BH por familia):

| | verificador | artefactos | retrieval RAG-vs-RAG | entre-modelos sig_bh |
|---|---|---|---|---|
| (A) **v3** | base | **excluidos** | n.s. (0) | **6/18** |
| (B) publicado v2 | small | incluidos | n.s. (0) | 0/18 |
| (C) N5-base | base | incluidos | n.s. (0) | 0/18 |

- (B)vs(C): mismo artefactos, distinto verificador → ambos 0/18 ⇒ el cambio small→base **NO**
  crea significancia entre-modelos.
- (C)vs(A): mismo verificador base, artefactos in→out → 0→6/18 ⇒ **la exclusión de artefactos (H1)
  es lo que vuelve significativas las diferencias entre-modelos.** Confound H1 CONFIRMADO con
  comparación controlada.

**Veredictos (números, no predicción):**
1. **"El método de retrieval no mueve la fidelidad" = ROBUSTO.** Ningún par RAG-vs-RAG significativo
   bajo los 3 instrumentos (small-publicado, base-con-artefactos, base-v3-sin-artefactos). La
   contribución central de la tesis se sostiene.
2. **"Entre-modelos TODO n.s." (afirmado en N5) NO sobrevive a la corrección H1.** Bajo v3 hay
   **6/18 pares significativos** (mayormente qwen/gemma, los artefacto-pesados): denso granite-vs-qwen
   d_z=+0,60 p_bh=0,008; léxico gemma-vs-granite −0,61 / gemma-vs-mistral −0,53 / granite-vs-qwen +0,49
   / mistral-vs-qwen +0,50. El contaminante asimétrico (gemma 23% / qwen 20% / mistral 2%) **enmascaraba**
   diferencias reales de fidelidad entre modelos. **Esto corrige un claim publicado (N5).**
3. **Niveles absolutos v3 (base):** granite lex/den/hib 0,192/0,204/0,210 · gemma 0,440/0,320/0,328 ·
   mistral 0,179/0,198/0,174 · qwen 0,359/0,374/0,333. Los artefacto-pesados (qwen/gemma) **suben**
   respecto a N5-base (qwen lex 0,192→0,359; gemma lex 0,367→0,440) al sacar los artefactos
   (que caían a contradicted/unsupported). H2 (vb_agree) NO mueve estos números (fidelidad
   H2-invariante: solo reordena contradicted↔unsupported); H2 corrige la **banda contradicted**
   (eval: 62%→17,5% falso-contradicted vs chunks aleatorios) y **q085** (26→5 contradicted).

**Pendiente para cifras comparables a la Tabla 6 publicada:** restaurar el verificador small offline
(curl --ssl-no-revoke, workaround N5) y re-correr `rescore_nli_v3.py --verifier small --variant vb_agree`.
El mecanismo (exclusión de artefactos) es verificador-independiente, así que se espera el mismo flip
entre-modelos bajo small; falta confirmarlo con números.

**Postura (Q1=ambas):** `_v3` corregido (base) como evidencia de la corrección; publicado v2 (small)
conservado como superseded documentado. La reescritura de A.3/`main.tex` con cifras v3 queda gated
tras el reporte a Enzo (con la decisión small-vs-base para las cifras citables).

Artefactos v3: `experiments/results/exp12_matrix/faithfulness_rescore_v3__base__vb_agree.json`,
`faithfulness_metrics_v3.json`, `output/audit/h2_variant_eval.json`.

### v3-small — verificador del PAPER (small restaurado offline), DEFINITIVO para citar

small restaurado vía `curl --ssl-no-revoke` a `data/models/nli-deberta-v3-small/` (loads offline,
id2label={0:contradiction,1:entailment,2:neutral} = base). rescore v3-small (vb_agree,
contradicted=1696 ⇒ NLI real) → `faithfulness_metrics_v3_small.json`. Comparación con el MISMO
verificador (small) que usó la Tabla 6 publicada:

| | retrieval RAG-vs-RAG | entre-modelos sig_bh |
|---|---|---|
| publicado v2-small (artefactos IN) | 0/12 | 0/18 |
| **v3-small (artefactos OUT)** | **0/12** | **2/18** |

**Veredictos comparables al paper (verificador small):**
1. **"El retrieval no mueve la fidelidad" = ROBUSTO** bajo el verificador del propio paper (0/12 en
   v2 y v3). Confirmado ahora bajo **los dos verificadores** (base 0/12 y small 0/12). La contribución
   central de la tesis se sostiene — resultado, no predicción.
2. **"Entre-modelos todo n.s." (afirmado en N5) NO se sostiene** ni bajo small: **2/18** pares
   significativos (denso granite-vs-mistral d_z=+0,42 p_bh=0,014 n=75; léxico gemma-vs-mistral
   d_z=−0,45 p_bh=0,044 n=49). Menos que bajo base (6/18) pero >0 en ambos. **Corrige el claim
   publicado N5.**
3. **Niveles absolutos (v2-small → v3-small):** granite/mistral casi no cambian (~±0,01);
   **gemma +0,05..+0,07 · qwen +0,06..+0,11** — los artefacto-pesados **suben** al sacar el 20-23%
   de artefactos que inflaban su denominador con contradicciones-basura. La Tabla 6 publicada
   **subestimaba** la fidelidad de gemma y qwen.

Artefactos small: `faithfulness_rescore_v3__small__vb_agree.json`, `faithfulness_metrics_v3_small.json`.

### Código/artefactos de esta ronda (commits 2026-06-30)
- `src/generation/hallucination_detector.py`: `classify_artifact`, `decide_nli_status`,
  denominador `not_a_claim`, campo `not_a_claim_claims`.
- `scripts/`: `_h2_variant_eval.py`, `rescore_nli_v3.py`, `compute_faithfulness_metrics.py`
  (`--faithfulness-source`), `build_claim_audit_sample.py` (`--out-suffix`, `pre_clasificacion`),
  `audit/smoke_extractor_artifacts.py`; fixes Parte 3.
- Docs: README (Parte 2), `docs/TRACEABILITY_nota3.md`.
- **Sin tocar** evidencia firmada exp1-13; nuevos resultados → archivos `_v3`.
- Nota CRLF: el árbol estaba limpio (la premisa de "231 archivos" no reprodujo); `autocrlf=true`
  sin `.gitattributes` → recurrencia posible (decisión de política pendiente de Enzo).

### Estado de decisiones de Enzo (2026-06-30/07-01)
**Resueltas:**
- Variante H2 = **vb_agree** (elegida tras el trade-off del sub-gate `_h2_variant_eval.py`).
- Re-score corrido con base **y** small: `faithfulness_metrics_v3.json` (base) y
  `faithfulness_metrics_v3_small.json` (small, comparable con la Tabla 6 publicada).
- `.gitattributes` = **aplicado** (`* text=auto eol=lf`).
- Rama (H7) = **seguir en `main`**; `pre-corpus-rebuild-2026-05-21` congelada (sin renombrar ni fusionar).
- Timing reescritura A.3/`main.tex` = **hay margen, proceder** (cifras v3 confirmadas); la prosa del
  paper sigue gated hasta OK explícito frase-por-frase.

**Genuinamente pendiente (lado Enzo):**
- Revisar a mano `output/audit/claim_audit_sample_v3.csv` (`juicio_humano` 0/50) — único ítem abierto.

**Gated (espera OK explícito, no es trabajo de Enzo):** `git push` de los commits N8 (nada pusheado).

---
