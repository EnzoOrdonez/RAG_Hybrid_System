# SUMMER_RESULTS — Informe de hallazgos de la fase de verano

**Estado: EN CURSO** (arranque 2026-07-22). Este documento acumula los resultados de la ablación
(Fase 1), el diagnóstico (Fase 1b) y las mejoras (Fase 2). Ledger de decisiones:
`paper/summer_ablation_log.md`. Línea base: tag `summer-baseline` (cifras v4 verificadas,
`output/audit/phase0_verification_summer_2026-07-22.md`).

## Pregunta central
¿Por qué una mejor recuperación (NDCG@5 0.740 híbrido vs 0.442 léxico, oráculo independiente)
NO mejora la fidelidad de la respuesta (0/12 pares RAG-vs-RAG significativos; Granite
0.235/0.247/0.299)? ¿Instrumento, generación, contexto, o techo real?

## Respuesta (2026-07-23 — sujeta a validación con gold humano)
**El nulo 0/12 NO es robusto al instrumento: un verificador de grounding limpio (HHEM) revela un efecto
retrieval→fidelidad para granite que el NLI ruidoso enmascara.** Test pareado between-scenario, familia
BH v4-consistente (24, incl sin_rag):
- NLI small: **0/12** (granite hib-vs-lex p_bh 0.085). NLI base: **0/12**.
- **HHEM: 1/12** — **granite hibrido-vs-lexico p_bh 0.020, d_z −0.35, SIGNIFICATIVO** (mistral 0.067,
  cerca). El efecto híbrido>léxico existe para el modelo determinista pero solo se detecta con un
  instrumento menos ruidoso.
- **Nivel instrument-relative:** HHEM (especificidad buena: falso-grounded 0.033) da +0.307 sobre NLI
  (granite 0.40-0.44 vs 0.23-0.30). El NLI marca **22% de texto ALEATORIO** como contradicted → baja el
  nivel Y enmascara el contraste. El "0.30" publicado es relativo al instrumento NLI.
**Correcciones de rigor (3 iteraciones):** entrada 6 ("todo artefacto NLI, HHEM 0.99") sobre-vendió
(bug de carga HHEM); entrada 8 ("0/12 robusto, sin efecto") sub-vendió (bug de familia BH excluyó
sin_rag). La verdad está en medio: NLI enmascara un efecto real pequeño granite-específico que HHEM
revela. Ver `hhem_vs_nli.md` + `tier3_negative_control_finding` (ledger 6,7,8,9). Report-before-prose:
el 0/12 depende del instrumento → material de discusión; NO cambia cifras firmadas.

## Línea base v4 (referencia congelada)
| Métrica | Valor | Fuente |
|---|---|---|
| NDCG@5 híbrido (oráculo indep.) | 0.7405 | exp11 |
| NDCG@5 híbrido (circular, no citar como real) | 0.9948 | exp11 |
| Fidelidad Granite lexico/denso/hibrido | 0.235(75)/0.247(85)/0.299(87) | exp12 v4_small |
| RAG-vs-RAG significativos | 0/12 | exp12 v4 ambos verificadores |
| Entre-modelos robusto | 0/18 | exp12 v4 small∩base |
| Expansión cross-cloud (25 q) | OFF≈ON (retirada) | exp13 |

## Tabla de ablación Tier A (2026-07-24, 5 brazos × 60q, exp15_ablation_tierA)
Contexto = ids firmados exp11 híbrido full-rerank, **transformados sin re-recuperar**. granite temp0 seed42.
Contraste pareado within-session vs `baseline_repro` (Wilcoxon+d_z+bootstrap seed42, familia BH de 4).
Fidelidad = rescore vb_agree τ0.7 (NLI) / max_chunk τ0.5 (HHEM). Ver `arm_stats__{small,base,hhem}.md`.

| Brazo | Transform | Fid. NLI small | NLI base | HHEM | Sig BH (los 3) | Veredicto |
|---|---|---|---|---|---|---|
| baseline_repro | identidad | 0.308 | 0.204 | 0.450 | ancla | reproduce exp12 (drift +0.03 n.s.) |
| reranker_off | RRF pre-rerank | 0.243 | 0.144 | 0.438 | 0/3 | reranking NO sube fidelidad |
| final_top_k_3 | top-3 | 0.282 | 0.222 | 0.438 | 0/3 | recorte de contexto NO mueve |
| context_reversed | orden invertido | 0.315 | 0.214 | 0.469 | 0/3 | orden NO importa |
| context_lost_middle | relevante al centro | 0.332 | 0.202 | 0.424 | 0/3 | **lost-in-the-middle NO** |

**0/4 brazos significativos bajo NLI-small, NLI-base Y HHEM → el nulo de ablación de contexto es ROBUSTO
al instrumento** (a diferencia del 0/12 entre-escenarios, que HHEM sí rompe a 1/12). Mecanística: la
fidelidad responde débilmente a QUÉ documentos elige el *método* (híbrido vs léxico), NO a cómo se arregla
un pool ya recuperado (rerank/top-k/orden/posición). Deriva H5 baseline_repro vs junio: +0.033 (n.s.
p=0.083), r=0.86, 39% queries idénticas → valida gate relajado y diseño pareado within-session.

## Diagnóstico (Fase 1b) — hipótesis y estado
| Hipótesis | Estado | Evidencia |
|---|---|---|
| Instrumento NLI ruidoso/descalibrado (¿0/12 artefacto del punto de operación?) | **Tier 0 + Tier 3-A COMPLETOS**: (Tier 0) κ 0.30–0.36; nulo robusto bajo base (0/64); bajo small granite hib-vs-lex sig con ent≤0.6, consistente 128/128. (Tier 3-A) small sobre-contradice 1.8× y es 2.5× más frágil al umbral → **el verificador runtime es el ruidoso**; 128 falso-contradicted; agregador `max` sub-acredita evidencia distribuida (noisy_or→granite hib>lex p_bh 0.009). Control negativo: NLI marca **22% de texto aleatorio como contradicted**. → parte del 0.30 es instrumental; **cuánto** pendiente de HHEM (bug de carga corregido, re-corriendo) + gold | `exp15_ablation_nli/{sweep,disagreement,negative_control}_*`; ledger 2,4,6,7 |
| Generación no ancla en la evidencia | **parcial (Tier A)**: degradar el contexto (rerank off, orden, posición) no baja la fidelidad → el anclaje no depende del arreglo del pool; el cuello está en generación/selección de contenido, no en presentación | `arm_stats__*` |
| Lost in the middle | **DESCARTADO (Tier A)**: context_lost_middle vs baseline 0/3 instrumentos (small +0.027, base −0.005, HHEM −0.025, todos n.s.) | `arm_stats__{small,base,hhem}.md` |
| Corte de contexto / nº fragmentos | **parcial (Tier A)**: final_top_k_3 (5→3) 0/3 n.s.; recorte de fragmentos no mueve la fidelidad a este tamaño | `arm_stats__*` |
| Reranking sube fidelidad | **DESCARTADO (Tier A)**: reranker_off 0/3, incluso tiende abajo (−0.05..−0.06 NLI, n.s.) | `arm_stats__*` |
| Declinación confunde la métrica | parcialmente tratado en v2/v4 | denominadores decline-aware |

## Matriz de factibilidad — Trabajos Futuros del A.3 (PRELIMINAR, 2026-07-23)
Viabilidad en esta laptop (RTX 3060 6 GB) antes de las encuestas. Se cierra al terminar Tier A/3.

| Línea (A.3) | Viable aquí | Costo | Payoff esperado | Veredicto preliminar |
|---|---|---|---|---|
| **1a. Decodificación anclada** (citar/atribuir evidencia, temperatura, prompt) | **Sí** | Bajo (infra lista) | **Nula (probado)** | **IMPLEMENTADA Y PROBADA 2026-07-24 (exp16) — SIN ganancia local.** anchored_cite + strict_abstain: 0/2 bajo NLI-small/base/HHEM; anchored tiende ABAJO (cita≠grounding), ambos suben declinación y recortan contenido. Descarta la línea como victoria local |
| **1b. Modelo de mayor capacidad** | **No en 6 GB** | — | Alto pero incuantificable local | **DISEÑO/NUBE** — granite@4096 ya no cabe 100% GPU (hallazgo); ≥13B exige otra máquina/nube. Reportar trade-off |
| **2. Anotación humana (relevancia + gold)** | **Parcial** (diseño sí, ejecución no) | ~4-5 h humano | Alto (rompe circularidad, arbitra instrumento) | **ENTREGADO EL DISEÑO** — `claim_audit_sample_v4` N≈200 listo; ejecuta Enzo/anotadores |
| **3. Verificador de fidelidad estable (Tier 3)** | **Sí** | Bajo-medio (CPU + descargas hechas) | **Alto** (κ 0.32; NLI 22% falso-contradicted) | **EN CURSO** — control negativo + ensembles + HHEM (corregido); selección espera gold |
| **4. Ablación de componentes (Tier A/B)** | **Sí** | Medio (GPU, gate determinismo relajado) | Alto (aísla qué mueve la fidelidad) | **Tier A HECHO 2026-07-24** — 0/4 robusto (rerank/top-k/orden/lost-middle no mueven fidelidad en 3 instrumentos); descarta lost-in-the-middle y reranking. Tier 0 hecho; Tier B oráculo listo |
| **5. Cross-cloud: reescritura/expansión densa** | **Sí (piloto)** | Bajo (25 q) | Medio (la inyección léxica falló, exp13) | **PILOTO** — exp16 sobre `cross_cloud_subset` |
| **6. Memoria semántica (tripletes/KG/versionada)** | **No (verano)** | Alto | Incierto | **SOLO DISEÑO** — excede el verano; entregar veredicto de factibilidad |

Nota clave (hallazgo que ata 1b + corte de contexto): granite@4096 es simultáneamente el techo que truncó
exp12 (input máx=4096 exacto) Y el máximo que casi-no-cabe en 6 GB → subir contexto O modelo exige salir
de esta laptop. Esto acota fuertemente qué "generación más fiel" es implementable localmente.

## Mejoras (Fase 2)
### exp16 — decodificación anclada (2026-07-24): RESULTADO NEGATIVO triangulado
3 brazos de prompt sobre el mismo pool híbrido (solo cambia system+sufijo), pareado within-session vs
baseline_repro fresco (`--no-cache`, co-temporal). Ver `exp16_anchored_finding_2026-07-24.md`, ledger 11.

| Brazo | Δ small | Δ base | Δ HHEM | Sig (3 instr.) |
|---|---|---|---|---|
| anchored_cite (cita [N] por claim) | −0.034 | −0.040 | −0.056 | 0/3 — tiende ABAJO |
| strict_abstain (omitir lo no explícito) | −0.002 | +0.031 | +0.038 | 0/3 — plano |

**La decodificación anclada NO mejora la fidelidad** (0/2 bajo NLI-small/base/HHEM). Guardas anti-gaming:
ambos brazos suben la declinación (51.7%→58/60%) y recortan contenido (palabras 352→223/191, claims
11.95→7.55/5.27); anchored_cite baja el solape (0.061, no copia) pero igual baja la fidelidad → **cita ≠
grounding**. Junto con Tier A (nulo de recuperación): ni contexto ni prompt mueven la fidelidad → techo de
capacidad del modelo (1b, fuera de 6GB) o instrumento (Tier 3, gold pendiente). Caveat: declinación
baseline 51.7% → n efectivo ≈29, underpowered; dirección + guardas argumentan contra un positivo oculto.

(Candidatas restantes: verificador estable [3, espera gold], piloto cross-cloud denso [5].)

## Configuración recomendada para SUS/Likert
(pendiente — cierre de Fase 3)
