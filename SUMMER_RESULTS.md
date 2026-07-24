# SUMMER_RESULTS — Informe de hallazgos de la fase de verano

**Estado: EN CURSO** (arranque 2026-07-22). Este documento acumula los resultados de la ablación
(Fase 1), el diagnóstico (Fase 1b) y las mejoras (Fase 2). Ledger de decisiones:
`paper/summer_ablation_log.md`. Línea base: tag `summer-baseline` (cifras v4 verificadas,
`output/audit/phase0_verification_summer_2026-07-22.md`).

## Pregunta central
¿Por qué una mejor recuperación (NDCG@5 0.740 híbrido vs 0.442 léxico, oráculo independiente)
NO mejora la fidelidad de la respuesta (0/12 pares RAG-vs-RAG significativos; Granite
0.235/0.247/0.299)? ¿Instrumento, generación, contexto, o techo real?

## Respuesta emergente (2026-07-23, PARCIAL — sujeta a HHEM re-corrido + gold humano)
**Evidencia válida:** los verificadores NLI marcan ~22% de texto NO relacionado como "contradicted"
(control negativo, small 0.237, base 0.215) → parte del ruido de fidelidad son contradicciones
inventadas por el instrumento. NO cuantifica cuánto del 0.30 es artefacto.
**CORRECCIÓN (rigor):** una versión previa afirmó "HHEM ve las respuestas grounded ≈0.99 → baja
fidelidad = artefacto NLI"; era ERRÓNEA (HHEM mal cargado, pesos `t5.` no aplicados → basura;
datos reales daban 0.038, no 0.99). Fix aplicado + test de cordura. HHEM re-corriendo con carga
correcta. Ver `output/audit/tier3_negative_control_finding_2026-07-23.md` (ledger entrada 7).
NO cambia cifras publicadas (report-before-prose).

## Línea base v4 (referencia congelada)
| Métrica | Valor | Fuente |
|---|---|---|
| NDCG@5 híbrido (oráculo indep.) | 0.7405 | exp11 |
| NDCG@5 híbrido (circular, no citar como real) | 0.9948 | exp11 |
| Fidelidad Granite lexico/denso/hibrido | 0.235(75)/0.247(85)/0.299(87) | exp12 v4_small |
| RAG-vs-RAG significativos | 0/12 | exp12 v4 ambos verificadores |
| Entre-modelos robusto | 0/18 | exp12 v4 small∩base |
| Expansión cross-cloud (25 q) | OFF≈ON (retirada) | exp13 |

## Tabla de ablación (se llena por brazo)
| Tier | Brazo | Retrieval (NDCG@5 indep) | Fidelidad (primary, NLI small) | n | Sig (BH) | Veredicto |
|---|---|---|---|---|---|---|
| — | baseline_repro | pendiente | pendiente | — | — | ancla |

## Diagnóstico (Fase 1b) — hipótesis y estado
| Hipótesis | Estado | Evidencia |
|---|---|---|
| Instrumento NLI ruidoso/descalibrado (¿0/12 artefacto del punto de operación?) | **Tier 0 + Tier 3-A COMPLETOS**: (Tier 0) κ 0.30–0.36; nulo robusto bajo base (0/64); bajo small granite hib-vs-lex sig con ent≤0.6, consistente 128/128. (Tier 3-A) small sobre-contradice 1.8× y es 2.5× más frágil al umbral → **el verificador runtime es el ruidoso**; 128 falso-contradicted; agregador `max` sub-acredita evidencia distribuida (noisy_or→granite hib>lex p_bh 0.009). Control negativo: NLI marca **22% de texto aleatorio como contradicted**. → parte del 0.30 es instrumental; **cuánto** pendiente de HHEM (bug de carga corregido, re-corriendo) + gold | `exp15_ablation_nli/{sweep,disagreement,negative_control}_*`; ledger 2,4,6,7 |
| Generación no ancla en la evidencia | pendiente | — |
| Lost in the middle | pendiente (Tier A) | — |
| Corte de contexto / nº fragmentos | pendiente (Tier A/B) | — |
| Declinación confunde la métrica | parcialmente tratado en v2/v4 | denominadores decline-aware |

## Matriz de factibilidad — Trabajos Futuros del A.3 (PRELIMINAR, 2026-07-23)
Viabilidad en esta laptop (RTX 3060 6 GB) antes de las encuestas. Se cierra al terminar Tier A/3.

| Línea (A.3) | Viable aquí | Costo | Payoff esperado | Veredicto preliminar |
|---|---|---|---|---|
| **1a. Decodificación anclada** (citar/atribuir evidencia, temperatura, prompt) | **Sí** | Bajo (infra Tier A lista) | Medio-alto si el techo NO es puro instrumento | **IMPLEMENTAR** — brazos de prompt/decoding en exp16 |
| **1b. Modelo de mayor capacidad** | **No en 6 GB** | — | Alto pero incuantificable local | **DISEÑO/NUBE** — granite@4096 ya no cabe 100% GPU (hallazgo); ≥13B exige otra máquina/nube. Reportar trade-off |
| **2. Anotación humana (relevancia + gold)** | **Parcial** (diseño sí, ejecución no) | ~4-5 h humano | Alto (rompe circularidad, arbitra instrumento) | **ENTREGADO EL DISEÑO** — `claim_audit_sample_v4` N≈200 listo; ejecuta Enzo/anotadores |
| **3. Verificador de fidelidad estable (Tier 3)** | **Sí** | Bajo-medio (CPU + descargas hechas) | **Alto** (κ 0.32; NLI 22% falso-contradicted) | **EN CURSO** — control negativo + ensembles + HHEM (corregido); selección espera gold |
| **4. Ablación de componentes (Tier A/B)** | **Sí** | Medio (GPU, gate determinismo relajado) | Alto (aísla qué mueve la fidelidad) | **EN CURSO** — Tier 0 hecho; Tier A pendiente re-corrida; Tier B oráculo listo |
| **5. Cross-cloud: reescritura/expansión densa** | **Sí (piloto)** | Bajo (25 q) | Medio (la inyección léxica falló, exp13) | **PILOTO** — exp16 sobre `cross_cloud_subset` |
| **6. Memoria semántica (tripletes/KG/versionada)** | **No (verano)** | Alto | Incierto | **SOLO DISEÑO** — excede el verano; entregar veredicto de factibilidad |

Nota clave (hallazgo que ata 1b + corte de contexto): granite@4096 es simultáneamente el techo que truncó
exp12 (input máx=4096 exacto) Y el máximo que casi-no-cabe en 6 GB → subir contexto O modelo exige salir
de esta laptop. Esto acota fuertemente qué "generación más fiel" es implementable localmente.

## Mejoras (Fase 2)
(pendiente — se definen tras cerrar Tier A/3; candidatas priorizadas: decodificación anclada [1a],
verificador estable [3], piloto cross-cloud denso [5])

## Configuración recomendada para SUS/Likert
(pendiente — cierre de Fase 3)
