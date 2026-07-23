# SUMMER_RESULTS — Informe de hallazgos de la fase de verano

**Estado: EN CURSO** (arranque 2026-07-22). Este documento acumula los resultados de la ablación
(Fase 1), el diagnóstico (Fase 1b) y las mejoras (Fase 2). Ledger de decisiones:
`paper/summer_ablation_log.md`. Línea base: tag `summer-baseline` (cifras v4 verificadas,
`output/audit/phase0_verification_summer_2026-07-22.md`).

## Pregunta central
¿Por qué una mejor recuperación (NDCG@5 0.740 híbrido vs 0.442 léxico, oráculo independiente)
NO mejora la fidelidad de la respuesta (0/12 pares RAG-vs-RAG significativos; Granite
0.235/0.247/0.299)? ¿Instrumento, generación, contexto, o techo real?

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
| Instrumento NLI ruidoso/descalibrado (¿0/12 artefacto del punto de operación?) | **Tier 0 + Tier 3-A COMPLETOS (2026-07-23)**: (Tier 0) κ 0.30–0.36; nulo robusto bajo base (0/64); bajo small granite hib-vs-lex sig con ent≤0.6, consistente 128/128. (Tier 3-A) small sobre-contradice 1.8× y es 2.5× más frágil al umbral (3.04% vs 1.19% flips) → **el verificador runtime es el ruidoso**; 128 falso-contradicted; **el agregador `max` sub-acredita evidencia distribuida** — `noisy_or` hace granite hib>lex significativo (p_bh 0.009). Dos artefactos de medición convergen → "baja fidelidad" en parte instrumental | `exp15_ablation_nli/sweep_*` + `disagreement_*`; ledger entradas 2, 4 |
| Generación no ancla en la evidencia | pendiente | — |
| Lost in the middle | pendiente (Tier A) | — |
| Corte de contexto / nº fragmentos | pendiente (Tier A/B) | — |
| Declinación confunde la métrica | parcialmente tratado en v2/v4 | denominadores decline-aware |

## Mejoras (Fase 2)
(pendiente — se definen tras la ablación)

## Configuración recomendada para SUS/Likert
(pendiente — cierre de Fase 3)
