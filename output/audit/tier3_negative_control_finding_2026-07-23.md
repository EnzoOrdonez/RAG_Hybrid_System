# Tier 3 — Control negativo del verificador NLI + CORRECCIÓN de bug HHEM

**Fecha:** 2026-07-23 · **Fase:** verano, Tier 3 · **Estado:** para revisión de Enzo
**Regla:** report-before-prose. NO cambia cifras firmadas. No tocar prosa sin OK.

## ⚠️ CORRECCIÓN — retracción parcial de una versión previa de este reporte
Una versión anterior (commit `1794f54`) afirmó, con base en un smoke de 2 queries, que **HHEM veía las
respuestas RAG como grounded ≈0,99** y por tanto "la baja fidelidad ≈0,30 es artefacto del instrumento
NLI". **Esa afirmación era ERRÓNEA: HHEM estaba mal cargado** (los pesos del safetensors llevan prefijo
`t5.` y se cargaron en `model.t5` en vez de `model` → `strict=False` descartó todos los pesos → T5 con
pesos aleatorios → scores basura). Detectado por un test controlado: la contradicción "the sky is red"
puntuaba 1,0 y el grounded "the sky is blue" 0,12 (invertido).

**Tras el fix** (`model.load_state_dict(state)`), el test controlado da lo correcto: sky-blue 0,856,
sky-red 0,005, "EKS supports Kubernetes" 0,968, no-relacionado 0,003. **Todos los números HHEM previos
(smoke 0,99; falso-grounded 0,010; fidelidad real 0,038) quedan VOID y se re-corren.** La conclusión
"baja fidelidad = artefacto NLI" **NO está respaldada**; queda pendiente de la re-corrida correcta + gold.

## Lo que SÍ es válido (verificadores NLI deberta — CrossEncoder, cargan bien)
Control negativo de validez de constructo (400 claims × 5 chunks ALEATORIOS no relacionados, seed 42;
reproduce `h2_variant_eval.json`): un verificador válido casi nunca debería etiquetar texto no
relacionado como contradicted.

| Verificador NLI | Falso-contradicted en texto ALEATORIO |
|---|---|
| small (runtime, vb_agree) | **0,237** |
| base (vb_agree) | **0,215** |
| v0 legacy (ambos) | 0,54–0,60 |

**Hallazgo válido:** los verificadores NLI marcan **~22 %** de pares aleatorios no relacionados como
"contradicted" — falso-positivo sistemático de contradicción. Converge con Tier 3-A: small es el
verificador runtime, el más ruidoso (2,5× más frágil al umbral, sobre-contradice 1,8× vs base, 128
falso-contradicted). Esto es evidencia real de que **parte del ruido de fidelidad viene de que el NLI
inventa contradicciones**, pero **por sí solo NO cuantifica cuánto** del 0,30 es artefacto — para eso
hace falta el verificador de grounding bien cargado (HHEM, re-corriendo) y/o el gold humano.

## Estado y siguiente
- HHEM re-corriendo con la carga corregida: control negativo + datos reales → dará la comparación real
  NLI-vs-grounding. Además necesita truncación/batch menor (era 1,9 h/config + OOM en 6 GB).
- Gold humano (`claim_audit_sample_v4`, N≈200) sigue siendo el árbitro objetivo: ¿tienen razón los NLI
  (contradicted) o el grounding? Selección de instrumento anclada en gold + control negativo, nunca en
  el contraste downstream (anti-p-hacking).
- **Lección de proceso:** el smoke de 2 queries dio 0,99 y me llevó a una conclusión apresurada; el dato
  de datos completos (0,038) reveló la inconsistencia y el test controlado localizó el bug. Ningún
  número de verificador nuevo se reporta sin (a) test controlado de cordura del modelo y (b) coherencia
  smoke-vs-full.
