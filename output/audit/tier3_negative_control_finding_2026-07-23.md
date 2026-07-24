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

## RESULTADO DEFINITIVO (HHEM corregido, 12/12 configs) — `hhem_vs_nli.{json,md}`

Con HHEM bien cargado (τ=0,5; especificidad negativa falso-grounded 0,033):

**1. NIVEL — NLI sub-acredita la fidelidad de forma sistemática.** HHEM > NLI-small en los 12 configs,
gap medio **+0,307** (rango +0,14..+0,43): granite 0,40-0,44 (vs NLI 0,23-0,30), mistral 0,49-0,58,
qwen 0,63-0,69, gemma 0,74-0,80. Coherente con el 22 % de falso-contradicted del NLI: **el instrumento
NLI baja el NIVEL absoluto de fidelidad.**

**2. CONTRASTE — el nulo 0/12 es ROBUSTO AL INSTRUMENTO.** El test pareado between-scenario bajo HHEM
(misma metodología v4: Wilcoxon + d_z + bootstrap + BH) da **0/12 RAG-vs-RAG significativos**, igual que
NLI-small y NLI-base. El par más fuerte (granite hibrido-vs-lexico) es direccionalmente consistente
(hib>lex) en TODOS los instrumentos pero nunca cruza BH: NLI-small p_bh 0,085, HHEM p_bh 0,11 — señal
débil sub-potenciada, **ni nula ni significativa**.

**Veredicto:** "mejor recuperación no mejora significativamente la fidelidad (0/12)" **SE SOSTIENE bajo
tres familias de verificador** (NLI small, NLI base, HHEM grounding ortogonal). El ≈0,30 publicado es
**relativo al instrumento** (un verificador de grounding limpio da ≈0,55 medio), pero el **contraste
entre escenarios es genuinamente pequeño/nulo, NO un artefacto de medición.** Esto **corrige mi
afirmación previa errónea** ("todo es artefacto NLI") y a la vez **refuerza** el hallazgo central de la
tesis: es robusto a la elección de instrumento.

## Implicación para A.3/LACCI (report-before-prose)
- El 0/12 gana robustez (3 instrumentos). Material para reforzar, no cambiar, el hallazgo central.
- Candidato a **Limitaciones**: la fidelidad absoluta es relativa al instrumento (NLI ≈0,30 vs HHEM
  ≈0,55); reportar los números como instrument-relative. NO tocar prosa sin OK frase por frase.

## Pendiente
- **Gold humano** (`claim_audit_sample_v4`, N≈200) para arbitrar el NIVEL (¿0,30 NLI o 0,55 HHEM está más
  cerca de la verdad?) y validar HHEM.
- deberta-large (8/12, resumible) como tercer voto NLI.
- **Lección de proceso:** el smoke de 2 queries (0,99) me llevó a una conclusión apresurada; los datos
  completos (0,038 con bug; 0,40 corregido) y el test controlado la corrigieron. Ningún verificador
  nuevo se reporta sin (a) test de cordura del modelo y (b) coherencia smoke-vs-full.
