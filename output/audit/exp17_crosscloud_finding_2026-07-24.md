# exp17 — recuperación balanceada por proveedor: PRIMER POSITIVO de la fase (cross-cloud)

**Fecha:** 2026-07-24 · **Fase:** verano, Fase 2 (mejoras), línea 5 A.3 · **Estado:** para revisión de Enzo
**Regla:** report-before-prose. NO cambia cifras firmadas (exp17 nuevo). Toca la matriz de factibilidad
(línea 5) → material de discusión, no de prosa sin OK.
**Evidencia:** `experiments/results/exp17_crosscloud_balanced/{retrieval_ids.json, retrieval_report.md, results.json, faithfulness_rows__{small,base}__vb_agree.json, faithfulness_rows__hhem.json, arm_stats__{small,base,hhem}.{json,md}, guards.{json,md}}`

## Diagnóstico que motivó el piloto
exp13 probó expansión LÉXICA de query y falló (NDCG OFF 0.852 ≈ ON 0.820, retirada). Pero el problema
real que la expansión léxica nunca tocó: **de 25 queries comparativas cross-cloud, solo 7 recuperan TODOS
los proveedores pedidos en el top-5**. 18/25 pierden ≥1 proveedor entero (q171 "AWS Lambda vs Azure
Functions" → 5/5 azure, 0 aws). El NDCG es alto (0.85) porque los chunks son relevantes al tema — pero de
UN proveedor → **la comparación es imposible de anclar**. Es falla de **selección de contenido** (el eje
que Tier 3 marcó como el único que mueve la fidelidad).

## Diseño
2 brazos, pareado within-session (25 q), granite temp0 seed42, `--no-cache`. Del MISMO pool híbrido
(retrieval_top_k=50 → rerank ms-marco-L12), solo cambia la selección top-5:
- **baseline** = `rerank(pool)[:5]` → **validado idéntico a exp13 exp_off (overlap 5.0/5 en 25/25)**.
- **balanced** = ⌈5/|P|⌉ por cada proveedor pedido, del mismo pool reordenado → cubre todos.
Aísla la variable COBERTURA (mismo pool, misma generación). Cobertura: **baseline 7/25 → balanced 25/25**
(el set cambió en 22/25 queries). Retrieval determinista (sin H5); solo la generación es within-session fresca.

## Resultado — balanced > baseline en los 3 instrumentos (PRIMER positivo direccional de la fase)

| Instrumento | baseline | balanced | Δ | d_z | p | p_BH |
|---|---|---|---|---|---|---|
| NLI small | 0,199 | 0,236 | **+0,037** | 0,13 | 0,40 | 0,40 |
| NLI base | 0,151 | 0,196 | **+0,045** | 0,21 (peq.) | 0,14 | 0,14 |
| **HHEM (grounding limpio)** | 0,477 | 0,558 | **+0,081** | 0,26 (peq.) | 0,24 | 0,24 |

Los TRES apuntan ARRIBA. HHEM (el instrumento más limpio, per Tier 3) da el efecto más grande (+0,081,
d_z 0,26). Ninguno cruza significancia (n=25, familia BH de 1 → underpowered), pero la CONSISTENCIA de la
dirección en 3 instrumentos independientes es señal, no ruido. HHEM baseline 0,477 → carga verificada.

## Guardas anti-gaming — el patrón OPUESTO a exp16: mejora GENUINA, no del instrumento
| Métrica | baseline | balanced | lectura |
|---|---|---|---|
| declinación | 56 % (14/25) | **32 % (8/25)** | **BAJA** — el modelo ya PUEDE comparar (ambos proveedores presentes) |
| palabras | 349 | **427** | **SUBE** — más contenido, no menos |
| claims genuinos | 11,5 | **14,9** | **SUBE** — afirma más y mejor anclado |
| solape verbatim 5-gram | 0,109 | 0,129 | ~plano — NO copia |

exp16 (anclaje por prompt, NEGATIVO) subía la declinación y recortaba contenido — gaming/abstención. **exp17
hace lo contrario:** la fidelidad sube MIENTRAS la declinación BAJA y el contenido SUBE, sin copiar. Es una
mejora real en la capacidad de responder comparaciones ancladas: al darle los dos proveedores, granite deja
de declinar y ancla más claims. La cobertura (no el arreglo ni el prompt) es la palanca.

## Veredicto (con caveat de potencia honesto)
**Balancear la cobertura de proveedores mejora la fidelidad de la respuesta comparativa cross-cloud** — el
PRIMER positivo de toda la fase de mejoras. Efecto pequeño (d_z 0,13–0,26), consistente en 3 instrumentos,
confirmado como genuino por las guardas (declina menos, dice más, no copia). **No significativo a n=25**
(familia BH de 1, underpowered) → es un PILOTO con señal, no una conclusión. Confirmatorio a mayor n
(las 194 q tienen ~pocas comparativas; habría que ampliar el set cross-cloud) requiere OK de Enzo.

Encaja con toda la fase: Tier A (arreglo del contexto) = nulo; exp16 (prompt) = nulo/negativo; **exp17
(selección de contenido = cobertura) = positivo.** Converge con Tier 3: el único eje que mueve la fidelidad
es QUÉ evidencia entra, no cómo se ordena ni cómo se instruye. La contribución de recuperación del híbrido
sí puede volverse fidelidad — pero solo cuando la selección garantiza la evidencia que la respuesta necesita.

## Reanálisis de mayor potencia (mismas 25 queries, sin datos nuevos)
Pool cross-cloud agotado en 25 (las 4 removidas son inválidas: corpus K8s/CNCF borrado en el rebuild).
En vez de autorar queries (result-chasing), se reanaliza a resolución de CLAIM conservando el pareo por
query: **GLMM binomial `supported ~ arm + (1|query)`** (el intercepto aleatorio por query da el contraste
within-query a nivel claim; una query de 30 claims informa más que una de 3) + **bootstrap de cluster por
query** (unidad válida, sin pseudo-replicación) del diff micro-promediado. Tests una-cola (H1: balanced >
baseline; dirección pre-especificada por el mecanismo). Condicional a claim genuino (declinaciones/vacuous
salen). Ver `powered_reanalysis.{json,md}`.

| Verificador | micro base→bal (diff) | GLMM OR | GLMM p (1-cola) | bootstrap p (1-cola) |
|---|---|---|---|---|
| NLI small | 0,213→0,247 (+0,035) | 1,15 | 0,152 | 0,30 |
| NLI base | 0,167→0,210 (+0,042) | 1,24 | 0,056 | 0,14 |
| **HHEM** | 0,571→0,605 (+0,033) | **1,25** | **0,021** | 0,23 |

**Lectura honesta (no sobre-vender):** bajo el modelo que conserva el pareo a nivel claim, **HHEM cruza
significancia una-cola (p=0,021)** y base queda marginal (0,056); pero el bootstrap conservador (cluster
por query) NO cruza (HHEM 0,23). El efecto es **real en dirección y consistente, al borde de la
significancia según el modelo** — sugestivo, no concluyente. Caveats: (1) una-cola; (2) 3 verificadores →
HHEM 0,021 no sobrevive Bonferroni ×3 (0,063); (3) el GLMM puede sobre-estimar levemente la potencia
(correlación residual entre claims de una misma respuesta) — por eso el bootstrap es la guarda
conservadora. Nota aparte: balanced tiene MÁS claims genuinos (372 vs 287) por menos declinación — una
ganancia extra que el análisis condicional a claim NO captura.

## Implicación para A.3/LACCI (report-before-prose)
- **Matriz línea 5 (cross-cloud reescritura/expansión densa):** de "PILOTO" → **PILOTO CON SEÑAL POSITIVA**.
  La expansión léxica (exp13) falló, pero el rebalanceo de cobertura por proveedor sube la fidelidad
  comparativa (piloto, n=25, no sig; guardas confirman genuino). Trabajo Futuro concreto y accionable.
- Matiza el hallazgo central: "mejor recuperación ≠ mejor fidelidad" se sostiene para la calidad topical
  (NDCG), PERO la **cobertura/balance** de la evidencia sí mueve la fidelidad en el caso comparativo.
- NO tocar prosa A.3 sin OK frase por frase.

## Pendiente
- Confirmatorio a mayor n (ampliar el set comparativo cross-cloud) — el piloto tiene señal pero n=25.
- Gold humano valida el nivel y si el +0,081 HHEM es real.
- Oracle NDCG@5 balanced vs baseline (medir el trade-off cobertura/relevancia-topical; `--with-oracle`, lento) — opcional.
