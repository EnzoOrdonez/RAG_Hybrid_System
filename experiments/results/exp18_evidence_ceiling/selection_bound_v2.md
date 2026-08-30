# exp18 — cota de seleccion (HHEM tau 0.5, n=188)

**Circular por construccion**: la seleccion se elige usando los claims que la propia respuesta escribio. Son COTAS, no estimaciones de efecto. No entran en la familia BH ni en el TOST.

| | media |
|---|---|
| baseline (su propio top-5) | 0.4552 |
| **alcanzable con k=5** (greedy, suelo=baseline) | **0.5834** |
| **cota superior** (cualquier chunk del pool) | **0.5879** |

Violaciones del bracket: **0** (debe ser 0; baseline <= alcanzable <= cota por construccion).

**Claims que NINGUN chunk del pool soporta:** 759. Su mejor score sobre el pool: media 0.2388, p50 0.2256, p90 0.4385 (tau 0.5). A menos de 0,1 del umbral: **123**.

Si esos scores estan muy por debajo de tau, el claim sencillamente NO esta en la evidencia recuperada y ninguna seleccion podria anclarlo; si se agolpan justo bajo tau, la cota es artefacto del umbral y no de la evidencia.

upper_bound ~ baseline => selection genuinely exhausted (ceiling is capacity or instrument); achievable_k5 >> baseline => headroom exists and is reachable with 5 chunks, so the oracle's null means relevance ranking cannot find it and the missing piece is a grounding-guided selector, which is LOCAL.