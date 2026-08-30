# exp19a — sonda offline del selector (n=188, tau 0.5)

Pregunta: reordenar por `(claim, chunk)` con ms-marco-L12, ¿encuentra los chunks que anclan los claims mejor que el ranking de produccion? **Cero generacion.**

**Sanity check:** solape medio con el top-5 real de exp18 = **5.0/5** (OK).

## Primaria — cobertura de claims a respuesta fija

| seleccion | cobertura de claims |
|---|---|
| baseline (top-5 de exp18) | 0.4552 |
| rerank por query (control del harness) | 0.4552 |
| **rerank por claim** | **0.4853** |
| cota alcanzable k=5 (techo) | 0.5834 |

Diferencia pareada (claim − baseline): **0.03** (IC95 0.0156 a 0.0448). Margen disponible: 0.1282. Fraccion del margen cerrada: **0.2344**.

the pool is stored in production-reranked order, so re-ranking it by (query, chunk) returns indices 0-4 = the baseline's own top-5. `query_rank` is therefore a harness check, not a second comparator; the contrast that matters is claim_rank vs baseline.

## COMPUERTA: **PASS**

declared before running: FAIL => exp19b is dead, the mechanism cannot find the evidence under conditions this favourable. PASS => the mechanism is not dead; it does NOT predict a faithfulness gain.

CIRCULAR: computed against the very claims the baseline wrote, so this is a statement about RETRIEVAL under a fixed answer. `frac_of_headroom_closed` is well defined ONLY inside that fixed-answer world -- it is NOT the fraction of exp18's +0.128 that exp19b would deliver, because a real selector changes the answer. Never an effect estimate.

Diagnostico (no primaria): recall@5 de chunks que anclan — baseline 0.2345, claim 0.2506. it saturates: most pool chunks support SOME claim (q001: 38 of 50), so recall@5 is ~5/|supporting| for any five supporting chunks and is blind to whether the right claims got covered

no faithfulness verifier is used to SELECT, so small/base/HHEM stay clean evaluators for exp19b; bge-reranker-large is untouched and remains the independent retrieval oracle