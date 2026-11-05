# P6: dependencia de PYTHONHASHSEED en la fusión (base 3bb9ca8)

Rama `claude/p6-fusion`. No se regenera ningún resultado firmado.

## 1. Qué pasa por iteraciones sin orden en el camino de retrieval y fusión [EJECUTADO: lectura y grep]

| Sitio | ¿Depende del hash? | Quién lo usa |
|---|---|---|
| `src/embedding/index/hybrid_index.py:147-155` (`_linear_fusion`: `all_ids` es un `set` y el `sort` es estable) | **Sí, con empates** | Solo los brazos `retrieval_hybrid_linear_03/05/07` de exp3 (`experiments/experiment_configs.py:200-240`) y `HybridRetriever.grid_search_alpha` |
| `hybrid_index.py:159-175` (`_rrf_fusion`, dict en orden de inserción) | No | `PROPOSED_HYBRID` (`src/pipeline/pipeline_config.py:92`, `fusion_method="rrf"`), es decir, exp8, exp11, exp12 y las siguientes |
| `src/embedding/index/bm25_index.py:111` (`argsort` sobre un array) | No: determinista con la misma entrada | BM25 |
| `src/reranking/cross_encoder_reranker.py:123` (`sort` estable sobre una entrada ordenada) | No | reranker |
| `src/retrieval/coverage_balancer.py:80-89,116` (devuelve `sorted(found)`; `pickset` solo para pertenencia) | No | exp17 |

## 2. Ejecución con PYTHONHASHSEED 0, 7, 42 y 1

- **Mecanismo [EJECUTADO, datos sintéticos con empates].** `_linear_fusion` con seis documentos empatados devuelve 4 top-4 distintos con 4 semillas, por ejemplo `['doc4','doc5','doc3','doc6']` con 0 y `['doc3','doc5','doc4','doc0']` con 7. RRF devuelve `['doc3','doc4','doc5','doc0']` con todas.
- **exp3 real: no verificado.** Reejecutarlo necesita los índices `data/indices` (no versionados) y BGE-large (no descargado). Además, los checkpoints firmados guardan solo `retrieved_ids` y no las puntuaciones fusionadas, así que no se puede medir la frecuencia real de empates.
- **Probabilidad estructural** [deducción del código, sin medir]: la normalización min-max da 0 al mínimo de cada lista. Un documento presente en una sola lista recibe `α·1` o `(1−α)·1`, y con α = 0,5 esos dos casos empatan exactamente.
- **Métricas de exp3:** no se pueden comparar. `output/tables/table_exp3.tex` informa NDCG@5, Recall@5 y MRR = 0 en los seis brazos, porque los checkpoints tienen `relevant_ids` vacío.

## 3. Arreglo [EJECUTADO]

- **Cambio:** `hybrid_index.py:155`: `combined.sort(key=lambda x: (-x[1], x[0]))`, que desempata por `chunk_id`. Sin empates, el orden no cambia.
- **Tests:** `tests/test_p6_linear_fusion_determinism.py` (2). Fallan los 2 contra 3bb9ca8 y pasan con el arreglo.

| Semilla | Top-4 lineal antes | Top-4 lineal después | RRF antes = después |
|---|---|---|---|
| 0 | doc4, doc5, doc3, doc6 | doc3, doc4, doc5, doc0 | doc3, doc4, doc5, doc0 |
| 7 | doc3, doc5, doc4, doc0 | doc3, doc4, doc5, doc0 | igual |
| 42 | doc4, doc3, doc5, doc1 | doc3, doc4, doc5, doc0 | igual |

## 4. ¿Dependen del hash las tablas del camera-ready?

- **Tabla de retrieval (exp11): no**, según la lectura del código. Es RRF, y BM25, FAISS y el reranker son deterministas. El ejecutable de RRF dio el mismo resultado con 3 semillas. **Reejecutar exp11: no verificado** (índices y modelos).
- **Tabla de fidelidad (exp12, v4_small): no por la fusión**, porque usa la misma ruta RRF. La agregación y el rescoring NLI no se reejecutaron (modelos), así que esa parte queda **no verificada**.
- **Discrepancia (S3).** El camera-ready (`docs/2026305869.pdf`, §IV, «Linear fusion with α ∈ {0.3, 0.5, 0.7} was also evaluated for comparison») presenta una evaluación de la que no hay ninguna cifra válida en el repo: `table_exp3.tex` está a ceros. La ganancia del arreglo solo afecta a esa evaluación.
- **Adopción:** el cambio toca `src/`, que está dentro de `i7_readiness.source_sha256`. Una imagen nueva necesitaría un recibo de readiness nuevo. Para RRF el resultado no cambia.
