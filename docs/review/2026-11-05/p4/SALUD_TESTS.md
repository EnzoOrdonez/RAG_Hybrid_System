# P4: salud de los tests (base 3bb9ca8)

Rama `claude/p4-tests`. No se modifica `scripts/study_operator/` ni `src/`: los arreglos y los tests nuevos están todos en `tests/`.

## a) F-016: `test_short_calls_always_check_host_telemetry` [EJECUTADO]

**Causa raíz.** El helper `supervisor()` de `tests/test_study_stimulus_supervisor.py` fijaba dos plazos de reloj real más cortos que la latencia de planificación bajo carga. Los dos vencen en el mismo `_receive` (`scripts/study_gate_supervisor.py:123`, «Independent worker deadline reached»), así que el mensaje del F-016 no los distingue:
- `preparation_seconds=3`: arranque de un worker `spawn` (`stimulus_supervisor.py:57`).
- `call_seconds=.1`: un viaje de ida y vuelta por pipe.

| Medida (4 CPU; carga = N procesos ocupados, `load.py`) | Sin carga | 64 procesos | 128 procesos |
|---|---|---|---|
| Arranque del worker (`f016_startup.py`) | p50 0,17 s / máx 0,19 s | — | p50 10,0 s / máx 10,66 s (5 de 5 > 3 s) |
| Viaje de una llamada (`f016_latency.py`, 300 llamadas) | p50 0,17 ms / máx 0,47 ms | p99 52 ms / máx 64 ms | p50 64 ms / p99 96 ms / máx 112 ms (2 > 100 ms) |

**Reproducción con 128 procesos.**
- **Antes:** el test original falló **30 de 30** (`TimeoutError` en `__enter__`, preparación de 3 s; registro `evidence/f016_before_30.txt`).
- **Arreglo, solo en el test:** `PREPARATION_SECONDS=60` y `SUCCESS_CALL_SECONDS=10`. El test de bloqueo conserva `HANG_CALL_SECONDS=.5`, porque necesita que venza.
- **Después:** **0 de 10** fallos con la misma carga, ejecutando el archivo completo (5 tests, unos 29 s cada vez; `evidence/f016_after_10.txt`).
- El test de bloqueo sigue comprobando que se mata al worker.

**Límite de la evidencia.** La carga es sintética, así que la causa bajo la suite real es una inferencia: la carga de la suite no se midió. El síntoma y el punto de fallo coinciden con el registrado.

## b) Mutación de los módulos que producen resultados [EJECUTADO]

- **Mutador:** [mutate_ast.py](mutate_ast.py), mutación AST de primer orden: `+/-`, `*`, `/`, `<`/`<=`, `>`/`>=`, `==`/`!=`, `and`/`or`, `in`/`not in`, constantes y `not`.
- **Criterio:** un mutante muere si falla algún test seleccionado. Cada archivo se restaura con `git checkout`.
- **Tests seleccionados:** antes, los que importan o mencionan el módulo; después, los mismos más `tests/test_p4_mutation_killers.py` (42 tests).

| Módulo | Qué produce | Antes | Después |
|---|---|---|---|
| `src/evaluation/retrieval_metrics.py` | P@k, R@k, MRR, NDCG, MAP | 46/63 (73 %) | 57/63 (90 %) |
| `src/embedding/index/hybrid_index.py` | fusión RRF y lineal, normalización | 0/66 (0 %): ningún test la ejercía | 32/66 (48 %); **las funciones de fusión, 32/36**: `search_hybrid` 6/6, `_rrf_fusion` 13/13, `_normalize_scores` 7/7 y `_linear_fusion` 6/10. Persistencia y validación (`build`, `save`, `load`, `validate`, `__init__`: 0/30) quedan fuera del objetivo |
| `src/retrieval/hybrid_retriever.py` | ruteo de la fusión (expansión, filtro de proveedor, reranker) | 0/22 (0 %) | 18/22 (82 %) |
| `src/evaluation/hallucination_metrics.py` | agregación de la fidelidad NLI | 0/12 (0 %) | 12/12 (100 %) |
| `src/evaluation/statistical_analysis.py` (60 de 115 sitios, espaciados) | tests pareados, d_z, BH y Holm, bootstrap, redondeo | 8/60 (13 %) | 37/60 (62 %) |

**Supervivientes que quedan.**
- `retrieval_metrics`, 6. Todos equivalentes o inalcanzables:
  - `:53` y `:86`: protegidos por la validación de `k` y de relevantes;
  - `:59`, `:71` y `:92`: `or`→`and` da el mismo 0,0.
  - `:65` (MRR sin aciertos) se cubrió después con `test_mrr_and_map_are_zero_when_nothing_relevant_is_retrieved`, que se añadió tras la corrida «después» y no está medido en el 57/63.
- `hybrid_retriever`, 4. Solo afectan al logging: `[:160]`, `[:50]` y `elapsed`; `:65` solo controla un `logger.info`.
- `statistical_analysis`, 23. Sin clasificar del todo por tiempo; hay redondeos de `to_dict` en campos no comprobados y ramas de normalidad.
- `hybrid_index`, 4 en `_linear_fusion`.

**Hallazgos del ejercicio** (caracterizados, no corregidos; `src/` no se toca en esta fase):
- **P4-H1 (S2, determinismo).** `hybrid_index.py:149`: `_linear_fusion` itera un `set`, y con puntuaciones empatadas el top-k cambia con PYTHONHASHSEED. Ejecutado con las semillas 0, 7, 42 y 1, da 4 órdenes distintos. Afecta a las corridas con fusión lineal (exp3). RRF no tiene el problema. **Arreglo propuesto:** `sorted(...)` o desempate por `cid`.
- **P4-H2 (S3).** `statistical_analysis.paired_comparison` con diferencias todas nulas toma el camino de la t pareada y devuelve `p = NaN` en lugar de la rama `wilcoxon_zero_diff` (p = 1). Nunca sale significativo. Fijado en `test_paired_comparison_insufficient_and_significance_threshold`.

## No verificado
- La carga real de la suite durante F-016.
- Windows: allí F-016 se observó sin medir la latencia.
- `statistical_analysis` completo: solo 60 de 115 sitios.
- `src/generation/hallucination_detector.py` (verificador NLI real): necesita modelos.
