[Fable 5.2] | 2026-11-15 20:55 | rama `fable/rag-auditoria` @ `692d6bd` | tarea F-propuesta (sin número, ver §9) | modo: ejecución
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (2303 pasan, 0 fallan, 6 omitidas) | 3 docs=exit 0
Cambios: solo este informe y la copia del prompt | Sin verificar: ningún modelo se ejecutó (D1); ver §10

# Auditoría y endurecimiento del código RAG (noviembre de 2026)

Encargo: [PROMPT_FABLE_RAG_2026-11-16.md](../PROMPT_FABLE_RAG_2026-11-16.md). Decisiones vigentes:
D1 no se ejecuta ningún modelo; D2 los cambios de clase B solo se proponen; D3 sin push, merge ni PR.
Este informe se actualiza en cada commit. El estado de cada hallazgo está en la tabla de §3.

## 1. Alcance y método

- Dentro: `src/evaluation`, `src/generation`, `src/pipeline`, `src/retrieval`, `src/reranking`,
  `src/embedding`, `src/chunking`, `src/preprocessing`, `src/ingestion` y `src/utils`
  (`reproducibility`, `local_models`, `signed_evidence`).
- Fuera: `src/ui/**`, `src/utils/deployment_artifacts.py`, `src/utils/checkout_boundary.py`, `config/`,
  `scripts/study_operator/**`, los scripts de I7, estudio y compuertas, y las tareas F-005, F-009, F-016,
  F-020 y F-021.
- Un hallazgo entra solo si se reproduce con un test o con una traza de código concreta. Lo ya
  registrado en los ledgers (`paper/audit_findings.md`, `audit_findings_cc_addenda.md` N1 a N9,
  `correction_log.md`, `summer_ablation_log.md`) no cuenta como hallazgo nuevo; se lista en §6.
- Clase A: el resultado no cambia en ninguna configuración de la evidencia ni del despliegue; se aplica
  con TDD. Clase B: cambia alguna salida; se entrega test `xfail(strict=True)`, parche sin aplicar,
  impacto y corrida `exp20+` necesaria.

## 2. Línea base (F0)

| Comprobación | Resultado |
|---|---|
| Rama y base | `fable/rag-auditoria`, cortada de `1574d22`, árbol limpio al empezar |
| Intérprete | `.venv-app` = Python 3.14.3; scipy 1.17.1, statsmodels 0.14.6, numpy 2.4.2 |
| `bash init.sh` capa 1 (secretos) | exit 0; 0 hallazgos, 6 excluidos por baseline |
| `bash init.sh` capa 2 (tests CPU) | exit 0; 2303 pasan, 0 fallan, 6 omitidas (conteo de la barra de progreso: la suite corre con `-qq` y no imprime la línea de resumen) |
| `bash init.sh` capa 3 (docs/harness) | exit 0; los seis controles de `check_docs.py` en OK |
| `scripts/verify_v4_offline.py` | exit 0; 24 controles OK, 0 FAIL |
| `scripts/verify_summer_offline.py` | exit 0; 65 controles OK, 0 FAIL; descubre exp15 (Tier A), exp16, exp17, exp18 y exp19b |
| Dependencias ausentes | `pytest-cov`, `coverage` e `hypothesis` no están instaladas; no se instalaron (límite 2). F4 no mide cobertura |

Los dos verificadores se leyeron antes de correrlos: importan `numpy`, `scipy`, `statsmodels` y funciones
puras de `src` y `scripts` (`decide_nli_status`, `paired_comparison`, `cohens_d`,
`apply_multiple_comparison_correction`, `compute_faithfulness_metrics`, `compute_tierA_arm_stats`).
Ninguno carga un modelo ni abre red. Cada uno escribe un reporte nuevo con la fecha del día en
`output/audit/`; no sobrescriben ningún archivo existente.

Referencia dorada (SHA-256 de la salida estándar):

| Verificador | SHA-256 normalizado | SHA-256 crudo |
|---|---|---|
| `verify_v4_offline.py` | `1624b2f530f1e588cd86ff39a3fa38644f374082cb5c44d6847fc839c2827f54` | `1e075aa2b5d6ee0c0df2d24b50f597c4ce60ca275284f782d29a76c13fc7953d` |
| `verify_summer_offline.py` | `eabda7648d2d5a91a6d2e2cc4f1b8f305eb9a3e1f8d2252148abb0761640f9c6` | `9ddd14d331c5f408796e39a2eab3492390b9a2597f1f6f97d5012208ba5cf5f3` |

El hash normalizado es el que se compara en F2 y F5. Normalización: se quitan los retornos de carro y las
líneas que empiezan por `Fecha: `, `Reporte: ` o `reporte -> `, porque llevan la fecha del día y la ruta
absoluta del checkout. El hash crudo solo es reproducible el mismo día y desde la misma carpeta.

```bash
python scripts/verify_v4_offline.py | tr -d '\r' | grep -vE "^Fecha: |^Reporte: |^reporte -> " | sha256sum
```

## 3. Hallazgos

Severidad: CRÍTICO = cifra o evidencia incorrecta, o riesgo para la evidencia firmada; ALTO =
comportamiento incorrecto en el camino desplegado; MEDIO y BAJO = el resto.

| ID | Sev. | Clase | Ubicación | Hallazgo | Evidencia | Estado |
|---|---|---|---|---|---|---|
| H-01 | CRÍTICO (riesgo) | B1 | `src/evaluation/benchmark_runner.py:199-215`, `:533-568`, `:703-707` | `BenchmarkRunner` escribe `results.json`, `aggregated_metrics.json` y los checkpoints sin pasar por `guard_write`. Relanzar un experimento firmado con todo ya completado reescribe sus artefactos sin cargar ningún modelo | Traza: `run_experiment` siempre termina en `_save_experiment_results`; `scripts/run_benchmark.py:164` y `scripts/run_llm_only_benchmark.py:138` usan el directorio de evidencia por defecto | Propuesto (B1); decide Enzo |
| H-02 | MEDIO (latente) | B2 | `src/evaluation/statistical_analysis.py:197-217`, `:297-307` | Dos sistemas con puntajes idénticos dan `p=NaN`; un solo NaN vuelve NaN todos los `p_bh` y `p_holm` de la familia | Reproducido con scipy 1.17.1. La evidencia firmada no contiene ningún `p` NaN | Propuesto (B2); decide Enzo |
| H-03 | MEDIO | B3 | `src/evaluation/benchmark_runner.py:258-324` | `_run_single_query` ignora `response.error`: una falla del pipeline queda como fila válida y su texto de error se puntúa como si fuera la respuesta | exp3, brazos lineales: q042 y q049 sin fragmentos recuperados en los 3 brazos, con `errors: 0` | Propuesto (B3); decide Enzo |
| H-04 | ALTO | B4 | `src/retrieval/coverage_balancer.py:86-88` | Los nombres de servicio genéricos del índice (`virtual machines`, `virtual network`, `vpc`, `iam`) añaden un proveedor que la pregunta no compara y le asignan cuota en `SURVEY_DEPLOY` | «Compare virtual machines on AWS and GCP» resuelve `aws, azure, gcp` con el índice real | Propuesto (B4); decide Enzo |
| H-05 | ALTO | B5 | `src/retrieval/query_processor.py:140-148` | La detección de proveedor es por subcadena: `laws` y `flaws` contienen `aws` y activan el filtro `single_provider` | «What data residency laws apply to object storage?» queda filtrada a AWS | Propuesto (B5); decide Enzo |
| H-06 | MEDIO | B6 | `src/evaluation/latency_metrics.py:46-51` | Se descartan los valores a más de 3 desviaciones antes de calcular p95, p99 y máximo: se recorta justo la cola que esas cifras deben describir | exp8b denso: p95 41 541 ms recortado frente a 45 630 ms sin recortar | Propuesto (B6); decide Enzo |
| H-07 | MEDIO (latente) | B7 | `src/evaluation/benchmark_runner.py:140` | La caché de índices usa `estrategia_tamaño` e ignora el modelo de embeddings: dos configuraciones con modelos distintos comparten el primer índice cargado | Traza de código; afecta a `exp2`, que no es evidencia firmada | Abierto |
| H-08 | MEDIO | B8 | `src/generation/llm_manager.py:304-311` | `generate()` guarda en caché una respuesta vacía y la sirve para siempre; `generate_stream()` sí la descarta | exp12 tiene 399 respuestas vacías de 3104 (N6.4 documenta 177 de qwen sin RAG) | Abierto |
| H-09 | BAJO | A | `src/generation/llm_manager.py:180-197`, `:319-328` | La caché se escribe sin atomicidad y una caché ilegible se descarta en silencio; el log dice «retrying» en el último intento | Traza de código y 3 tests en rojo antes del arreglo | Corregido (A1) |
| H-10 | BAJO | A | `src/evaluation/benchmark_runner.py:7,38,41-43` | El docstring promete un timeout de 120 s por consulta que no existe; `TimeoutError` tapa el builtin y nadie lo usa | Traza de código; `grep` da 0 usos y 0 `import *` | Corregido (A2) |
| H-11 | BAJO | A | `src/retrieval/coverage_balancer.py:99-102` | El docstring afirma que el orden de relevancia se conserva «dentro y entre» selecciones; el relleno puede dejar un fragmento mejor rankeado detrás | `A1,A2,A3,A4,B1` con cuota 3 da `A1,A2,A3,B1,A4` | Corregido (A3) |

## 4. Commits de clase A

Cada arreglo siguió el ciclo test rojo, cambio mínimo, test verde y tests del módulo. La columna de
equivalencia dice por qué ningún resultado cambia.

| ID | Commit (asunto) | Hallazgo | Rojo antes | Prueba de equivalencia |
|---|---|---|---|---|
| A1 | `fix: caché LLM atómica, aviso si es ilegible y log veraz del último intento` | H-09 | 3 de 5 tests de `tests/test_fable_llm_cache.py` fallaban: caché truncada tras un corte de escritura, caché ilegible sin aviso y «retrying in 4s» en el último intento | `test_saved_bytes_are_identical_to_the_legacy_serialisation` compara byte a byte con la serialización anterior; `tests/test_benchmark_parity.py` (clave de caché congelada y llamada a Ollama idéntica) sigue verde; 108 tests de los 8 módulos que usan `LLMManager` pasan. No cambian la clave, el contenido de la caché, los reintentos ni sus esperas (`sleeps == [2]`) |
| A2 | `fix: el runner deja de anunciar un timeout por consulta que no aplica` | H-10 | `test_the_module_does_not_advertise_a_timeout_it_does_not_apply` fallaba: el docstring decía «120-second timeout per query» y existía la clase `TimeoutError` | Solo cambian el docstring, un comentario y se borra una clase sin uso (0 referencias en `src`, `scripts` y `tests`; 0 `import *`). `test_query_timeout_is_accepted_but_never_enforced` fija que el parámetro se sigue aceptando y sigue sin aplicarse. `scripts/audit/smoke_hall_n_effective.py` pasa. Verificadores offline: SHA-256 normalizado igual a la línea base |
| A3 | `docs: el docstring de balance() describe el orden real del relleno` | H-11 | No aplica: solo cambia un docstring. El test `test_fill_pass_appends_leftovers_after_the_quota_picks` fija el comportamiento y pasa antes y después | Ninguna línea ejecutable cambia. `tests/test_coverage_balancer.py` completo pasa, incluida la aceptación que reproduce los `balanced_ids` de exp17 en 25/25 consultas |

Notas de A1:

- La escritura usa un archivo temporal hermano y `os.replace`. Si el archivo de caché fuera un enlace
  simbólico de archivo, `os.replace` sustituiría el enlace por un archivo normal. En este repositorio
  `data/llm_cache/` contiene archivos normales, así que el caso no se da.
- El docstring del módulo ahora dice la verdad sobre el timeout: 60 s para OpenAI, y para Ollama solo con
  `enforce_timeout=True`. El camino por defecto de los benchmarks llama a `ollama.chat` sin timeout de
  cliente. El comportamiento no se tocó.

## 5. Propuestas de clase B

Ninguna está aplicada. Cada una trae: el test que muestra el defecto (ya en la rama, marcado
`@pytest.mark.xfail(strict=True, reason="FABLE-B<n>: pendiente de decisión de Enzo")`), el parche listo en
`docs/reports/fable-rag-2026-11/patches/`, el impacto estimado releyendo las salidas firmadas y la corrida
nueva que haría falta.

Cómo se verificó cada parche: se aplicó de forma temporal, se corrió su test con `--runxfail` (pasa), se
corrió el resto de tests del módulo (pasan), se guardó el `git diff` y se revirtió. Con los ocho parches
aplicados a la vez, los 94 tests de `tests/test_fable_*.py`, `test_coverage_balancer.py`,
`test_signed_evidence_guard.py`, `test_benchmark_parity.py` y `test_retrieval_contracts.py` pasan con
`--runxfail`, y los dos verificadores offline dan el mismo SHA-256 normalizado que la línea base. Es
decir: ninguno de los ocho parches cambia una cifra que los verificadores rederivan.

Para aplicar uno: `git apply docs/reports/fable-rag-2026-11/patches/B<n>.diff` y quitar la marca `xfail`
de su test (con `strict=True`, el test falla como XPASS si se olvida).

### B1 (H-01, CRÍTICO por riesgo): `BenchmarkRunner` puede reescribir evidencia firmada

**Defecto.** `run_experiment` termina siempre en `_save_experiment_results`, que escribe `results.json`
y `aggregated_metrics.json` con `write_text`, sin `guard_write`. `scripts/run_benchmark.py` y
`scripts/run_llm_only_benchmark.py` construyen el runner con el directorio de evidencia por defecto.
Con los checkpoints completos, `python scripts/run_benchmark.py --experiment exp8` no carga ningún
modelo («All queries already completed») y aun así reescribe los dos archivos. ADR-0009 declara como
limitación que los runners de verano no llaman a `guard_write`; este runner, que es el de exp3 a exp9,
no estaba en esa lista. El botón de la UI sí se protegió en N9.

**Cuánto cambiaría.** Simulado en memoria y en un directorio temporal, sin tocar la evidencia: se
cargaron las filas de cada `results.json` firmado y se re-agregaron con el código actual.

| Experimento | Campos de `aggregated_metrics.json` que cambiarían | Ejemplo |
|---|---|---|
| exp3 | 24 | aparecen los campos `hall_n_*` |
| exp4 | 16 | ídem |
| exp5 | 60 | `hall_faithfulness_mean` de `llm_llama3.1`: 0,3442 pasa a 0,3024 |
| exp6 | 100 | `hall_faithfulness_mean` de un brazo de ablación: 0,3708 pasa a 0,3234 |
| exp7 | 36 | aparecen `hall_faithfulness_n` y los conteos de claims |
| exp8 | 60 | `hall_faithfulness_mean` del híbrido: 0,3687 pasa a 0,3248 |
| exp8b | 60 | `hall_faithfulness_mean` del híbrido: 0,5141 pasa a 0,4249 |
| exp9 | 0 | ya está agregado con el código actual |

La diferencia es legítima (los archivos firmados son anteriores al filtro de los Flags 137 y 140), pero
el punto es otro: un relanzamiento inocente los pisaría en silencio. Hoy la única red es `git status`.

1. **Test:** `tests/test_fable_benchmark_runner.py::test_rerunning_a_completed_protected_experiment_does_not_rewrite_its_artifacts`.
2. **Parche:** [B1.diff](fable-rag-2026-11/patches/B1.diff). Al empezar `run_experiment` se pasa por
   `guard_write` cada archivo que la corrida reescribiría (`results.json`, `aggregated_metrics.json` y un
   checkpoint por configuración). Si el directorio está protegido y el archivo existe, sale con
   `SystemExit` antes de hacer nada.
3. **Impacto:** ningún resultado, ranking ni métrica cambia. Cambia el flujo en un caso: reanudar un
   experimento nuevo (`exp20+`) escrito bajo `experiments/results/` exige declararlo vivo en
   `src/utils/signed_evidence.py`, que es el flujo que ese módulo ya documenta. Los tests y los scripts
   de `scripts/audit/` usan directorios temporales y no se ven afectados. Con el parche,
   `tests/test_signed_evidence_guard.py` sigue verde.
4. **Corrida nueva:** ninguna.

Recomendación: aplicar. Es la única propuesta que protege la evidencia en lugar de cambiar un cálculo.
Se dejó como clase B porque altera qué hace un comando que hoy «funciona».

### B2 (H-02, MEDIO latente): `p=NaN` con puntajes idénticos contamina la familia BH/Holm

**Defecto.** Si dos sistemas tienen exactamente los mismos puntajes en una métrica, `paired_comparison`
devuelve `p_value = NaN`: Shapiro sobre diferencias constantes da «normal», y la t pareada con varianza
cero da NaN (con scipy 1.17.1 la rama de Wilcoxon también puede dar NaN). El `except ValueError` que
debía cubrir «todas las diferencias son cero» ya no se alcanza con esta versión de scipy. Después,
`multipletests` propaga ese NaN a todos los `p_bh` y `p_holm` de la familia, y deja `sig_bh` incoherente
(`True` junto a `p_bh = NaN`).

1. **Tests:** `tests/test_fable_metrics.py::test_identical_scores_give_p_one_not_nan` y
   `::test_one_tied_comparison_does_not_turn_the_whole_family_into_nan`.
2. **Parche:** [B2.diff](fable-rag-2026-11/patches/B2.diff). Si todas las diferencias son cero,
   `paired_comparison` devuelve `p = 1.0`, `test_name = "all_pairs_tied"` y no llama a scipy.
3. **Impacto:** ninguna cifra firmada cambia. Dos comprobaciones: (a) no hay ningún `p_value`, `p_raw`,
   `p_bh` ni `p_holm` igual a NaN en `experiments/results/**` ni en `paper/audit_outputs/`; (b) con el
   parche aplicado, `verify_summer_offline.py` (que importa `paired_comparison`) y
   `verify_v4_offline.py` dan el mismo SHA-256 normalizado que la línea base. Lo único que cambia es la
   etiqueta `test_name` en un par totalmente empatado.
4. **Corrida nueva:** ninguna. Protege cualquier análisis futuro (`exp20+`) en el que dos brazos
   coincidan en una métrica, cosa probable en métricas binarias con pocos casos.

### B3 (H-03, MEDIO): los errores del pipeline se registran como respuestas válidas

**Defecto.** `RAGPipeline.query` no lanza excepción cuando falla: devuelve un `RAGResponse` con
`error` y un texto fijo en `answer` («Error generating response: …», «Pipeline error: …» o «No relevant
documentation found for this query.»). `_run_single_query` no mira `response.error`, así que la fila
queda con `error=None`, cuenta como válida y, si hay métricas de generación o de fidelidad, el texto
fijo se puntúa como si fuera la respuesta del modelo. Un fallo de Ollama a mitad de una corrida
entraría al promedio de fidelidad como un 0 con `method="nli"`.

**Rastro en la evidencia.** exp3, brazos `retrieval_hybrid_linear_03/05/07`: q042 y q049 (ambas sobre
Azure Container Instances) no recuperaron ningún fragmento y su `answer` es el texto fijo, con
`errors: 0` en los tres brazos. Son 6 filas. En esas filas `retrieval_metrics` está vacío, así que no
movieron ninguna media. En el resto de `experiments/results/**/*.json` no hay ninguna `answer` que
empiece por uno de los tres textos fijos.

1. **Test:** `tests/test_fable_benchmark_runner.py::test_a_pipeline_error_is_recorded_as_an_error_not_scored_as_an_answer`.
2. **Parche:** [B3.diff](fable-rag-2026-11/patches/B3.diff). La fila conserva `retrieved_ids` y sus
   métricas de recuperación, lleva `error=response.error` y no calcula métricas de generación ni de
   fidelidad sobre el texto fijo.
3. **Impacto:** ninguna cifra firmada cambia. Si exp3 se re-agregara, sus tres brazos lineales pasarían
   de `errors: 0` a `errors: 2`.
4. **Corrida nueva:** ninguna para la evidencia actual. Cualquier corrida `exp20+` con este runner
   reportaría sus fallos como fallos.

Punto que debes decidir: una consulta sin fragmentos recuperados, ¿es un error o un cero de
recuperación? El parche la trata como error, igual que hoy se tratan las excepciones, y por tanto queda
fuera de las medias de recuperación. Si prefieres contarla como cero, hace falta además separar en
`_save_aggregated_metrics` las filas válidas para recuperación de las válidas para generación.

### B4 (H-04, ALTO): nombres de servicio genéricos añaden un proveedor no pedido

**Defecto.** `resolve_wanted_providers` suma el proveedor de cualquier nombre de servicio del índice que
aparezca como palabra completa en la pregunta. El índice tiene 22 servicios; cuatro de sus nombres son
también el término genérico que usan los tres proveedores: `virtual machines` y `virtual network`
(Azure), `vpc` e `iam` (AWS). Con el índice real:

| Pregunta | Proveedores que compara | Resuelve hoy |
|---|---|---|
| Compare virtual machines on AWS and GCP | aws, gcp | aws, **azure**, gcp |
| How does a GCP VPC differ from an Azure virtual network? | azure, gcp | **aws**, azure, gcp |
| Compare IAM in Azure and GCP | azure, gcp | **aws**, azure, gcp |

Con tres proveedores la cuota es `ceil(5/3) = 2`, así que el proveedor que nadie pidió puede ocupar
hasta 2 de los 5 fragmentos que ve el modelo. Solo ocurre en `SURVEY_DEPLOY`
(`balance_cross_cloud_providers=True`) y en consultas tipadas `cross_cloud`.

1. **Test:** `tests/test_fable_providers.py::test_generic_service_names_do_not_add_an_unrequested_provider` (3 casos).
2. **Parche:** [B4.diff](fable-rag-2026-11/patches/B4.diff). Una lista corta, `_GENERIC_SERVICE_NAMES`,
   deja fuera esos cuatro nombres al buscar servicios. El proveedor sigue detectándose por su palabra
   clave o alias.
3. **Impacto:** ninguno sobre la evidencia. De las 194 consultas, 51 se tipan `cross_cloud`; en las 51 el
   conjunto resuelto es idéntico con y sin el parche. Con el parche, la aceptación de
   `tests/test_coverage_balancer.py` sigue reproduciendo los `balanced_ids` de exp17 en 25/25. Cambia
   solo el comportamiento ante preguntas libres de los participantes.
4. **Corrida nueva:** ninguna obligatoria. Si quieres medirlo antes de desplegar, un `exp20` de cobertura
   (sin generación) sobre un conjunto de preguntas comparativas que usen términos genéricos. Llevarlo a
   la web exige otra Fase B.

Alternativa al parche: derivar la ambigüedad del propio corpus (un nombre es ambiguo si aparece con
frecuencia en el texto de fragmentos de otros proveedores). Evita la lista escrita a mano, que es la
clase de lista que el módulo quería evitar, pero cuesta un pase sobre los 24 481 fragmentos al arrancar.

### B5 (H-05, ALTO): la detección de proveedor es por subcadena

**Defecto.** `_detect_providers` usa `if kw in q`. La palabra clave `aws` está dentro de `laws`, `flaws`,
`draws` o `bylaws`. Una pregunta sin proveedor queda detectada como `aws`, se tipa `single_provider` y
los tres recuperadores descartan todo fragmento que no sea de AWS:

- «What data residency laws apply to object storage?» resuelve `['aws']`, filtro `['aws']`.
- «What are common flaws in serverless function design?» resuelve `['aws']`, filtro `['aws']`.

Es distinto de F2 del ledger de verano, que registra que falta `google` a secas y que `k8s`/`cncf`
apuntan a un corpus borrado. F2 sigue pendiente de tu decisión y este parche no lo toca.

1. **Test:** `tests/test_fable_providers.py::test_provider_detection_does_not_match_inside_other_words` (2 casos).
2. **Parche:** [B5.diff](fable-rag-2026-11/patches/B5.diff). Coincidencia por palabra completa.
3. **Impacto:** ninguno sobre la evidencia. En las 194 consultas del conjunto y en las 200 del respaldo
   previo, el detector por subcadena y el de palabra completa devuelven exactamente lo mismo (0
   diferencias), así que `query_type`, filtro y plantilla de prompt no cambian en ninguna. Afecta al
   camino desplegado y a cualquier consulta libre.
4. **Corrida nueva:** ninguna. `QueryProcessor` alimenta el `query_type` de `run_generation_matrix.py`;
   como no cambia en ninguna de las 194, una re-corrida de exp11 o exp12 no se vería alterada.

### B6 (H-06, MEDIO): las latencias se recortan antes de calcular p95, p99 y máximo

**Defecto.** Con más de 10 valores, `compute_latency_stats` elimina los que están a más de 3
desviaciones de la media y calcula todo sobre el resto. Las consultas más lentas son justo lo que p95,
p99 y `max` deben describir.

| Experimento, sistema | n | conservados | p95 recortado | p95 real | máx. recortado | máx. real |
|---|---|---|---|---|---|---|
| exp8, léxico | 200 | 197 | 54 365 ms | 57 004 ms | 67 215 ms | 79 174 ms |
| exp8, denso | 200 | 197 | 71 834 ms | 72 766 ms | 96 359 ms | 133 988 ms |
| exp8, híbrido | 200 | 199 | 190 621 ms | 192 480 ms | 248 585 ms | 257 052 ms |
| exp8b, léxico | 200 | 196 | 42 098 ms | 43 820 ms | 48 764 ms | 68 592 ms |
| exp8b, denso | 200 | 197 | 41 541 ms | 45 630 ms | 57 061 ms | 70 472 ms |
| exp8b, híbrido | 200 | 199 | 81 282 ms | 86 290 ms | 109 005 ms | 127 166 ms |
| exp9, sin RAG | 200 | 199 | 65 024 ms | 65 943 ms | 79 926 ms | 87 880 ms |

1. **Test:** `tests/test_fable_metrics.py::test_tail_latencies_are_not_trimmed_before_computing_the_tail_statistics`.
2. **Parche:** [B6.diff](fable-rag-2026-11/patches/B6.diff). Quita el recorte.
3. **Impacto:** afecta a los campos `lat_*` de los `aggregated_metrics.json` de exp3 a exp9 y a
   `fig_latency_breakdown`. No afecta a la tabla de latencias citable de Nota 3
   (`output/tables/nota3/latency__exp12_matrix`), que la produce `scripts/_latency_p50p95.py` sin recorte
   (`docs/TRACEABILITY_nota3.md`). No encontré ninguna cifra de latencia de exp8 citada en
   `docs/reports/results-summary.md` ni en `docs/TRACEABILITY_nota3.md`; no revisé la prosa del paper.
4. **Corrida nueva:** ninguna. Si alguna cifra heredada de latencia se cita, se recalcula offline a un
   archivo `_vN`.
