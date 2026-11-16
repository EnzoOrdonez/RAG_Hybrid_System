[Fable 5.2] | 2026-11-15 21:24 | rama `fable/rag-auditoria` @ `2df11c3` | tarea F-propuesta (sin número, ver §9) | modo: ejecución
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (2373 pasan, 0 fallan, 6 omitidas, 12 xfail estrictos) | 3 docs=exit 0
Cambios: 17 commits sobre `1574d22` más el de cierre, que solo toca este informe (§11) | Sin verificar: ningún modelo se ejecutó (D1); `init.sh` completo no se repitió tras el commit de cierre; ver §10 y §11

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
| H-07 | MEDIO (latente) | B7 | `src/evaluation/benchmark_runner.py:140` | La caché de índices usa `estrategia_tamaño` e ignora el modelo de embeddings: dos configuraciones con modelos distintos comparten el primer índice cargado | Traza de código; afecta a `exp2`, que no es evidencia firmada | Propuesto (B7); decide Enzo |
| H-08 | MEDIO | B8 | `src/generation/llm_manager.py:304-311` | `generate()` guarda en caché una respuesta vacía y la sirve para siempre; `generate_stream()` sí la descarta | exp12 tiene 399 respuestas vacías de 3104 (N6.4 documenta 177 de qwen sin RAG) | Propuesto (B8); decide Enzo |
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
- `os.replace` añade dos modos de fallo que la escritura directa no tenía: en Windows falla si otro
  proceso tiene abierto el archivo de caché en ese instante, y necesita permiso de escritura en el
  directorio, no solo en el archivo. En ambos casos se registra el aviso de siempre y esa entrada no se
  persiste; el texto generado no cambia. No hay un segundo escritor en `src` y el despliegue corre con la
  caché desactivada.
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

### B7 (H-07, MEDIO latente): la caché de índices ignora el modelo de embeddings

**Defecto.** `_get_hybrid_index` guarda el índice bajo la clave `estrategia_tamaño`. Dos configuraciones
del mismo experimento que solo difieren en el modelo de embeddings reciben el primer índice cargado. Es
el diseño de `exp2` («variable: embedding_model»). Además, los nombres cortos que arma el propio método
(`bge-base`, `all-MiniLM-L6`) no existen en `MODEL_CONFIGS` de `embedding_manager.py`, de modo que esas
configuraciones fallarían al cargar y se saltarían con un log de error.

1. **Test:** `tests/test_fable_benchmark_runner.py::test_index_cache_distinguishes_embedding_models`.
2. **Parche:** [B7.diff](fable-rag-2026-11/patches/B7.diff). El modelo entra en la clave. No corrige los
   nombres cortos: hacer ejecutable `exp2` es otra decisión.
3. **Impacto:** ninguno sobre la evidencia. `exp1` y `exp2` no existen como evidencia
   (`docs/guides/guardrails.md`) y todas las configuraciones de exp3 a exp19b usan `bge-large` con
   `adaptive/500`, el único índice construido. Para ellas la clave nueva identifica el mismo índice.
4. **Corrida nueva:** ninguna hoy. Sería requisito de una ablación de embeddings futura (`exp20+`), que
   además exige construir los índices.

Ver la pregunta 1 de §8: este es el único parche B que, a mi juicio, podría pasar a clase A.

### B8 (H-08, MEDIO): `generate()` congela en la caché las respuestas vacías

**Defecto.** `generate()` guarda la respuesta si no hubo error, aunque el texto esté vacío. A partir de
ahí la misma clave devuelve siempre la cadena vacía sin volver al modelo. `generate_stream()` ya evita
el caso (`if self.cache_enabled and text`).

**Rastro en la evidencia.** exp12 tiene 399 respuestas vacías de 3104, y las cachés versionadas tienen
exactamente 399 entradas vacías:

| Modelo | Respuestas vacías en exp12 | Entradas vacías en su caché |
|---|---|---|
| qwen3.5:9b | 345 (sin RAG 177, léxico 64, denso 52, híbrido 52) | 345 de 776 |
| gemma4:e4b | 54 (léxico 16, denso 19, híbrido 19) | 54 de 776 |
| granite4.1:8b, mistral:7b | 0 | 0 |

N6.4 documenta las 177 de qwen sin RAG y dice que una respuesta vacía queda con `method="none"`, fuera
de las medias. No encontré registradas las otras 222 (168 de qwen con RAG y 54 de gemma) y no volví a
comprobar que se excluyan igual. La causa de que el modelo devuelva vacío no la investigué (D1).

1. **Test:** `tests/test_fable_llm_cache.py::test_an_empty_answer_is_not_frozen_into_the_cache`.
2. **Parche:** [B8.diff](fable-rag-2026-11/patches/B8.diff). No se escribe en caché una respuesta sin
   texto.
3. **Impacto:** ninguna cifra firmada cambia y la reproducción desde caché de exp12 se conserva: el
   parche solo afecta a la escritura, y las 399 entradas vacías ya versionadas se siguen sirviendo. Cambia
   el futuro: una respuesta vacía nueva se vuelve a pedir al modelo en la siguiente llamada.
4. **Corrida nueva:** ninguna obligatoria. Si quieres saber si esas 399 respuestas vacías eran un fallo
   transitorio o una propiedad del modelo con ese prompt, hace falta un `exp20` que regenere solo esas
   consultas con qwen y gemma, sin caché. Sin esa corrida no se puede afirmar ninguna de las dos cosas.

## 6. Lo que se examinó y no es hallazgo

### 6.1 Ya registrado en los ledgers

| Tema | Dónde está registrado | Nota de esta auditoría |
|---|---|---|
| El detector de proveedores devuelve `k8s` y `cncf`, cuyo corpus se borró, y no reconoce `google` a secas | `paper/summer_ablation_log.md`, entrada 16, F2 (decisión tuya pendiente) | Sigue vigente en el camino desplegado: «What is a Kubernetes Pod?» se tipa `single_provider` con filtro `['k8s']`, se queda sin fragmentos y el pipeline responde «No relevant documentation found for this query.» |
| `RAGPipeline.query` solo enruta el prompt con `prompt_routing=True` | entrada 16, F1; ADR-0008 | Fijado por `tests/test_fable_pipeline.py` en las dos configuraciones |
| `multidimensional_scoring`, `terminology_normalization` y `reranker_top_k` no tienen efecto | N9; `docs/architecture/system-design.md` §8 | Sin cambios |
| La fidelidad sin RAG es 0 por construcción y las declinaciones generan afirmaciones | N3 y N5 | Fijado por tests |
| Artefactos de formato fuera del denominador y filas vacuas | N8 (H1) y N9 (P0) | Fijado por tests |
| `pure_decline` mide un prefijo, no un rechazo | ledger de verano, entrada 22, defecto #7 | Sin cambios |
| `NLI_TIMEOUT` se declara y nadie lo usa | `paper/audit_findings.md` | Mismo patrón que H-10 |
| p = 0,0 ficticio en la rama Anderson-Darling | `paper/audit_findings.md`, Flag 112 | Sin cambios |
| Boilerplate «Stay organized with collections» en el 33 % del corpus | N8 (H6) y N9 | Sin cambios |
| 177 respuestas vacías de qwen sin RAG | N6.4 | Ver B8 para las otras 222 |

### 6.2 Examinado y descartado

- **Patrones de declinación y apóstrofo tipográfico.** Seis de los 14 patrones llevan apóstrofo recto. En
  las 4 646 respuestas de exp8, exp9, exp12, exp13, exp17 y exp18, aceptar también el apóstrofo curvo no
  cambia la clasificación de ninguna. Sin propuesta.
- **`SKIP_PATTERNS` y `^the.*context`.** El patrón descarta cualquier oración que empiece por «the» y
  contenga «context». En esas mismas respuestas coincide con 1 272 oraciones; una heurística marcó 35
  como posibles afirmaciones genuinas, y las 12 que leí son también comentarios sobre el contexto
  recuperado. No hay evidencia de que se pierdan afirmaciones reales. Sin propuesta.
- **Empates en la fusión lineal.** `_linear_fusion` recorre un `set`, así que el orden entre empates
  depende de `PYTHONHASHSEED`. El entorno obligatorio ya lo fija en 42 y la fusión RRF, que es la de la
  evidencia principal y la del despliegue, usa un diccionario con orden de inserción. Sin propuesta.
- **Empates en BM25.** `argsort()[::-1]` no es estable, pero es determinista para una misma versión de
  numpy. Sin propuesta.
- **Búsqueda densa.** 24 481 vectores, por debajo del umbral de 50 000: `IndexFlatIP` exacto, como ya
  confirma N6.3.
- **`balance()` con más proveedores pedidos que cuota.** Con 4 proveedores y k = 5 uno podría quedarse
  sin fragmento. No es alcanzable: el corpus tiene 3 proveedores.
- **Fragmentos sin proveedor.** Un `cloud_provider` nulo rompería `_build_sources` y el contexto agrupado.
  En el índice real los 24 481 fragmentos tienen proveedor (`aws`, `azure`, `gcp`) y servicio. No se toca.
- **`study_statistics.py` y `study_analysis.py`.** Leídos completos. Sin hallazgos.
- **`signed_evidence.py`, `local_models.py` y `reproducibility.py`.** Leídos completos y sin cambios. Dos
  notas: `local_models.resolve` identifica el modelo por el último segmento del ID y no comprueba el
  manifiesto, de modo que dos IDs de organizaciones distintas con el mismo nombre colisionarían;
  `ensure_hashseed_at_startup` usa `os.execv`, que en Windows no entrecomilla argumentos con espacios (la
  ruta del repositorio no los tiene).

### 6.3 Cadena de construcción del índice (P3): observaciones sin parche

Lectura parcial: `adaptive_chunker.py`, `text_cleaner.py`, `deduplicator.py` y `embedding_manager.py`
completos. Todo cambio
aquí es de clase B, obliga a reconstruir el índice e invalida la comparabilidad con exp10 a exp19b, así
que no preparé parches ni tests. Quedan como observaciones con su traza:

- `adaptive_chunker.py:356-391`: cuando una sección pequeña se fusiona con la siguiente, el resultado
  conserva el `heading_path` de la primera. El texto de la segunda sección queda etiquetado con el
  encabezado de la anterior, y ese encabezado es el que el modelo cita en `[Source: …]`.
- `adaptive_chunker.py:244-248`: si el modelo de embeddings del chunker no carga, `_semantic_split` pasa a
  partir por tokens dentro de un `except Exception` sin log. Los límites de los fragmentos dependen en
  silencio de que ese modelo esté disponible al construir.
- `text_cleaner.py:25`: `View.*?on GitHub` no exige límite de palabra, así que borra desde el «view» de
  «Overview» o «Review» hasta «on GitHub» dentro de la misma línea. Los patrones de `Feedback` (línea 22) y
  de cookies (línea 37) también pueden borrar prosa legítima.
- `text_cleaner.py:115-131` colapsa los espacios repetidos fuera de bloques cercados, incluida la
  indentación; `:139-156` elimina párrafos repetidos de 20 caracteres o más aunque estén dentro de código.

- `deduplicator.py:153-172`: cada grupo de casi-duplicados conserva el documento más largo, pero el grupo
  se arma con todo lo que devuelve la consulta LSH, incluidos documentos que ya se marcaron para borrar.
  Si B ya cayó frente a A y luego C se agrupa con B, y B es más largo que C, C también se borra aunque
  no se parezca a A. Un documento puede eliminarse en favor de otro que tampoco sobrevive.
- `src/ingestion/doc_parser.py:46,74`: `doc_id` y `chunk_id` son `uuid4()` aleatorios. Reconstruir el
  corpus genera identificadores nuevos, así que los `retrieved_ids` de la evidencia solo se pueden
  resolver contra el índice ya construido. La reproducibilidad de la recuperación depende de conservar
  `data/indices`, no de poder regenerarlo.

No leídos: `terminology_normalizer.py`, `metadata_extractor.py`, el resto de `doc_parser.py`, los
crawlers, `ingestion_pipeline.py`, los otros cuatro chunkers y `test_queries.py`. De `results_exporter.py`
solo revisé dónde calcula estadísticos.

## 7. Huecos de tests cubiertos (F4)

Antes de esta auditoría no había ningún test de comportamiento para `benchmark_runner`,
`generation_metrics`, `latency_metrics`, `hallucination_metrics`, `prompt_templates`, `hybrid_retriever`,
`reproducibility`, ni para `RAGPipeline.query` de punta a punta. `multidimensional_scorer` sigue sin tests
a propósito: N9 ya registra que nunca tuvo un punto de llamada.

Todos los tests nuevos corren en CPU con dobles (índice, reranker, LLM y modelo NLI falsos). Ninguno carga
un modelo ni lee `data/indices`. Son tests de caracterización: fijan lo que el código hace hoy, incluido
lo discutible, para que un cambio se note.

| Archivo | Tests | Qué fija |
|---|---|---|
| `tests/test_fable_pipeline.py` | 26 | SHA-256 de los seis textos de prompt; contexto estándar y agrupado por proveedor; RRF (suma de rangos recíprocos, empates a favor de BM25, independencia de los puntajes crudos); fusión lineal; ruteo de la consulta cruda y expandida a cada pierna; filtro `single_provider`; niveles de confianza y citas del formateador; `RAGPipeline.query` completo en la configuración heredada, en la desplegada (ruteo de prompt, balanceo, pool de 50 para el reranker), sin candidatos, con fallo del LLM, con excepción y sin RAG |
| `tests/test_fable_detector.py` | 15 | estados por afirmación y denominador sin artefactos; `vb_agree` con un solo chunk; fila vacua (N9); `method="none"`; camino sin evidencia (N3); caída del NLI a palabras clave (`mixed`); extracción de listas con viñetas; envoltorio `compute_hallucination_metrics`; `set_all_seeds` y la guarda de `PYTHONHASHSEED` |
| `tests/test_fable_metrics.py` | 14 | `exact_match` (solape del 80 %), `f1_token` sobre conjuntos, ROUGE-L, estadísticos de latencia, BH y Holm calculados a mano, convención `b - a` del tamaño de efecto, bootstrap con semilla fija; más los `xfail` de B2 y B6 |
| `tests/test_fable_benchmark_runner.py` | 8 | ejecución de una consulta, fila de error por excepción, agregación con exclusión de `none` y `error` (Flags 137 y 140), H-10; más los `xfail` de B1, B3 y B7 |
| `tests/test_fable_providers.py` | 5 funciones, 12 casos | orden del relleno de `balance()` (H-11), clasificación de consultas; más los `xfail` de B4 y B5 |
| `tests/test_fable_llm_cache.py` | 7 | H-09 y la reutilización de caché; más el `xfail` de B8 |

Cobertura: no medida. `pytest-cov` no está instalado y el límite 2 prohíbe instalarlo.

Dos comportamientos que estos tests fijan y conviene tener presentes, sin ser hallazgos nuevos:

- `exact_match` no es igualdad de cadenas: devuelve 1,0 si el 80 % de los tokens del gold aparece en la
  respuesta. `f1_token` trabaja sobre conjuntos, así que las repeticiones cuentan una vez (no es el F1
  de SQuAD). Las cifras citables de Nota 3 no usan estas dos métricas.
- Una respuesta vacía da fidelidad 0,0 en `compute_hallucination_metrics` y una de solo espacios da 1,0
  en el detector. Ambas quedan con `method="none"` y fuera de las medias. En exp8, exp9, exp12, exp13,
  exp17 y exp18 no hay ninguna respuesta de solo espacios.

## 8. Decisiones que te tocan y preguntas abiertas

Una decisión por propuesta. La recomendación es mía y no está aplicada.

| Propuesta | Decisión | Recomendación |
|---|---|---|
| B1 | ¿`BenchmarkRunner` debe negarse a reescribir artefactos de un experimento protegido? | Aplicar. No cambia ningún resultado y cierra el único camino que hoy pisa evidencia sin aviso |
| B2 | ¿Un par totalmente empatado debe dar `p = 1` en vez de NaN? | Aplicar. Los verificadores dan el mismo SHA-256 con el parche puesto |
| B3 | ¿Un fallo del pipeline es una fila de error? Y antes: ¿una consulta sin fragmentos es un error o un cero de recuperación? | Decidir primero la segunda pregunta. Si es error, aplicar tal cual |
| B4 | ¿Se excluyen los cuatro nombres de servicio genéricos del balanceo? | Aplicar antes de construir la próxima imagen de despliegue, o registrar el caso como limitación del estudio |
| B5 | ¿La detección de proveedor pasa a palabra completa? | Igual que B4. Conviene decidirla junto con F2 (`k8s`, `google`), que toca la misma función |
| B6 | ¿Se quita el recorte de 3 desviaciones en las latencias? | Aplicar. Solo afecta a agregados heredados que hoy no se citan |
| B7 | ¿La caché de índices distingue el modelo de embeddings? | Aplicar o dejar: hoy no tiene efecto práctico |
| B8 | ¿`generate()` deja de cachear respuestas vacías? | Aplicar para corridas futuras. Aparte, decidir si un `exp20` regenera las 399 vacías de exp12 |

Preguntas que no dependen de una propuesta:

1. **B7 como clase A.** Ninguna configuración firmada ni desplegada cambia de índice con el parche. Lo
   dejé en clase B porque `exp2`, tal como está escrito, sí cambiaría de comportamiento. ¿Lo reclasificas?
2. **Las 222 respuestas vacías no registradas** (168 de qwen con RAG y 54 de gemma en exp12). ¿Se anotan
   en el ledger junto a N6.4?
3. **Reportes diarios de los verificadores.** Los dos de hoy (`output/audit/*_offline_check_2026-11-15.md`)
   están en el segundo commit de la rama, como en sesiones anteriores. Si no los quieres, basta quitarlos al
   integrar.
4. **Cabecera de telemetría.** Pide `tarea F-NNN` y no podía editar `.harness/feature_list.json`. Usé
   «F-propuesta (sin número)»; el texto de la entrada está en §9.
5. **D1 alternativa.** No la escribiste, así que no cargué ningún encoder. Toda equivalencia de clase A
   está probada con dobles y con los verificadores offline, no con recuperación real.

## 9. Textos propuestos (sin número)

Ninguno de los dos archivos de destino se editó (límite 5).

### 9.1 ADR

```markdown
## ADR-XXXX: Auditoría del código RAG de noviembre de 2026

- **Estado:** Propuesto. **Fecha:** 2026-11-15.
- **Contexto:** con la lógica RAG congelada (ADR-0004) y el despliegue I7 en fase humana, se auditó `src/`
  en una rama aislada (`fable/rag-auditoria`, base `1574d22`) sin ejecutar modelos. La auditoría separó
  los cambios que no alteran ningún resultado (clase A) de los que sí (clase B).
- **Decisión:** se integran los tres arreglos de clase A (caché LLM atómica y con aviso, docstring veraz
  del runner sin timeout, docstring de `balance()`) y los tests `tests/test_fable_*.py`. Las ocho
  propuestas de clase B se deciden una por una; cada una aceptada se aplica desde su parche en
  `docs/reports/fable-rag-2026-11/patches/` y se le quita la marca `xfail` a su test. Decisiones tomadas:
  B1 <aceptada o rechazada>, B2 <…>, B3 <…>, B4 <…>, B5 <…>, B6 <…>, B7 <…>, B8 <…>.
- **Consecuencias:** los tests de caracterización fijan el texto de los prompts, la fusión y el camino de
  `RAGPipeline.query`; un cambio en cualquiera de ellos ahora rompe un test y obliga a declararlo. Un
  parche de clase B aceptado que toque el camino desplegado no llega a la web sin una Fase B nueva.
- **Fuentes:** `docs/reports/fable-rag-auditoria-2026-11.md`, `docs/PROMPT_FABLE_RAG_2026-11-16.md`.
```

### 9.2 Entrada de `.harness/feature_list.json`

```json
{
  "id": "F-0NN",
  "category": "quality",
  "description": "Auditoría y endurecimiento del código RAG (rama fable/rag-auditoria, base 1574d22). Tres arreglos de clase A aplicados y ocho propuestas de clase B con test xfail estricto y parche sin aplicar. Informe: docs/reports/fable-rag-auditoria-2026-11.md.",
  "steps": [
    "Hecho 2026-11-15: línea base (init.sh en 0 y SHA-256 de los dos verificadores offline), auditoría de P1, P2 y P4, y lectura parcial de P3.",
    "Hecho 2026-11-15: arreglos A1 (llm_manager), A2 (benchmark_runner) y A3 (coverage_balancer) con TDD; 82 casos de test nuevos en tests/test_fable_*.py.",
    "Pendiente de Enzo: una decisión por propuesta B1 a B8 (informe, sección 8).",
    "Por cada B aceptada: git apply del parche, quitar la marca xfail, bash init.sh y los dos verificadores con el mismo SHA-256.",
    "Pendiente de Enzo: integrar la rama (sin push hasta su OK, ADR-0013) y registrar el ADR.",
    "No toca experiments/results, paper, config, src/ui ni scripts/study_operator."
  ],
  "status": "active",
  "passes": false
}
```

## 10. Lo que no se verificó y por qué

- **Ningún modelo se ejecutó** (D1): ni Ollama, ni BGE, ni el cross-encoder, ni NLI, ni HHEM. La
  equivalencia de los arreglos de clase A descansa en tests con dobles y en que los dos verificadores
  offline dan el mismo SHA-256. No hay una comparación de recuperación real antes y después. Para los tres
  arreglos aplicados el riesgo es bajo: no tocan recuperación, prompt ni métricas.
- **El efecto real de B4 y B5 sobre lo recuperado** no está medido. Sé qué proveedores se resuelven y qué
  filtro se activa; no sé cuánto cambia la respuesta final, porque eso exige el índice y el generador.
- **Cobertura de tests:** no medida (`pytest-cov` ausente).
- **Cadena de construcción del índice (P3):** lectura parcial, detallada en §6.3. No ejecuté ningún
  chunker ni limpiador: el tokenizador y el modelo de embeddings del chunker no se cargaron.
- **`results_exporter.py`:** solo revisé dónde calcula estadísticos. Las figuras no se regeneraron.
- **Prosa del paper y del A.3:** no la leí. La afirmación de que ninguna latencia de exp8 se cita se limita
  a `docs/reports/results-summary.md` y `docs/TRACEABILITY_nota3.md`.
- **Causa de las 399 respuestas vacías de exp12:** no investigada; haría falta generar.
- **Otros sistemas operativos:** todo corrió en Windows con `.venv-app` (Python 3.14.3). No corrí la suite
  en Linux ni en CI.
- **Combinación de parches B:** los ocho aplican juntos sobre el HEAD de esta rama y la suite `fable` pasa
  con todos puestos. No corrí `init.sh` completo con los parches aplicados.
- **Simulación de B1:** la tabla de campos que cambiarían sale de re-agregar en memoria las filas de cada
  `results.json` firmado. No ejecuté `scripts/run_benchmark.py` sobre la evidencia.

## 11. Cierre (F5)

Verificación final sobre `2df11c3`, el último commit que toca código o tests. El commit de cierre solo
edita este informe.

| Comprobación | Resultado |
|---|---|
| `bash init.sh` | exit 0 en las tres capas |
| Capa 2, conteo por los puntos de progreso | 2373 pasan, 0 fallan, 6 omitidas, 12 `xfail` estrictos, 0 `xpass` |
| Diferencia con la línea base (2303 pasan) | 70 casos nuevos que pasan y 12 `xfail`, todos en `tests/test_fable_*.py` |
| `scripts/verify_v4_offline.py` | exit 0, SHA-256 normalizado igual al de §2 |
| `scripts/verify_summer_offline.py` | exit 0, SHA-256 normalizado igual al de §2 |
| `git diff --stat 1574d22..HEAD -- experiments paper config src/ui scripts/study_operator .harness` | vacío |
| `scripts/scan_secrets.py --baseline secrets_baseline.json` | exit 0 (capa 1) |
| Árbol tras correr los verificadores | limpio: los dos reportes diarios se reescriben idénticos |

Revisión independiente del diff de `src/` y `tests/` (una pasada, agente revisor de código): 0 hallazgos
críticos y 0 altos. Uno medio y dos bajos:

- Medio, corregido en `2df11c3`: el test de la guarda de `PYTHONHASHSEED` dejaba la variable centinela
  `_CLOUDRAG_HASHSEED_REEXEC=1` en el entorno del proceso de tests. Ahora se fija y se borra con
  `monkeypatch`, que la restaura al terminar.
- Bajo, corregido en `2df11c3`: el margen temporal del test de `query_timeout` pasó de 50 ms a 40 ms,
  porque `sleep(0.05)` puede volver antes en Windows.
- Bajo, documentado en §4 (notas de A1): `os.replace` añade modos de fallo en Windows si otro proceso
  tiene abierto el archivo de caché. El fallo se registra con un aviso y la corrida sigue, igual que antes.

Lo que este cierre no cubre:

- `init.sh` completo no se repitió después del commit de cierre. Sobre el árbol final sí corrieron la
  capa 1 y la capa 3, que son las que un cambio de documentación puede romper.
- Una corrida anterior de `init.sh`, sobre `848dfff`, dio los mismos conteos (2373, 0, 6, 12).
- Nada se publicó: sin push, sin merge y sin PR (D3). La rama solo existe en este equipo.

Para revisar: `git log --oneline 1574d22..fable/rag-auditoria`.
