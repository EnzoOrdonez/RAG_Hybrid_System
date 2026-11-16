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
| H-01 | CRÍTICO (riesgo) | B1 | `src/evaluation/benchmark_runner.py:199-215`, `:533-568`, `:703-707` | `BenchmarkRunner` escribe `results.json`, `aggregated_metrics.json` y los checkpoints sin pasar por `guard_write`. Relanzar un experimento firmado con todo ya completado reescribe sus artefactos sin cargar ningún modelo | Traza: `run_experiment` siempre termina en `_save_experiment_results`; `scripts/run_benchmark.py:164` y `scripts/run_llm_only_benchmark.py:138` usan el directorio de evidencia por defecto | Abierto |
| H-02 | MEDIO (latente) | B2 | `src/evaluation/statistical_analysis.py:197-217`, `:297-307` | Dos sistemas con puntajes idénticos dan `p=NaN`; un solo NaN vuelve NaN todos los `p_bh` y `p_holm` de la familia | Reproducido con scipy 1.17.1. La evidencia firmada no contiene ningún `p` NaN | Abierto |
| H-03 | MEDIO | B3 | `src/evaluation/benchmark_runner.py:258-324` | `_run_single_query` ignora `response.error`: una falla del pipeline queda como fila válida y su texto de error se puntúa como si fuera la respuesta | exp3, brazos lineales: q042 y q049 sin fragmentos recuperados en los 3 brazos, con `errors: 0` | Abierto |
| H-04 | ALTO | B4 | `src/retrieval/coverage_balancer.py:86-88` | Los nombres de servicio genéricos del índice (`virtual machines`, `virtual network`, `vpc`, `iam`) añaden un proveedor que la pregunta no compara y le asignan cuota en `SURVEY_DEPLOY` | «Compare virtual machines on AWS and GCP» resuelve `aws, azure, gcp` con el índice real | Abierto |
| H-05 | ALTO | B5 | `src/retrieval/query_processor.py:140-148` | La detección de proveedor es por subcadena: `laws` y `flaws` contienen `aws` y activan el filtro `single_provider` | «What data residency laws apply to object storage?» queda filtrada a AWS | Abierto |
| H-06 | MEDIO | B6 | `src/evaluation/latency_metrics.py:46-51` | Se descartan los valores a más de 3 desviaciones antes de calcular p95, p99 y máximo: se recorta justo la cola que esas cifras deben describir | exp8b denso: p95 41 541 ms recortado frente a 45 630 ms sin recortar | Abierto |
| H-07 | MEDIO (latente) | B7 | `src/evaluation/benchmark_runner.py:140` | La caché de índices usa `estrategia_tamaño` e ignora el modelo de embeddings: dos configuraciones con modelos distintos comparten el primer índice cargado | Traza de código; afecta a `exp2`, que no es evidencia firmada | Abierto |
| H-08 | MEDIO | B8 | `src/generation/llm_manager.py:304-311` | `generate()` guarda en caché una respuesta vacía y la sirve para siempre; `generate_stream()` sí la descarta | exp12 tiene 399 respuestas vacías de 3104 (N6.4 documenta 177 de qwen sin RAG) | Abierto |
| H-09 | BAJO | A | `src/generation/llm_manager.py:180-197`, `:319-328` | La caché se escribe sin atomicidad y una caché ilegible se descarta en silencio; el log dice «retrying» en el último intento | Traza de código y 3 tests en rojo antes del arreglo | Corregido (A1) |
| H-10 | BAJO | A | `src/evaluation/benchmark_runner.py:7,38,41-43` | El docstring promete un timeout de 120 s por consulta que no existe; `TimeoutError` tapa el builtin y nadie lo usa | Traza de código; `grep` da 0 usos y 0 `import *` | Corregido (A2) |
| H-11 | BAJO | A | `src/retrieval/coverage_balancer.py:99-102` | El docstring afirma que el orden de relevancia se conserva «dentro y entre» selecciones; el relleno puede dejar un fragmento mejor rankeado detrás | `A1,A2,A3,A4,B1` con cuota 3 da `A1,A2,A3,B1,A4` | Abierto |

## 4. Commits de clase A

Cada arreglo siguió el ciclo test rojo, cambio mínimo, test verde y tests del módulo. La columna de
equivalencia dice por qué ningún resultado cambia.

| ID | Commit (asunto) | Hallazgo | Rojo antes | Prueba de equivalencia |
|---|---|---|---|---|
| A1 | `fix: caché LLM atómica, aviso si es ilegible y log veraz del último intento` | H-09 | 3 de 5 tests de `tests/test_fable_llm_cache.py` fallaban: caché truncada tras un corte de escritura, caché ilegible sin aviso y «retrying in 4s» en el último intento | `test_saved_bytes_are_identical_to_the_legacy_serialisation` compara byte a byte con la serialización anterior; `tests/test_benchmark_parity.py` (clave de caché congelada y llamada a Ollama idéntica) sigue verde; 108 tests de los 8 módulos que usan `LLMManager` pasan. No cambian la clave, el contenido de la caché, los reintentos ni sus esperas (`sleeps == [2]`) |
| A2 | `fix: el runner deja de anunciar un timeout por consulta que no aplica` | H-10 | `test_the_module_does_not_advertise_a_timeout_it_does_not_apply` fallaba: el docstring decía «120-second timeout per query» y existía la clase `TimeoutError` | Solo cambian el docstring, un comentario y se borra una clase sin uso (0 referencias en `src`, `scripts` y `tests`; 0 `import *`). `test_query_timeout_is_accepted_but_never_enforced` fija que el parámetro se sigue aceptando y sigue sin aplicarse. `scripts/audit/smoke_hall_n_effective.py` pasa. Verificadores offline: SHA-256 normalizado igual a la línea base |

Notas de A1:

- La escritura usa un archivo temporal hermano y `os.replace`. Si el archivo de caché fuera un enlace
  simbólico de archivo, `os.replace` sustituiría el enlace por un archivo normal. En este repositorio
  `data/llm_cache/` contiene archivos normales, así que el caso no se da.
- El docstring del módulo ahora dice la verdad sobre el timeout: 60 s para OpenAI, y para Ollama solo con
  `enforce_timeout=True`. El camino por defecto de los benchmarks llama a `ollama.chat` sin timeout de
  cliente. El comportamiento no se tocó.
