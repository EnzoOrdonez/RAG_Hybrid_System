# Cierre operativo: registro de trabajo del 1 de octubre de 2026

Última verificación: 2-oct-2026 (timestamps de los logs externos). El directorio
de evidencia conserva la fecha de inicio, no oculta la fecha final.

## Alcance y crítica previa

Base verificada: `fix/interview-readiness`, `7e9fc9c`. No se modifican la
configuración sellada, SURVEY_DEPLOY, dependencias, evidencia histórica ni artículo.
No se ejecutan cohortes NLI/compuerta ni se conceden decisiones GO.

El plan requiere distinguir cuatro cosas que no son equivalentes:

- Una suite sin seguimiento no demuestra un bloqueo. Cada corrida tiene log,
  identificador recuperable y un límite de 600 s para su propio árbol de procesos.
- `pure_decline` en v2 es un marcador dentro de los primeros 300 caracteres,
  no prueba de abstención sin afirmaciones. No se cambia esa definición histórica.
- Una prueba con dobles no verifica la red de Zoom, el aislamiento de Windows ni
  la identidad física de un disco. Esos controles necesitan evidencia humana.
- La evidencia retrospectiva genera hipótesis; no autoriza sustituir tareas ni
  cambiar umbrales a partir de las respuestas observadas.

La alternativa de modificar la venv o desactivar servicios para lograr que pase
la suite no permite atribuir el defecto. Se mantiene el entorno y se investiga
la frontera concreta de salida de los subprocesos de PowerShell.

## Baseline ejecutado, antes de editar código

Evidencia externa: `C:/CloudRAG/study-closure-20261001/`.

| Corrida | Resultado verificado |
|---|---|
| `baseline-fast.log`, UTF-8, `not slow and not gpu` | 735 aprobados, 5 fallos, 5 excluidos; 149,86 s |
| Comprobación de los cinco fallos, codificación nativa | 5 aprobados; 2,35 s |
| `baseline-full-native.log`, sin filtro | 745 aprobados, 9 subpruebas, 3 avisos SWIG; 228,13 s |
| `baseline-ruff.log` | Todos los Python cambiados desde 0153ffa: aprobado |
| `baseline-secrets.log` | Cero hallazgos activos; 17 exclusiones del baseline existente |
| `git diff --check`, `git status --short` | Sin diferencias antes de editar |

Los cinco fallos son `test_non_admin_manager_refuses_without_files`, las dos
instancias de `test_manager_rejects_checkout_as_evidence_root`,
`test_non_admin_cannot_attempt_system_registration` y
`test_memory_probe_refuses_non_admin_before_creating_files`. Los lectores de
`subprocess.run(..., text=True)` fallan al decodificar el byte 0xA0 como UTF-8;
`result.stderr` queda en None y la aserción falla con TypeError. Al cambiar sólo
PYTHONUTF8 a 0, pasan los cinco. Enzo autorizó expresamente corregir las pruebas
para ambas modalidades. La solución compara el contrato ASCII sobre bytes, sin
adivinar una página de códigos ni descartar errores con `errors=ignore`.

La suite completa no se colgó en esta reproducción. La prueba más lenta fue
`test_local_model_resolution.py::test_deployed_retrieval_reproduces_exp18_ids_exactly`
(101,17 s). El volcado a los 60 s es diagnóstico, no un test fallido ni un aborto.
La causa de las horas consumidas por los PID históricos no puede certificarse
retrospectivamente: no se conserva aquí su stack. La pérdida de identificadores
de seguimiento en llamadas anteriores es un defecto de supervisión, no evidencia
de un fallo del intérprete. No se recreó ni reinstaló la venv.

## Criterios previos de aceptación

Cada pieza debe superar suite completa, Ruff de sus Python, escaneo contra
`secrets_baseline.json` y `git diff --check`, conservando logs externos.
La corrección de codificación además se comprueba con PYTHONUTF8=0 y 1.

Registro de declinación: misma función v2 de la evidencia, texto de presentación
idéntico, clase y versión en las ocho respuestas, error técnico no clasificado,
sin reintento por declinación y sin registro de familiarización. Los conteos por
condición/tarea acompañan a F4 sin alterar los contrastes SUS/F/U.

Smoke real: P999, purpose=smoke, SMOKE_NOT_GATE; seis tareas, dos libres, SUS y
Likert dos veces, C1–C4, cegamiento, exportación y respaldo verificados, cero
contenido/latencias/errores persistidos de práctica. Formularios sintéticos,
modelos reales. Sin reintento automático después de un fallo. Si no se puede
asegurar un máximo de quince minutos, se entrega lanzamiento humano; las pruebas
sintéticas y AppTest no sustituyen una sesión real en navegador.

## Decisiones y resultados de implementación

Enzo confirmó explícitamente en esta sesión reutilizar exclusivamente v2 y
mantener separadas sus tres clases junto a F4; no modificar el formateador ni
denominar abstención a su suma. La función fue movida, no reimplementada:
`src/evaluation/decline_classifier.py::classify_response`. El script histórico
`scripts/compute_faithfulness_metrics.py` reexporta el mismo objeto.

Cadena verificable: `compute_faithfulness_metrics.py` genera `decline_census_v2`;
`_export_paper_tables_nota3.py` genera `tabla6c_clasificacion_v2`;
`_make_figures_nota3.py` genera `f3_census_declinacion_v2` desde ese censo.
La numeración «Tabla 7/Figura 6 del artículo V2.14» es DECLARADA por Enzo, no
una nueva revisión del artículo en esta tanda. No se editó `paper/`.

| Pieza | Commit | Verificación completa y dirigida |
|---|---|---|
| Codificación de cinco pruebas Windows | `472aa46` | 745 aprobados + 9 subpruebas, UTF-8; cinco también en modo nativo |
| Clase/versión v2, exportación, análisis y UI | `8e31c5c` | 767 completos / 762 filtrados; 9 subpruebas en ambos; 22 casos nuevos |
| Recuperación de errores y guardados | `af674c4` | 783 completos / 778 filtrados + 9 subpruebas; 16 casos nuevos |
| Lanzador humano P999 y cierre documental | Commit de esta entrega, registrado en audit-git.log | 809 completos / 804 filtrados; 9 subpruebas en ambos; 26 casos nuevos |

Logs por pieza: `encoding-*`, `declination-*`, `robustness-*` en el directorio
externo de evidencia. Ruff, secretos con baseline y diff-check pasaron en las
tres piezas. Los cinco puntos excluidos por los marcadores se ejecutaron en
cada suite completa. No se modificaron dependencias.

Fallos conservados: `declination-red.log` tiene 20 fallos previos a implementar;
`robustness-red.log` tiene 9 fallos (uno era un error del propio test: tratar una
ruta string como Path, corregido) y 3 aprobados. `robustness-backup-red.log`
reprodujo el defecto de no registrar pending cuando falla mkdir: 1 fallo y
15 aprobados. Los controles de publicación usan `os.replace` real con inyección
de ENOSPC/PermissionError, no una simulación de un booleano de éxito.

## Evidencia retrospectiva — HIPÓTESIS, no validación prospectiva

Inventario reproducible en `C:/CloudRAG/study-closure-20261001/retrospective.json`:
ruta, SHA-256, fecha, clase y diferencias de configuración por respuesta.
Se leyeron los patrones allí registrados: `*/latency-cohort/hybrid-*.json`,
`*/measurements/attempts/*/result.json`, `*/cohort/attempts/*/result.json`.
Las 16 respuestas encontradas tienen status success, sin error, y configuración
idéntica al `SURVEY_DEPLOY.model_dump()` actual, incluido balance por proveedor.
Esto no demuestra equivalencia de todas las condiciones de ejecución históricas.

| Tarea | Fuentes / fechas UTC | Configuración | Clase v2 observada |
|---|---|---|---|
| q001 | routing 5-sep; cohort-61664 6-sep; warm-protocol 9–10-sep; lexical-clean 11-sep; diag-run 20-sep de 2026 | SURVEY_DEPLOY, Granite 4.1:8b, balance activo | 5 answered, 3 hedged_partial |
| q010 | Las mismas cohortes y fechas | Igual | 8 pure_decline (marcador temprano, no prueba de abstención) |
| q064 | `output/audit/survey_config_latency_2026-08-04.json`, 4-ago-2026 | SURVEY_DEPLOY k=5 y variante k=10; Granite | No determinable: sólo latencias, longitud y conteos, sin texto |
| q070 | No encontrada en las fuentes inspeccionadas | No verificable | Sin evidencia clasificable |
| q171 | No encontrada en las fuentes inspeccionadas | No verificable | Sin evidencia clasificable |
| q172 | No encontrada en las fuentes inspeccionadas | No verificable | Sin evidencia clasificable |

No se extrapolan estas frecuencias al estudio, no se cambian tareas y no se
presentan como estimaciones de probabilidad de fallo. Los hashes identifican
los archivos leídos; no vuelven válidas ventanas históricas abortadas.

## Hallazgos heredados que impiden lanzar la compuerta

Revisión del runner `scripts/run_study_gate.py`, sin ejecutarlo en modo real:

1. `main` no conecta `make_app_adapter`; sin `--dry-run` el primer intento llega
   a `Real adapter required`, se etiqueta terminal y no mide la aplicación.
2. `_preflight` valida booleanos declarados, no Zoom vivo, sesenta segundos de
   carga CPU/GPU ni ausencia de procesos ajenos.
3. `run(window=2)` admite un directorio nuevo sin verificar ventana 1;
   `run_cohort` sólo comprueba su estado, no revalida identidad/configuración.
4. El límite monotónico se comprueba entre llamadas; no hay supervisor independiente
   para una llamada bloqueada. Los intentos se publican al terminar el bucle, no
   como checkpoint durable previo a cada consulta.

Por ello no se proporciona un comando real que aparente cumplir el pre-registro.
La suite sintética verde **no cierra estos requisitos**. Repararlos y demostrar
sus controles es trabajo pendiente antes de las ventanas humanas; esta tanda
no cambió el runner ni los criterios aprobados para ocultar esos hallazgos.

## Archivo de aislamiento en D:

Se localizó y leyó `D:/scripts/isolate_compute_env.ps1`. No se ejecutó Isolate en
esta continuación. Su propio texto advierte que un kill forzado evita `finally`;
además incluye todos los navegadores en su lista y no restaura aplicaciones no
declaradas en RestoreApps. No es por sí solo una admisión ni una garantía B.4.
No debe ejecutarse indiscriminadamente durante Zoom o con formularios abiertos.
AnyDesk, servicios, procesos ajenos y configuración del sistema no se modificaron.

## Smoke y cierre humano

Enzo aceptó explícitamente el lanzador con recorrido humano y verificación
automática, sin instalar automatización de navegador. `scripts/study_smoke.py`
usa la UI y fábrica existentes, responde sólo a `launch` humano y no sustituye
la sesión por llamadas directas. La prueba de cableado sustituye el servidor y
el recorrido por dobles: es **sintética**, no un smoke real. Sus 26 casos dirigidos
pasaron; la suite final queda registrada en `final-full.log` y `final-fast.log`.
El límite predeterminado humano de 45 minutos es un límite operativo configurable,
no una predicción ni un umbral de latencia del estudio.

`final-full.log`: 809 aprobados, 9 subpruebas, 3 avisos SWIG, 285,91 s.
`final-fast.log`: 804 aprobados, 5 excluidos, 9 subpruebas, 3 avisos SWIG, 139,52 s.
Total añadido desde 7e9fc9c: 64 casos parametrizados; no se afirma que sean
64 funciones. `final-all-ruff.log`, `final-secrets.log` y `final-diff-check.log`
documentan los controles estáticos, sin hallazgos activos nuevos.

El lanzador sólo posee su servidor local y descendientes; no gestiona servicios.
La limpieza kill-on-close usa el helper existente, cuya propiedad de descendientes
está probada por `test_managed_gate.py::test_killing_job_owner_terminates_grandchild`.
La prueba del lanzador comprueba que sólo se solicita terminar el PID propio vivo.
No se ha ejecutado aquí una muerte forzada de un servidor Streamlit real.

El respaldo de P999 resuelve DiskNumber y puntos de montaje y rechaza topologías
virtuales/pools/red no demostrables. La función genérica `study_backup.backup_export`
conserva su callback por defecto basado en letra de unidad: **fuera del lanzador
P999 sigue siendo necesario verificar físicamente el destino con el procedimiento
humano**. No se presenta ese control genérico como una comprobación física automática.
Además, el control heredado de pending está en `StudyStore.issue`, no en `admit`:
una invitación emitida antes de un fallo de copia requiere bloqueo humano de
admisión. Esa limitación sigue abierta y debe corregirse antes de uso con personas.
La sonda sólo-lectura del 2-oct (`disk-inventory-readonly.log`) identificó C: en
DiskNumber 0 y D: en DiskNumber 1, ambos NVMe. No se seleccionó ni escribió un
destino real de respaldo durante esta tanda.

Ver [instrucciones P999](STUDY_SMOKE_P999.md), [notas UX](STUDY_UX_NOTES.md) y
[checklist humano en orden obligatorio](STUDY_HUMAN_CHECKLIST.md). Smoke real,
NLI, compuerta, ensayo P998, piloto P900/P901 y estudio siguen sin ejecutar en
esta tanda. Sólo las pruebas de software y el análisis retrospectivo están verificados.

## Matriz de riesgos residuales

| Modo de fallo | Probabilidad / efecto | Control o test | Riesgo que queda | Responsable |
|---|---|---|---|---|
| Codificación PowerShell | Desconocida; falsos fallos de tests | Cinco guardas en bytes; pruebas en UTF-8 y nativo | Otros consumidores de texto externo no auditados exhaustivamente | Mantenedor |
| Caída de Ollama | Desconocida; tarea sin respuesta | `test_ollama_failure_is_durable_technical_error_and_retry_is_explicit` | Servicio/modelo reales pueden fallar; no reintento automático | Operador |
| Recarga/doble envío | Desconocida; duplicación/pérdida de estado | Test con trabajador vivo y FileLock real, AppTest de reconexión | Red/Zoom no comprobados con dobles | Operador + mantenedor |
| Disco lleno/denegado | Desconocida; imposibilidad de guardar | Inyección en publicación de solicitud/resultado/instrumentos; rollback | Texto aún no escrito y formularios no enviados no son recuperables por garantía | Operador |
| Respaldo inaccesible/mismo disco | Desconocida; pérdida ante fallo primario | Pending/reintento/hash; mapa físico en launcher P999 | Función genérica fuera de P999 requiere control humano; fallo simultáneo | Operador |
| Clasificación v2 mal interpretada | Desconocida; sobreestimar abstención | Misma función y tres clases separadas con F4 | Marcador no es un juicio semántico de contenido | Investigador |
| Práctica persistida fuera de la app | Desconocida; incumplimiento del protocolo | Regresión de silencio/ephemeralidad y estructura de exportación | Grabaciones/capturas externas requieren control humano | Operador |
| Escape de Zoom/Windows Home | Desconocida; exposición de archivos/programas | Checklist B.4, cuenta y permisos verificables | Sin verificación operativa no hay garantía de aislamiento | Enzo |
| Compuerta incompleta | Desconocida; falso GO operativo | Bloqueo explícito por hallazgos del runner | Telemetría, supervisor, revalidación y CLI real pendientes | Mantenedor; autorización Enzo |
| Deriva del sello/reloj | Desconocida; cambio de tratamiento/latencia errónea | Rechazo antes de consultar; regresión reloj monotónico | Timestamps civiles pueden cambiar; auditoría humana del entorno | Operador |

No se asignaron porcentajes de probabilidad sin medición. No se reclutaron personas,
no se activó Isolate del archivo de aislamiento, no se modificaron servicios, cuentas,
políticas, digest, modelo, driver, preregistros, corpus, gold, evidencia congelada
ni artículo. Sin push, merge, rebase, amend ni cambios de venv.
