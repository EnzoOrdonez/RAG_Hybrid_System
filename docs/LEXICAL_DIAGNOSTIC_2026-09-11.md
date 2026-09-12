# Diagnóstico léxico prospectivo — 2026-09-11

## Tercera tanda: segunda ventana detenida por Brave (2026-09-11, Lima)

**VERIFICADO: diagnóstico incompleto; NO-GO vigente.** Se ejecutaron 29/40
posiciones con el build exacto `658b7487bc570033067a1c012d8023fc05f9a2c9`:
28 respuestas válidas (14 pares), una respuesta completada con condiciones
inválidas y 11 posiciones sin iniciar. Cero errores de generación, timeouts o
intentos abortados. La posición inválida no entra en percentiles ni se reemplaza.
La ventana terminó por incumplimiento de condiciones, no por defecto del reloj.
Los 14 pares previos se conservan como evidencia parcial, sin convertirlos en la
cohorte confirmatoria prevista de 20 pares ni mezclarlos con cohortes históricas.

Evidencia nueva: `C:/CloudRAG/lexical-clean-20260911T2020Z/` (en adelante `R`).
El análisis es `R/partial-analysis.json`, SHA-256
`f58c09fa46ddc5d50a833e8bd2f6b30ee4daa347b28708c2e41cde294d3a1f8c`.
Incluye filas, hashes, llamadas NLI reales y observaciones que interpolan cada
p95; `R/analysis-inventory.json` fija sus fuentes antes de derivar resultados.
El análisis y las cinco pruebas externas de reloj/análisis quedan fuera del
checkout; no se alteró el build medido para añadirlas.

### Contrato, baseline y supervisión

- `R/measurement-contract.md`: crítica y fronteras fijadas antes del corte.
  El reloj incluye selección/comprobación de preparación, consulta completa,
  serialización y extracción de trazas a memoria. Excluye precalentamiento,
  instalación de probes, inicio/cierre del observador, validación ambiental
  posterior y publicación final. Incluye el inicio durable y arranque del
  heartbeat. `R/clock-boundaries-02.txt` prueba nueve fronteras del ejecutor y
  la del wrapper; complementa la regresión versionada de 2 s + 3 s = 5 s.
- `R/baseline-pytest.txt`: 538 pasan / 5 excluidos, tres avisos SWIG, 70,65 s.
  Ruff y diff aprobados. El primer escaneo se lanzó sin baseline y devolvió
  los 17 hallazgos históricos; `R/baseline-secrets-with-baseline.txt` verifica
  cero activos con 11 exclusiones de corpus y 6 de evidencia firmada.
  Un fallo inicial de captura de stderr de PowerShell y otro de comillas del
  lanzador precedieron las pruebas completas; no fueron fallos del pipeline.
- `R/environment-before.json`, 20:25:25 UTC: driver 616.64, Ollama 0.22.1,
  digest Granite esperado `444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852`,
  AnyDesk detenido. Snapshot y verificaciones del bundle aprobados; manifiesto
  SHA-256 `0bb1957f6ce0426214c4ca121f162638db0947e9f95b22fc214fad7034caec10`.
- Supervisor independiente probado antes del corte: deadline a las
  20:27:22 UTC y pérdida del controlador a las 20:27:29 UTC, en
  `R/proof/{deadline,controller}/selftest-passed.json`. Servicios simulados
  durante estas dos pruebas, no una afirmación de restauración real anticipada.
- Corte a las 20:27:52 UTC. Se detuvieron Overlay PID 3080, 3120, 4312,
  13088 y 29076. El reinicio PID 32084 quedó vinculado a NvContainer en
  `R/window-01/overlay-relaunch-proof.json`; solo entonces se intervino
  `NvContainerLocalSystem` (PID previo 31332). Se desactivó temporalmente la tarea
  `NVIDIA App SelfUpdate_{B2FE1952-0186-46C3-BAEC-A80AA35AC5B8}`.
  AnyDesk y NVDisplay quedaron fuera de la intervención.
- Admisión: 13 muestras, aprobada, sin razones de invalidez. Contraste a las
  20:34:31 UTC: diez pares, mediana -0,6104 %, límite superior unilateral 95 %
  -0,02277 % frente a 5 %, aprobado. No se descontaron tiempos RAG.

### Causa concreta del cierre y restauración

**VERIFICADO:** intento híbrido índice 14, ID
`ab41ff5e41654570ae215632c93848e9`, respuesta de 20,1455 s completada,
`conditions_invalid=true`, razón `prohibited_process`. La primera muestra de
su telemetría, 21:12:09.849592 UTC, registra Brave y procesos GPU Brave PID 2212
y 35520. Archivo:
`R/cohort/telemetry/85869fad697f4959b453f42195eff97d.jsonl`.
Esto demuestra presencia del proceso prohibido, no quién lo abrió ni cuánto
afectó la latencia. El ejecutor detuvo la secuencia al registrar el resultado.
Invalidez de condiciones: 1/29 (3,45 %); fallos de generación/abortos: 0/29.

Restauración real a las 21:13:12.9649008 UTC (16:13:12 Lima), antes del límite
duro 22:27:34 UTC. NvContainer restaurado/verificado a las 21:13:09 UTC;
tarea habilitada a las 21:13:11 UTC. `R/restoration-audit.json` vuelve a
verificar el estado a las 01:52:32 UTC del 12-sep: Overlay PID 12976, 28720,
32976, 35556 y 35724; NvContainer Running/Auto; NVDisplay Running/Auto;
AnyDesk Stopped/Auto; tarea habilitada; watchdog retirado. El registro
`controller_returned` acredita el retorno del controlador y la restauración,
**no** el éxito del diagnóstico completo. No se abrió una tercera ventana.

### Resultados parciales descriptivos, no confirmatorios

Cada celda muestra p50 / p95 sobre 14 respuestas válidas por sistema. Los
percentiles de etapas son independientes y no se suman.

| Medida | Híbrido | Léxico |
|---|---:|---:|
| Respuesta completa (s) | 33,79 / 64,71 | 25,03 / 68,99 |
| Retrieval (s) | 0,1307 / 0,1455 | 0,0330 / 0,0407 |
| Re-ranking (s) | 4,2272 / 4,9509 | <0,0001 / <0,0001 |
| Generación (s) | 19,55 / 46,46 | 19,27 / 49,95 |
| Verificación, etapa NLI (s) | 3,46 / 12,66 | 3,60 / 16,95 |
| Chunks al generador | 5 / 5 | 5 / 5 |
| Tokens de salida | 230,5 / 563,55 | 212,5 / 659,5 |
| Claims extraídos | 3,5 / 9 | 4 / 13,15 |
| Llamadas NLI reales | 2,5 / 9 | 3,5 / 12,8 |
| Pares NLI reales | 12,5 / 45 | 17,5 / 64 |
| Prompt de usuario (caracteres) | 8885 / 13587,4 | 7165,5 / 10428,8 |
| Respuesta (caracteres) | 1191,5 / 2474 | 1073 / 2662,65 |

El p95 parcial léxico de 68,9869 s interpola q001/q016 con pesos 0,65/0,35.
Al aplicar esos mismos pesos a sus componentes: generación 49,9474 s,
verificación 16,9477 s, retrieval 0,0329 s y otros 2,0589 s (más componentes
residuales inferiores a 0,00001 s). Esto contabiliza el total y su exceso
descriptivo de 8,9869 s sobre 60; **no** demuestra qué intervención ahorraría
esos segundos. No se adjudica causalmente el exceso repartiendo percentiles.

En q016, léxico/híbrido: 77,3821/49,6160 s, 666/378 tokens, 19/9 claims,
90/45 pares NLI. Generación añade 20,2255 s y verificación 12,2538 s al léxico;
su ahorro en retrieval/re-ranking compensa aproximadamente 4,7131 s. Es una
observación prospectiva compatible con la hipótesis, no confirmación de la cola
completa. q057, otra candidata histórica, no llegó a medirse en esta ventana.
Tampoco se oculta la respuesta híbrida q027 de 92,7320 s, 845 tokens y 40 pares
NLI: el p95 parcial híbrido también supera 60 s. No reemplaza su piloto previo
ni autoriza a ignorar esta nueva señal.

**Crítica:** AB/BA y verificaciones de residencia acotan orden y recarga, pero
no eliminan interacción con contexto/plantilla, variación de generación ni toda
carga residual. El cierre temprano deja seis pares sin completar y no permite
confirmar la hipótesis exigida. No atribuir estos resultados exclusivamente al
driver, a RAM o a hardware, ni usar la aparición de Brave para explicar respuestas
previas donde el observador no lo registró.

### Parada y continuación necesaria

No se redacta una optimización como si su hipótesis estuviera confirmada: el
pre-registro de mejora sigue bloqueado. No se implementaron optimizaciones, P2s
ni nube; P900 y su evidencia de resiliencia no se modificaron.
Una continuación que aspire a veinte pares válidos requiere autorización
explícita para **otra ventana y una cohorte nueva completa**; reponer el índice
14 inválido o juntar estos catorce pares con otra ventana violaría el protocolo.
Antes de esa nueva ventana: cerrar Brave y comprobar ausencia con
`Get-Process brave -ErrorAction SilentlyContinue`; mantenerlo cerrado durante
toda la medición. Esta es una acción futura propuesta, no ejecutada ni autorizada
por silencio. El pre-registro de la compuerta y el NO-GO vigente no cambian.

## Segunda tanda: crítica y contrato antes de implementar

Baseline `94e73a5`: 522 tests pasan / 5 excluidos. Evidencia nueva:
`C:/CloudRAG/lexical-clean-20260911T1245Z/baseline-*`.
El usuario autorizó **una** ventana NVIDIA de diagnóstico. Aclaró después que
AnyDesk debe permanecer **apagado e intacto**, porque está presente localmente.

El supervisor anterior modifica NvContainer incondicionalmente y contempla Epic;
no cumple esta autorización estrecha. Se añadirá un perfil diagnóstico explícito:
solo Overlay, tareas SelfUpdate identificadas y servicio NvContainer si se observa
reinicio del overlay vinculado a él. La cadena parental demuestra origen, no
reinicio. No se tocarán AnyDesk, Epic, NVDisplay ni el driver. Primero se prueban
deadline y pérdida del controlador con restauración simulada independiente; luego
se arma la ventana real. Cada intención se persiste antes de modificar su objeto.
Si el servicio no se interviene, tampoco se reinicia al restaurar. Se verifica la
reaparición del overlay raíz, sin exigir PID o cantidad de subprocesos idénticos.

El ejecutor anterior exige una cohorte de tres sistemas; se añadirá una entrada
diagnóstica separada, sin cambiar su interpretación histórica. Los cuarenta slots
AB/BA definidos abajo comparten proceso y preparación. Un manifiesto nuevo fija
fuentes, consultas, receta, hashes y entorno; la recuperación conserva abortos y
prohíbe duplicados. El diagnóstico **no emite GO**. La ventana no se renueva ni
extiende al reintentar: cualquier ejecución posterior a restauración exige otra
autorización. No se ejecuta un segundo intento ante una condición inválida.

Aceptación antes de la ventana: suite completa, Ruff, diff y secretos; regresiones
de alcance/ancestría, orden, receta, residencia, slots inválidos, deadline y
restauración; pruebas del watchdog sobre este mismo commit. Contraste sintético
pareado con instrumentación, 10 pares y límite superior unilateral bootstrap del
95 % <=5 %, sin descontar tiempos. El análisis informará componentes de las dos
observaciones que interpolan el p95 total, sin sumar percentiles independientes
ni llamar ahorro causal al tiempo observado. La mejora permanece sin autorizar.

### Defecto detectado durante la primera ventana, antes de los slots

En `b47af43`, el coordinador llamaba `Preparation.pipeline()` antes de
`measure_traced_attempt`. El ejecutor histórico hace esa selección dentro de su
callback cronometrado. Aunque ambas rutas verifican residencia, quitar ese costo
del total impide comparar el mismo reloj. La revisión lo detectó después del
contraste sintético y antes de iniciar las cuarenta posiciones. No se corrige
sumando retrospectivamente una constante ni reinterpretando el umbral.

Se pidió restauración inmediata; terminó a las 13:08:36Z. La ventana se consumió.
Se observaron cero request/result de posiciones diagnósticas y una preparación
iniciada sin cierre. No se atribuye duración de inferencia a esa interrupción.
La corrección mantendrá la instalación de probes fuera del reloj, pero ejecutará
la selección/validación de preparación dentro del callback, como la compuerta.
Una regresión de reloj determinista exigirá que 2 s de selección + 3 s de consulta
produzcan 5 s de respuesta completa. No se abre otra ventana sin autorización.

Evidencia VERIFICADA en la raíz de esta segunda tanda:

- `proof/deadline/selftest-passed.json`, 13:00:16Z, y
  `proof/controller/selftest-passed.json`, 13:00:23Z: watchdog independiente,
  servicios simulados; SHA del supervisor
  `3383b0d7c5214bbc7d05a83a8a84826f22ee7f48629481d8728cf991472b0a5e`.
- `window-01/overlay-relaunch-proof.json`: tras el corte a las 13:00:45Z apareció
  PID 30888, hijo de nvcontainer 30956, hijo del PID 8616 del servicio autorizado.
  Solo entonces se desactivó temporalmente NvContainerLocalSystem.
- `window-01/admission-before-contrast/result.json`: admisión aprobada.
- `window-01/contrast/result.json`, 13:07:21Z: diez pares, mediana -0,0379 %,
  límite superior unilateral 95 % de 0,1899 %, umbral 5 %. Es contraste sintético,
  no una medición de latencia RAG ni un permiso para descontar tiempos.
- `restore-request.json` y `window-01/restored.json`, 13:08:36Z: restauración
  real completada con AnyDesk intacto. El launcher registró `restored=false`
  cinco segundos antes mientras la restauración independiente seguía trabajando;
  no se sobrescribe ese registro intermedio ni se confunde con el cierre posterior.
- A las 13:09:43Z se observaron cinco procesos NVIDIA Overlay nuevos (3080,
  3120, 4312, 13088, 29076), NvContainer en Running y NVDisplay en Running;
  AnyDesk en Stopped. Los PID son evidencia temporal, no identificadores reutilizables.
- `clock-red.txt` conserva el fallo de regresión previo a la corrección.

No se creó `LEXICO_P95_PREPRREGISTRO_2026-09-11.md`: falta el contraste RAG
prospectivo que debe sustentar sus apartados. Los resultados históricos siguen
siendo hipótesis candidatas. Tampoco se ejecutaron optimizaciones, P2s ni nube.

## Baseline y crítica antes de implementar

VERIFICADO: baseline `7c449bf`, rama `fix/interview-readiness`; 507 tests pasan,
5 excluidos, 3 avisos SWIG. Evidencia nueva, exclusivamente técnica:
`C:/CloudRAG/lexical-diagnostic-20260911T1030Z/baseline-*`.
Este documento registra el diagnóstico; **no autoriza una optimización**.

La mayor cantidad de operaciones de retrieval híbrido no implica un mayor costo
total: los contextos y el enrutamiento de prompts difieren. El léxico tiene menor
mediana histórica, pero mayor p95. Debe explicarse la cola, no afirmar que todas
las consultas léxicas son más lentas. Los p95 por etapa no se suman.

Hipótesis ordenadas, aún SUPUESTAS:

1. Contextos y plantillas distintos inducen más tokens de salida en las consultas
   de la cola léxica. Predicción: esas posiciones concentran más tokens y tiempo
   de generación, aun con rendimiento por token comparable.
2. Más claims verificables implican más pares NLI y mayor tiempo de verificación.
   Predicción: los pares realmente enviados al modelo explican la diferencia por
   posición; no basta multiplicar claims por chunks porque existen artefactos,
   declinaciones y fallbacks.
3. Orden, residencia o carga residual contribuyen. Predicción: la diferencia no
   persiste de forma consistente al contrabalancear las mismas posiciones y
   verificar las condiciones de ejecución.

Alternativa rechazada por ahora: acortar respuestas o cambiar batch NLI antes de
medir. Mezclaría explicación con optimización y podría alterar calidad. Se usará
instrumentación externa al código de producción, después del pre-calentamiento:
envoltorios temporales que conservan argumentos, resultados y excepciones; sin
forzar carga de modelos. Los registros se publican fuera del cronómetro al cerrar
el intento. Si falta la prueba de no interferencia, no se afirma que la medida sea
equivalente a la receta sin instrumentación.

## Protocolo del diagnóstico

- Veinte posiciones del manifiesto anterior, idénticas para híbrido y léxico:
  40 intentos calientes nuevos, separados de toda cohorte histórica.
- Índices pares: híbrido→léxico; impares: léxico→híbrido. Preparación de los tres
  pipelines en el mismo proceso; caché desactivada, lectura 180 s, seed 42,
  mismo modelo, digest, driver y entorno. Comprobar residencia antes de cada intento.
- Admisión y observador existentes; contraste pareado antes de inferencias.
  No descontar interferencia de latencias. Fallos, abortos e inválidas se conservan
  y no entran en percentiles ni se reemplazan silenciosamente.
- Persistir tiempos por etapa, entrada exacta al generador con hash, tamaños en
  caracteres y tokens reportados por Ollama, chunks, extracción de claims y cada
  llamada predict con número real de pares, batch_size, softmax y resultado/fallo.
- Conservar diferencias emparejadas y distribuciones de cada sistema. Los registros
  históricos sirven para sugerir mecanismos, no sustituyen el diagnóstico nuevo.

Aceptación de instrumentación: tests de transparencia sobre la ruta real del
detector, restauración incluso ante excepciones, conteo de pares/artefactos/fallback,
rechazo de modelos fríos, y evidencia durable sin sobrescritura. No se cambia la
interfaz pública de RAGResponse ni se modifica la aplicación del participante.

## Aprobaciones y parada obligatoria

Confirmado por el usuario en planificación: «datos nuevos» significa ejecuciones
nuevas del conjunto fijo; el P2 de descarga corresponde al ejecutor; elevaciones
para evidencia externa, telemetría y pruebas autorizadas. AnyDesk/NVIDIA permanecen
activos hasta demostrar necesidad y obtener aprobación concreta de la ventana.

Después del diagnóstico: presentar un pre-registro separado con una única mejora,
efecto esperado, calidad, medición prospectiva y rollback. **Esperar el visto bueno
del usuario antes de implementar la mejora.** Los tres P2 se corrigen después de
su validación aislada. Fase B no se inicia en esta tarea, incluso si hay GO.

## Resultado parcial, 2026-09-11 10:38 UTC

VERIFICADO: `scripts/lexical_diagnostic.py` aporta un contexto temporal de medición
y `measure_traced_attempt`, que publica los probes con el registro terminal del
ejecutor durable. No carga modelos fríos ni altera `src/`. La preparación,
admisión, programación de las 40 posiciones y contraste siguen siendo obligaciones
del coordinador; **no se ha ejecutado ni cerrado la cohorte diagnóstica nueva**.

`scripts/analyze_lexical_diagnostic.py` reconstruyó los 60 resultados calientes
anteriores comprobando sus hashes y los de las llamadas HTTP contra
`final-memory-analysis-01.json` (SHA-256
`2625c89066164580f0ea6718d4075849de70f27464bf8a7e23c5eaa9e5e6df42`).
Salida nueva: `C:/CloudRAG/lexical-diagnostic-20260911T1030Z/historical-workload-01.json`.
Los conteos reales de llamadas/pares NLI históricos quedan `null`: no se inventan.

| Histórico caliente, 20 por sistema | Híbrido | Léxico |
|---|---:|---:|
| Retrieval p50 / p95 s | 0,134 / 0,151 | 0,032 / 0,042 |
| Reranking p50 / p95 s | 4,332 / 5,076 | <0,001 / <0,001 |
| Generación p50 / p95 s | 18,743 / 34,823 | 14,960 / 43,184 |
| NLI p50 / p95 s | 2,794 / 12,562 | 1,385 / 19,132 |
| Prompt caracteres p50 / p95 | 8193 / 12542,2 | 6916,5 / 10216,4 |
| Tokens de salida p50 / p95 | 218 / 423,45 | 178,5 / 600,45 |
| Claims p50 / p95 | 3 / 9 | 2 / 14,05 |

| Consulta / sistema | Chunks | Prompt caracteres | Tokens salida | Claims (artefactos) | Generación s | NLI s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| q016 híbrido | 5 | 8291 | 378 | 9 (0) | 31,12 | 12,64 | 50,49 |
| q016 léxico | 5 | 7574 | 742 | 14 (1) | 52,44 | 17,75 | 72,29 |
| q057 híbrido | 5 | 12102 | 323 | 7 (0) | 29,10 | 9,80 | 46,06 |
| q057 léxico | 5 | 8713 | 593 | 15 (0) | 42,70 | 20,61 | 65,40 |

VERIFICADO: las dos posiciones lentas léxicas tienen más tokens y claims, con
cinco chunks en ambos sistemas y prompts léxicos más cortos. SUPUESTO: esta carga
explica causalmente la diferencia prospectiva. La historia no contrabalanceó el
orden ni registró llamadas predict; falta descartar diferencias de contenido,
residencia y velocidad por token mediante el diagnóstico nuevo. No hay todavía
hipótesis de optimización aprobada ni estimación defendible de su efecto.

VERIFICADO: `/api/version` 0.22.1 y digest esperado, driver 616.64, a las
10:24:30Z. Admisión nueva `admission-01/result.json`, muestras desde
10:37:39Z hasta 10:38:42Z: 14 muestras, GPU media 3,14 %, AC en todas;
rechazo exclusivo `prohibited_process`. NVIDIA Overlay: PID 3716, 30912,
33344, 38144, 38548 en esa ventana. Los PID deberán revalidarse antes de actuar.
AnyDesk permaneció activo; no se justifica cortarlo con estos datos.

No se repitió la ventana fallida ni se ejecutó inferencia. Para continuar se
requiere autorización concreta para NVIDIA Overlay y su mecanismo de reinicio
(`NvContainerLocalSystem` y tareas `NVIDIA App SelfUpdate_*` aplicables), mediante
el supervisor existente, conservando AnyDesk y restaurando el estado previo.
La instrumentación aún necesita contraste de interferencia en la ventana limpia.
