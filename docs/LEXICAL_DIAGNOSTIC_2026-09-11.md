# Diagnóstico léxico prospectivo — 2026-09-11

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
