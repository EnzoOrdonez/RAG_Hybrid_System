# Compuerta local: registro de ejecución

## Crítica previa a la corrección

El navegador real mostró rutas de operador antes del login en modo participante.
La evidencia inicial permanece en
`C:/CloudRAG/operational-20260905T1428Z/finding-participant-navigation.json`.
`st.stop()` no impide el descubrimiento de `pages/`. Tampoco basta invocar
`st.navigation` dentro de la entrada mientras existe esa carpeta: una primera
visita directa puede ejecutar la página autodetectada antes de la entrada.

La corrección acordada mueve las pantallas a `src/ui/views/` y registra una sola
página de participante. Mantiene las herramientas de desarrollo privadas y el
protocolo existente, incluidos descansos y preguntas abiertas. No se cambian
modelos, recuperación, cuestionarios ni el umbral de latencia.

AppTest puede ejecutar el caso de una página (comprobado en memoria con Streamlit
1.54.0), pero no sustituye la prueba de rutas desde navegador. Las regresiones
comprobarán el registro antes/después del login y la ausencia de módulos
autodetectables; el navegador probará también la primera URL tras reiniciar.

El traslado incorpora siete avisos Ruff preexistentes en pantallas de desarrollo.
Se eliminan únicamente imports y asignaciones sin uso, conservando los controles;
no se aprovecha esta corrección para cambiar la lógica de esas pantallas.

Baseline provisionado: 383 pruebas aprobadas, 5 excluidas, ninguna omitida.
Las dos regresiones de rutas fallaron antes de la corrección; después pasaron junto
con la suite completa: 385 aprobadas, 5 excluidas (12,07 s).

## Criterio y alcance

GO local requiere cuatro pasos satisfactorios: bundle, sesión completa, latencia y
resiliencia. Medición acordada: 20 consultas frías y 20 calientes por sistema,
p95 ≤60 s en cada combinación, sin excluir fallos ni observaciones lentas.
Las sesiones son sintéticas y se almacenan fuera del checkout. La preparación de
nube solo comienza después de GO local; no autoriza despliegue ni entrevistas remotas.

## Evidencia operativa verificada

Build ejecutado: `4af97286d43bc54af4ac7e17203b5ec3b8119c1e`. Ollama 0.22.1,
Granite `granite4.1:8b`, digest
`444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852`.
El manifiesto provisionado tiene SHA-256
`0bb1957f6ce0426214c4ca121f162638db0947e9f95b22fc214fad7034caec10`;
permanece en `C:/CloudRAG/operational-20260905T1428Z/deployment-manifest.json`.
Los artefactos de esta ejecución están fuera del repositorio, en
`C:/CloudRAG/operational-20260905-routing/`.

El navegador probó las cinco URL antiguas (`chat_page`, `dashboard_page`,
`experiments_page`, `explorer_page`, `evaluation_page`): todas redirigieron al login,
sin enlaces de operador. La primera URL tras iniciar el proceso fue una ruta de
operador. Un token inválido fue rechazado y una segunda invitación no pudo comenzar
mientras P900 estaba activo. P900 completó tres prácticas con modelos reales.

La respuesta de la primera consulta (`q003`) sobrevivió tanto a desconexión y nuevo
login como a reinicio de Streamlit. `disconnection-check.json` y
`restart-recovery-check.json` verifican igualdad de ID de intento, respuesta, fuentes,
número de intentos y valoraciones. El desplegable de fuentes se abrió en navegador.

La sesión quedó bloqueada en la consulta 8 (`q027`, AWS RDS): dos timeouts reales
consecutivos, de 68,007 s y 67,809 s. No fueron fallos inyectados. El navegador mostró
el mensaje de recuperación, un botón de reintento y ningún formulario ni botón de
valoración. `real-timeout-no-rating.json` y `session-blocked.json` conservan la
evidencia. Hay nueve intentos y siete valoraciones sintéticas; no hay exportación
`full_session.json`. No se alcanzaron los descansos, SUS ni preguntas abiertas.
El progreso persistido permanece en el directorio temporal indicado por `run.json`.

## Medición independiente

Streamlit se detuvo antes de iniciar la cohorte, preservando el checkpoint. Se usan
las mismas 20 consultas en cada celda: posiciones `floor(i*30/20)`, `i=0..19`, de
`_get_evaluation_queries()`. Se conserva la receta de la app, sin caché LLM, seed 42,
contexto 4096 y máximo 1024 tokens; no se aumenta el timeout para el ensayo.

Frío significa un trabajador Python nuevo y descarga de Granite comprobada con
`/api/ps`. El reloj empieza antes de importar/inicializar los auxiliares, verificar
el bundle y ejecutar retrieval, generación y NLI. No se vacía la caché de archivos
del sistema operativo. Caliente usa un trabajador persistente por sistema, con una
consulta de calentamiento completa excluida y veinte consultas medidas.

`latency-cohort/driver.py` se detuvo en preflight, antes de medir: PowerShell elimina
las variables de entorno vacías. `driver-v2.py` admite esa representación y comprueba
que PyTorch sea la compilación CPU. Se conservan ambas versiones. El ensayo no
modifica el entorno de la app para mejorar resultados. Se comprobó la sintaxis del
ejecutor y su interpolación lineal de percentiles con una secuencia de referencia.

Cada intento terminado tiene JSON propio, incluso si falla, junto con logs de su
trabajador. `protocol.json` fija consultas, alcance y versiones. El ejecutor original
preveía un `summary.json` con todos los intentos, pero la interrupción impidió
crearlo. Esa población para percentiles quedó descartada por decisión del usuario:
**solo respuestas completas exitosas entran a p50/p95; errores y abortos cuentan
en la tasa de fallos**. Un fallo impide GO aunque el tiempo hasta el error sea corto.

## Reanudación autorizada: crítica antes de implementar

Baseline previo al nuevo ejecutor: HEAD `4af9728`, 385 pruebas aprobadas,
5 excluidas, 3 advertencias SWIG, 50,41 s. Persisten las tres modificaciones de
documentación de la ejecución anterior; se conservarán hasta el cierre documental.

El ejecutor externo original solo escribe al terminar: un cierre pierde el tiempo
parcial. Reutilizar el nombre de archivo de una consulta confunde observación con
intento físico. La corrección necesita eventos durables antes/durante el trabajo,
IDs únicos de intento y exclusión mutua. Una señal de cierre no prueba quién la
originó; el aborto histórico tendrá duración desconocida y causa indeterminada.

Decisión histórica, posteriormente sustituida por la cohorte independiente 616.64:
terminar primero las 70 observaciones con HTTP de lectura
60 s. Los 50 resultados existentes se importan por referencia y hash. El aborto
histórico cuenta como fallo adicional: 121 intentos de cohorte si no hay más abortos.
Los fallos nunca entran a p50/p95, tengan o no duración final. Calentamientos y
diagnóstico se contabilizan aparte. La instrumentación añade un pequeño coste de
registro; se declara en lugar de afirmar identidad temporal con el ejecutor previo.

El nuevo ejecutor registrará progreso cada segundo cuando el proceso pueda ejecutar
el hilo de registro. No promete una cota de pérdida durante suspensión del sistema
o bloqueo del intérprete. Tras un cierre, el último tiempo persistido será solo un
límite inferior; nunca se imputará como latencia de respuesta completa. El proceso
se lanzará separado de la consola interactiva, sin prometer supervivencia a apagados.

Primero se probarán cierres reales de procesos con una carga sintética, sin modelos.
No se relanzará la cohorte hasta demostrar recuperación idempotente, bloqueo de
recuperación durante inferencia activa y exclusión de errores de los percentiles.

### Resultado del ejecutor y bloqueo de preflight

VERIFICADO: commit `46690adcdd7d4162ce5b3641672522cef911cd14`, 401 pruebas
aprobadas, 5 excluidas, 3 advertencias SWIG (31,54 s); Ruff, diff check y escaneo de
secretos aprobados. Las 16 regresiones nuevas incluyen cierre real de trabajador y
coordinador. En Windows el lanzador del venv puede sobrevivir o morir separado del
intérprete: las pruebas cierran el PID real del proceso, registrado en su petición.
La recuperación rechaza un trabajador vivo incluso si el coordinador ya murió.

El modo `plan` encontró 70 posiciones pendientes, 50 intentos terminados y el aborto
histórico adicional, sin ejecutar modelos ni modificar la evidencia original.

VERIFICADO en `2026-09-06T01:34:38Z`: el controlador NVIDIA cambió de
`32.0.16.1062` (610.62) a `32.0.16.1664` (616.64). CPU, GPU y RAM coinciden con el
hardware previo. Ollama no responde en localhost:11434; el cliente instalado indica
0.22.1, pero no se pudo volver a verificar el servidor ni el digest. El fallo de
conexión no demuestra un cambio de digest.

Evidencia reconstruible de preflight y auditoría:
`C:/CloudRAG/resume-preflight-20260906T013610Z/runtime-comparison.json`, junto con
`git-status.txt` y `git-log.txt`. No se relanzó la cohorte: hay que decidir si se
conservan estratos separados por controlador o se mide una cohorte nueva homogénea.
No hay evidencia para atribuir el `window-CLOSE` al cambio de controlador.

## Cohorte independiente 616.64: crítica y criterios antes de implementar

Decisión del usuario: sustituir la reanudación de 70 posiciones por una cohorte
nueva de **120 intentos físicos**, sin importar los 51 históricos. Cada aborto
nuevo consume su posición; no se reintenta automáticamente. Calentamientos y
diagnóstico van aparte. El modo histórico conserva su semántica anterior.

El ejecutor actual exige una fuente histórica y deduce el digest de sus resultados:
no sirve para iniciar una cohorte vacía. Reutilizarlo sin separar esas reglas
mezclaría entornos o excedería los 120 intentos. Se añadirá inicialización explícita,
manifiesto inmutable de entorno y controles antes/después de medir. Un cambio de
entorno invalida el intento, lo excluye de percentiles y detiene la ejecución.
Los controles de identidad quedan fuera del reloj de respuesta, pero se registran.

Criterios medibles: inicialización con 120 pendientes y cero importaciones;
repetirla no altera el manifiesto; cambio de entorno bloquea reanudación; aborto
recuperado consume una sola posición; ninguna duración fallida entra a p50/p95.
Se conserva HTTP 60 s, seed 42, caché desactivada y la selección de 20 consultas.
Primero se verifica el bundle contra su referencia anterior; solo entonces se crea
una nueva copia del manifiesto. Los auxiliares siguen en CPU y Granite usa Ollama.

Baseline VERIFICADO: `46690adcdd7d4162ce5b3641672522cef911cd14`, 401 pasan,
5 excluidas, 3 advertencias SWIG, 31,35 s. Evidencia nueva separada:
`C:/CloudRAG/cohort-61664-20260906T015527Z/`. El reinicio de bandeja no restauró
la API; el único fallback `ollama serve` sí respondió en
`2026-09-06T01:58:51.1594274Z`: servidor 0.22.1 y digest esperado coincidente.
Véanse `tray-restart.json`, `serve-launch.json`, `serve-outcome.json`,
`serve.stderr.log` y `ollama-tags.json`. No se actualizó Ollama ni el controlador.

### Inicialización verificada de la cohorte nueva

VERIFICADO: `fd747f818ffbbad61427abee58e62223d76b27d8` incorpora seis
regresiones adicionales. Suite completa: 407 pasan, 5 excluidas, 3 advertencias SWIG,
33,12 s; Ruff y diff check aprobados; escaneo sin hallazgos nuevos, con 17 exclusiones
del baseline (11 corpus y 6 evidencia firmada). Antes de implementar, las seis
regresiones nuevas fallaron por ausencia del comportamiento requerido.

`bundle-original-verify.txt`, `bundle-new-verify.txt` y `bundle-comparison.json`
confirman que el manifiesto nuevo tiene el mismo SHA-256 que el anterior:
`0bb1957f6ce0426214c4ca121f162638db0947e9f95b22fc214fad7034caec10`.
`measurements/source-manifest.json` registra el entorno, versiones de paquetes,
hashes de fuentes, consultas, modelo y bundle. Su SHA-256 inicial es
`923e9b79b8bbe5118228cb68f9c15cdafd512cc563c9b085d88dca68772e9bf7`.
`initial-plan.json` contiene 120 pendientes y ninguna importación histórica.

Ejecución iniciada en `2026-09-06T05:42:17.5339380Z` según `cohort-launch.json`.
La invocación registra PID real 34592; PID 384 corresponde al lanzador del venv.
Los registros de intentos son inmutables y un aborto nuevo consume su posición.
Los calentamientos quedan fuera de los 120 intentos y se contabilizan aparte.
Los controles de identidad antes/después no entran al reloj de respuesta.
Hallazgo del registro: la clave heredada `interrupted_cohort=true` es incorrecta
para esta primera invocación nueva. No se utiliza para calcular poblaciones ni
percentiles; `mode=fresh`, los 120 pendientes y la ausencia de importaciones fijan
su procedencia. Se conserva el registro original y se documenta el error; corregir
esa etiqueta requerirá su regresión después de finalizar la medición, para no
cambiar el hash del ejecutor mientras una cohorte está activa.

### Cierre de cohorte y crítica del diagnóstico

VERIFICADO: 120 intentos terminados, 105 éxitos, 15 timeouts, cero abortos;
3 calentamientos exitosos adicionales. `completed-cohort-summary.json` recalcula
los percentiles y `completed-cohort-inventory.json` conserva hashes de los archivos.
Las seis condiciones incumplen p95 <=60 s. Baseline posterior: 407 pasan,
5 excluidas, 3 advertencias SWIG, 31,92 s (`before-diagnostics-pytest.txt`).

El texto genérico «timed out» pierde la clase de excepción y la fase HTTP.
Modificar directamente el timeout de la app confundiría diagnóstico con corrección.
Se instrumentará un ejecutor separado que delegue sin modificar las peticiones:
duración y excepción de `list`/`chat`, petición exacta y métricas devueltas por Ollama.
Se comparará q027 con 60/180 s, frío/caliente; calentamientos y diagnósticos no
entran a la cohorte. Predicciones: si agota lectura de generación, `chat` tendrá
ReadTimeout cerca de 60 s y 180 s podrá terminar; si falla identidad, lo mostrará
`list`; si domina capacidad, aparecerán carga/offload y costes de generación/NLI.
El timeout visible de la app no se cambia sin aprobación del usuario.

### Latencias finales: cohorte 616.64

VERIFICADO, commit medido `fd747f818ffbbad61427abee58e62223d76b27d8`:
Intel Core i5-12450H, 8 núcleos/12 procesadores lógicos, 16.891.633.664 bytes de RAM,
RTX 3060 Laptop 6144 MiB, driver 616.64, servidor Ollama 0.22.1. Digest Granite:
`444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852`.
El manifiesto de entorno registra Python, paquetes y hashes de fuentes. Cada intento
terminado pasó controles de identidad antes/después. Ningún resultado histórico se
importó. Las duraciones siguientes incluyen respuesta completa hasta NLI.

| Sistema | Condición | Intentos | n exitosos | p50 s | p95 s | Fallos |
|---|---|---:|---:|---:|---:|---:|
| Híbrido | Frío | 20 | 19 | 90,67 | 126,01 | 1 (5 %) |
| Híbrido | Caliente | 20 | 18 | 40,41 | 75,90 | 2 (10 %) |
| Léxico | Frío | 20 | 16 | 71,50 | 96,67 | 4 (20 %) |
| Léxico | Caliente | 20 | 17 | 24,19 | 83,28 | 3 (15 %) |
| Semántico | Frío | 20 | 18 | 76,29 | 111,99 | 2 (10 %) |
| Semántico | Caliente | 20 | 17 | 26,20 | 67,35 | 3 (15 %) |

Total: 105 respuestas completas y 15 fallos de 120 intentos (12,5 %), cero abortos.
Tres calentamientos exitosos adicionales, excluidos de la cohorte. Todos los fallos
tienen el mensaje `All 1 retries failed: timed out`. La última respuesta terminal
corresponde a semántico caliente índice 19; los seis bloques quedaron completos.
Las seis condiciones incumplen p95 <=60 s. Esto **no demuestra por sí solo** que la
única causa sea hardware ni habilita Fase B mientras la sesión real siga incompleta.

Caso RDS q027, intento `3a018bc85161436f8aeea356710f8870`: duración 112,93 s,
generación 63,00 s, NLI 0. El servidor registra `GET /api/tags` exitoso en 10,6 ms y
`POST /api/chat` fallido a 60 s. q016, intento `484666121b0946eba1e4bc8a17e07ae1`,
sí terminó: 141,22 s, generación 60,87 s y NLI 32,32 s. Un total >60 s no implica
necesariamente un timeout HTTP. El log de Ollama muestra offload parcial de capas;
la contribución del hardware sigue pendiente de atribución suficiente.

### Evidencia histórica 610.62: conjunto separado

Los 51 intentos históricos (50 terminados y un aborto) permanecen en
`C:/CloudRAG/operational-20260905-routing/latency-cohort/`, sobre la app
`4af97286d43bc54af4ac7e17203b5ec3b8119c1e`, driver 610.62 y Ollama 0.22.1.
No se mezclan con la cohorte nueva. Los calentamientos no cuentan en esos 51.

| Sistema | Condición | Intentos | n exitosos | p50 s | p95 s | Fallos |
|---|---|---:|---:|---:|---:|---:|
| Híbrido | Frío | 20 | 19 | 74,91 | 88,93 | 1 |
| Híbrido | Caliente | 20 | 18 | 35,15 | 67,44 | 2 |
| Léxico | Frío parcial | 11 | 8 | 65,12 | 82,12 | 2 + 1 aborto |

Los demás bloques no se ejecutaron. El aborto `window-CLOSE` conserva duración final
desconocida y causa originaria indeterminada; nunca se imputa a percentiles. La
diferencia observada entre cohortes no permite atribuir causalidad al driver: no es
un experimento controlado del controlador y no se mantuvieron iguales todas las
condiciones de carga del sistema operativo.

## Cierre operativo del 2026-09-07: NO-GO

### Diagnóstico separado y causa raíz

VERIFICADO: ejecutor diagnóstico `769d09111ad04ffca8a80b291a0f8300f83a11e3`;
corrección de etiqueta de invocación `c53a29af860a38990308ab971169774e2ec8521d`.
La etiqueta se deriva ahora de la existencia de intentos previos; su regresión
reprodujo el fallo antes del cambio. Los registros antiguos no se reescribieron.
Suite posterior: 411 pasan, 5 excluidas, 3 advertencias SWIG, 27,29 s; Ruff,
diff check y secretos aprobados. Las tres regresiones de trazado preservan petición,
respuesta y excepción original, y distinguen `list` de `chat`.

Los probes bajo `c53a29a` se ejecutaron entre `2026-09-06T19:27:10.4018283Z` y
`2026-09-06T19:42:25.1686334Z`, según `diagnostic-outcome.json`. El resultado del
ejecutor es «complete»: significa cuatro probes registrados, **no cuatro éxitos**.
Están fuera de los 120 intentos y tienen dos calentamientos exitosos adicionales.
No se modificó el timeout del código de la app.

| Probe híbrido q027 | HTTP chat s | Resultado | Respuesta completa s |
|---|---:|---|---:|
| Frío, timeout 60 | 60,014 | `httpx.ReadTimeout` | — |
| Frío, timeout 180 | 128,395 | Éxito, generación y NLI | 199,046 |
| Caliente, timeout 60 | 60,015 | `httpx.ReadTimeout` | — |
| Caliente, timeout 180 | 180,010 | `httpx.ReadTimeout` | — |

VERIFICADO: `list` fue exitoso en los cuatro casos (2,066–2,126 s de cliente).
`chat` agotó la lectura a los límites configurados. El cliente es construido con
`httpx.Timeout(self.timeout, connect=min(5, self.timeout))` en
`src/generation/llm_manager.py::_ollama_chat`; la app fija 60 s en
`src/ui/components/index_loader.py::load_pipeline`. El mecanismo inmediato del
fallo es el timeout de lectura durante generación. El probe frío exitoso de 180 s
reportó 686 tokens de salida y 114,788 s de evaluación de tokens en Ollama
(aproximadamente 5,98 tokens/s). No se calculan percentiles con los tres fallos.

VERIFICADO: el log del servidor durante el probe caliente de 180 s registra RAM
libre de 3,3 GiB, VRAM disponible para asignación de 3,6 GiB, 23/41 capas en GPU y
pesos compartidos entre CPU (2,2 GiB) y GPU (2,6 GiB). Evidencia en
`serve.stderr.log`, líneas 15346–15354 y 15478; peticiones, clases de excepción y
métricas en `diagnostics/{cold,warm}-{60,180}/measured-http/*.json`.

SUPUESTO pendiente de contrastar: presión de memoria/offload y carga de cómputo
explican parte de la demora. Estos datos **no aíslan su contribución causal** ni
descartan efectos del estado previo de generación. No se afirma «NO-GO solo por
hardware». Elevar a 180 s no es una corrección suficiente y no se aplica a la app.
El `window-CLOSE` histórico sigue con causa originaria indeterminada: no hay
evidencia que lo vincule al driver, al usuario ni al servicio de Ollama.

### Compuerta y trabajo detenido

| Paso | Estado | Evidencia |
|---|---|---|
| Bundle, identidad y servidor | VERIFICADO ✅ | Verificaciones original/nueva, manifiestos idénticos; API 0.22.1, driver 616.64 y digest esperado |
| Sesión real completa | VERIFICADO incompleta ❌ | P900: revisión 46, tres prácticas, nueve intentos, siete ratings; consulta 8 pendiente, sin SUS ni exportación |
| Latencia completa | VERIFICADO incumplida ❌ | 120 intentos, 15 timeouts; seis p95 >60 s; resumen e inventario final |
| Resiliencia | VERIFICADO parcialmente ⚠️ | Evidencia previa de desconexión/reinicio y dos fallos sin valoración; falta cierre/exportación real de la sesión |

El checkpoint de P900 se volvió a leer el 2026-09-07 y conserva el estado anterior.
No se reintentó a ciegas esa consulta ni se fabricaron ratings, SUS o exportación.
La recuperación previa se acredita con `disconnection-check.json`,
`restart-recovery-check.json`, `real-timeout-no-rating.json` y `session-blocked.json`
en `C:/CloudRAG/operational-20260905-routing/`; no se repitieron esas pruebas.

**NO-GO; Fase B bloqueada.** La sesión está incompleta y 180 s no resolvió los
errores de generación. No se crearon artefactos de nube, no se cotizó ni se desplegó.
No se modificaron modelo, driver, contexto, límite de tokens ni timeout participante.
No hubo push, merges ni modificaciones de evidencia congelada/corpus/gold/paper.

Siguiente decisión propuesta al usuario: perfilar de forma controlada generación,
RAM/VRAM y offload antes de modificar la receta. Aceptación propuesta: aislar la
causa con una sola variable por comparación, conservar fallos, demostrar respuestas
completas en los casos que fallaron y acordar por separado cualquier cambio visible
de la app. Un timeout mayor, por sí solo, no demuestra p95 <=60 s ni autoriza entrevistas.

### Auditoría final

VERIFICADO en `2026-09-07T14:04:45.7177927Z`: servicio 0.22.1, driver 616.64 y
digest esperado; 557 archivos de la cohorte conservan los hashes del cierre.
Evidencia final en
`C:/CloudRAG/cohort-61664-20260906T015527Z/final-audit-20260907T140440Z/`:
`environment-and-integrity.json`, snapshots de logs de Ollama, resumen histórico
recalculado con hashes, `pytest.txt`, `ruff.txt`, `diff-check.txt` y `secrets.txt`.
La suite final obtuvo **411 pasan, 5 excluidas, 3 advertencias SWIG, 55,92 s**;
Ruff y diff check aprobados, secretos sin hallazgos nuevos con el baseline vigente.
El comando inicial de resumen histórico falló por quoting de PowerShell, antes de
escribir el resumen o ejecutar la suite; se corrigió y quedó registrado en
`audit-command-correction.json`. No se presenta ese primer comando como exitoso.

Pregunta planteada al cierre anterior: ¿el equipo estuvo conectado a corriente y sin juegos
u otras cargas de GPU durante la cohorte y los probes del 6 de septiembre? No se
registraron esas condiciones de manera suficiente para atribuir la diferencia de
rendimiento exclusivamente al hardware o al driver.

## Reanudación controlada del 2026-09-07

**DECLARADO por el usuario:** durante parte de la cohorte 616.64 la laptop estuvo
desconectada de corriente y Brave se abrió varias veces. La cohorte se conserva
como evidencia histórica con condiciones no controladas; no es una referencia de
capacidad local. No se reconstruyen condiciones retrospectivas ni se mezclan sus
percentiles con los del nuevo piloto. La comparación con 610.62 será descriptiva:
no permite atribuir diferencias al driver. No se cambia el controlador.

### Crítica del plan y criterios fijados antes de inferir

Extender el ejecutor durable evita crear un segundo mecanismo de recuperación.
El riesgo principal es confundir una condición inválida con un fallo del sistema,
o eliminar respuestas lentas y mejorar artificialmente los percentiles. Por ello,
el resultado técnico se conserva, `conditions_invalid` es independiente, y todos
los intentos consumen su posición. Fallos/abortos cuentan en la tasa de fallos;
solo éxitos completos con controles válidos entran a p50/p95. No hay reemplazos.

El piloto acordado tiene **40 posiciones híbridas: 20 frías y 20 calientes**, con
las mismas veinte consultas y orden históricos. Veinte observaciones por condición
permiten una comparación descriptiva emparejada, pero el p95 depende de la cola
de una muestra pequeña y no certifica un percentil poblacional. Un resultado
satisfactorio del piloto no equivale al GO de los tres sistemas; la extensión a
los otros 80 intentos requiere decisión posterior.

En este equipo Modern Standby solo se encontró el plan Equilibrado. Se registra
AC + Equilibrado + overlay efectivo «Mejor rendimiento» (GUID
`ded574b5-45a0-4f42-8737-46345c09c238`); no se modifican ajustes de energía.
Antes de medir se exigen 60 s de observación, CPU y GPU medias inferiores al 10 %,
sin navegadores, launchers u overlays visibles. No se cierran procesos ajenos.

`scripts/observe_interview_gate.py` observa cada 5 s alimentación, modo efectivo,
RAM, CPU actual calculada por diferencias, procesos con PID/RAM, GPU/VRAM,
temperatura GPU, frecuencias y limitadores disponibles; registra `ollama ps` y
`/api/ps`. Los contadores no soportados se conservan como N/A; la temperatura CPU
se declara no disponible. Utilización GPU y reparto de pesos GPU/CPU son magnitudes
distintas. Telemetría obligatoria ausente, cambio de energía, otro modelo o carga
externa sostenida invalidan el control y pausan tras finalizar el intento activo.
Los límites térmicos/energéticos propios de la inferencia y la presión de memoria
son resultados observados: por sí solos no invalidan una respuesta lenta.

Las observaciones tienen UTC, reloj monotónico, PID y errores; se guardan como
JSONL con flush/fsync. El request enlaza el journal incluso si el proceso aborta.
Los HTTP traces conservan peticiones, clase de excepción y métricas devueltas por
Ollama. El cliente caliente mantiene su conexión entre consultas. Ni la receta
del modelo ni el timeout participante de 60 s cambian.

### Avance verificado y bloqueo de admisión

Baseline `fc0537765e469e34e202cb2c99434400000ea83f`, árbol inicialmente limpio:
**411 pasan, 5 excluidas, 3 advertencias SWIG, 51,72 s**. Evidencia nueva y separada:
`C:/CloudRAG/controlled-pilot-20260907T190630Z/` (`baseline-status.txt`,
`baseline-log.txt`, `baseline-pytest.txt`).

Las pruebas del observador sin inferencia (`observer-smoke.json`,
`observer-command-profile.json`, `observer-native-smoke.json`) detectaron un coste
inicial de 8,91/4,34 s por muestra. El perfil midió 1,99 s al lanzar PowerShell y
un coste residual aproximado de 2 s en HTTP. Se sustituyó el subproceso de inventario
por getters nativos y solo la URL del observador usa IPv4. Las siguientes lecturas
tardaron 0,39/0,50 s y no tuvieron errores. Estos son tiempos de observación, **no
latencias RAG ni una prueba de ausencia de interferencia durante inferencia**.

VERIFICADO en esas muestras: AC, overlay esperado, pero Epic Games Launcher y
overlays activos. Se pidió al usuario cerrarlos.

La admisión formal posterior falló: `admission-01/identity.json`, `samples.jsonl`
y `result.json` bajo el directorio externo anterior. Commit observado
`7b48d3792f6870f4f5a673d443fa3a7a33013bf6`, observer SHA-256
`f247e8c8ae790103985429b20d5eb009b382ca17cf34c4f587c86bd71656a5da`.
Ventana UTC `2026-09-07T20:13:36.417953+00:00` a
`2026-09-07T20:14:36.820888+00:00`: **13 muestras, 60,40 s**, CPU media **3,05 %**,
GPU media **15,08 %**, sin errores de sensores. Edge, Epic Games Launcher y overlays
de NVIDIA/Epic seguían presentes. Motivos: `idle_gpu` y `prohibited_process`.
El comando terminó con error explícito de admisión; no es un timeout RAG y no
consume ninguna de las cuarenta posiciones. Coste del observador en esta ventana:
media **0,315 s**, máximo **0,399 s** por muestra. No demuestra ausencia de
interferencia durante inferencia; falta el contraste sobre trabajo sintético.

El cambio de instrumentación incluye **14 regresiones nuevas**: selección inmutable,
40 posiciones reales del coordinador con un calentamiento aparte, reanudación sin
duplicación, exclusión de controles inválidos conservando la respuesta, bloqueo
antes de inferir, sensores nativos y detección de energía/carga/telemetría ausente.
Auditoría: **425 pasan, 5 excluidas, 3 advertencias SWIG, 30,67 s**;
Ruff, diff check y escaneo con baseline aprobados (`implementation-*-02.txt`).
El pase final repitió **425 pasan/5 excluidas en 29,96 s**, con los mismos tres
avisos SWIG y Ruff/diff/secretos aprobados (`final-pytest.txt`, `final-ruff.txt`,
`final-diff.txt`, `final-secrets.txt`).

No se inició el piloto ni se reintentó P900. Quedan pendientes una admisión aprobada,
el coste sobre trabajo sintético bajo las condiciones fijadas y los 40 intentos.
**NO-GO; Fase B sigue bloqueada.** No hay evidencia nueva para atribuir la demora
exclusivamente al hardware, ni para cambiar la espera visible del participante.
