# Compuerta local: registro de ejecución

**Estado vigente: NO-GO (2026-09-10, Lima); Fase B bloqueada.** La nueva cohorte
completó 120 posiciones y P900 terminó SUS/exportación. El p95 léxico caliente
de 65,75 s incumple el criterio prospectivo de 60 s. Híbrido y semántico calientes
pasan; el frío se informa sin bloquear por latencia. Las secciones anteriores al
[cierre actual](#cierre-de-la-cohorte-preparada-y-p900-2026-09-10-lima) conservan la
cronología histórica y no describen el estado más reciente.

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

## Contraste del observador y nuevo prechequeo (2026-09-07)

DECLARADO por el usuario: cerró las aplicaciones indicadas, mantiene AC/Mejor
rendimiento y dispone de unas dos horas sin interrupciones. Esta declaración no
sustituye los controles. La reanudación parte de `87fc16f`, árbol limpio, baseline
**425 pasan/5 excluidas en 30,80 s**, Ruff/diff/secretos aprobados.
Evidencia nueva: `C:/CloudRAG/clean-pilot-20260907T203624Z/` (`baseline-*`).

### Crítica y decisiones previas

Descontar un coste sintético de la latencia RAG produciría una duración que ningún
participante experimentó y supondría interferencia aditiva sin demostrarla. El
usuario eligió **no descontar**: aceptar el observador si el límite superior
unilateral del 95 % del aumento mediano pareado es <=5 %. Se fijaron diez pares,
orden alternado AB/BA, hashing de un buffer de 64 MiB, cantidad de trabajo calibrada
una vez a unos diez segundos, bootstrap de 10.000 remuestreos y seed 42. Ambos
brazos usan el mismo registrador durable; preparación/cierre del observador quedan
fuera del reloj de respuesta, como en el ejecutor RAG. Los pares fallidos o inválidos
no se descartan para aprobar: el contraste se detiene y conserva evidencia parcial.
Un coste sintético aceptable no demuestra ausencia de interferencia en GPU.

Hipótesis para la etapa de inferencia, aún no ejecutada: offload parcial asociado
con generación lenta; presión de RAM/VRAM; limitación térmica/energética; o duración
en etapas fuera del chat HTTP. El usuario aceptó declarar causalidad indeterminada
si la observación no aísla el mecanismo. No se atribuirá retrospectivamente al
driver la diferencia entre cohortes con condiciones históricas no verificables.

VERIFICADO: `cf0328d4ddbcf77716404c3c097987ef6308ccad` incorpora el contraste en
`scripts/contrast_interview_observer.py` y once regresiones: criterio bootstrap,
variabilidad de la cola, pares incompletos/duplicados/inválidos, veinte brazos del
registrador real, rechazo de repetición y conservación de un brazo fallido.
Suite **436 pasan/5 excluidas en 31,64 s**, tres avisos SWIG; Ruff/diff/secretos
aprobados (`contrast-pytest.txt`, `contrast-ruff.txt`, `contrast-diff.txt`,
`contrast-secrets.txt`).

### Resultado operativo: bloqueo previo al contraste

VERIFICADO: `observer-contrast-01/result.json`, UTC
`2026-09-07T20:44:58.506716+00:00`, registra `status=blocked`, `passed=false`,
motivo `prohibited_process`. El protocolo contiene commit y hashes de los tres
scripts; `initial-state.json` conserva los sensores y procesos observados.
**Cero pares sintéticos, cero nuevas ventanas de admisión, cero intentos RAG.**
No se repitió automáticamente el prechequeo ni se consumieron posiciones del piloto.

Los cinco procesos NVIDIA Overlay seguían activos (PID 2860, 10356, 16896, 21664,
33192); `nvidia-smi` listó los PID 10356 y 16896 como usuarios de GPU. El inventario
está en `gpu-processes.txt`, `gpu-compute-processes.txt` y su timestamp asociado.
`gpu-engines.json` registró 19 % en un motor 3D para PID 31384, identificado después
como AnyDesk; también registró DWM y Windows Terminal. Esa muestra de motor **no
es el porcentaje total de utilización de la RTX** ni prueba una causa exclusiva.
WDDM reportó memoria por proceso N/A. No se cerró AnyDesk: podría sostener el acceso
remoto del usuario. Se solicitó confirmar si puede cerrarse y desactivar la
superposición NVIDIA, con comandos para comprobarlo.

**NO-GO; Fase B bloqueada.** Continúan pendientes el contraste, la admisión, los
cuarenta intentos, pruebas de semántica HTTP y el cierre P900/SUS/exportación.
No hubo cambios de timeout, modelos, driver, energía ni sesiones. El límite de dos
ventanas de admisión fallidas de esta reanudación sigue sin consumirse.

Auditoría final: **436 pasan/5 excluidas en 30,31 s**, tres avisos SWIG;
Ruff, diff check y secretos aprobados (`final-pytest.txt`, `final-ruff.txt`,
`final-diff.txt`, `final-secrets.txt`). Git y el inventario SHA-256 de la evidencia
nueva se registran en `final-git.json` y `final-inventory.json` al cerrar los commits.

## Cierre del piloto y reanudación de P900 (2026-09-09)

### Evidencia reconstruible y entorno

En esta sección R = `C:/CloudRAG/managed-pilot-20260908T012621Z/`.
No confundir la fecha del directorio con las fechas efectivas de cada prueba.
**VERIFICADO:** piloto build `a761fa28e7af691f2e2bdaee880e20f987ae1a59`,
Ollama 0.22.1, driver 616.64, i5-12450H, 16 GB nominales, RTX 3060 Laptop 6 GB;
Python 3.14.3, auxiliares CPU, seed 42, caché de respuestas desactivada, contexto
4096 y máximo 1024 tokens. Versiones y digest completo en
`R/window-real-05/cohort/source-manifest.json`. Granite conserva
`444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852`.
Se verificó el manifiesto original, se tomó snapshot nuevo y se verificó de nuevo:
`verify-original.log`, `snapshot.log`, `verify-new.log`, `deployment-manifest.json`
de esa ventana. Hash del manifiesto: `0bb1957f6ce0426214c4ca121f162638db0947e9f95b22fc214fad7034caec10`.

Contraste aprobado a las 00:31:49Z: 10 pares, mediana -0,179 %, límite superior
unilateral bootstrap 95 % +0,0463 %, por debajo del criterio 5 %. Dos admisiones
aprobadas, AC y overlay efectivo Mejor rendimiento, sin cambiar energía.
La reserva ETW fue verificada en runtime (256 buffers de 64 KiB); las capturas
utilizadas no perdieron eventos. Piloto terminado 01:24:20Z, restauración real
01:24:33Z con AnyDesk primero. Fallos previos de ventanas 01–04 se conservan y
no se mezclan con la ventana aprobada. Acciones/PIDs/timestamps:
[MANAGED_GATE_WINDOW.md](MANAGED_GATE_WINDOW.md#resultado-de-la-ventana-05-verificado).

### Latencias: poblaciones separadas

**VERIFICADO:** éxitos completos y controles válidos exclusivamente en percentiles;
fallos/abortos cuentan aparte y nunca reciben duración inventada. Frío significa
worker nuevo y descarga previa de Granite, conservando caché de archivos del SO.
Caliente mantiene el worker después de un calentamiento independiente.

| Cohorte / sistema | Condición | Éxitos / intentos | p50 s | p95 s | Fallos (abortos incluidos) |
|---|---|---:|---:|---:|---:|
| Piloto 616.64 / híbrido | Frío | 20/20 | 50,13 | 83,24 | 0 |
| Piloto 616.64 / híbrido | Caliente | 20/20 | 23,21 | 57,62 | 0 |
| Histórica 616.64 / híbrido | Frío | 19/20 | 90,67 | 126,01 | 1 |
| Histórica 616.64 / híbrido | Caliente | 18/20 | 40,41 | 75,90 | 2 |
| Histórica 616.64 / léxico | Frío | 16/20 | 71,50 | 96,67 | 4 |
| Histórica 616.64 / léxico | Caliente | 17/20 | 24,19 | 83,28 | 3 |
| Histórica 616.64 / semántico | Frío | 18/20 | 76,29 | 111,99 | 2 |
| Histórica 616.64 / semántico | Caliente | 17/20 | 26,20 | 67,35 | 3 |
| Histórica 610.62 / híbrido | Frío | 19/20 | 74,91 | 88,93 | 1 |
| Histórica 610.62 / híbrido | Caliente | 18/20 | 35,15 | 67,44 | 2 |
| Histórica 610.62 / léxico | Frío parcial | 8/11 | 65,12 | 82,12 | 3 (1 aborto) |

Piloto: 40 éxitos, cero fallos/abortos/condiciones inválidas y un calentamiento
excluido. `R/window-real-05/payload-complete.json` y `cohort/reports/`.
Los p95 son percentiles descriptivos de 20 observaciones por condición, no un
límite de confianza ni una garantía para futuras sesiones; hubo una respuesta
caliente de 61,47 s aun cuando su p95 muestral cumple el umbral.
Histórica 616.64: build `fd747f8`, 120 intentos/105 éxitos/15 timeouts,
tres calentamientos aparte; `C:/CloudRAG/cohort-61664-20260906T015527Z/`.
**DECLARADO por el usuario:** hubo batería y aperturas de Brave; no usarla como
capacidad autoritativa del hardware. Histórica 610.62: build `4af9728`,
51 intentos físicos (45 éxitos, 5 errores, un aborto), un calentamiento aparte;
`C:/CloudRAG/operational-20260905-routing/latency-cohort/`. Otras condiciones no
medidas en esa cohorte. `window-CLOSE` sigue de causa indeterminada.

La reducción observada no identifica un efecto causal del driver: las condiciones
históricas no son retroverificables. No se mezclan ni recalculan esas cohortes.
El piloto de 40 no sustituye los controles léxico/semántico requeridos para GO.

### Causa y RAM

`R/memory-analysis-01.json` (2026-09-09T10:45:35Z), generado con `f4a60f2`,
verifica hashes y separa chat HTTP de carga/observación. **VERIFICADO:** offload
parcial, costos de carga en frío, generación y NLI; sin agotamiento sostenido de
RAM durante chat. **SIN AISLAR:** cuánto aporta cada límite físico y si explica
exclusivamente la demora. Actividad residual del SO y de `upc.exe` está registrada;
pasar la lista de admisión no prueba ausencia absoluta de interferencias.
No se cambió retrospectivamente la clasificación de las observaciones.
La recomendación es **no comprar 32 GB como solución acreditada del p95**: no hay
estimación defendible de aceleración; detalle en
[RAM_INTERVIEW_DIAGNOSTIC.md](RAM_INTERVIEW_DIAGNOSTIC.md).

### Recuperación real de P900 y nuevo bloqueo

**VERIFICADO:** servidor Streamlit de build `f4a60f2a420c63b8d4e04c143a36590bdba49326`,
registro `R/p900-server-01.json`, mismo digest y manifiesto. El token existente
recuperó revisión 46 en consulta 8 (q027), tres prácticas hechas, siete ratings y
dos errores. Se reintentó una vez con timeout 60 s, después de restaurar procesos:
esta prueba UI está fuera del piloto controlado y nunca entra en sus percentiles.

Intento `370c95e728eb45d7b1dfdd15090b3ba1`, 10:48:37–10:50:29Z,
**111,9988 s de respuesta fallida**, `pipeline_error`; stderr informa `timed out`.
No se registró el subtipo HTTP en este intento UI: los traces anteriores sí
confirmaron `httpx.ReadTimeout`, pero no se inventa un trace nuevo.
Checkpoint revisión 49: 10 intentos, 7 ratings, ningún SUS ni exportación.
`R/p900-recovery-failure-01.json`, copia `p900-checkpoint-revision49.json`
(SHA-256 `eca08ef8a01ba2aa70db2e2e87a00dd037dfe9d102a86e6a4ba683e96b936afa`),
`p900-streamlit-01.stderr.log`, `p900-browser-failure-01.json`.
El navegador comprobó cero grupos de valoración y cero botones de envío de rating.
No se calificó el fallo ni se fabricó SUS para forzar la exportación.
Se detuvo exclusivamente el Streamlit propio; `p900-server-stop-01.json`.

Desconexión y reinicio ya probados no se repitieron como si fueran evidencia nueva:
`C:/CloudRAG/operational-20260905-routing/disconnection-check.json`,
`restart-recovery-check.json`, `real-timeout-no-rating.json`.
La recuperación actual refuerza persistencia, pero sigue faltando el cierre integral.

### Timeout: verificado y propuesta pendiente

La semántica queda aclarada con código y tres pruebas HTTP reales en
`tests/test_http_timeout_semantics.py` (commit `75dfa3e`). `httpx.Timeout(60,
connect=5)` limita silencio de lectura/escritura y espera de pool por operación;
no es deadline de retrieval + generación + NLI. `list` y `chat` son peticiones
distintas. En el probe frío/180 histórico, chat duró 128,3947 s, list 2,0800 s,
y la respuesta completa 199,0460 s. No se superó un deadline total de 180 s porque
ese deadline no existe. [Semántica oficial HTTPX](https://www.python-httpx.org/advanced/timeouts/).

**PROPUESTA, NO IMPLEMENTADA:** probar read timeout de 180 s solo en participante,
con conexión 5 s, reloj visible de tiempo transcurrido y avisos a los 60/120 s,
sin porcentajes de progreso ni prometer término antes de 180 s. Conservar respuesta
persistente, lock de inferencia, recuperación por token y prohibición de valorar
fallos. Recargar o cerrar el navegador no debe anunciar cancelación del modelo.
La UI debe explicar que una espera larga puede incluir carga y verificación.
Esto requiere aprobación específica del cambio visible, aun teniendo autorización
para ejecutar modelos. No equivale a resolver el criterio p95 <=60 s.

Tests exigidos antes de usarlo: timeout configurado correctamente sin cambiar otros
modos/modelos; feedback real del navegador a 60/120 s sin bloquear refresco; error
sin rating; recarga sin inferencias duplicadas; suite completa. Después, prueba
real P900 y exportación. No ejecutar nuevos reintentos a ciegas.

### Veredicto y auditoría

| Paso | Estado vigente | Evidencia |
|---|---|---|
| Bundle/identidad | ✅ VERIFICADO | snapshot/verify y manifiesto de ventana 05 |
| Sesión real completa | ❌ VERIFICADO incompleta | P900 revisión 49, q027 fallida, sin SUS/exportación |
| Latencia <=60 s en todas las condiciones | ❌ VERIFICADO | Piloto híbrido frío 83,24 s; caliente 57,62 s; controles limpios no ejecutados |
| Resiliencia integral | ⚠️ VERIFICADO parcialmente | Persistencia/recuperación y exclusión de ratings erróneos; falta exportación final |

**NO-GO; Fase B bloqueada.** No se cumple la excepción «solo latencia por hardware»:
falta cierre real y la exclusividad causal del hardware no está demostrada.
No se diseñó ni preparó despliegue de nube, no se cotizaron proveedores ni se
desplegó. Tampoco se modificó timeout, modelos, driver, energía o evidencia congelada.

Auditoría de `f4a60f2`: **476 pasan, 5 excluidas, tres advertencias SWIG, 69,60 s**;
Ruff y secretos aprobados, diff sin errores. Evidencia `R/analysis-pytest.txt`,
`analysis-ruff.txt`, `analysis-secrets.txt`, `analysis-diff.txt`. El test de journal
alterado detectó primero una fixture sin política de posición; se corrigió la
fixture y las cinco regresiones pasaron. No se presenta el primer fallo como pase.
La auditoría final de documentos y el inventario de hashes se guardan en R con
prefijo `closure-`, sin sobrescribir artefactos anteriores.
Pase de cierre verificado: **476 pasan, 5 excluidas, tres avisos SWIG, 43,78 s**;
Ruff/secretos aprobados. No hay diferencias versionadas contra `670f8e5` en
`experiments/results`, `data`, `paper` ni `output` (`closure-protected-paths.txt`).

## Enmienda caliente y preparación de nueva cohorte — 2026-09-09

Esta sección sustituye las propuestas pendientes y el criterio de la sección
anterior; no reinterpreta las mediciones históricas. El usuario autorizó lectura
participante de 180 s, reloj y avisos a 60/120 s; pre-calentamiento obligatorio y
una cohorte nueva homogénea de 120 posiciones (tres sistemas, frío/caliente, 20 por
celda). GO exige p95 caliente <=60 s y cero errores, abortos o condiciones inválidas
en cada celda caliente, además de P900 completa, exportación y resiliencia integral.
El frío se informa sin bloquear por su latencia. La anterior excepción de hardware
ya no aplica. Crítica, receta y textos UX: [WARM_GATE_PREREGISTRATION.md](WARM_GATE_PREREGISTRATION.md).

**VERIFICADO:** `ffceaff` implementa timeout y reloj; 482 tests pasaron. `3744b9b`
implementa preparación real de tres pipelines y NLI en el proceso participante,
residencia de Granite por 30 minutos renovados y pausa si se pierde preparación;
494 tests pasaron. Los tests son offline y no acreditan aún latencia real.
Evidencia externa: `C:/CloudRAG/warm-protocol-20260909T152518Z/`, baseline
`baseline-head.txt`/`baseline-pytest.txt` (476 pasan), auditorías `wait-*` y
`preparation-final-*`. `wait-browser-controlled.json` verifica el reloj HTML
con tiempo de navegador simulado; no es una inferencia ni una medición real.

El ejecutor prospectivo mantiene una sola identidad por cohorte y permite ventanas
por condición. La preparación de cada proceso caliente conserva tres respuestas
y pruebas NLI fuera de las posiciones medidas; cada respuesta conserva su
comprobación de residencia. La recuperación de capturas WPR compartidas queda
acotada por ID de ventana. No se repiten ventanas fallidas a ciegas.

**NO-GO pendiente de validación nueva:** estas implementaciones no completan P900
ni acreditan todavía el criterio caliente de los tres sistemas. No se han preparado
artefactos de nube ni desplegado recursos como parte de esta enmienda.

## Seguimiento del bloqueo léxico: 2026-09-11

Continúa **NO-GO**. Se inició instrumentación externa sin cambiar la app ni la
receta medida. La reconstrucción histórica muestra más tokens/claims en las dos
posiciones que forman la cola léxica; todavía no prueba causalidad exclusiva.
La admisión nueva falló por NVIDIA Overlay, sin ejecutar inferencias ni cerrar
AnyDesk. Las cohortes históricas y P900 no se modificaron. Véase
[el diagnóstico fechado](LEXICAL_DIAGNOSTIC_2026-09-11.md) para protocolo, crítica,
evidencia y autorización pendiente. Esto no constituye el pre-registro de una
optimización ni cambia los criterios de aceptación siguientes.

## Cierre de la cohorte preparada y P900: 2026-09-10 Lima

### Pre-registro y trazabilidad

**VERIFICADO:** ejecutor y app `b6f3fea006ce2ae2555bd63ffc4a1e78c37f703a`.
Raíz R: `C:/CloudRAG/warm-protocol-20260909T152518Z/`. Se mantuvieron sin cambios
el ejecutor, fuentes, paquete de modelos y pre-registro durante las 120 posiciones.
La documentación de cierre es posterior; no cambia el build medido.

Crítica aplicada: declarar caliente como condición operativa solo es comprobable
si la app prepara los tres pipelines en su propio proceso y detecta pérdida de
residencia. Ese contrato se implementó antes de medir. Veinte posiciones por celda
permiten aplicar el criterio descriptivo acordado, no garantizar el p95 poblacional
ni identificar por sí solas un efecto causal del driver. No seleccionar reintentos
favorables, descontar observador ni modificar umbral después de conocer resultados.
El pre-registro exige cero fallos/abortos/inválidas y p95 <=60 s en cada celda caliente;
el frío se completa y reporta sin bloquear por su latencia. Fase B solo con GO.

| Identidad | Valor VERIFICADO |
|---|---|
| Equipo | i5-12450H, 16 GB RAM, RTX 3060 Laptop 6 GB |
| Driver / Ollama | 616.64 / 0.22.1 |
| Granite | `granite4.1:8b`, digest `444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852` |
| Receta | participante, offline, seed 42, temperatura 0, contexto 4096, máximo 1024 tokens, caché desactivada, auxiliares CPU |
| HTTP / residencia | read 180 s, connect 5 s, write/pool 60 s; keep_alive 30m |
| SHA-256 manifiesto bundle | `0bb1957f6ce0426214c4ca121f162638db0947e9f95b22fc214fad7034caec10` |
| SHA-256 manifiesto cohorte | `fca9054866656cfb801c040e6a673924836386efcad8d3edb255b15021b70c40` |
| SHA-256 pre-registro | `81b5b27f805089c2b57cdcbc308d39c19809a5e6f35c59f1cb10b377fcd82258` |

El bundle se verificó en cada ventana. Las muestras válidas acreditan AC, plan
efectivo y ausencia de las cargas excluidas según el observador; no ausencia
absoluta de actividad del SO. `final-memory-analysis-01.json`, generado
2026-09-10T20:37:58Z, verifica hashes de resultados, journals, telemetría, HTTP y
capturas ETW. SHA-256 `2625c89066164580f0ea6718d4075849de70f27464bf8a7e23c5eaa9e5e6df42`.

### Latencias completas de la receta nueva

Respuesta completa: retrieval + generación + NLI y costos externos al cronómetro
interno. Frío: proceso nuevo y descarga de Granite, conservando caché de archivos
del SO. Caliente: tres pipelines preparados en el mismo worker y residencia
comprobada por consulta. Los tres intentos de preparación contienen nueve consultas
y nueve probes NLI explícitos en total; están fuera de estas 120 posiciones.

| Sistema | Condición | Posiciones | n válido | p50 s | p95 s | Errores/abortos | Inválidas |
|---|---|---:|---:|---:|---:|---:|---:|
| Híbrido | Frío | 20 | 20 | 49,30 | 83,86 | 0 / 0 | 0 |
| Híbrido | Caliente | 20 | 20 | 28,97 | 51,57 | 0 / 0 | 0 |
| Léxico | Frío | 20 | 20 | 43,92 | 79,58 | 0 / 0 | 0 |
| Léxico | Caliente | 20 | 20 | 18,67 | 65,75 | 0 / 0 | 0 |
| Semántico | Frío | 20 | 19 | 39,71 | 72,31 | 0 / 0 | 1 |
| Semántico | Caliente | 20 | 20 | 19,89 | 45,09 | 0 / 0 | 0 |

Cierre del ejecutor: `window-semantic-warm-01/payload-complete.json`,
2026-09-10T12:44:51Z, `cohort_complete=true`, `report.passed=false`.
Las 120 respuestas terminaron, pero solo 119 entran en percentiles. Fallos técnicos
0/120; condiciones inválidas 1/120 (0,83 %), reportadas aparte. No convertir una
duración inválida en cero ni sustituir su posición.

### Incidencias y restauración

1. `window-semantic-cold-01` falló antes de iniciar la posición 2 por la guarda
   de descarga de Granite: no consumió un intento de consulta. Los logs muestran
   fallos simultáneos de GPU discovery, sin demostrar causalidad exclusiva.
   En Ollama 0.22.1, `expireRunner` programa la descarga y el HTTP 200 no espera
   necesariamente a que `/api/ps` quede vacío ([routes.go](https://github.com/ollama/ollama/blob/v0.22.1/server/routes.go#L330),
   [sched.go](https://github.com/ollama/ollama/blob/v0.22.1/server/sched.go#L857)).
   `unload-diagnostic-01/result.json` (2026-09-10T11:18:22Z) verificó precarga GPU
   sin generación y descarga completa. Se reanudó sin cambiar la guarda ni receta;
   sigue siendo una limitación operativa a corregir antes de otra cohorte.
2. `window-semantic-cold-02`: Brave apareció durante el índice 7; primera muestra
   2026-09-10T11:36:38Z, intento `d56a21dac9e943bea15ac28b03863e57`, 62,4932 s.
   Se conservó la respuesta como inválida y se pausó. `brave-interruption-01.json`
   conserva procesos, timestamps y hashes. No se cerró Brave programáticamente.
   Ausencia verificada antes de `semantic-final-start.json` a las 11:51:32Z;
   nueva admisión, sin reemplazar el índice 7. Presencia de Brave invalida el
   control acordado; no cuantifica su efecto causal en esa duración.

Las ocho ventanas tienen `restored.json` real, con snapshots y acciones en
`window.json`/`events/`. AnyDesk solo se detuvo en las dos ventanas híbridas;
las demás usaron `KeepAnyDesk` y conservaron el acceso remoto. Las restauraciones UTC:

| Ventana | Restauración UTC |
|---|---|
| hybrid-cold-01 | 2026-09-09T21:16:40Z |
| hybrid-warm-01 | 2026-09-10T01:59:32Z |
| lexical-cold-01 | 2026-09-10T02:37:21Z |
| lexical-warm-01 | 2026-09-10T03:05:38Z |
| semantic-cold-01 | 2026-09-10T03:17:13Z |
| semantic-cold-02 | 2026-09-10T11:38:32Z |
| semantic-cold-03 | 2026-09-10T12:16:53Z |
| semantic-warm-01 | 2026-09-10T12:44:56Z |

Las dos ventanas finales aprobaron contrastes de diez pares: límite superior
unilateral 95 % de interferencia 0,1341 % / 0,0681 %, frente al 5 % fijado.
La restauración no demuestra que el usuario haya leído el aviso o se haya reconectado.

### P900, timeout y cierre de resiliencia

Sesión ficticia `5f98cae82a544089a50616789f5095fa` bajo
`C:/Users/enziz/AppData/Local/Temp/cloudrag-technical-20260905-routing/`.
App del build medido; inicio `p900-streamlit-180-start-01.json`,
2026-09-10T20:39:10Z. Retomó revisión 49, consulta 8 de A, tres prácticas conservadas.
Terminó en revisión **168**, estado `complete`, **33 intentos / 30 ratings / 3 errores
históricos sin rating**. Las 23 respuestas nuevas terminaron sin fallos, con lectura
180 s y configuración/preparación persistidas. Diez ratings por sistema.

`p900-ui-evidence-final-01.json` conserva acceso por token, navegación de operador
ausente, fuentes abiertas, valoraciones, SUS y pantalla final. No se publica el token.
SUS global: diez respuestas sintéticas neutrales 3, score 50; respuestas abiertas
marcadas **PRUEBA TECNICA**. No es evidencia de usabilidad humana.

Dos pausas prolongadas agotaron residencia: invalidaciones 2026-09-10T22:26:17Z
y 2026-09-11T01:33:48Z. La app exigió preparar antes de crear nuevos intentos;
progreso intacto. Tres preparaciones completas y dos invalidaciones en `_preparation/`.
Estas pausas y configuraciones históricas impiden tratar P900 como sesión homogénea
de latencia o entrevista humana continua. No se mezcló con la cohorte.

La lectura 180 s no es un deadline total. Consulta 8 de A completó en 117,8537 s;
el navegador mostró el aviso de 60 s durante inferencia. La prueba separada
`wait-blocking-probe-01.py` reutilizó `render_wait` real con bloqueo síncrono de
125 s, sin modelos ni reloj simulado. `wait-blocking-probe-result-01.json` demuestra
aviso de 120 s a 02:00/02:02/02:04 mientras el servidor seguía bloqueado. No contar
este bloqueo sintético como latencia RAG. Textos exactos y límites de validez UX
permanecen en el pre-registro; el feedback puede influir en SUS y es parte de la UI evaluada.

Exportación más checkpoint: ocho archivos. `p900-export-reconstruction-01.json`
(2026-09-11T01:54:41Z) y su script `verify_p900_closure_01.py` verifican igualdad de
intentos, ratings, SUS, preguntas, fuentes y timestamps; unicidad de 30 ratings;
ausencia de enlaces a errores; lectura por el analizador; hashes inalterados.
SHA-256 de `full_session.json`:
`5a19d6a213e6cff9200bd3b09bc732de24994d7a797656b7cb0f5fe135a46074`.
La copia `R/p900-export-copy-01/` conserva los ocho archivos con esos mismos hashes;
`p900-export-copy-verified-01.json` acredita la copia sin modificar los originales.
La reconstrucción ocurrió después de detener la app, desde otro proceso.
Se reutiliza, sin repetir, la evidencia de desconexión/reinicio y errores:
`C:/CloudRAG/operational-20260905-routing/disconnection-check.json`,
`restart-recovery-check.json`, `real-timeout-no-rating.json`.
`p900-and-probe-stop-01.json` (2026-09-11T01:47:20Z) identifica las dos apps cerradas,
puertos 8501/8502 sin listeners y AnyDesk/NvContainer en Running/Auto.

### Hallazgos y límites del cierre

| ID | Severidad | Evidencia | Acción propuesta, no ejecutada |
|---|---|---|---|
| W01 | P1, bloquea el GO fijado | Léxico caliente p95 65,7494 s, dos respuestas >60 s | Propuesta de mejora con tests y nueva validación prospectiva; no modificar criterio ni seleccionar reintentos |
| W02 | P2, medición | Guarda de descarga inmediata; `measure_interview_gate.py:610`, ventana semantic-cold-01 | Espera acotada de descarga con registro de cada respuesta y regresión antes de otra cohorte; conservar este fallo |
| W03 | P2, presentación | P900 C3/C10: `[Source: None]` y `[Source: N/A]` aparecen como `None / /` y `N / A /`; `response_formatter.py:185`, `evaluation_page.py:238` | Separar citas no identificadas de fuentes asociadas; no inventar chunks ni alterar respuestas exportadas; revisar nota UX antes de cambiar la UI |
| W04 | P2, analizador | `load_all_sessions()` advierte `_preparation - no full_session.json`; acepta P900 completa | Distinguir directorios auxiliares de sesiones en el escaneo, con regresión; no generar estadísticas humanas con P900 |

W01 concentra tiempo en generación/NLI, no retrieval: q016 total 72,29 s,
generación 52,44 y NLI 17,75; q057 total 65,40 s, generación 42,70 y NLI 20,61.
RAM mínima durante esos chats 3,62/3,55 GiB y fallos duros globales 0,38/2,36 por
segundo. Se observó 21 % CPU / 79 % GPU, pero no se aisló el costo del offload ni
una causa exclusiva de hardware. No hay mejora estimable por comprar 32 GB;
[nota RAM actualizada](RAM_INTERVIEW_DIAGNOSTIC.md). `window-CLOSE` histórico sigue
indeterminado. Las comparaciones históricas no separan driver de carga/receta.

### Cohortes históricas separadas

No se recalcularon ni mezclaron. La diferencia con estos p95 no es un experimento
causal de driver. La cohorte 616.64 de septiembre 6 tuvo batería y Brave según
**DECLARACIÓN del usuario**, no condiciones retro-verificadas.

| Cohorte / build | Sistema/condición | Éxitos/intentos | p50 / p95 s | Errores / abortos |
|---|---|---:|---:|---:|
| 610.62 / 4af9728 | Híbrido frío | 19/20 | 74,91 / 88,93 | 1 / 0 |
| 610.62 / 4af9728 | Híbrido caliente | 18/20 | 35,15 / 67,44 | 2 / 0 |
| 610.62 / 4af9728 | Léxico frío parcial | 8/11 | 65,12 / 82,12 | 2 / 1 |
| 616.64 contaminada / fd747f8 | Híbrido frío | 19/20 | 90,67 / 126,01 | 1 / 0 |
| 616.64 contaminada / fd747f8 | Híbrido caliente | 18/20 | 40,41 / 75,90 | 2 / 0 |
| 616.64 contaminada / fd747f8 | Léxico frío | 16/20 | 71,50 / 96,67 | 4 / 0 |
| 616.64 contaminada / fd747f8 | Léxico caliente | 17/20 | 24,19 / 83,28 | 3 / 0 |
| 616.64 contaminada / fd747f8 | Semántico frío | 18/20 | 76,29 / 111,99 | 2 / 0 |
| 616.64 contaminada / fd747f8 | Semántico caliente | 17/20 | 26,20 / 67,35 | 3 / 0 |
| Piloto híbrido limpio / a761fa2 | Híbrido frío | 20/20 | 50,13 / 83,24 | 0 / 0 |
| Piloto híbrido limpio / a761fa2 | Híbrido caliente | 20/20 | 23,21 / 57,62 | 0 / 0 |

Fuentes históricas: `C:/CloudRAG/operational-20260905-routing/latency-cohort/`,
`C:/CloudRAG/cohort-61664-20260906T015527Z/` y
`C:/CloudRAG/managed-pilot-20260908T012621Z/window-real-05/`. Calentamientos aparte:
uno, tres y uno respectivamente. No ejecutar ni sobrescribir evidencia histórica.

### Veredicto contra el pre-registro

| Criterio | Estado | Evidencia |
|---|---|---|
| Bundle/identidad | ✅ VERIFICADO | Manifiestos y verificaciones por ventana |
| p95 caliente <=60 s, tres sistemas, cero fallos/inválidas | ❌ VERIFICADO | Híbrido 51,57; léxico 65,75; semántico 45,09 s; 20 válidas en cada celda caliente |
| Frío medido y pre-calentamiento obligatorio | ✅ VERIFICADO | 60 posiciones frías, una inválida preservada; preparación real de app y cohortes |
| P900 completa con SUS/exportación | ✅ VERIFICADO | Revisión 168, reconstrucción y hashes |
| Resiliencia integral | ✅ VERIFICADO | Historial de desconexión/reinicio + preparación perdida + errores sin rating + lectura tras parar servidor |

**NO-GO local; Fase B bloqueada.** El frío dejó de bloquear conforme al pre-registro,
pero el léxico caliente no cumple. No se transfirió GO a staging, no se diseñó nube,
no se generaron Docker/compose/env/provisionamiento, no se cotizó ni desplegó.
No se optimizaron modelos ni NLI a posteriori, no se cambió driver ni energía,
no se editaron datos exportados, evidencia experimental, corpus, gold o paper;
sin push ni merge. Los cambios finales son documentales.

Auditoría previa del build medido: `cohort-final-pytest.txt`, **507 pasan / 5
excluidas**, tres advertencias SWIG, 57,13 s; Ruff/diff/secretos aprobados en los
archivos `cohort-final-*`. Baseline original de esta enmienda: 476 pasan / 5
excluidas en `baseline-*`; 482 tras timeout, 494 tras preparación y 507 tras
congelar el protocolo. Auditoría final de documentación registrada con prefijo
`closure-final-*` en R; esos archivos conservan comandos, salidas y estado git.

**Pase final VERIFICADO:** 507 pasan / 5 excluidas, tres advertencias SWIG,
72,12 s (`closure-final-pytest.txt`). Ruff sobre los 18 archivos Python de la
enmienda, `git diff --check` y secretos con baseline pasan; 17 exclusiones
históricas del escáner, ningún hallazgo nuevo. `closure-final-checks.txt` registra
códigos de salida 0 y ninguna diferencia en `experiments/results`, `data`, `paper`
ni `output` respecto de `670f8e5`. Este cierre modifica cinco documentos; no hay
cambios adicionales de código ni inferencias después de la validación de P900.
