# Ventana local administrada (prueba técnica)

**Estado al 2026-09-10, Lima:** nueva cohorte preparada completa, 120 posiciones;
119 válidas y una inválida por Brave, sin errores de generación/abortos. Las ocho
ventanas quedaron restauradas, incluidas dos interrumpidas. P900 completó SUS y
exportación. Continúa NO-GO por p95 léxico caliente de 65,75 s. Resultados y rutas:
[registro operativo](OPERATIONAL_GATE_2026-09-05.md#cierre-de-la-cohorte-preparada-y-p900-2026-09-10-lima).
Los registros de ventana real 01–05 que siguen son históricos y permanecen separados.

## Crítica y criterios previos

La autorización del usuario permite cerrar AnyDesk y NVIDIA Overlay temporalmente.
Un `finally` en el mismo proceso no basta: un cierre de terminal o reinicio lo evita.
Por eso `scripts/manage_gate_window.ps1` registra primero un watchdog SYSTEM con
trigger de inicio del equipo y confirma su ejecución antes de cualquier corte.
Su política RemoteSigned se limita al proceso de la tarea; no modifica políticas
globales. No se detiene el servicio de pantalla NVIDIA ni protección del equipo.

El watchdog restaura AnyDesk primero ante vencimiento (máximo 120 minutos), pérdida
del controlador o heartbeat de más de 60 segundos. Después termina exclusivamente
el worker identificado por PID y fecha de creación: un Job Object Windows cierra
sus descendientes. Recupera capturas WPR por su instancia única y restaura el estado
previo de NvContainerLocalSystem, tarea de autoactualización NVIDIA y aplicaciones
interactivas autorizadas. Conserva intenciones, resultados y errores externos.
El equipo apagado no puede restaurar servicios hasta arrancar. El estado Running
de AnyDesk y la reaparición de su proceso no prueban una reconexión humana exitosa.

Antes del uso real se exige: suite completa, Ruff, diff y secretos limpios; pruebas
reales sintéticas de deadline y pérdida de controlador; Job Object con terminación
del descendiente; captura ETW con cero eventos/buffers perdidos. Las simulaciones
no constituyen prueba de restauración de servicios reales: esa se verifica al cerrar
la ventana real. El aviso visible indica la hora límite y ruta de estado antes del
corte; no se afirma que el usuario lo haya leído.

## Instrumentación y límites

`gate_memory.py` registra compromiso y límite globales, RAM disponible y contadores
PDH. Page Faults/sec, Page Reads/sec y Pages Input/sec no equivalen a fallos duros/s.
`gate_memory.wprp` captura HardFaults/ProcessThread/Filename, y `gate_etw.py` atribuye
los fallos duros observados por intervalos del reloj real de respuesta. Conserva
atribución desconocida y rechaza pérdida de eventos. En Windows 11, tracerpt puede
fallar en campos posteriores de Thread v3: el parser solo decodifica su prefijo
documentado ProcessId/TThreadId. No infiere prioridad ni stacks.

Iniciar/detener y decodificar WPR queda fuera del reloj de respuesta; la actividad
de captura durante la respuesta sí forma parte del contraste del observador. Se
exigen diez pares AB/BA, límite superior unilateral bootstrap 95% <=5%, seed 42 y
10.000 remuestreos. No se resta un costo sintético de las latencias RAG.
Una captura inválida conserva la respuesta y excluye su duración de percentiles.

La admisión exige 60 s observados en AC, Equilibrado/Mejor rendimiento, CPU/GPU
medias <10% y ausencia de cargas excluidas. El ejecutor no repite automáticamente
una admisión fallida. Guarda lista completa `nvidia-smi` para identificar usuarios
gráficos WDDM, además de procesos, offload, memoria y temperatura GPU. Temperatura
CPU se declara no disponible si no hay sensor autorizado accesible.

Los cuarenta slots híbridos históricos (20 frío/20 caliente) no sustituyen la nueva
cohorte de 120 posiciones ni el cierre de P900. Un fallo, aborto o condición inválida
consume su slot, sin reemplazo ni inclusión en p50/p95. La app participante usa
ahora lectura de 180 s y residencia renovable de 30 minutos; el protocolo prospectivo
está en [WARM_GATE_PREREGISTRATION.md](WARM_GATE_PREREGISTRATION.md).

Cada `Run` requiere `-Cohort RUTA_EXTERNA -System hybrid|lexical|semantic -Phase cold|warm`.
Inicializar una sola vez con `measure_interview_gate.py init --output RUTA_EXTERNA
--systems hybrid lexical semantic --controlled`, después de fijar y verificar las
variables de la guía y el commit auditado. El manifiesto incluye su ruta de bundle;
las ventanas siguientes verifican ese mismo archivo sin recrearlo. `-KeepAnyDesk`
evita tocar servicio e interfaces de acceso remoto cuando las condiciones lo permiten.
Sin ese switch, siguen siendo obligatorios el aviso y la necesidad observada del corte.
El supervisor conserva el límite de 120 minutos y el ejecutor deja margen antes de
otra consulta/preparación. El cierre del payload distingue condición y cohorte completas.
Las capturas compartidas llevan ID de ventana: el watchdog solo recupera las propias.

## Evidencia de preparación (VERIFICADO)

Raíz: `C:/CloudRAG/managed-pilot-20260908T012621Z/`. Las fechas de archivo son UTC;
el nombre de la raíz identifica el inicio del trabajo, no el comienzo del piloto.

- `recovery-prerequisite/`: ejecución de tarea SYSTEM y eliminación de la tarea.
- `watchdog-selftest-01/` y `watchdog-selftest-02/`: fallos de arranque, sin servicios
  detenidos. El segundo conserva el rechazo por política Restricted de SYSTEM.
- `watchdog-selftest-03/selftest-passed.json`: watchdog SYSTEM disparado por deadline
  el 2026-09-08T10:06:32Z, restauración **simulada** AnyDesk primero.
- `watchdog-controller-loss-01/selftest-passed.json`: pérdida de un controlador
  sintético detectada por SYSTEM el 2026-09-08T10:10:28Z; mismo orden de recuperación.
- `capture-smoke-01/smoke-result.json`: ciclo WPR real con instancia propia, cinco
  segundos de observación, decodificación y ausencia de pérdidas comprobadas.
- `memory-probe/`: 592 fallos duros globales sin pérdidas. El proceso Python sintético
  tuvo cero; endpointprotection.exe leyó primero el archivo nuevo y produjo 209
  fallos sobre él. Esto valida captura/atribución, no demuestra presión RAM del RAG.

No se cerró AnyDesk ni se ejecutó inferencia en estas pruebas de preparación.
Consultar el informe operativo para el resultado posterior de la ventana real.

El primer inicio real (`window-real-01/`, build `9616379`) falló antes de armar:
`msg.exe` no existe en esta instalación. No se detuvo AnyDesk ni se ejecutaron
consultas. Además, `Select-Object path` sobre hashtables produjo una ruta nula de
restauración; la regresión reprodujo el valor nulo antes de corregirlo. Se retiró
el watchdog que reintentaba esa restauración, verificando servicios y PIDs originales
en `window-real-01/precut-cleanup.json` (2026-09-08T10:19:24Z).
El aviso se cambió a `WScript.Shell.Popup`: `notice-test-01/notice-test.json` registra
la prueba real de cinco segundos (2026-09-08T10:22:08Z), terminada por timeout. Esto
no confirma lectura humana. La serialización conserva ahora las rutas mediante
objetos explícitos y elimina duplicados por aplicación/sesión.

El segundo inicio (`window-real-02/`, build `d28a790`) sí cerró AnyDesk y los overlays
y aprobó la admisión. El contraste se detuvo antes de completar su primer par:
sus comprobaciones de extremos aplicaban incorrectamente la continuidad de 15 s
al intervalo que incluía detener/decodificar WPR. Las tres muestras durante el
trabajo observado eran válidas; el error fue `endpoint_control_reasons=telemetry_gap`.
La regresión reprodujo el fallo antes de separar los controles de extremos de la
telemetría continua. Esta última mantiene su límite de 15 s, y la admisión exige
continuidad. No se reinterpretan como aprobados los resultados de ese intento.
AnyDesk se detuvo a las 10:25:08Z y se verificó Running/Auto a las 10:27:16Z del
2026-09-08; el resto de la restauración terminó a las 10:27:27Z y el watchdog se
eliminó. Evidencia: `window-real-02/events/` y `restored.json`. No hubo consultas RAG.

El tercer inicio (`window-real-03/`, build `e00b8b7`) aprobó admisión y avanzó en el
contraste, pero su consola mostró `UnicodeDecodeError`: `tracerpt` emitía bytes
locales incompatibles con `PYTHONUTF8=1`. Aunque XML/ETL seguían disponibles, la
salida de consola podía perderse en el hilo lector de subprocess. Se abortó la
ventana sin autorizar el piloto; AnyDesk y los demás componentes quedaron restaurados
a las 10:36:26Z del 2026-09-08 (`restored.json`). Una regresión con stdout/stderr
no UTF-8 reprodujo ambos errores. La captura guarda ahora los bytes íntegros en
base64 y una vista textual explícitamente escapada, sin depender de la página de
códigos de Windows ni silenciar pérdidas de evidencia.

El cuarto inicio (`window-real-04/`, build `c52d2dc`) aprobó admisión, completó cinco
pares y rechazó el brazo observado del sexto por **1.463 eventos perdidos** y cero
buffers perdidos, según XML y `contrast/telemetry/05.etw/summary.txt`. La guarda
funcionó; no hay un contraste aprobado ni consultas RAG. Restauración terminada
el 2026-09-08T10:45:50Z. La suficiencia del búfer es una **hipótesis**, no una causa
demostrada: se aumenta la reserva solicitada de 64 a 256 buffers de 64 KiB (16 MiB),
sin eliminar proveedores. El test limita la solicitud a 16-32 MiB; esto no afirma
la asignación efectiva del kernel ni confunde el BufferSize del ETL fusionado con
la reserva configurada. Criterios sin relajar: cero pérdidas y diez pares válidos
con límite superior de interferencia <=5%. La siguiente ventana vuelve a medir;
no completa ni mezcla los cinco pares anteriores.

VERIFICADO el 2026-09-09T00:20:59Z: `wpr -status collectors -details` informó modo
File, Buffer Size 64 KB y Number of Buffers 256 para la instancia propia de
`capture-buffer-probe-01/`. En esta versión local sí se aplica la solicitud,
aunque la página Microsoft de `Buffers` limita su descripción al modo memoria.
Se conserva stdout binario y vista textual en `command-*.json`; el smoke consulta
la configuración activa y detiene su captura en `finally`. Referencias del ajuste:
[Sessions](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/sessions) y
[Buffers](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/buffers).

## Resultado de la ventana 05 (VERIFICADO)

Raíz `C:/CloudRAG/managed-pilot-20260908T012621Z/window-real-05/`; build `a761fa2`.
Se entregó Popup el 2026-09-09T00:25:06Z y archivo `notice.json` con deadline
02:24:54Z. Lectura humana del aviso no verificada. Corte remoto terminado 00:25:12Z.

| Componente previo | Acción reversible | Restauración verificada UTC |
|---|---|---|
| AnyDesk servicio PID 29780, interfaz PID 20652 | Deshabilitar/detener servicio y cerrar interfaz | Servicio 01:24:22Z, interfaz PIDs 12632/36472 a 01:24:31Z |
| NvContainerLocalSystem PID 16372 | Deshabilitar/detener autoarranque durante ventana | Running/Auto 01:24:24Z |
| NVIDIA Overlay PIDs 7804/22236/23316/23524/33728 | Salen al detener su servicio; ausencia comprobada por admisión | Se restaura su mecanismo de arranque; no se afirma identidad de PIDs anteriores |
| NVIDIA App SelfUpdate, previamente habilitada | Deshabilitación temporal | Habilitada 01:24:26Z |
| Epic/launchers autorizados | Ningún proceso de esos nombres en snapshot inicial | No fue necesario relanzarlos |

`window.json` contiene rutas, identidad PID/creación y estado inicial; `events/`
contiene intenciones y comprobaciones. `restored.json` confirma restauración real
con AnyDesk primero a las 01:24:33Z; watchdog propio eliminado 01:24:36Z. No se
cambiaron NVDisplay.ContainerLocalSystem, FvSvc, protección del endpoint ni energía.
Recomprobación a las 11:01:17Z: `R/closure-restoration-audit.json`, donde R es el
directorio padre de ventana 05. AnyDesk PIDs 32752/36472, NVIDIA Overlay
7708/19812/20580/22008/26600; servicios AnyDesk/NvContainer Running/Auto,
tarea NVIDIA habilitada, watchdog propio ausente y Streamlit de prueba detenido.

Admisiones antes del contraste (14 muestras) y antes del piloto (13) aprobadas.
Contraste: diez pares completos, costo mediano -0,179 %, límite superior unilateral
95 % bootstrap +0,0463 % (<5 %), sin descuento de tiempo. `contrast/result.json`.
Los 40 intentos y un calentamiento tienen captura ETW sin pérdidas; análisis
posterior comprueba los hashes. RAG terminó 01:24:20Z (`payload-complete.json`).

«Controlado» significa aprobado según los controles registrados, no ausencia
absoluta de actividad del sistema. ETW conservó actividad de servicios Windows,
protección del endpoint y `upc.exe` (146 fallos en el primer chat frío). Este último
nombre no pertenece a la lista de exclusión implementada: no se inventa una causa
ni se reescriben los estados de la cohorte para ocultarlo. Sus efectos no están
aislados; esta limitación impide interpretar la admisión como prueba causal total.

## Registro histórico de semántica del timeout (antes de la enmienda de 180 s)

`tests/test_http_timeout_semantics.py` verifica con HTTP real de loopback que un
silencio de lectura agota `httpx.ReadTimeout`, que bytes sucesivos reinician la
espera de lectura y que `list` y `chat` tienen presupuestos separados. El límite
de lectura no es un deadline total de consulta.

El probe histórico frío/180 s no contradice esa semántica: su trace `chat` duró
128,3947 s, mientras el intento completo duró 199,0460 s; aproximadamente 70,65 s
ocurrieron fuera de `chat` (incluyendo `list`, 2,0800 s). Evidencia original:
`C:/CloudRAG/cohort-61664-20260906T015527Z/diagnostics/cold-180/`.
El timeout de participante permanece en 60 s. Elevarlo no acredita p95 <=60 s;
una propuesta de espera más larga y feedback requiere revisar primero el piloto
y autorización específica del comportamiento visible.
