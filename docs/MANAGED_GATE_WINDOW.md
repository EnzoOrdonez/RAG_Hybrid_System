# Ventana local administrada (prueba técnica)

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

Los cuarenta slots híbridos (20 frío/20 caliente) no sustituyen los otros sistemas
ni el cierre de P900. Un fallo, aborto o condición inválida consume su slot, sin
reemplazo ni inclusión en p50/p95. Timeout de la app: 60 s por lectura, sin cambios.

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

## Semántica del timeout (VERIFICADO, sin cambio de la app)

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
