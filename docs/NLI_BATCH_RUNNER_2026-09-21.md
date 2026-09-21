# Runner de validación NLI compartido — 2026-09-21

**NO-GO vigente.** Implementación autorizada por el usuario el 21-sep; eficacia y
equivalencia numérica con modelos reales todavía pendientes. Este runner ejecuta
el [pre-registro aprobado](NLI_SHARED_PREREGISTRATION_2026-09-20.md).
La app conserva `per_claim`. El candidato `cross_claim` sólo se selecciona en
el ejecutor técnico; no cambian textos UX, prompts, retrieval, re-ranking,
modelos, umbrales, dispositivo ni tamaño interno de lote (32).

## Diseño y crítica

Una posición se identifica por sistema, índice y brazo. Las 120 posiciones son
20 consultas × tres sistemas × dos brazos, en calendario determinista rotado,
diez órdenes control/candidato y diez candidato/control por sistema. Se firma
antes de medir. Control y candidato usan el mismo build instrumentado. Las
cohortes anteriores no se mezclan con ésta.

El cronómetro conserva la frontera corregida: comienza en `measure_attempt`,
incluye el chequeo `preparation.pipeline` dentro de la operación, consulta
completa y exportación de trazas; termina al volver la operación. Preparación
inicial de los tres sistemas, admisión, controles de identidad posteriores,
persistencia final y replays de calidad quedan fuera. Los relojes por etapa
no suman sus percentiles; `other_s` es el residual por respuesta.

Riesgos considerados: agrupar puede aumentar padding/memoria o cambiar scores.
Se registran scores crudos durante ambos brazos, etiquetas, claims, fuentes,
prompts y carga. No se presume aceleración por reducir llamadas. Un fallo de
lote conserva la recuperación secuencial pero invalida el éxito técnico.
Los mocks prueban contratos; no prueban rendimiento ni equivalencia numérica
del modelo real.

Después del calendario de latencias se verifican las 40 respuestas fuente y
las 120 nuevas con ambos algoritmos sobre el mismo texto **original del LLM**
y sus chunks. No se regenera texto. Se comparan exactamente etiquetas,
evidencia, scores publicados y agregados, excluyendo sólo `processing_time_ms`.
Se guardan scores sin redondear y hashes de las entradas. No se introduce una
tolerancia. No se intercalan replays entre brazos de un par. Cualquier diferencia
o error se conserva y bloquea aceptación. Un replay aprobado no se repite al
reanudar; uno fallido requiere revisión, no un reintento para obtener aprobación.

## Ventanas, pausa y fallos

Se reutiliza el supervisor de Windows: prueba independiente antes del corte,
restauración ante muerte del controlador, cierre normal o límite de 120 minutos,
NVIDIA Overlay y mecanismos identificados únicamente. `NVDisplay` y AnyDesk
quedan intactos. Cada ventana exige `-AuthorizeNewWindow`, incluida la primera.
El flag expresa autorización puntual del humano; no renueva por silencio una
ventana histórica. Pre-flight permite sólo el overlay objetivo antes del corte;
después exige su ausencia. Navegadores y procesos GPU ajenos bloquean inicio.
El comando interno de inferencia rechaza ejecución independiente sin manifiesto
de ventana armada, pruebas del supervisor y plazo vigente de hasta 120 minutos.

Planificación conservadora: antes de un bloque de seis respuestas se exige
margen de 600 s por respuesta restante en el bloque más 120 s de restauración.
Este margen **no** redefine el timeout HTTP, que sigue siendo por lectura.
Si falta margen, se guarda una pausa y se restaura; la próxima ventana repite
admisión, contraste del observador y preparación. Antes de cada respuesta se
comprueba margen de restauración. El supervisor independiente conserva la cota
total aunque una llamada quede bloqueada. Los replays exigen 600 s de margen.

Una contaminación ordinaria durante un intento lo marca inválido en vivo;
no se reemplaza ni se descarta su respuesta lenta, y el calendario continúa.
Fallo técnico, pérdida de identidad, telemetría rota o evento de energía detienen
la ejecución y conservan evidencia. Ante disco lleno se conserva la reserva
de emergencia del supervisor. Suspensión automática se inhibe sólo en la ventana.
Con Windows apagado no puede garantizarse restauración inmediata; al volver se
restaura, se prohíben inferencias fuera de plazo y la cohorte energética queda
abortada, sin reutilizar sus datos como válidos.

**Caso de protocolo pendiente de decisión del usuario:** si se interrumpe un par
después de su primer brazo, está prohibido completar el segundo en otra ventana.
El runner rechaza esa reanudación **antes de intervenir NVIDIA** y pide revisión;
no declara por su cuenta la cohorte terminal ni salta el hueco. La pregunta
pendiente es terminar como insuficiente o continuar sólo pares posteriores.
Los intentos existentes quedan intactos; el brazo no iniciado no se imputa.
La reanudación entre pares completos sí está implementada.

## Lanzamiento humano (no ejecutado por el agente)

1. Usar físicamente el equipo, AC y Mejor rendimiento, sin otros trabajos.
   Cerrar Brave/Chrome/Edge/Firefox, launchers y overlays ajenos; AnyDesk debe
   estar detenido. No cambiar driver, Ollama, modelo ni configuración.
   Verificar en PowerShell:

   ```powershell
   Get-Process brave,chrome,msedge,firefox,opera,EpicGamesLauncher,steam,AnyDesk -ErrorAction SilentlyContinue
   Get-Service AnyDesk
   nvidia-smi
   ```

   El primer comando debe quedar vacío y AnyDesk en `Stopped`. No cerrar
   procesos desconocidos a ciegas: el pre-flight imprime sus PID para revisión.
   El overlay NVIDIA objetivo puede estar presente: lo controla el supervisor.

2. Abrir **PowerShell como administrador** y comprobar rama y árbol limpio:

   ```powershell
   Set-Location 'C:\Users\enziz\Projects\hybrid-rag-system\.worktrees\interview-readiness'
   git branch --show-current
   git status --short
   git log -1 --oneline
   ```

3. Lanzar una ventana autorizada:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_diagnostic_window.ps1 -NliExperiment -AuthorizeNewWindow
   ```

   Se imprime la ruta nueva `C:/CloudRAG/diag-run-<timestamp>/`, estado de
   supervisor/admisión y `intento i/120 | sistema | brazo | elapsed | ETA`.
   Después aparece progreso de calidad. La ETA de consultas no incluye replays.
   ESTIMADO de planificación: varias horas y posiblemente varias ventanas,
   según generación/NLI; no se promete completar 120 más 160 replays en 120 min.
   Nunca abrir navegadores/juegos durante una ventana de medición.

4. Una pausa por margen restaura primero. Con nueva autorización explícita:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_diagnostic_window.ps1 -NliExperiment -Resume 'C:\CloudRAG\diag-run-REEMPLAZAR' -AuthorizeNewWindow
   ```

   Mismo build/entorno/paquete, sin editar archivos ni rellenar posiciones.
   Código 3: calendario pendiente tras pausa; código 2: evidencia insuficiente
   o calidad pendiente/fallida; código 1: fallo. Revisar `summary.json` y los
   logs antes de reanudar. Si hay fallos de calidad o par incompleto, detenerse
   y traer el paquete; no forzar ni abrir una nueva cohorte para ocultarlos.
   Máximo dos admisiones fallidas por la misma causa: identificarla antes de
   otra autorización. Código 0 sólo indica paquete listo para análisis, **no GO**.

5. En un fallo, conservar toda la carpeta. Traer `summary.json`, ruta del paquete
   y el mensaje de error (foto sólo como apoyo). Evidencia principal:
   `windows/<id>/cohort-run.log`, `payload-error.json`, `restored.json`, pruebas
   en `proof/`, `cohort/attempts/`, `cohort/quality/`. Si falta restauración
   verificada, no relanzar ni retirar manualmente el watchdog.

6. Al terminar, devolver `summary.json`, `manifest.json`, `manifest.sha256` y la
   ruta completa del paquete. Para comprobar hashes sin modelos ni intervención:

   ```powershell
   .\.venv-app\Scripts\python.exe scripts/unattended_diagnostic.py verify --root 'C:\CloudRAG\diag-run-REEMPLAZAR'
   ```

## Prueba sintética y lectura del resumen

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_diagnostic_window.ps1 -NliExperiment -DryRun
```

No requiere administrador, no carga modelos ni toca servicios. Comprueba rechazo
de contaminación, expiración sin inferencias, supervisor **simulado** independiente
tras kill de un hijo, autorización de reanudación, 120 posiciones sin duplicación,
alternancia real de estrategia con modelo simulado, 160 replays, invalidez y hashes.
No sustituye las pruebas reales del Programador de tareas antes del corte.

`summary.json` separa seis celdas brazo/sistema: n, fallos, inválidos y
p50/p90/p95/mín/máx/media por etapa y carga. Incluye 60 pares con ventana/orden,
bootstrap pareado exploratorio (10.000, seed 42) y discrepancia frente al ahorro
NLI supuesto del 65 %. Fallos/abortos/invalidaciones nunca entran en cuantiles.
`complete` sólo significa calendario consumido; `quality_pass` exige equivalencia
de las 160 entradas; `confirmation_ready` exige 20 válidos por cada celda y calidad.
`latency_pass_candidate` exige p95 ≤60 s en los tres sistemas candidatos.
El control se reporta aunque exceda 60 s. Ningún flag cambia automáticamente el
veredicto: quedan la auditoría del paquete y contratos P900/resiliencia.

La evidencia de desarrollo (no latencias reales) está en
`C:/CloudRAG/nli-build-20260921T004410616Z/`. El commit de producción es `c86d734`.
No se ejecutaron aquí la cohorte real, P2s, infraestructura ni despliegues.

Auditoría de alcance: Ruff sobre archivos tocados aprobado. Ruff global encontró
149 incidencias en 60 archivos versionados idénticos al baseline `976a379`, según
`runner-ruff-baseline-comparison.json`; no se corrigen fuera de este alcance ni
se afirma que el repositorio completo esté limpio de lint. Suite, secretos,
diff y dry-run se registran en los logs externos `runner-final-*`.
