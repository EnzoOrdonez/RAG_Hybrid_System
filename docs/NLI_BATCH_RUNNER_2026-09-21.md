# Runner de validación NLI compartido — 2026-09-21

**NO-GO vigente.** Implementación autorizada por el usuario el 21-sep; eficacia y
equivalencia numérica con modelos reales todavía pendientes. Este runner ejecuta
el [pre-registro aprobado](NLI_SHARED_PREREGISTRATION_2026-09-20.md).
La app conserva `per_claim`. El candidato `cross_claim` sólo se selecciona en
el ejecutor técnico; no cambian textos UX, prompts, retrieval, re-ranking,
modelos, umbrales, dispositivo ni tamaño interno de lote (32).

## Diseño y crítica

### Enmienda autorizada de reanudación (21-sep, séptima tanda)

DECISIÓN DEL USUARIO, previa a la medición: continuar los pares restantes
conservando el hueco `INCOMPLETO_INTERRUMPIDO`. Crítica previa a implementar:
un registro sintético de intento ausente inventaría una observación; borrar el
brazo existente ocultaría un fallo. Se usará un anexo inmutable por par con
referencias y hashes de los intentos originales y lista de brazos no ejecutados.
Se publica sólo después de recuperar intentos y verificar restauración, nunca
entre los brazos de un par que aún se está ejecutando en la misma ventana.

El calendario podrá quedar consumido con menos de 120 ejecuciones reales. Eso
no significa suficiencia: se conservan **20 pares completos válidos por sistema**
como n mínimo registrado, cero fallos y p95 candidato ≤60 s. No se ha añadido un
cálculo de potencia ni rebajado el n. Contrastes, bootstrap y equivalencia pareada
usan sólo pares con ambos brazos válidos en la misma ventana; los brazos huérfanos
siguen disponibles como evidencia individual descriptiva, fuera del contraste.
Una interrupción de energía sigue abortando toda la cohorte según el protocolo.

Criterios verificables: reanudación autorizada sin duplicados ni imputación;
anexo idéntico al volver a empaquetar; rechazo de intento posterior en un hueco;
conteos separados y NO-GO por insuficiencia incluso con latencias restantes bajas.

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

**Regla autorizada:** al restaurar se recuperan los intentos abortados, sin
inventar duración final, y se sella cada par roto en
`cohort/gaps/<sistema>-<índice>.json` como `INCOMPLETO_INTERRUMPIDO`. El anexo
referencia los intentos existentes con SHA-256 y enumera `missing_arms`; no crea
un intento para el brazo que nunca comenzó. Si ambos comenzaron pero uno quedó
abortado, el par también se excluye, aunque `missing_arms` esté vacío.

Al reanudar se verifica el manifiesto, los hashes y `-AuthorizeNewWindow`, y se
avanza al siguiente par pendiente. Está prohibido repetir los intentos previos,
completar el compañero en otra ventana, sustituir posiciones o recalcular sus
duraciones. El anexo se publica una sola vez, entra al manifiesto y permanece
idéntico al reempaquetar. No se aplica esta política retroactivamente a paquetes
construidos sin `interrupted_pair_policy=continue-pairs-preserve-gap-v1`.

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
   supervisor/admisión y `intento i/120 | posicion j/120 | sistema | brazo | elapsed | ETA`.
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
   logs antes de reanudar. Tras una interrupción y restauración verificada se
   permite continuar los pares restantes con el mismo comando; el hueco ya
   sellado no se rellena. Si hay fallo de calidad, identidad o restauración,
   detenerse y traer el paquete; no forzar ni abrir una cohorte para ocultarlo.
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
También crea `scenarios/interrupted-pair/`: interrupción tras la novena respuesta,
restauración simulada, rechazo sin autorización y reanudación autorizada. El
resultado es **119 intentos reales simulados, un brazo no ejecutado, 59 pares
completos y NO-GO por insuficiencia**. Compara hashes antes/después y no rellena
el intento décimo. Es un escenario separado del camino feliz de 120 posiciones.
No sustituye las pruebas reales del Programador de tareas antes del corte.

`summary.json` distingue `planned_slots=120`, `executed_attempts` y
`not_executed_slots`. El campo heredado `total_attempts=120` es el tamaño planeado,
no el número observado. `groups` conserva seis celdas individuales descriptivas
con n, fallos, inválidos y p50/p90/p95/mín/máx/media por etapa y carga.
`paired_groups`, `pairs` y el bootstrap usan **sólo pares completos válidos**;
un brazo huérfano válido puede aparecer en `groups`, nunca en el contraste.
`pair_states` separa `COMPLETO_VALIDO`, `INVALIDO_CONTAMINACION`,
`INCOMPLETO_INTERRUMPIDO`, `FALLIDO_TECNICO` y posiciones pendientes. Las causas
adicionales siguen en los registros originales. `interrupted_pairs` incluye los
anexos; `complete_valid_pairs_by_system` y `sufficiency` declaran suficiencia.

Bootstrap exploratorio (10.000, seed 42) sobre los pares disponibles, con n
efectivo explícito y etiqueta insuficiente si n<20; no cambia el umbral GO.
Fallos/abortos/invalidaciones nunca entran en cuantiles ni se imputan. La
discrepancia respecto del ahorro NLI supuesto del 65 % también es descriptiva.

`complete` sólo significa calendario consumido, incluidos huecos registrados.
Los replays de respuestas nuevas se restringen a brazos de pares válidos; los
40 replays fuente permanecen separados. `quality.replay_complete` indica que
terminaron los replays elegibles; evita abrir ventanas sin fin cuando existe
un hueco. `quality_pass` sigue exigiendo las 160 entradas; sin 20 pares válidos
por sistema no puede cumplirse. Los replays excluidos previos se conservan y
enumeran, sin incorporarlos a la equivalencia pareada.
`confirmation_ready` exige suficiencia y calidad.
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

La auditoría de esta enmienda de huecos se registra separadamente en
`C:/CloudRAG/nli-gap-20260921-060446391/`, baseline `fb45949`, 653 tests.
Los nuevos logs finales y el paquete sintético no alteran la evidencia anterior.
