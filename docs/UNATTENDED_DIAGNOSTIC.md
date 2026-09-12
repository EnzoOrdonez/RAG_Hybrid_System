# Diagnóstico emparejado desatendido — 2026-09-12

Estado: **NO-GO vigente**. Este runner prepara evidencia diagnóstica; nunca
autoriza entrevistas ni implementa una optimización. El humano lanza la cohorte.
El agente ejecuta solamente tests y el dry-run sintético.

## Diseño aprobado y crítica

Se reutilizan el coordinador emparejado y el supervisor existente. Duplicar la
lógica de inferencia habría creado otra frontera temporal y otra definición de
validez. El protocolo nuevo se identifica como `unattended-paired-v1`; los
protocolos históricos conservan su comportamiento y sus archivos.

Las tres decisiones del usuario quedan aplicadas:

1. **Pre-flight en dos fases.** Antes del corte solamente se permite el overlay
   objetivo, con imagen exacta
   `C:\Program Files\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA Overlay.exe`.
   Brave, Chrome, Edge, Firefox, Opera, Steam, Epic Games Launcher, AnyDesk y
   otros overlays provocan rechazo. Un PID GPU desconocido también. La lista
   GPU permitida está cerrada en `process_reasons`: procesos Windows con ruta
   bajo Windows, Ollama en su ruta instalada y Windows Terminal firmado por
   su ubicación de instalación esperada. Estas comprobaciones de ruta son una
   política operativa, no una verificación criptográfica de ejecutables.
   Tras el corte no se permite ningún overlay. No se mata un proceso desconocido.
2. **Reanudación explícita.** Cada ventana tiene identidad, estado previo, prueba
   independiente del supervisor y restauración propios. Tras restauración se
   exige `-Resume ... -AuthorizeNewWindow`. Antes de continuar se verifican el
   inventario y los hashes; el build y la receta deben coincidir. No se duplican
   posiciones ni se sustituyen inválidas. Una cohorte con 40 posiciones consumidas
   no abre otra ventana, aunque tenga posiciones inválidas.
3. **Suspensión.** El supervisor mantiene `ES_SYSTEM_REQUIRED` solamente durante
   la ventana, sin cambiar el plan energético. Libera la solicitud al terminar.
   Compara tiempo transcurrido con tiempo despierto y el arranque de Windows;
   una diferencia superior a 2 segundos o un reinicio clasifica la cohorte como
   abortada por energía. La tarea arranca también al iniciar Windows. No se
   inician inferencias tras el plazo vencido. Los resultados parciales originales
   permanecen intactos; un archivo adicional invalida la cohorte completa.

Windows permite impedir la suspensión automática por inactividad, pero no una
suspensión impuesta por el usuario. La restauración **no es garantizable con el
sistema apagado o suspendido**: se ejecuta cuando Windows vuelve a funcionar.
La detección tiene una resolución de aproximadamente dos segundos. Referencias:
[SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate),
[tiempo de interrupción](https://learn.microsoft.com/en-us/windows/win32/sysinfo/interrupt-time).

| Escenario | Respuesta y límite declarado |
|---|---|
| Se cierra la consola o muere el ejecutor | Supervisor SYSTEM independiente, identidad PID+inicio y heartbeat; restaura por muerte o heartbeat obsoleto. La prueba real del supervisor es obligatoria en cada lanzamiento, antes del corte. |
| Ollama cae o falla la inferencia | Persiste fallo, consume esa posición, detiene y restaura. No inventa duración para abortos ni los mete en percentiles. |
| Aparece un navegador durante una consulta | El observador publica invalidez persistente en vivo. Conserva respuesta y posición; continúa el calendario ante contaminación ordinaria. No reemplaza la posición. |
| Identidad, energía, residencia o telemetría fallan | Detiene: continuar implicaría un protocolo diferente o evidencia no fiable. |
| Disco lleno | Admisión exige 10 GiB libres; reserva real de 16 MiB para recuperación. La restauración de NVIDIA precede a limpiar ETW. Un error del log no impide intentar restaurar. No puede garantizarse escribir evidencia en un disco averiado; el error impide declarar un paquete íntegro. |
| ETW no se detiene | Sólo se intenta detener la instancia propia identificada; nunca cancelación global. Se informa `cleanup_pending`; no se permite otra intervención hasta resolverla. |
| Dos lanzamientos simultáneos | Bloqueo exclusivo por ruta antes de preparar; no hay dos controladores para el mismo paquete. |
| Se agotan 120 minutos | El corte de inferencia/restauración comienza a los 118 minutos, dejando dos minutos de margen. Un SO o servicio bloqueado puede impedir completar la restauración: no se afirma éxito sin `restored.json`. |

Riesgos restantes: rutas de instalación distintas se rechazan, no se adivinan;
la carga puede cambiar entre muestras; no se garantiza disponibilidad de disco,
del Programador o del sistema operativo. El dry-run no demuestra que el
Programador y los servicios reales funcionarán: las dos pruebas independientes
del lanzamiento real son la condición previa al corte. No se cambia la velocidad
de modelos ni se descuenta arbitrariamente el costo del observador: el contraste
pareado sintético real precede a la cohorte y debe aprobar.

## Frontera temporal y criterios

40 posiciones diagnósticas: léxico e híbrido, mismas 20 consultas, calientes,
orden AB/BA del coordinador existente. Preparación de los tres sistemas y
contraste van separados. Cada consulta incluye la selección/verificación de
preparación que hace el flujo real, consulta completa y serialización/exportación
instrumentada. La instalación y desmontaje del instrumento quedan fuera del
cronómetro. `tests/test_diagnostic_clock_boundaries.py` y la regresión 2+3=5 de
`tests/test_paired_diagnostic.py` fijan la frontera corregida de `658b748`.

No se eliminan lentos. Fallos, abortos y condiciones inválidas quedan fuera de
p50/p95, con sus conteos aparte; no ejecutados no se imputan. `complete` significa
calendario consumido sin aborto energético; `confirmation_ready` exige 20 válidos
por sistema. Ninguno equivale a GO. Los percentiles por etapa **no se suman**.
No se mezcla ninguna cohorte histórica. Modelo/digest/driver, seed 42, offline,
timeout de lectura 180 s, caché desactivada y preparación mantienen la receta.

## Lanzamiento humano

1. En la máquina local, conecta AC y comprueba el plan de alto rendimiento:
   `powercfg /getactivescheme`. No cambies de driver, modelo ni entorno durante
   la ventana. Deja al menos 10 GiB libres en C: (`Get-PSDrive C`).
2. Cierra navegadores y launchers. Comprueba:

   ```powershell
   Get-Process brave,chrome,msedge,firefox,opera,steam,EpicGamesLauncher,AnyDesk -ErrorAction SilentlyContinue
   Get-Service AnyDesk
   nvidia-smi
   ```

   El primer comando debe quedar sin procesos y AnyDesk debe estar `Stopped`.
   Los overlays objetivo pueden estar presentes antes del lanzamiento; los
   gestiona el script. Si aparece otro consumidor, identifícalo y ciérralo tú.
3. Abre **PowerShell como administrador**, entra al worktree y ejecuta:

   ```powershell
   Set-Location 'C:\Users\enziz\Projects\hybrid-rag-system\.worktrees\interview-readiness'
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_diagnostic_window.ps1
   ```

   La política Bypass afecta sólo ese proceso. No cambia una política global.
   El árbol debe estar limpio. La guía operativa y el bundle deben estar instalados.
4. Verás la ruta `C:/CloudRAG/diag-run-<timestamp>`, las pruebas del supervisor,
   admisión/contraste y `intento i/40 | sistema | elapsed | ETA` al completar cada
   intento. El archivo de estado de la ventana incluye inicio y plazos.
   **ESTIMADO: 60–90 minutos**, extrapolación operativa de la ventana parcial;
   límite máximo de intervención 120 minutos, sujeto al límite de SO descrito.
   No abras otras aplicaciones ni suspendas el equipo.
5. Al terminar, comprueba `windows/<id>/restored.json` y devuelve **`summary.json`
   más la ruta completa del paquete**. Conserva también `manifest.json` y
   `manifest.sha256`; el agente verificará todo antes de analizar.

Códigos: 0 = 40 posiciones válidas y limpieza cerrada (sigue siendo diagnóstico);
2 = calendario consumido sin suficiencia/limpieza completa; 1 u otro no cero =
fallo técnico o rechazo. El dry-run tiene código 0 si todas sus comprobaciones
sintéticas pasan, aunque fabrica una posición inválida deliberadamente.

## Reanudar y recuperar

Una interrupción no energética consume el intento abierto como aborto, sin
duración inventada. Después de restauración y paquete íntegro:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_diagnostic_window.ps1 -Resume 'C:/CloudRAG/diag-run-RUTA_REAL' -AuthorizeNewWindow
```

`-AuthorizeNewWindow` es la autorización humana de esa nueva intervención, no
una autorización permanente. No se repite una ventana fallida automáticamente.
Si hay evento de energía, la cohorte es terminal: conserva el paquete y consulta
antes de una cohorte nueva. No cambies archivos para forzar la reanudación.

Si falla el comando, conserva el texto de consola y la carpeta; comprueba primero
la restauración. Si falta `restored.json`, no lances otra ventana. Para solicitar
restauración idempotente de **esa** ventana (sin nuevo corte):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\manage_gate_window.ps1 -Mode Restore -Root 'C:/CloudRAG/diag-run-RUTA_REAL/windows/ID_REAL'
```

Trae los eventos y cualquier error. Si sólo falta empaquetar después de restaurar:

```powershell
.\.venv-app\Scripts\python.exe scripts/unattended_diagnostic.py package --root 'C:/CloudRAG/diag-run-RUTA_REAL'
.\.venv-app\Scripts\python.exe scripts/unattended_diagnostic.py verify --root 'C:/CloudRAG/diag-run-RUTA_REAL'
```

Un rechazo inicial conserva `preflight/*.json` y no abre la ventana; no produce
una cohorte completa. Corrige lo indicado antes de un nuevo lanzamiento. Nunca
borres evidencia para hacer pasar el pre-flight. Tras dos rechazos por la misma
causa, trae la lista de procesos y el rechazo; no repitas a ciegas.

## Evidencia y pruebas de esta entrega

Baseline VERIFICADO: `1385c640019eaf5499db1a715d9d8783e9dcb2d7`, 538 tests,
5 excluidos, en `C:/CloudRAG/unattended-build-20260912T023124Z/`.
El directorio contiene la auditoría final y el dry-run `dry-run-final/`.
`dry-run.json` demuestra rechazo sintético, muerte de un proceso hijo propio y
restauración simulada por otro proceso, autorización de reanudación, cero
inferencias con plazo vencido, 40 posiciones sin duplicación, invalidez retenida,
hashes y aborto terminal por energía. **No se tocaron servicios, tareas NVIDIA
ni modelos en el dry-run.** El supervisor real se prueba en el lanzamiento humano.

Mapa de regresiones: `tests/test_unattended_diagnostic.py` cubre políticas,
observador real con muestras simuladas, continuación del ejecutor real ante una
posición inválida, integridad de inventario/hashes, disco lleno, limpieza ETW,
subproceso terminado y funciones PowerShell aisladas. Los tests de reloj cubren
las inclusiones/exclusiones temporales. Los JSON por intento y ventanas son
inmutables; `packages/<id>/` conserva versiones anteriores de manifiesto y
resumen. Las vistas superiores se actualizan atómicamente. Se excluyen del hash
únicamente archivos de bloqueo, reserva temporal y autorreferencia del manifiesto.

DECLARADO por el usuario el 11-sep: él abrió Brave en el intento 29 de la segunda
ventana. Ese intento sigue inválido, los 11 no ejecutados no se imputan y los
14 pares parciales siguen descriptivos. No cambia el NO-GO ni confirma una causa.
