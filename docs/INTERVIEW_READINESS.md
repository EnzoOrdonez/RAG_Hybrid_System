# Preparación técnica de Evaluation Mode

**NO-GO para entrevistas (2026-09-09).** El piloto controlado con driver 616.64
terminó con 40 respuestas híbridas completas y cero fallos: p95 frío 83,24 s,
caliente 57,62 s. Pasaron admisión y contraste del observador; AnyDesk y mecanismos
NVIDIA fueron restaurados. P900 volvió a fallar en consulta 8 al recuperarla desde
la app, fuera del piloto: revisión 49, diez intentos, siete ratings, sin SUS ni
exportación. Se conservan los tres errores sin valoración. El timeout sigue en
60 s; su semántica por lectura ya está comprobada, no limita el tiempo total RAG.
La preparación de nube continúa bloqueada: faltan cierre integral, controles limpios
léxico/semántico y atribución exclusivamente a hardware. El registro vigente está en
[OPERATIONAL_GATE_2026-09-05.md](OPERATIONAL_GATE_2026-09-05.md).
No se ha desplegado en nube. La evidencia experimental congelada, el corpus, el gold
y el trabajo académico no forman parte de los cambios.
El usuario declaró uso con batería y aperturas de Brave durante esa cohorte: sus
latencias no acreditan capacidad local bajo condiciones controladas.

## Entorno instalable

`requirements-app.in` declara dependencias directas; `requirements-app.txt` fija las
transitivas y sus hashes para Python 3.14 y auxiliares CPU. Es un entorno nuevo de
aplicación: no reemplaza las versiones históricas de `requirements-lock.txt`.
Desde la raíz del checkout, con Python 3.14 y uv 0.11.21 instalados:

```powershell
uv venv --python 3.14 .venv-app
uv pip sync --python .venv-app/Scripts/python.exe --torch-backend cpu --require-hashes requirements-app.txt
uv pip check --python .venv-app/Scripts/python.exe
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONHASHSEED = '42'
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = ''
& .venv-app/Scripts/python.exe -m pytest -m 'not slow and not gpu' -ra
```

En Linux, usar `.venv-app/bin/python` y `export VARIABLE=valor`. CI usa el mismo lock
con `uv pip sync --system --torch-backend cpu --require-hashes requirements-app.txt`.
La instalación descarga paquetes; no descarga pesos de modelos.

## Artefactos y configuración de despliegue

La app no crea índices ni descarga modelos al admitir participantes. El operador debe
provisionar un bundle de confianza con los cinco archivos de `INDEX_FILES` y los tres
snapshots de `MODEL_DIRS` declarados en `src/utils/deployment_artifacts.py`. El bundle
incluye las consultas reales; no se sustituyen silenciosamente por ejemplos.
No cargar archivos BM25 pickle de procedencia desconocida: el manifiesto comprueba
integridad respecto de una referencia confiable, no autenticidad de su origen.

Con los artefactos existentes, crear una referencia en una ruta nueva fuera de los
directorios de evidencia y verificarla después de copiar el bundle al despliegue:

```powershell
& .venv-app/Scripts/python.exe scripts/check_deployment_artifacts.py snapshot --manifest C:/CloudRAG/deployment-manifest.json
& .venv-app/Scripts/python.exe scripts/check_deployment_artifacts.py verify --manifest C:/CloudRAG/deployment-manifest.json
```

Estos comandos solo leen y calculan hashes de los artefactos; `snapshot` crea el
manifiesto exclusivamente y rechaza sobrescritura. Crear antes su directorio padre.
La verificación completa se realiza al cargar el índice en cada proceso y queda
cacheada. Montar el bundle y el manifiesto como solo lectura durante las entrevistas;
reiniciar el proceso y verificar de nuevo para cualquier cambio autorizado.

Variables requeridas antes de iniciar la aplicación:

| Variable | Uso |
|---|---|
| `CLOUDRAG_MODE=participant` | Modo predeterminado: solo Evaluation, sin navegación de operador |
| `CLOUDRAG_SESSION_DIR` | Directorio persistente y privado de sesiones, fuera del checkout |
| `CLOUDRAG_ARTIFACT_MANIFEST` | Ruta absoluta al manifiesto verificado |
| `CLOUDRAG_MODEL_DIGEST` | SHA-256 completo, 64 caracteres hexadecimales minúsculos, del modelo Granite provisionado |
| `CLOUDRAG_BUILD_ID` | Commit desplegado (`git rev-parse HEAD`) para la trazabilidad |
| `OLLAMA_HOST` | Endpoint privado del servidor Ollama |

Obtener y comprobar el digest contra el modelo provisionado requiere la validación
operativa autorizada. La app consulta la identidad en Ollama antes de cada generación
y rechaza un cambio de digest. Mantener las variables offline del bloque anterior.

La aplicación utiliza Granite `granite4.1:8b`, temperatura 0, seed 42, caché desactivada,
salida máxima 1024 tokens y contexto 4096. Hybrid usa `SURVEY_DEPLOY`; los controles
conservan su recuperación lexical/dense. Las diferencias con exp19b están declaradas
en [APP_VS_EXPERIMENTO.md](APP_VS_EXPERIMENTO.md): no es un replay del experimento.
El cliente tiene un intento y timeout HTTP de lectura de 60 s/conexión de 5 s. **Ese
timeout no limita el tiempo total de retrieval + generación + NLI ni demuestra p95.**

## Operación de una entrevista a la vez

Usar un único despliegue con almacenamiento local persistente compartido por sus
procesos. Los locks de archivo no constituyen coordinación entre máquinas con discos
independientes. Publicar únicamente Streamlit detrás de HTTPS; mantener Ollama y el
directorio de sesiones privados. `development` habilita Chat y herramientas de operador
y debe ejecutarse solo en un entorno privado separado.

El operador emite una invitación desde la misma configuración de almacenamiento:

```powershell
& .venv-app/Scripts/python.exe scripts/manage_interviews.py invite P01
& .venv-app/Scripts/python.exe -m streamlit run src/ui/app.py
```

El token se imprime una vez; entregarlo por el canal privado acordado. En disco solo
se guarda su hash. IDs válidos: `P` y 2–6 dígitos. No reutilizar participantes de datos
históricos. Una segunda entrevista no puede comenzar mientras otra siga activa.
El mismo token permite retomar la respuesta guardada sin regenerarla; una sesión
completa no se reinicia ni sobrescribe. El orden técnico de sistemas no aparece al participante.

Las vistas viven en `src/ui/views/`, fuera del descubrimiento automático de `pages/`.
La entrada registra una sola página con `st.navigation(..., position="hidden")` en
modo participante, antes y después del login. Las URL antiguas de operador no abren
esas vistas: Streamlit redirige a la entrada y puede mostrar «Page not found».

Una consulta interrumpida queda pendiente. El participante puede recuperar su estado
cuando el lock confirme que ya no hay inferencia activa; un fallo técnico permite
reintentar, nunca calificar una respuesta vacía o una verificación degradada.
Para abandonar definitivamente una sesión, el operador consulta su UUID en el
almacenamiento y ejecuta `scripts/manage_interviews.py abandon UUID`. Esto revoca el
token y libera la admisión sin borrar registros; falla si sigue habiendo inferencia.
Un token perdido no puede recuperarse desde su hash: coordinar la recuperación con el
operador y no crear una segunda observación con el mismo ID.

## Registros y análisis

Cada UUID tiene checkpoint con revisión y escrituras atómicas. Se guarda cada intento
antes de generar: participante, consulta, sistema, configuración, timestamps y estado.
Al terminar se persisten respuesta exacta, fuentes, chunks, verificación, digest del
modelo, hash del manifiesto y duración. La UI ofrece las fuentes en un desplegable;
guardar esa lista no demuestra que el participante haya abierto o leído el desplegable.
`shown_at` registra la presentación desde el servidor, no una confirmación de render
del navegador. Las prácticas mantienen solo progreso, tal como indica la UI existente.

Una calificación exige una respuesta exitosa guardada. El checkpoint rechaza escrituras
de una pestaña desactualizada. La exportación marca `complete` solo después de guardar
todos sus archivos; un error conserva el estado pendiente y permite reintentar.
`full_session.json` incluye intentos y ratings enlazados; el analizador solo acepta
exportaciones v2 cuya revisión coincide con un checkpoint completo. No editar los
archivos manualmente durante una entrevista. Respaldar el directorio privado completo.

Las sesiones históricas se leen para análisis, sin migrarlas ni inventar respuestas
ausentes. `scripts/analyze_user_sessions.py` empareja por participante, rechaza IDs
duplicados, conserva SUS=0 y declara participantes excluidos. El contraste principal
de utilidad usa Wilcoxon y BH para la familia de tres comparaciones. Diferencias
constantes no nulas dejan d_z indefinido; no se reportan como efecto cero.
Las duraciones incluyen conteo observado y p50/p95; un dato ausente no se convierte
en latencia cero. El resumen de tiempos corresponde a consultas calificadas: revisar
además `attempts` para fallos, reintentos y desconexiones antes de evaluar la latencia.
Las salidas del analizador van a `output/interviews/` o a `CLOUDRAG_ANALYSIS_DIR` si se
configura una ruta privada distinta, sin sobrescribir las tablas históricas de `output/`.

Los contratos de retrieval rechazan IDs repetidos y k inválido. Se preserva la
convención histórica de precisión sobre resultados devueltos (`min(k,n)`), explicitada
en su docstring; no se recalculan resultados archivados. La corrección estadística de
entrevistas vive en su analizador. El módulo estadístico genérico ahora rechaza con
`ValueError` el d_z indefinido de diferencias constantes no nulas; los casos definidos
conservan sus valores y no se modifica ningún resultado histórico archivado.

## Evidencia de validación y compuerta pendiente

Baseline en `main` 670f8e5: 337 pasan, 4 omitidas, 5 excluidas. El primer pase del nuevo
entorno limpio obtuvo 364 pasan, 4 omitidas, 5 excluidas. El pase final del 2026-09-05
obtuvo inicialmente **379 pasan, 4 omitidas, 5 excluidas** en 15,59 s (Python 3.14.3, Windows;
3 advertencias de deprecación de FAISS/SWIG). Las cuatro omisiones requieren
chunk map/artefactos exp17 y snapshots BGE/MiniLM ausentes en el worktree; no son pruebas
de modelos ejecutadas. Los casos nuevos incluyen AppTest real de Streamlit con dobles
de pipeline: tres prácticas, treinta consultas, recarga antes de calificar, dos
descansos, SUS y exportación. También cubren admisión, conflictos, corrupción,
interrupción, exportación parcial, fallos de NLI, integridad y denominadores estadísticos.
Ruff pasa en todos los archivos Python modificados, `git diff --check` está limpio,
los 99 paquetes del entorno son compatibles y el escaneo de secretos con baseline no
presenta hallazgos nuevos (6 exclusiones de evidencia firmada). No hay cambios en
`experiments/results/`, `data/`, `paper/` ni `output/` respecto de la base del worktree.
CI se verificó estáticamente y con su suite local; no se afirma una ejecución remota.

Auditoría posterior a la cohorte 616.64 y sus herramientas de diagnóstico:
**411 pasan, 5 excluidas**, 3 advertencias SWIG, 55,92 s en el cierre del 2026-09-07;
Ruff, diff check y escaneo
con baseline aprobados (17 exclusiones: 11 corpus y 6 evidencia firmada).
El bundle provisionado permite ejecutar los casos antes omitidos. No se mezclan
versiones ni resultados de tests históricos con la medición actual.

La ejecución local de modelos está autorizada. El despliegue real en nube no lo
está. Para emitir GO deben quedar verificados los cuatro puntos siguientes:

1. Provisionar y verificar el bundle real, commit, digest y versiones de servidor.
2. Ejecutar una sesión completa con la app del despliegue objetivo; comprobar acceso,
   recuperación tras desconexión, fuentes y reconstrucción de la sesión exportada.
3. Medir respuesta completa (incluye NLI) en los tres sistemas, con caché desactivada,
   cargas frías/calientes y pausas; reportar p50/p95, fallos y hardware. Exigir p95 ≤60 s.
4. Confirmar que errores técnicos no entran como valoraciones y que el almacenamiento
   sobrevive reinicios. Emitir GO solo con esa evidencia; no inferirlo de los mocks.

### Registro histórico del piloto pendiente (2026-09-07)

**DECLARADO por el usuario:** la cohorte histórica 616.64 incluyó uso con batería y
aperturas de Brave. Sus resultados se conservan, pero no acreditan la capacidad
local bajo condiciones controladas. El estado continúa **NO-GO**, con P900 sin
cierre integral y Fase B bloqueada. Véase la reanudación y evidencia externa en
`docs/OPERATIONAL_GATE_2026-09-05.md`.

El ejecutor permite iniciar una cohorte independiente con
`scripts/measure_interview_gate.py init --output RUTA_EXTERNA_NUEVA --systems hybrid --controlled`.
Requiere primero configurar las variables de esta guía y verificar bundle, digest,
servidor y commit. El manifiesto fija 40 posiciones (20 frías/20 calientes);
`plan --output RUTA` es de solo lectura y `run --output RUTA` exige admisión observada
de 60 s en AC, Equilibrado con overlay efectivo Mejor rendimiento, CPU/GPU medias
<10 %, sin procesos de la lista de exclusión implementada (navegadores comunes,
Epic, Steam y nombres con `overlay`). Esa lista no detecta toda carga posible.
El comando directo no cambia ajustes ni cierra procesos.
El proceso frío descarga Granite y conserva la caché de archivos del sistema;
el caliente mantiene un worker y separa un calentamiento exitoso de las 20 posiciones.

Todo fallo o aborto consume su posición sin reemplazo. Una condición inválida
conserva respuesta, error y duración, excluye esa duración de percentiles y pausa
la cohorte; revisar la evidencia antes de cualquier reanudación. La observación
incluye logs incrementales de recursos y traces HTTP, sin cambiar el timeout de
la app. El piloto híbrido no autoriza por sí solo entrevistas ni preparación de
nube: faltan los controles de los otros sistemas y el cierre real de P900.

VERIFICADO: instrumentación en `7b48d37`, 425 tests pasan/5 excluidos, Ruff/diff/secretos
aprobados. La primera admisión controlada falló el 2026-09-07 a las 20:14 UTC:
CPU media 3,05 %, GPU 15,08 %; Edge, Epic y overlays seguían activos. No hubo
inferencias nuevas. Evidencia: `C:/CloudRAG/controlled-pilot-20260907T190630Z/admission-01/`.
Cerrar esas cargas y mantener AC/Mejor rendimiento antes de pedir otra admisión;
no reintentar automáticamente ni interpretar esta observación de reposo como latencia RAG.

### Contraste: método y registro histórico del 2026-09-07

`scripts/contrast_interview_observer.py --output RUTA_EXTERNA_NUEVA` compara diez
pares de trabajo sintético idéntico con/sin observador, alternando el orden. Se
exige límite superior unilateral bootstrap del 95 % <=5 % de aumento mediano,
con 10.000 remuestreos y seed 42. No se descuentan tiempos del RAG. Fallos, pares
incompletos o condiciones inválidas impiden aprobar; la evidencia parcial se conserva.
Cada ejecución requiere una ruta nueva y no modifica la app ni ejecuta modelos.

VERIFICADO: implementación `cf0328d`, 436 tests pasan/5 excluidos y checks aprobados.
El prechequeo del 2026-09-07 a las 20:44 UTC se bloqueó por NVIDIA Overlay todavía
activo, antes del primer par. Se detectó además actividad de un motor GPU asociada
a AnyDesk; falta confirmar si el usuario puede cerrarlo sin perder acceso remoto.
Evidencia: `C:/CloudRAG/clean-pilot-20260907T203624Z/observer-contrast-01/` y el informe
operativo. En ese prechequeo no se ejecutaron ventanas de admisión ni consultas RAG.

### Evidencia vigente del 2026-09-09

VERIFICADO: `C:/CloudRAG/managed-pilot-20260908T012621Z/window-real-05/` contiene
manifiesto verificado, dos admisiones aprobadas, contraste de diez pares aprobado,
40 intentos completos y restauración de procesos. Build medido `a761fa2`.
`memory-analysis-01.json` de su directorio padre separa carga y chat, comprueba
hashes y excluye el calentamiento. La suite posterior al analizador (`f4a60f2`)
obtuvo 476 pasan/5 excluidas. La app bajo prueba no cambió durante el análisis.

| Compuerta | Estado |
|---|---|
| Bundle y modelos reales | ✅ VERIFICADO |
| Sesión real completa | ❌ P900 revisión 49, sin SUS/exportación |
| p95 <=60 s, tres sistemas frío/caliente | ❌ Híbrido frío 83,24 s; caliente 57,62 s; faltan controles limpios |
| Resiliencia integral | ⚠️ Persistencia y errores sin rating verificados; cierre/exportación pendientes |

El corte autorizado de AnyDesk, su watchdog y las restauraciones están documentados
en [MANAGED_GATE_WINDOW.md](MANAGED_GATE_WINDOW.md). No repetir esas operaciones
sin una ventana autorizada; no hace falta repetir las pruebas de desconexión y
reinicio ya conservadas. Las cohortes históricas permanecen separadas.

El diagnóstico no acredita 32 GB como solución del p95 ni permite cuantificar una
mejora por ampliación: [nota de RAM](RAM_INTERVIEW_DIAGNOSTIC.md). Offload parcial
está observado; causa exclusivamente de hardware no demostrada. La propuesta de
lectura 180 s con reloj y avisos al participante sigue **sin implementar**, pendiente
de aprobación específica; el criterio p95 no cambia. No hay artefactos de nube
nuevos ni despliegue autorizado. Véase el cierre vigente del registro operativo
para timestamps, hashes, límites de causalidad y el nuevo fallo UI de 112,00 s.
