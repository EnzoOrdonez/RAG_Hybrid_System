# Preparación técnica de Evaluation Mode

**NO-GO para entrevistas.** La cohorte independiente terminada el 2026-09-06 sobre
driver 616.64 completó 120 intentos: 105 respuestas completas, 15 timeouts y cero
abortos; las seis condiciones incumplen p95 <=60 s. La sesión técnica P900 permanece
incompleta en la consulta 8, conservando sus dos timeouts sin permitir valorarlos.
El diagnóstico separado confirmó `httpx.ReadTimeout`: elevar a 180 s permitió el
probe frío, pero el caliente volvió a agotar 180 s. No es una corrección validada;
el timeout de la app sigue en 60 s. La preparación de nube sigue bloqueada:
todavía no se ha demostrado que la latencia sea el único impedimento y su causa sea
exclusivamente capacidad de hardware. El registro de ejecución y sus mediciones está en
[OPERATIONAL_GATE_2026-09-05.md](OPERATIONAL_GATE_2026-09-05.md).
No se ha desplegado en nube. La evidencia experimental congelada, el corpus, el gold
y el trabajo académico no forman parte de los cambios.

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
