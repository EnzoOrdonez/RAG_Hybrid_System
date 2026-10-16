# Automatizador de navegador I7

El criterio previo está en el paquete privado de A2, CRITERIOS_R1.md. Se exige
recorrido completo de la interfaz real por HTTPS local, sin modificar la app,
con dobles de modelos y respaldo remoto. Se confirman los valores guardados por
la app: consulta libre, SUS, ocho ítems UEQ-S, Likert, comparación y cierre.
Fuera de formularios, los textarea reciben Ctrl+Enter y se espera el rerun.
Dentro de formularios se editan y se desenfocan; solo el botón explícito envía
el formulario. Después se cotejan los valores exactos del checkpoint. El export debe
tener ocho intentos, dos bloques completos y SHA-256 igual al respaldo.
Se rechazan entrada no registrada, export incompleto e identidad de boot distinta.

La prueba local usa Playwright y Chromium sin intervención manual, con certificado
autofirmado solo para localhost. El despliegue exige TLS válido. El código de
participante permanece intacto; el servidor de prueba dobla preparación,
generación y transporte del respaldo. Los resultados son sintéticos.

`requirements-i7-browser.txt` fija versiones y hashes de ruedas oficiales. Es
una dependencia del dueño, separada del lock de la app. `config/i7-browser-pin.json`
fija el árbol instalado del paquete y el ejecutable por plataforma; una
diferencia bloquea el smoke. El recibo guarda hashes de código, pin, navegador,
valores confirmados, export e identidad. Una revisión de código invalida un
recibo local anterior hasta repetir la prueba.

Fuentes de implementación: [acciones de Playwright](https://playwright.dev/python/docs/input)
y [navegadores de Playwright](https://playwright.dev/python/docs/browsers).
Los hashes de dependencias proceden de los metadatos de cada versión en
[PyPI](https://pypi.org/project/playwright/1.58.0/), archivados en el paquete privado.

Ejecutar con el intérprete del estudio y el navegador previamente instalado:
`python -m pytest tests/test_study_i7_browser.py -ra`. La prueba positiva
escribe browser-validation.json en CLOUDRAG_BROWSER_RECEIPT_DIR si se define.
Copiar ese recibo verificado a la raíz privada del operador. El comando smoke
de B llama al mismo automatizador y confirma el backend por el puente IAP.
Los tokens y respuestas sintéticas permanecen en memoria o evidencia privada.

## Instalación reproducible de Chromium 1208

Abrir una consola nueva `powershell.exe -NoLogo -NoProfile`. Desde la raíz del
worktree, con el entorno de la app ya creado en `.venv-app` y Python 3.14.3,
ejecutar el bloque completo. No requiere activar el venv ni heredar variables
de una consola anterior. pytest y cryptography deben estar instalados en ese
entorno de pruebas del estudio. El lock del navegador solo añade dependencias
del dueño; no cambia el lock de la app.

```powershell
$ErrorActionPreference = 'Stop'
Remove-Item Env:PLAYWRIGHT_BROWSERS_PATH,Env:CLOUDRAG_BROWSER_RECEIPT_DIR,Env:PYTHONPATH,Env:PYTHONHOME,Env:PYTEST_ADDOPTS,Env:PYTEST_PLUGINS -ErrorAction SilentlyContinue
$env:PYTHONUTF8 = '1'
$py = (Resolve-Path .venv-app/Scripts/python.exe).Path
$env:PLAYWRIGHT_BROWSERS_PATH = 'C:/CloudRAG/i7-browser-runtime'
$env:CLOUDRAG_BROWSER_RECEIPT_DIR = 'C:/CloudRAG/i7-local-verification-stability'
python -m pip --python $py install --require-hashes -r requirements-i7-browser.txt
if ($LASTEXITCODE -ne 0) { throw 'Browser dependency installation failed' }
& $py -m playwright install --no-shell chromium
if ($LASTEXITCODE -ne 0) { throw 'Chromium installation failed' }
& $py -m scripts.study_operator.i7_browser_preflight
if ($LASTEXITCODE -ne 0) { throw 'Pinned Chromium preflight failed' }
& $py -m pytest tests -k study_i7 -ra
if ($LASTEXITCODE -ne 0) { throw 'I7 local suite failed' }
```

Playwright **1.58.0** queda fijado por versión y hashes en
`requirements-i7-browser.txt`. Su manifiesto `driver/package/browsers.json`
asigna a Chromium la revisión **1208**, versión **145.0.7632.6**. El comando
`install` obtiene esa revisión por el manifiesto de la versión fijada; no se
pasa `chromium@1208`. No actualizar Playwright para buscar otro navegador.
`--no-shell` instala el Chromium completo que el automatizador abre por ruta
explícita, sin sustituirlo por Chrome/Edge del sistema o el headless shell.

Con la variable del ejemplo, el ejecutable Windows es:
`C:/CloudRAG/i7-browser-runtime/chromium-1208/chrome-win64/chrome.exe`.
SHA-256 exigido:
`ff0c99fc5bb2301cad77930aa76e5354cf2a5fefb7efdcf3b3bc10ba0550c015`.
Comprobación manual opcional: `Get-FileHash "$env:PLAYWRIGHT_BROWSERS_PATH/chromium-1208/chrome-win64/chrome.exe" -Algorithm SHA256`.

En Linux, definir `PLAYWRIGHT_BROWSERS_PATH` con una ruta absoluta antes de
instalar y ejecutar; el binario es
`$PLAYWRIGHT_BROWSERS_PATH/chromium-1208/chrome-linux64/chrome`, con SHA-256
`481fea1516a1f2b76454664272f12cd9dd1f20117b21e1f1498e08bc7f872c00`.
El contenedor local de pruebas ya aporta las bibliotecas Linux necesarias.

La variable debe tener el mismo valor durante instalación, tests y todos los
comandos Fase B. No se acepta el caché implícito del perfil del usuario.
`i7_browser_preflight` comprueba revisión, ruta bajo `chromium-1208`, árbol del
paquete, versión y hash completo del ejecutable contra
`config/i7-browser-pin.json`. Solo `PINNED_BROWSER_READY` permite continuar.
Ausencia, cambio de ruta o bytes distintos bloquean con
`PINNED_BROWSER_RUNTIME_REQUIRED`. El check del runbook lo enumera con dueño
`code`; todo comando vivo de `i7_phase_b` lo comprueba antes de crear el cliente
de nube. El preflight lee archivos locales y arranca el driver local, sin
navegar ni crear recursos GCP.

La gestión de rutas y la elección del Chromium completo siguen la
[documentación oficial de Playwright](https://playwright.dev/python/docs/browsers#managing-browser-binaries).
Los hashes de este documento proceden del pin local y los binarios verificados,
no de una tarifa ni de una medición en nube. Guardar stdout/stderr de la consola
nueva junto con el recibo local generado en la ruta explícita.

## Sincronización observable y fallos

La visibilidad de un botón y un checkpoint nuevo no prueban que el frontend
haya terminado el rerun. `StreamlitSync` observa los mensajes protobuf del
WebSocket de Streamlit 1.54.0: una petición `rerun_script` posterior a la acción
y su `script_finished = FINISHED_SUCCESSFULLY`. También exige que el DOM de
`stApp` indique `data-test-script-state=notRunning` y
`data-test-connection-state=CONNECTED`. Una finalización temprana para otro
rerun no satisface esa condición. La conexión WebSocket permanece abierta.

Cada botón se acciona una sola vez. El automatizador espera esas condiciones,
con límite de 25 s en local y 600 s en el recorrido desplegado; la preparación
conserva su límite existente de 900 s. El sondeo de estado no repite acciones,
consultas ni envíos. No se aumentaron tiempos para encubrir la carrera.
Los formularios se editan sin Ctrl+Enter porque ese atajo envía el formulario
completo; se comprueba el estado seleccionado de cada radio y el valor visible
del texto. Las comprobaciones exactas de checkpoint, export e identidad siguen
siendo obligatorias después del único envío explícito.

Las regresiones retrasan la entrega de la finalización al frontend, conservando
el orden de mensajes, y comprueban que no se edite durante `running`. También
rechazan petición, finalización o render ausentes sin repetir la acción.
Las negativas de valor, export e identidad exigen el error correspondiente,
no cualquier excepción. Ver
[interacción con formularios de Streamlit](https://docs.streamlit.io/develop/concepts/architecture/forms).

Los registros de reproducción y validación de estabilidad quedan en el paquete
privado `C:/CloudRAG/iteration7-browser-stability-20261016/`. La admisión requiere
30 corridas consecutivas de las dos pruebas positivas en Windows y 10 en Linux,
con los fallos previos conservados. Cada corrida contiene dos recorridos completos;
no se sustituyen corridas fallidas dentro de una serie. La evidencia local no
mide la probabilidad de fallos de generación o infraestructura en la Fase B.
