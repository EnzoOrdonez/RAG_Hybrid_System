# Automatizador de navegador I7

El criterio previo está en el paquete privado de A2, CRITERIOS_R1.md. Se exige
recorrido completo de la interfaz real por HTTPS local, sin modificar la app,
con dobles de modelos y respaldo remoto. Se confirman los valores guardados por
la app: consulta libre, SUS, ocho ítems UEQ-S, Likert, comparación y cierre.
Los textarea reciben Ctrl+Enter y luego se coteja el checkpoint. El export debe
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
