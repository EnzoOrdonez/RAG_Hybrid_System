# Anexo de entorno: imagen con observación de residencia corregida

Autorización: Enzo, iteración 3, cláusulas 21, 38 y 42. Este anexo se commitea
antes de iniciar la colección corregida. Complementa el anexo del 2026-10-03 y
[la corrección prospectiva](STUDY_GATE_OBSERVATION_CORRECTION_2026-10-04.md).

## Entorno de la nueva colección

Proyecto `pure-loop-474323-a8`, VM original `cloudrag-study-l4-20261002`, región
`us-central1`, zona `us-central1-a`, `g2-standard-4` estándar no Spot con una L4.
Se mantienen Ubuntu 24.04, disco de arranque persistente de 100 GiB retenido,
protección de borrado, bucket privado regional con acceso uniforme e IP efímera.
No hay reubicación, segunda GPU, instantánea o balanceador nuevo.

La imagen candidata validada tiene etiqueta `cloudrag-study:i3-build-03`.
Su ID y commit completo se leen de `cloud-i3-build-03/image-id.json`, dentro del
paquete externo de la iteración 3; no se introducen hashes manuales como prueba.
El contexto procede de la fuente limpia y del vendor privado verificado, con
todos los pins conservados. Los recibos de build y descarga verifican ambas
suites, `pip check` y durabilidad POSIX sobre el disco persistente de la VM.

La imagen cambia el reloj del guard de residencia y sus regresiones. No cambia
la app, sus recetas, generación, instrumentos, pesos o índices. NumPy conserva
`NPY_DISABLE_CPU_FEATURES=X86_V4`, auxiliares CPU y Ollama en la L4. Continúan
las diferencias Linux/Windows y hardware frente a exp12 documentadas en el
anexo anterior; no se presume equivalencia general de salidas.

## Fuente de aplicación y documentación de control

Este documento se incorpora después de construir la imagen. Por ello se
distinguen explícitamente el commit **dentro de la imagen** y el commit posterior
que añade exclusivamente este anexo al checkout. No se declara que sean iguales
ni se modifica el recibo del build para hacerlos coincidir.

El operador exige checkout limpio, imagen/descarga/suites verificadas y que la
diferencia entre la fuente de la imagen y el checkout de control sea únicamente
este archivo. Una diferencia de código, test, receta, dependencia u otro archivo
rechaza el arranque. El preflight y runner consumen la identidad efectiva de la
imagen, no el commit del documento. El reporte registra ambos commits mediante
recibos generados por Git; el código ejecutado en UI y runner es el mismo.

## Identidad y medición

Tras encender, el contenedor genera un inventario externo nuevo en
`/srv/cloudrag/iteration3/deployment/<ejecucion>/environment_identity.json`.
Imagen, commit de aplicación, sello, recetas, paquetes, driver/GPU, Ollama,
digest completo, pesos e índices se verifican contra el entorno vivo. Los
settings, entrada, preflight, runner y reporte referencian ese único inventario;
el paquete recibe su copia por generación y SHA-256. La URL/certificado se
regeneran y verifican con la IP efímera, sin costo HTTPS en reposo.

La cohorte anterior sigue terminal y excluida. Se admiten únicamente las dos
ventanas nuevas de la colección corregida: 60 intentos cada una, inventario
idéntico, cero fallos/inválidos y p95 lineal agregado <=60 segundos por condición.
Frontera, tareas, supervisores y presupuestos no cambian. No hay participantes;
ningún resultado elimina el requisito de aprobación ética ni habilita R2.

## UX

El anexo no cambia pantallas, textos, respuestas o instrumentos de la app.
