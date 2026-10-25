# Build Linux de la iteración 5

El controlador `scripts.study_operator.cpu_build` utiliza exclusivamente el clon
CPU cuyo ID coincide con el recibo de restauración real del paquete. No genera
respuestas ni decide la aceptación del estímulo o de la compuerta.

Antes de lanzar, exige un checkout limpio publicado, un contexto generado por
`scripts.study_operator.build_context` del mismo commit, tres intentos como
máximo, presupuesto con reserva de cierre y un clon detenido sin IP externa.
El inventario del contexto se llama `build-context-<etiqueta>-inventory.json`;
el nombre `-receipt.json` queda reservado para el registro del comando.

La subred debe tener Private Google Access. El clon recibe la SA dedicada con
scope de almacenamiento y los bindings mínimos existentes: lectura y creación
en el bucket de sesiones y creación en el prefijo técnico autorizado. Esta
comprobación no afirma aislamiento de metadatos de la app: no hay app en ejecución.

El invitado usa el caché Docker verificado, con `--network=none`, `--pull=false`,
`--rm=false` y las dependencias fijadas. Un fallo de caché se conserva como fallo
de build; no se cambian versiones ni se borra el intento. Las suites Linux,
`pip check` y las siete pruebas POSIX usan el disco persistente del clon. Un
recibo PASS descargado exige las siete pruebas sin exclusiones y un montaje
ext4 del disco raíz, además de los recibos de ambas suites con código cero.

El supervisor invitado corta a los 90 minutos; cada comando tiene su límite.
También rigen el STOP nativo de dos horas, el apagado invitado de 110 minutos y
el cierre global independiente del paquete. La respuesta de lanzamiento solo
indica `CPU_BUILD_LAUNCHED_NOT_ACCEPTED`. La colección verifica generación,
tamaño y SHA-256 de cada objeto técnico, y confirma la VM detenida antes de
sellar su prueba. No sube árboles de sesiones ni el árbol completo del despliegue.

Las rutas y los recursos temporales se declaran antes del lanzamiento. Los
contenedores, contextos, imagen y evidencia fallida se conservan hasta el cierre
autorizado. La imagen nueva deberá calificarse posteriormente en la L4 con
identidad viva, congelamiento, aceptación y compuerta propios.
