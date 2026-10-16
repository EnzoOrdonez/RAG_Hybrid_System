# Cierre de sesiones técnicas y efectos UNKNOWN

El trabajo de compuerta conserva su límite nativo de 9000 s. Después de cerrar
la medición deja hasta 600 s, dentro de ese límite, para descargar la prueba,
restaurar un respaldo representativo y retirar/purgar datos sintéticos. El ACK
de descarga no acredita por sí solo la limpieza. Se exige gate-finalized con
el hash de la prueba conservada; al vencer el límite se apaga y se conserva
todo lo que falte. Una limpieza interrumpida requiere continuación explícita,
sin repetir la cohorte.

Un terminal tiene descarga propia por hash y permanece fuera de los almacenes
que se purgan. La operación technical-close, con la app en mantenimiento,
cierra una sesión activa como interrumpida y abandonada; exporta y respalda
explícitamente todos los registros técnicos por el agente aislado. No modifica
los tiempos ni convierte un intento fallido en válido.

El ciclo comprueba export idéntico tras restauración, nueva app de recuperación,
original intacto e invitación revocada. Prueba retiro del primer código sintético
y purga de todas las copias, exigiendo los cuatro controles de borrado. Las
descargas privadas preceden al borrado. Un error API conserva recibos y datos.

Continuación de limpieza, después de lectura de identidad y READY técnico:
`python -m scripts.study_operator.i7 --root <raiz> --sdk <sdk> --live technical-cycle`.
Esa continuación requiere reserva suficiente y presupuesto; no genera consultas.

Toda creación propia GPU y del disco saneado conserva intención antes del
efecto. La restauración CPU mantiene su propio registro de intenciones.
`reconcile-all` lee listados exitosos de VM, discos, snapshots y regla temporal
y comprueba nombre, zona, marcador, ID si existe y snapshot fuente cuando
corresponde. Un VM propio recuperado debe cumplir sus guardas antes de poder
adoptarse para STOP. No se adopta un recurso por nombre solamente.

Comando:
`python -m scripts.study_operator.i7 --root <raiz> --sdk <sdk> --live reconcile-all`.
El recibo devuelve para cada bloqueo el comando describe exacto con proyecto,
nombre y zona. API fallida, ausencia ambigua o identidad distinta conservan
UNKNOWN y prohíben recrear a ciegas. La conciliación no borra ni arranca recursos.
Las reservas conservan su conciliador específico y autoeliminación nativa.
