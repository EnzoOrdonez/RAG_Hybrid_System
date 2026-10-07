# Supervisión del operador de la iteración 5

Durante esta iteración, `installation.json` del operador nuevo debe incluir
`audit_run` con la ruta de su paquete activo. El operador registra ahí las
intenciones antes de crear recursos, sus IDs después de la lectura API y la
exposición de costo. No copia contenido de sesiones, códigos, tokens ni datos
personales: conserva campos técnicos y un hash del estado del operador.

El supervisor independiente puede detener una VM creada cuya respuesta se
perdió, comprobando nombre, zona y marcador propios. Un disco queda registrado
antes de solicitar la VM; un fallo de capacidad no lo deja sin propietario.
Los nombres nuevos pertenecen a `cloudrag-i5-` y sus marcadores a `CloudRAG-I5-`.
Las instalaciones 3 y 4 permanecen congeladas.

La admisión suma el costo del paquete, otras reservas y la exposición del
operador una sola vez. Rechaza nuevas operaciones pagadas tras el cierre,
al llegar a la reserva de cierre o al alcanzar el corte. Los comandos de
apagado siguen pudiendo guardar su recibo en el operador. Un paquete CLOSED o
SEALED no se modifica, ni se crea un lock dentro de él.

Para las sesiones futuras, después del cierre y sello de esta iteración, la
instalación humana debe dejar de apuntar a ese paquete mediante la eliminación
de `audit_run` de su configuración mutable. El operador conserva su propio
ledger y los límites nativos de STOP. Esto no autoriza una sesión: siguen
exigiéndose aceptación medida, auditoría independiente y el registro ético
creado por Enzo. El runbook final debe documentar y probar esta transición.

`ip-release` solo declara liberación después de un listado API vacío. La
confirmación del comando delete por sí sola conserva el ID y las reservas,
permitiendo reconciliar una respuesta perdida.
