# Borrado verificable: contrato de la iteración 5

El operador nuevo no interpreta HTTP 400 como listado vacío de objetos
soft-deleted. Antes de inventariar y borrar, lee la política viva del bucket:
retención soft delete cero y versionado desactivado. Un proceso del propietario
consulta todo el historial de Admin Activity desde la creación del bucket.
La VM conserva su SA mínima; no se amplían sus permisos para leer ese historial.

La lectura paginada continúa aun cuando una página no tenga entradas si devuelve
un token siguiente. Rechaza tokens repetidos, límites vencidos, métodos no
reconocidos, creación ausente, mutaciones adicionales y metageneraciones
distintas del ancla de configuración previamente auditada. Esta decisión
conservadora puede bloquear cambios inocuos hasta una nueva auditoría del ancla;
mantiene los datos en lugar de prometer un borrado que no pueda verificarse.
Las páginas completas solo viven en memoria. Se conservan hashes, método, hora
de servidor, insertId y estado técnico; no IP, correo, User-Agent ni contenido
arbitrario del request. La fuente de paginación es la
[API oficial de Cloud Logging](https://docs.cloud.google.com/logging/docs/reference/v2/rest/v2/entries/list).

Con la admisión congelada, cada generación se descarga y se verifica por
SHA-256 antes de eliminar cualquier objeto. Se vuelve a comprobar la política
antes de cada borrado y al terminar. La aceptación exige listado normal vacío,
listado de versiones vacío y limpieza de disco verificada. El recibo incluye
descargas, generaciones y prueba de política; no afirma haber listado
soft-deleted cuando esa API no admite el listado con retención cero.
Una reanudación completa vuelve a comprobar los archivos descargados: un
respaldo local alterado no recibe una nueva acreditación de borrado.

En esta iteración la ejecución se restringe a datos sintéticos propios. El
propósito study continúa requiriendo la aprobación ética creada por Enzo y
confirmación interactiva para purgar o retirar. Las comprobaciones unitarias y
la lectura de política no sustituyen los recibos de purga y retiro reales sobre
esas sesiones sintéticas, pendientes hasta la aceptación del despliegue final.
