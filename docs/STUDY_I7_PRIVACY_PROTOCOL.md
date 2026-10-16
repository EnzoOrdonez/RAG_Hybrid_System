# Canarios y disco de servicio I7

Estado: código probado localmente. El censo desplegado se ejecuta en Fase B.

El disco de servicio propio conserva SO y driver de un clon secundario,
con saneamiento offline y selección del contenido de servicio por SHA-256.
Se vacían los árboles de runtime, sesiones y respaldos. Antes, los nueve archivos
del inventario I6 se archivan fuera de ese disco mediante `archive_legacy`:
creación exclusiva, hash de cada original y comparación de los bytes dentro del
tar. Las instantáneas originales y los archivos fuente se conservan intactos.

El cliente introduce por entrada privada una IP propia y un User-Agent único.
No aparecen en argv, URL, scripts guardados ni recibos. La IP propia permite
buscar también la dirección real de la conexión; los encabezados reenviados
prueban su eliminación en el proxy. La petición recorre raíz, salud y favicon.
El smoke sintético completo recorre el navegador, la app, el runner y el respaldo.
El mismo automatizador completa la sesión con un UA sintético y la IP canario
en encabezados. Chromium no propaga los encabezados HTTP arbitrarios al
WebSocket: la IP canario viaja también como sufijo del UA. El recibo CDP
confirma ambos en el handshake real, sin persistirlos en el informe. Caddy
suprime ese encabezado completo antes de la app; no se añaden identificadores
a consultas, prompts ni respuestas.

La prueba local de control positivo deja pasar deliberadamente esos encabezados
al doble de generación y de respaldo y comprueba detección en app, runner,
respaldos y bucket. Los dominios proxy, journald y serial son dobles de API
en esa prueba. Los siete controles positivos deben detectarse y la falta de
un dominio produce INCOMPLETE. Esto no acredita cobertura del SO del clon.
En B, el comando privacy ejecuta el navegador, exige ambos canarios en el
WebSocket y luego recorre los siete dominios reales. Exige cero aciertos.

Fuentes: montajes escribibles efectivos y capas superiores de todos los
contenedores retenidos del servicio, logs de Docker si existen, árbol del
operador y periodos, respaldos locales, journald persistente completo, consola
serial desde la creación y todas las generaciones del bucket. La consola
serial y el bucket se leen desde el dueño; no se concede permiso nuevo a la VM.
Los tar se inspeccionan sin extraerlos al disco. Un enlace o archivo ilegible
requiere revisión y produce INCOMPLETE. Un objeto con borrado recuperable
impide PASS hasta comprobar su contenido o la política histórica de retención.

No hay exclusiones predeterminadas. Cada excepción exige dominio, localizador
exacto, SHA-256, motivo y fecha; su contenido debe coincidir. No se aceptan
exclusiones globales de I3 ni de respaldos. El archivo privado histórico está
fuera del disco de servicio. Los resultados públicos muestran conteos y hashes,
sin identificadores ni contenido de archivos.

PASS exige los siete dominios completos y cero aciertos. La detección se prueba
con fugas deliberadas en cada dominio y canarios divididos entre bloques.
El resultado de privacidad no sustituye las cuatro comprobaciones de borrado
ni la validación del retiro en todas las copias del almacén.
