# Orden de regiones para la capacidad L4

Primero se prueban a, b y c en us-central1, en hasta tres rondas separadas por
45 minutos o más. Una entrada de `accelerator-types list` demuestra que existe
el tipo de acelerador en una zona; no demuestra stock. No se crea capacidad
reservada ni una segunda GPU encendida.

Si no hay capacidad, el orden de las otras regiones estadounidenses se obtiene
desde el equipo de Enzo con cinco GET secuenciales por región a `/api/ping` de
los endpoints publicados por [GCPing](https://github.com/GoogleCloudPlatform/gcping).
GCPing no es un producto oficial de Google. Su fuente define GET hasta las
cabeceras HTTP200. La medición incluye DNS, TCP y TLS, usa certificado validado
y timeout10s; conserva las cinco posiciones sin descartar el primer acceso.
Se ordena por la mediana y, ante empate, por el nombre de región. Un fallo queda
registrado por clase, sin dirección del cliente ni texto de la excepción; una
región con fallos queda pendiente sin atribuirle una latencia inventada.

La prueba no mide el RAG ni concede GO. Tras cambiar región se exige subred,
IP y certificado propios, identidad nueva y anexo publicado antes de medir.
El bucket de sesiones sigue en us-central1. La instantánea regional restaurada
en otra región de Norteamérica incurre en transferencia de USD0,02/GiB según
[el precio de instantáneas](https://cloud.google.com/compute/disks-image-pricing);
el ledger debe reservar el límite de datos antes de crear el disco.
