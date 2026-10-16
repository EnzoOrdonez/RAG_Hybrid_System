# Preparación del host para la siguiente versión

Estado: propuesta documentada, pendiente de implementación y medición desplegada.
No modifica el startup, la instantánea final03 ni una VM existente. No concede GO.
Fuente: Adendas 3, 4, 5 y 8 de `ANEXO_ENTORNO_I6.md`, anclado en el paquete
`C:/CloudRAG/iteration6-cowork-20261015/`.

## Contrato de arranque

Aplicar en cada arranque destinado a estímulos, smoke o sesiones, antes de abrir
la admisión. Un READY de infraestructura no acredita esta preparación. Registrar
un recibo por boot, sin consultas, respuestas, IP de cliente ni User-Agent.
Ante un control fallido, mantener cerrada la admisión y detener el ensayo.

1. Verificar imagen, modelo, GPU, controlador y configuración sellada. I6 midió
   L4, controlador 580.178.04, Ollama 0.22.1 y Granite 4.1 8B con digest
   `444af1c4b2fedd6b54041aca558e7300b0b3d5c0468c44619126240323ba2852`.
2. Mantener `nvidia-persistenced` con `--persistence-mode` mediante una unidad
   complementaria revisada. Conservar la unidad original y sus hashes. Comprobar
   servicio activo y `nvidia-smi` con `persistence_mode=Enabled` en cada boot.
3. Con la admisión cerrada, leer a `/dev/null` el blob verificado del modelo.
   Comprobar por `mincore` todas sus páginas residentes, sin ejecutar generación
   ni cargar el modelo en Ollama. Registrar bytes, páginas y tiempo de lectura.
   La precarga no elimina del cronómetro ninguna carga hecha durante una consulta.
4. Disponer de una caché JIT CUDA por GPU, controlador, versión de Ollama e imagen.
   Prepararla fuera de los ensayos, registrar procedencia y un manifiesto de cada
   archivo. I6 declaró 16 archivos, unos 41 MB y manifiesto
   `195e8cf7e92d72ce5ad2dffcee6f3132880eda1b7beadbb0a464f68171c5d43a`.
   Verificar el manifiesto completo en origen y destino. Un nombre de directorio
   con el controlador no basta para demostrar compatibilidad.
5. Para la siguiente receta, montar un directorio persistente dedicado de caché
   en Ollama y fijar `CUDA_CACHE_PATH` a ese destino. No mezclarlo con pesos ni
   datos de sesiones. Revisar permisos del usuario efectivo, espacio y montaje
   antes de iniciar el servicio. Invalidar la caché si cambia su identidad;
   regenerarla fuera del ensayo y volver a medir. No aceptar silenciosamente
   una caché ausente, corrupta o parcialmente copiada.
6. Comprobar el marcador del boot actual en STARTING, sequence=1 y `models=[]`.
   Conservar la admisión fría de 60 s, la preparación de 900 s, los límites de
   carga y respuesta, y todas las reglas del monitor. No añadir excepciones de
   CPU, GPU o procesos para hacer pasar la preparación.
7. Vincular el recibo al boot, imagen y hashes de entrada. Solo después continuar
   con la admisión prevista por el protocolo. Si se reinicia Ollama o se recrea
   su contenedor, repetir la comprobación del montaje y el estado inicial.

## Validación previa a sesiones

Probar un reinicio real con caché ausente y corrupta: ambos deben impedir la
admisión. Probar tres reinicios preparados, identidad y certificado persistentes,
primera carga dentro del límite y una sesión sintética completa. Conservar los
intentos fallidos y sus costos. Ejecutar la compuerta pendiente conforme al
preregistro; las latencias de I6 no cubren una caché de páginas o JIT vacía.

La restauración corregida en `session_control` necesita una imagen nueva y una
prueba desplegada con respaldo por generación y SHA-256. No presentar el commit
local como reparación de final03. Revisar el alcance de la aceptación sobre la
nueva identidad antes de conceder GO. No reconstruir imagen en este cierre.

## Operación y reversión

Verificar la reserva por la API antes del arranque y tras su borrado. Registrar
ocupación e intervalos sin VM; una VM detenida no evita el cargo de la reserva.
No registrar «borrada» por ejecutar el comando: exigir éxito y ausencia posterior.
El error de zona de la guardia de I6 queda como incidente, con costo por conciliar.

Ante fallo, detener admisión y preservar recibos. Una reversión del drop-in o del
montaje exige nueva preparación y verificación; no habilita sesiones sobre un
estado distinto del medido. Mantener final03 como evidencia histórica inmutable.
