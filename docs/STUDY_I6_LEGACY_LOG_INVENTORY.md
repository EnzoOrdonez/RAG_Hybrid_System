# Registros Caddy heredados en final03

Inventario reconstruido del recibo `g_privacy_dates.sh` de RUN_LOG.jsonl,
2026-10-15T23:34:53.222089+00:00. Nueve archivos con la IP del investigador,
según el escaneo registrado. No se reproducen direcciones ni contenido.
Los tamaños y fechas son del recibo; los bytes de estos nueve archivos no
están disponibles en esta comprobación local. Sus SHA-256 quedan NO VERIFICADOS.
No se borró ni modificó ningún archivo de la VM o de la instantánea.

| Directorio bajo `/srv/cloudrag/iteration3/runtime/` | Archivo | Bytes |
|---|---|---:|
| `access-20261004T023425Z` | `caddy-public-log.stderr` | 20402 |
| `access-20261004T024929Z` | `caddy-public-log.stderr` | 40505 |
| `access-20261004T033336Z` | `caddy-public-log.stderr` | 48710 |
| `access-20261004T035910Z` | `caddy-public-log.stderr` | 51665 |
| `access-20261004T140448Z` | `caddy-public-log.stderr` | 59579 |
| `access-20261004T143041Z` | `caddy-public-log.stderr` | 74547 |
| `access-20261004T172058Z` | `caddy-public-log.stderr` | 132871 |
| `access-20261004T190612Z` | `caddy-public-log.stderr` | 135826 |
| `access-20261004T211114Z` | `caddy-public-log.stderr` | 142858 |

## Retiro propuesto antes de sesiones

1. Mantener final03 y la evidencia original de I3 inmutables y con acceso
   restringido al investigador y su asesor. No publicarlas ni incorporarlas
   a un paquete de evidencia pública. Registrar quién conserva la copia.
2. En una tarea posterior autorizada, obtener estos nueve archivos de la
   evidencia privada, calcular sus hashes y verificar una copia archivada
   fuera del volumen que atenderá sesiones. Conservar fechas, procedencia
   y manifiesto. Una versión redactada no sustituye esa evidencia original.
3. Crear el siguiente entorno de sesiones desde una receta revisada que
   excluya los archivos heredados. Identificarlo con hashes nuevos, preservar
   el mapeo con I3 y verificar que la exclusión no modifica el RAG ni sus activos.
   No reescribir la instantánea final03 ni cambiar su identidad histórica.
4. Comprobar en ese entorno, con canarios sintéticos y cobertura de todas
   las rutas, journald, serial y respaldos, que no persisten identificadores
   de cliente. Documentar cualquier exclusión por ruta y fecha; excluir todo
   `iteration3` no demuestra ausencia de datos nuevos dentro de ese árbol.
5. Mantener bloqueada la admisión hasta cerrar la validación desplegada
   de privacidad, restauración y la compuerta exigida por el preregistro.

El script final de I6 excluye `iteration3` y `iteration6-cowork-backups`.
Su conteo cero acredita únicamente el ámbito inspeccionado. La cobertura de
consola serial y objetos remotos debe enlazarse con recibos propios; este
inventario no la infiere del escaneo de archivos locales.
