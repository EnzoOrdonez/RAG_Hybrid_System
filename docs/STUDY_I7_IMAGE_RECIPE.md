# Imagen I7 y restauración desplegada

Estado: receta local, pendiente de construcción y validación en noviembre.
Final03 y las tres instantáneas heredadas se conservan sin cambios.

La fuente de la imagen será un checkout limpio de `fix/interview-readiness`,
con HEAD publicado tras permiso de Enzo. No se descarta el cambio local de
`CLAUDE.md`: se prepara otro checkout limpio para construir. Se mantienen
Dockerfile, pins, modelos, índices, calendario y los 19 módulos congelados.
El contexto incluye `session_control.py` con sellado de la configuración
verificada antes de `StudyStore.check()` y lectura UTF-8 de restauraciones.

El disco de servicio será un clon secundario propio, saneado mientras está
desmontado del servicio. Conserva el sistema operativo y el driver de la fuente.
El constructor exporta las tres imágenes por digest, descarga el archivo privado
con los nueve originales y confirma su SHA-256 antes del saneamiento. Conserva
solo los artefactos enumerados: pesos, índices, código nuevo, protocolo, TLS y
semilla JIT compatible con el driver. Reinicia los árboles de Docker, registros,
temporales y runtime del clon. Las instantáneas y el disco fuente quedan intactos.
El censo con canarios de noviembre debe comprobar la cobertura residual del SO.
La configuración I7 exige `iteration=7`, preparación del host y caché JIT
persistente. Un fallo impide READY. No se declara identidad heredada de I6.

Antes de GPU: calificar la imagen en CPU, comprobar inventario, usuarios,
dependencias, pruebas POSIX y diferencia cero del RAG congelado. Conservar
recibos, imagen exportada, digest y commit; no etiquetar final03 de nuevo.

La prueba desplegada recibe un respaldo sintético cerrado con generaciones y
SHA-256. El comando I7 de restauración llama al puente por IAP: mantenimiento,
`disk_dispatch`, carpeta `recoveries/<hash>/sessions`, contenedor de control
con la misma imagen y red `none`, configuración sellada, restauración y export
idéntico por SHA-256. Las invitaciones restauradas quedan revocadas y el
almacén original queda intacto. Luego se verifican retiro y purga en todas
las copias y generaciones conforme al contrato de borrado.

Prueba local del recorrido: `python -m pytest tests/test_study_i7_restore.py
tests/test_study_session_restore.py tests/test_study_restore_probe.py -q`.
El runbook de Fase B fija los comandos pagados y los criterios de cierre.
