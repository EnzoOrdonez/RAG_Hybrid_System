# Cierre de la iteración 2 (2026-10-03)

La admisión de participantes sigue bloqueada. No se ejecutó la compuerta L4:
no hay un GO ni un NO-GO medido nuevo. El paquete de auditoría externo es
`C:/CloudRAG/autonomous-run-20261003T111748Z/`; su reporte y ledger contienen
los comandos, sellos y fallos conservados. Los resultados sintéticos no habilitan
admisión ni R2.

## Implementado y comprobado

- La escritura POSIX sincroniza el archivo, reemplaza atómicamente y sincroniza
  el directorio padre. Los errores se propagan. Un error posterior al reemplazo
  no promete restaurar el registro anterior. Windows conserva su comportamiento.
- El inventario del entorno se genera fuera del checkout; entrada del contenedor,
  preflight y runner verifican el mismo inventario sellado. El anexo documenta
  generación y frontera de confianza del recibo de imagen producido por el host.
- Los contratos nativos de Windows tienen marcador explícito. Las pruebas NLI
  usan el snapshot offline presente y fallan si este no carga.
- Última verificación funcional antes del cierre: Windows, 857 aprobadas,
  6 omitidas y 9 subpruebas; filtrada, 852 aprobadas, 6 omitidas,
  5 excluidas y 9 subpruebas. Las omisiones incluyen tres contratos POSIX y
  tres comprobaciones de rechazo de un token sin administrador.
- Un contenedor Linux independiente con la base fijada del Dockerfile aprobó
  las siete pruebas de durabilidad, incluido matar el escritor tras sincronizar
  el archivo y antes del reemplazo. Usa almacenamiento overlay de Docker Desktop;
  no sustituye las suites de la imagen definitiva ni el disco persistente GCP.
- El automatizador compartido usa etiquetas visibles, Tab y espera de botones
  habilitados; comprueba cada radio seleccionado. Pasaron el recorrido sintético
  y una regresión con preparación superior a cinco segundos.
- El segundo smoke local real completó seis tareas, dos consultas libres,
  SUS y Likert por bloque, C1–C4, cegamiento y clasificación v2 en ocho respuestas.
  La exportación y su respaldo en otro disco físico se verificaron por SHA-256.
  El primer smoke real falló por el timeout de espera de preparación; se conserva.

## Bloqueos y límites

El único intento de encender la VM heredada falló con
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`. No comenzó ningún build Linux en
GCP. Se desarmó el nuevo arranque y se verificó `TERMINATED`, protección de
borrado y disco retenido. No se creó otra VM, balanceador, certificado, IP estática
ni regla temporal. HTTPS, smoke remoto, identidad efectiva de imagen y compuerta
quedaron pendientes de ese prerrequisito.

El NLI real no inició. El inventario identificó carga GPU ajena de Google Drive,
Surfshark, Recortes y Docker Desktop. Recortes no aceptó el cierre normal;
forzarlo sin comprobar capturas pendientes arriesga contenido del usuario.
No se cortó AnyDesk ni se cambió energía o servicios para medir. El supervisor
complementario externo aprobó pruebas sintéticas de muerte del controlador y
deadline bajo SYSTEM, sin intervención real. Sus versiones anteriores con
comparación UTC incorrecta se conservan como pruebas insuficientes.

La búsqueda literal única de q180 no encontró evidencia directa bilateral en
los candidatos elegibles del catálogo. Se conservan q180, su reserva y el sello
revisado; no se consultaron respuestas del sistema para elegir tareas.

## Pendientes de Enzo

Antes de medir NLI, guardar y cerrar Recortes y permitir una ventana explícita de
aislamiento y restauración de las aplicaciones identificadas. Antes de entrevistas,
completar Linux, HTTPS público, smoke remoto y compuerta según el preregistro,
además de obtener la aprobación ética. Enzo actualizará 4.8, B.4 y Declaraciones
con proveedor, región, entorno, diferencias frente a exp12, reserva de q180,
enmienda e inventario efectivo. No se editaron esos documentos ni se reclutó.
