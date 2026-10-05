# Operador de iteración 4 — borrador sujeto a ensayo literal

Este documento no acredita todavía un despliegue apto. Los comandos requieren
la instalación nueva sellada en `C:/CloudRAG/operator-iteration4`, con su
`installation.json`, imagen preparada y recibos. No se usa el operador anterior.
El ensayo literal, la aceptación del estímulo y la compuerta nueva están pendientes.

## Preparación de una sesión, al menos tres días antes

Abre PowerShell sin administrador. No uses `Start-Transcript`, redirecciones ni
capturas al invitar: el token se imprime una vez y no se guarda. La lista que
vincula nombres y códigos se custodia por separado, fuera de Google Cloud y del
operador. No ingreses nombres en ningún comando.

Para `study`, Enzo guarda a mano `ethics/ethics_approval.pdf` y
`ethics/ethics_approval.json` en la instalación. El JSON tiene exactamente
`committee`, `approval_code`, `date` (AAAA-MM-DD), `pdf_sha256` (64 dígitos
hexadecimales) y `approved_b4_version`. El operador compara el PDF real con el
hash, rechaza una fecha futura y no crea ese registro.

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 status
& C:/CloudRAG/operator-iteration4/operator.ps1 ip-reserve
$firstSession = Read-Host 'Fecha de la primera sesión (AAAA-MM-DD), al menos 3 días después de hoy'
& C:/CloudRAG/operator-iteration4/operator.ps1 tls-prepare --first-session $firstSession
```

`tls-prepare` enciende con propósito técnico, verifica el certificado público y
detiene la VM y prepara una instantánea regional ligada a esa imagen, IP y
certificado. Antes de copiar el disco, exige que todos los periodos de iteración
4 y todas las instancias de recuperación estén vacíos: descarga y purga los
datos del periodo anterior antes de esta preparación. No crea una instantánea
con datos de participantes. La IP sigue asociada y el certificado privado queda en el disco
persistente, fuera del repositorio y del paquete de auditoría. Caddy lo reutiliza
y renueva. Si se libera la IP, una reserva posterior puede devolver otra: repite
la preparación con anticipación. No aceptes una excepción de certificado.

## Día de sesión: encender 60 minutos antes

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 start --purpose study
& C:/CloudRAG/operator-iteration4/operator.ps1 preflight
```

Si el primer preflight informa que el invitado aún no está READY, espera y
repítelo dentro de 15 minutos desde `start`; no repitas `start`. Se exige READY
en 15 minutos, TLS, identidad, IAM, metadatos inaccesibles y respaldo despejado.
Cada invitación vuelve a comprobarlo. No se admite una sesión con menos de
70 minutos hasta el primero de los dos límites independientes de apagado.

Si preflight informa `BOOTSTRAP_FAILED` u otro rechazo definitivo, no sigas
esperando ni repitas start. Conserva los recibos técnicos de ese arranque:

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 diagnostics
& C:/CloudRAG/operator-iteration4/operator.ps1 stop
```

`diagnostics` guarda solo los recibos técnicos permitidos, incluidos los
comandos de arranque y sus códigos de salida. No copia sesiones, logs de
acceso, claves de certificado ni el árbol del despliegue. Revisa el primer
comando fallido en la ruta que devuelve el recibo antes de corregir y reiniciar.

```powershell
$participantCode = Read-Host 'Código congelado de participante, por ejemplo P01'
& C:/CloudRAG/operator-iteration4/operator.ps1 invite $participantCode
```

Entrega al participante la URL de preflight y el token al iniciar. La invitación
caduca a las 24 horas o al completar la sesión. Se puede revocar sin borrar datos:

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 revoke $participantCode
```

## Falta de capacidad

Al cerrar cada jornada, después de completar y respaldar todas sus sesiones,
archiva las copias locales del disco. GCS sigue guardando sus respaldos por
código durante el periodo; también quedan descargas privadas verificadas en
el equipo del investigador. Una sesión abierta o un respaldo distinto bloquea
el archivo. En `study`, confirma escribiendo `ARCHIVAR`. El comando cierra la
app; ejecuta `stop` después, antes de dejar el servidor para el día siguiente.

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 archive-local
& C:/CloudRAG/operator-iteration4/operator.ps1 archive-local --execute
& C:/CloudRAG/operator-iteration4/operator.ps1 stop
```

Así el disco de origen queda vacío y conciliado antes de un posible stockout
en el próximo arranque. El operador consulta GCS antes de emitir una invitación
y rechaza un código ya respaldado aunque su copia de disco esté archivada.

Ante `ZONE_RESOURCE_POOL_EXHAUSTED`, conserva el error. Una conmutación exige que
el operador haya comprobado que no quedan sesiones, invitaciones ni copias
pendientes en el disco activo. La política conservadora impide conmutar durante
una sesión o abandonar datos sin recuperarlos. Nunca enciendas dos VM con GPU.

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 failover --zone us-central1-b
& C:/CloudRAG/operator-iteration4/operator.ps1 start --purpose study
& C:/CloudRAG/operator-iteration4/operator.ps1 preflight
```

Si tampoco hay capacidad, prueba `us-central1-c` con el mismo comando de
`failover`. Si ninguna zona admite la VM, no cambies región ni reintentes a
ciegas. Texto para Enzo: «El servidor de la sesión no está disponible. Te
propondré otra fecha; hoy no realizaremos la sesión ni recogeremos respuestas».
Este aviso lo comunica Enzo; el operador nunca contacta participantes.

Para volver a la zona original, con el mismo requisito de conciliación:

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 failback
& C:/CloudRAG/operator-iteration4/operator.ps1 start --purpose study
& C:/CloudRAG/operator-iteration4/operator.ps1 preflight
```

## Retiro y cierre del periodo

Los comandos de borrado cierran la app primero y dejan la admisión bloqueada.
Sin `--execute` solo inventarían. La ejecución descarga todas las generaciones
y las copias de disco al almacenamiento privado local, verifica SHA-256, borra
el ámbito y verifica listados normales, con versiones y soft-deleted vacíos.
Las descargas privadas siguen bajo custodia del investigador: retirar datos del
servidor no equivale a autorizar conservarlos para análisis tras un retiro.
Enzo debe resolver también esas copias privadas y su lista separada conforme a
la aprobación ética, y registrar esa decisión fuera del servidor.

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 withdraw $participantCode
& C:/CloudRAG/operator-iteration4/operator.ps1 withdraw $participantCode --execute
& C:/CloudRAG/operator-iteration4/operator.ps1 export-anonymized
& C:/CloudRAG/operator-iteration4/operator.ps1 purge-study
& C:/CloudRAG/operator-iteration4/operator.ps1 purge-study --execute
```

En `study`, `withdraw --execute` exige escribir el código y `purge-study
--execute` exige `PURGAR` en consola. En esta iteración solo se prueban datos
sintéticos propios. La exportación es seudonimizada por código: las consultas
libres, C4 y razones de cegamiento requieren revisión manual. No la publiques
automáticamente. El recibo indica el directorio privado de descarga.

```powershell
& C:/CloudRAG/operator-iteration4/operator.ps1 stop
& C:/CloudRAG/operator-iteration4/operator.ps1 status
& C:/CloudRAG/operator-iteration4/operator.ps1 ip-release
```

Exige `TERMINATED_VERIFIED` y conserva disco, instantánea y buckets. La IP se
libera al cerrar si no hay sesión agendada. No borres recursos originales.
Un GO técnico no autoriza reclutar: requiere aprobación ética y B.4 aprobada.

## Recuperación: información tomada del recibo, no inventada

La restauración crea una instancia nueva de la app, en un almacén separado del
original y sin admisión pública, con el mismo propósito, periodo, imagen y
protocolo. No exige vaciar el almacén original. Toma ID y generaciones de los objetos del recibo de
respaldo. La invitación restaurada queda revocada; se exporta de nuevo y se
comprueba que el SHA-256 sea idéntico. Repetir el mismo respaldo verifica la
copia anterior; una copia alterada se rechaza. Retiro y purga inventarían y
borran también estos almacenes de recuperación. No se sobrescribe una sesión existente.

```powershell
$sessionId = Read-Host 'ID de sesión del recibo de respaldo (32 caracteres hexadecimales)'
$fullGeneration = Read-Host 'Generación de full_session.json del recibo'
$manifestGeneration = Read-Host 'Generación de export_manifest.json del recibo'
& C:/CloudRAG/operator-iteration4/operator.ps1 restore $participantCode --session-id $sessionId --full-generation $fullGeneration --manifest-generation $manifestGeneration
```

Pendiente antes de dar este runbook por probado: instalar la imagen final y
comprobar todos los comandos, rutas y mensajes con recibos reales.
