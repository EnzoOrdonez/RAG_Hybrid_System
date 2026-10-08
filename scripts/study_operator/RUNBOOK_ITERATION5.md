# Operador CloudRAG de iteración 5

Este runbook está pendiente de ensayo literal completo en la nube. Una prueba
de comandos o un resultado sintético no acredita aceptación del despliegue.
Usa PowerShell sin administrador. No cierres aplicaciones del investigador.
Los operadores 3 y 4 y sus paquetes permanecen congelados.

La instalación es `C:/CloudRAG/operator-iteration5`. El archivo `release.json`
selecciona una versión inmutable; el lanzador comprueba todos sus hashes antes
de ejecutarla. `installation.json` contiene configuración y `active.json`
contiene el estado reanudable. No edites esos archivos para sortear un rechazo.

## Preparación técnica de un primario final

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 --help
& C:/CloudRAG/operator-iteration5/operator.ps1 status
$config = Get-Content -LiteralPath C:/CloudRAG/operator-iteration5/installation.json -Raw | ConvertFrom-Json
& C:/CloudRAG/operator-iteration5/operator.ps1 iap-prepare
& C:/CloudRAG/operator-iteration5/operator.ps1 ip-reserve
& C:/CloudRAG/operator-iteration5/operator.ps1 bootstrap --zone $config.zone
& C:/CloudRAG/operator-iteration5/operator.ps1 preflight
```

`bootstrap` exige una instantánea restaurada y calificada, una tarifa oficial,
una subred revisada y una reserva de IP de su región. Nunca crea una segunda
GPU activa. Si la creación anterior tiene resultado desconocido, conserva sus
recibos y concilia instancias, operaciones y discos antes de otra creación.
Un intento desconocido no es un stockout. No repitas `bootstrap`, cambies la
zona ni borres su intención para adivinar la causa.

`start` y `preflight` son operaciones distintas. Si el invitado aún se prepara,
espera 30 segundos y repite solo `preflight`, hasta 15 minutos desde la petición
de encendido. READY debe verificar imagen, identidad, TLS, IAM, aislamiento de
metadatos y respaldo despejado. Ante un rechazo definitivo:

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 diagnostics
& C:/CloudRAG/operator-iteration5/operator.ps1 stop
& C:/CloudRAG/operator-iteration5/operator.ps1 status
```

Conserva la ruta del recibo y resuelve la primera operación fallida antes de
otro intento. `diagnostics` no descarga sesiones ni logs de acceso. Una VM
detenida y una imagen construida no equivalen a READY.

## IP y certificado, al menos tres días antes

No hay fecha de sesión establecida en esta iteración. Para un ensayo, usa una
fecha ficticia cuatro días después; para una sesión, Enzo usa su fecha real.

```powershell
$firstSession = (Get-Date).ToUniversalTime().Date.AddDays(4).ToString('yyyy-MM-dd')
& C:/CloudRAG/operator-iteration5/operator.ps1 tls-prepare --first-session $firstSession
& C:/CloudRAG/operator-iteration5/operator.ps1 status
```

La IP es estática regional y tiene costo aun con la VM detenida. El certificado
de Let's Encrypt se conserva en el disco privado y Caddy lo reutiliza. No
aceptes excepciones de certificado ni otra cuenta ACME. `tls-prepare` exige
vacíos todos los periodos y recuperaciones antes de copiar el disco: ninguna
instantánea de contingencia debe contener datos de personas. Una IP liberada
puede cambiar al reservarla otra vez; prepara nuevamente TLS con anticipación.

## Ensayo sintético y día de sesión

Enciende 60 minutos antes. Este ensayo utiliza propósito `technical` y código
`P999`. Si ya existe ese código, conserva sus datos y usa otro código sintético
libre; no lo retires para repetir una observación fallida.

```powershell
$participantCode = 'P999'
& C:/CloudRAG/operator-iteration5/operator.ps1 start --purpose technical
& C:/CloudRAG/operator-iteration5/operator.ps1 preflight
```

Solo después de READY, en una consola interactiva SIN transcripción, captura,
redirección ni registro de salida:

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 invite $participantCode --cell 1 --profile without_experience
```

El token aparece una vez y solo su SHA-256 viaja a la VM. Caduca a las 24 horas
o al completar la sesión. Entrega la URL verificada y el token en memoria al
automatizador previamente aprobado. Nunca los incluyas juntos en archivos,
argumentos de proceso, capturas o evidencia. La invitación exige al menos
70 minutos antes del primer límite de apagado del invitado y de la VM.

El smoke debe comprobar familiarización efímera, tres tareas y una consulta
libre por bloque, ocho respuestas v2, cegamiento, SUS, UEQ-S inmediatamente
después del SUS en cada bloque, Likert, comparativas, cierre y exportación.
Verifica el respaldo automático por generación y SHA-256 antes de admitir
otra sesión. Un respaldo pendiente mantiene cerrada la admisión.

Para una sesión con personas, Enzo debe obtener primero el dictamen de la
auditoría independiente y la aprobación ética. Solo él crea en `ethics/` el
PDF `ethics_approval.pdf` y el registro `ethics_approval.json` con `committee`,
`approval_code`, `date`, `pdf_sha256` y `approved_b4_version`. Este agente no
crea ese registro ni invita a personas. Entonces el comando de encendido es:

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 start --purpose study
```

La revocación no borra datos:

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 revoke $participantCode
```

## Recuperación y exportación

Lee la generación del objeto completo y del manifiesto en el recibo de respaldo
del código. Usa los valores devueltos, nunca una generación inventada:

```powershell
$sessionId = Read-Host 'session_id del recibo de respaldo verificado'
$fullGeneration = Read-Host 'Generación del objeto completo del recibo'
$manifestGeneration = Read-Host 'Generación del manifiesto del recibo'
& C:/CloudRAG/operator-iteration5/operator.ps1 restore $participantCode --session-id $sessionId --full-generation $fullGeneration --manifest-generation $manifestGeneration
& C:/CloudRAG/operator-iteration5/operator.ps1 export-anonymized
```

La recuperación crea una instancia nueva de la app y conserva el original.
Comprueba que la exportación recuperada tenga el mismo SHA-256. Las descargas
quedan en `private/` del operador. La exportación por código es seudonimizada:
la consulta libre requiere revisión manual antes de publicar cualquier dato.
Los resultados agregados pueden publicarse solo tras esa revisión de Enzo.

## Retiro y fin del periodo

En esta iteración ejecuta el borrado únicamente sobre datos sintéticos propios.
Cada comando simula por defecto. Revisa el inventario y el recibo de descarga
con SHA-256 antes de ejecutar:

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 withdraw $participantCode
& C:/CloudRAG/operator-iteration5/operator.ps1 withdraw $participantCode --execute
& C:/CloudRAG/operator-iteration5/operator.ps1 purge-study
& C:/CloudRAG/operator-iteration5/operator.ps1 purge-study --execute
```

En `study`, confirma el código o `PURGAR` en consola interactiva. Se exige
política soft delete=0, sin versionado, historial de auditoría sin cambios de
esa política, listado normal vacío y descarga verificada antes del borrado.
No se usa `--soft-deleted`, que GCS rechaza cuando la retención es cero.
Se inventarían todas las copias de disco, recuperaciones y objetos del periodo.
El bucket técnico no recibe sesiones `study`. La lista de nombres y códigos
de personas la custodia Enzo por separado, fuera de la aplicación.

## Contingencia y vuelta

Solo conmuta desde una instantánea vacía preparada por `tls-prepare`. Después
de cada jornada, respalda y archiva las copias locales para preparar el disco:

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 archive-local
& C:/CloudRAG/operator-iteration5/operator.ps1 archive-local --execute
& C:/CloudRAG/operator-iteration5/operator.ps1 stop
```

Una sesión abierta o un respaldo pendiente bloquea el archivo. En `study`,
confirma `ARCHIVAR`. Para el simulacro técnico usa una zona de la misma región
que la instalación revisada; por ejemplo, en us-central1:

```powershell
$alternateZone = 'us-central1-b'
& C:/CloudRAG/operator-iteration5/operator.ps1 failover --zone $alternateZone
& C:/CloudRAG/operator-iteration5/operator.ps1 start --purpose technical
& C:/CloudRAG/operator-iteration5/operator.ps1 preflight
& C:/CloudRAG/operator-iteration5/operator.ps1 stop
& C:/CloudRAG/operator-iteration5/operator.ps1 failback
& C:/CloudRAG/operator-iteration5/operator.ps1 start --purpose technical
& C:/CloudRAG/operator-iteration5/operator.ps1 preflight
& C:/CloudRAG/operator-iteration5/operator.ps1 stop
```

La misma región permite conservar IP, URL y certificado. Una IP estática no
se mueve entre regiones. Otra región de EE. UU. exige instalación regional
revisada, subred, IP y certificado nuevos, anexo e identidad verificados y
smoke propio. El bucket de sesiones permanece en us-central1 y se registra
la transferencia. No cambies a mano el estado ni uses una zona de otra región
en la instalación actual. El GO, cuando se mida, solo cubre la misma imagen,
software y tipo de máquina; no acredita entornos sin identidad y smoke.

Si hay `ZONE_RESOURCE_POOL_EXHAUSTED`, usa la contingencia preparada. Si ninguna
alternativa llega a READY, Enzo comunica: «El servidor no tiene capacidad
disponible. Reprogramaremos la sesión; tus datos y tu participación no se ven
afectados». El agente no envía ese mensaje a ninguna persona.

## Cierre sin sesión agendada

```powershell
& C:/CloudRAG/operator-iteration5/operator.ps1 stop
& C:/CloudRAG/operator-iteration5/operator.ps1 status
& C:/CloudRAG/operator-iteration5/operator.ps1 ip-release
& C:/CloudRAG/operator-iteration5/operator.ps1 iap-release
```

Exige TERMINATED, IP y regla IAP ausentes, discos y evidencia preservados y
ledger actualizado. No borres la VM ni el disco original, los buckets o
evidencia. Conserva los fallos del ensayo y cada recibo: completar una parte
no vuelve aprobado el runbook completo.
