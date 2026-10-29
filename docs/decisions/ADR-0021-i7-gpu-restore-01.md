# ADR-0021: Incidente gpu-restore-01

[DECLARADO] Aceptado el 2026-10-29 para corrección y auditoría local sobre
468eb255dbe8abb5086308b1b8d0043e94619bcb. Tarea F-020. No autoriza despliegue,
publicación, instalación de componentes ni gasto. Complementa
[ADR-0020](ADR-0020-i7-final-freeze.md).

## Incidente y estado corregido

[VERIFICADO por registro histórico] El intento gpu-restore-01 falló porque el SDK
no tenía beta instalado; reservations create GA no ofrecía delete-at-time.
El recibo 0003 y una lista posterior vacía fundamentaron que no se creó reserva.
Esto no demuestra una reserva facturada por el defecto temporal: ese defecto se
reprodujo localmente con nube falsa.

Fuente: `C:/CloudRAG/i7-preparacion-20261028-468eb25/incident-gpu-restore-01/STATE_CORRECTION_RECORD.json`.
SHA-256 del registro:
`b2b95eabc8a37f667c5b0e1c1c46a920719389d7065c977823c71e589c5a08eb`.

[DECLARADO en el registro] Enzo autorizó «Corrección declarada». Se retiró el
intent UNKNOWN, open_exposure_usd pasó de 8.73198706 a 0 y las horas reservadas
de 12 a 0; spent_usd y budget_started_utc se conservaron. Identidades del estado:

- Antes: `4161b1079824ce75d50409565302471663b80f21927b089822e2ce7495d17a02`.
- Después: `c8fa3e60828bbbb485f2aa3d12a62da45b44925a241b8c5b863fdba6bcf6b952`.

[VERIFICADO] El hash anterior coincide con active.before.json archivado junto al
registro. El posterior se cita del registro: no se dispone allí de una copia
after independiente y no se compara con un estado operativo posterior.
En esta tarea no se corrige ni se reescribe ningún archivo de estado privado.

[VERIFICADO por registro histórico] Se instaló beta localmente con exit 0.
El launcher archivado launch_whole_second.py inyectaba un reloj de segundos
completos sin cambiar 468eb25. No se ejecutó ese launcher en esta corrección.

## Decisión

[CORREGIDO por REPORTE_REVISION3] La reproducción original comparaba .968532
local con .968 generado por el fake truncador; no era una lectura real de GCE.
El SDK 587.0.0 usa FormatDateTime %3f: redondea half-up (.968532 a .969),
manteniendo .999 si el redondeo desborda. El fake se corrige en esta ronda.

[DECLARADO] Las nuevas reservas usan UTC a segundos completos, preservando la
precisión de requested_utc y exigiendo igualdad exacta del vencimiento. Para
intents heredados, la verificación compartida con reconcile acepta únicamente
el esperado truncado o el redondeado half-up del SDK, sin acarreo de .999 al
segundo siguiente. No admite un intervalo de tolerancia. Para .968532 se aceptan
.968 y .969 por compatibilidad explícita; fuera de esos candidatos se bloquea.
Un cambio de ±1 ms en reservas nuevas o quitar la zona sigue bloqueando.
Recuperar una reserva existente no ejecuta otro CREATE ni duplica la exposición.

[DECLARADO] Un preflight de components list --only-local-state exige beta antes
de reserve/admit_paid y del despacho de entradas pagadas. Falla cerrado con
GCLOUD_BETA_REQUIRED, sin instalar nada. Stop, guard, cierre y conciliación quedan
disponibles; se conserva la parada CPU por rechazo de imagen.

[VERIFICADO] La auditoría local de 15 familias/95 flags encontró un flag oculto,
listen-on-stdin, registrado por el parser del SDK; no se cambia el transporte.
Las comparaciones temporales y las pruebas de imagen/commit se registran en el
[informe del incidente](../reports/i7-gpu-restore-01-audit.md).

## Publicación histórica y cierre

Referencia: `C:/CloudRAG/i7-preparacion-20261028-468eb25/publication/FORCE_PUSH_RECORD.json`
y su resumen FORCE_PUSH_RECORD.md. SHA-256 del JSON:
`a87943b6ebba3830873bcc689bf079df6df1189dd5832c27ee90f6b3c9f6524a`.
El registro documenta autorización de Enzo, respaldo de d0ba95a y force-with-lease
de fix/interview-readiness a 468eb25 el 2026-10-29T00:02:40Z. Es una referencia
histórica, no una autorización ni una operación realizada en esta tarea.

[DECLARADO] Los seis commits locales se cierran con init.sh 0/0/0 sobre el hash
final y git diff --stat contra 468eb25. F-020 sigue active, passes=false; no se
relajan las identidades de fuente, imagen, evidencia ni GO humano.
