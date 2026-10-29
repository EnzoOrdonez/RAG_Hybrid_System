[Codex] | 2026-10-29T17:18:19.751435+00:00 | rama `fix/interview-readiness` @ `1f3b5ff` | F-020 | modo: ejecución local
Capas: init.sh final pendiente; auditoría de ayuda=exit 0. Cambios: informe. Sin verificar: efectos GCP reales.

# Auditoría local de gpu-restore-01

## SDK y método

[VERIFICADO] Google Cloud SDK 587.0.0. Beta instalado según components list --only-local-state.
Se ejecutaron únicamente --version, components list local y los 15 comandos --help de la tabla. Sin recursos, autenticación, instalación de componentes ni llamadas con efecto.
Inventario por AST de las funciones alcanzables, revisado contra el despacho de las once etapas y sus auxiliares. Incluye limpieza, parada de CPU por rechazo de imagen y transporte SSH/IAP de Windows.
Los flags globales de Cloud.command se comprueban en cada comando. El túnel nativo usa sus propios flags. Se reconocen negaciones como --[no-]deletion-protection.
Control negativo: un flag inexistente se detecta; una negación documentada se reconoce.

Evidencia local reproducible: `C:\Users\enziz\AppData\Local\Temp\i7-sdk-audit-20261029T121533` (audit-runner.py, manifest.json, componentes, versión y ayuda completa).
SHA-256 de manifest.json: `7b176bc23218e009838249d920b6234ede8ad599492826f7d437c7b26b703805`.
SHA-256 del ejecutor: `3fcf49cfb27e95b62e51ec77e72ddcda55a547eea550cd4ea6fd1fcaa163048a`.
Cada entrada del manifiesto guarda comando, flags, ubicaciones, exit y SHA-256 de su ayuda.

## Flags y discrepancias

| Comando gcloud | Flags comprobados | Discrepancias |
|---|---|---|
| `beta compute reservations create` | `--accelerator`, `--delete-at-time`, `--description`, `--format`, `--machine-type`, `--project`, `--quiet`, `--require-specific-reservation`, `--vm-count`, `--zone` | Ninguna; help exit 0 |
| `compute disks create` | `--description`, `--format`, `--project`, `--quiet`, `--source-snapshot`, `--type`, `--zone` | Ninguna; help exit 0 |
| `compute disks delete` | `--format`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute firewall-rules delete` | `--format`, `--project`, `--quiet` | Ninguna; help exit 0 |
| `compute instances add-metadata` | `--format`, `--metadata`, `--metadata-from-file`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute instances create` | `--address`, `--deletion-protection`, `--description`, `--disk`, `--format`, `--instance-termination-action`, `--machine-type`, `--maintenance-policy`, `--max-run-duration`, `--metadata`, `--metadata-from-file`, `--network`, `--project`, `--quiet`, `--reservation`, `--reservation-affinity`, `--scopes`, `--service-account`, `--subnet`, `--tags`, `--zone` | Ninguna; help exit 0 |
| `compute instances delete` | `--format`, `--keep-disks`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute instances set-scheduling` | `--format`, `--instance-termination-action`, `--max-run-duration`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute instances start` | `--format`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute instances stop` | `--format`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute instances update` | `--format`, `--no-deletion-protection`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute reservations delete` | `--format`, `--project`, `--quiet`, `--zone` | Ninguna; help exit 0 |
| `compute snapshots delete` | `--format`, `--project`, `--quiet` | Ninguna; help exit 0 |
| `compute ssh` | `--command`, `--dry-run`, `--format`, `--project`, `--quiet`, `--ssh-flag`, `--ssh-key-expire-after`, `--tunnel-through-iap`, `--zone` | Ninguna; help exit 0 |
| `compute start-iap-tunnel` | `--listen-on-stdin`, `--project`, `--verbosity`, `--zone` | --listen-on-stdin ausente de ayuda; parser local hidden=True; help exit 0 |

[VERIFICADO] 15 familias; 95 comprobaciones de flags por comando; 94 documentadas en ayuda y una oculta. No hubo corrección de flags de producción.
[VERIFICADO] --listen-on-stdin no figura en --help, pero el parser instalado lo registra con hidden=True en surface/compute/start_iap_tunnel.py:118-120. Retirarlo cambia el transporte stdin/stdout del ProxyCommand; se conserva.
SHA-256 del parser local: `6a458c61797b7ab6d1406a960e841be94ff2b0e93073156689a1228e48bf7cbb`.
Esta comprobación acredita existencia sintáctica, no autorización IAM, valores concretos, capacidad, gasto ni éxito operativo.

## Cobertura por etapa

| Etapa | Comandos con efecto y auxiliares |
|---|---|
| gpu-restore | beta reservations create; disks create; instances create; instances stop para CPU rechazada. |
| ready | compute ssh; start-iap-tunnel; publicación de clave SSH administrada por SDK. |
| tls-three | instances add-metadata/start/stop; SSH/IAP de ready. |
| contexts | instances add-metadata/start/stop; SSH/IAP para preflight, mantenimiento y contextos. |
| smoke | instances add-metadata/start/stop; SSH/IAP para preflight, invitación y observación. |
| restore | SSH/IAP para mantenimiento y restauración; operaciones HTTP/GCS indicadas abajo. |
| privacy | instances add-metadata/start/stop; SSH/IAP para smoke, mantenimiento y prueba privada. |
| withdraw-purge | SSH/IAP para mantenimiento, retirada y purga; HTTP/GCS. |
| stimulus | instances set-scheduling, add-metadata/start/stop; reservations delete; SSH/IAP. |
| gate | reserva condicional beta; set-scheduling, add-metadata/start/stop; reservations delete; SSH/IAP; HTTP/GCS. |
| close | instances stop/update/delete; reservations delete; disks/snapshots/firewall-rules delete de recursos CPU propios. |

Los prefijos abreviados de la cobertura corresponden a las filas compute de la tabla anterior.
Las ramas se inventarían aunque un estado concreto no las ejecute. Lecturas IAM, describe/list, atributos y serial no crean recursos; no se confunden con los comandos con efecto.
Backups, restore, privacy y purge también usan Storage mediante HTTP/GCS; el invitado ejecuta código Python/Docker y publica atributos por HTTP de metadatos. Estas operaciones no tienen flags gcloud: no se ejecutaron y no se atribuye una validación --help a ellas.
CPU-build, technical-cycle y reconcile no forman parte de las once etapas solicitadas; las limpiezas transitivas invocadas por gate y close sí están incluidas.

## Comparaciones temporales local/API

[VERIFICADO por lectura y sondas locales] Se buscaron deleteAtTime, creationTimestamp,
lastStartTimestamp, lastStopTimestamp y sus consumidores en scripts/study_operator.
Las ubicaciones nombran funciones para mantener la referencia al cambiar líneas.

| Ubicación | Valores y precisión | Comparación / zona | Resultado y alcance |
|---|---|---|---|
| i7.Operator.reserve | until local tenía 6 decimales; GCE recibe milisegundos por gcloud FormatDateTime | Igualdad de datetime aware | Defecto reproducido en 468eb25; corregido en 6bfbec9. Nuevas solicitudes emiten UTC con segundos completos. |
| i7.Operator.reconcile, UNKNOWN encontrado | delete_at_utc histórico con microsegundos frente a deleteAtTime API | Igualdad estricta original | Mismo defecto reproducido; 6bfbec9 trunca solo el esperado a milisegundos y exige igualdad de instantes. Cambiar ±1 ms bloquea; no hay tolerancia de un segundo. |
| i7.Operator.start | deleteAtTime API menos reloj local | Resta de datetime aware y margen mínimo native_stop_s + 60 | Z y offsets equivalentes conservan el instante; no se compara texto ni se pierde precisión en el margen. Sin arreglo adicional. |
| i7.Operator.reconcile, reserva conocida | now frente a deleteAtTime; min(now, TTL) menos created_utc API | Orden, mínimo y resta de instantes aware | No requiere igualdad de representación. El TTL limita el cargo; permanece la precisión recibida. Sin arreglo adicional. |
| i7.Operator.guard | now menos created_utc de creationTimestamp | Resta de datetime.fromisoformat | Python 3.14 admite Z y offsets; no hay igualdad exacta con un instante local previo. El método no aplica aquí min(now, TTL): observación de contabilidad fuera del arreglo de precisión; no se modifica. |
| i7_costs.retained_gib_hours y lifecycle.reserve_cost / ip_release | creationTimestamp guardado como created_utc o ip_reserved_utc frente a reloj local | Restas y cotas de retención | Admiten instantes aware con distinta precisión; no necesitan redondeo. Algunos consumidores no codifican explícitamente el error de un timestamp sin zona; no se añade migración ni corrección transversal. |
| lifecycle.settle_alternate_creation | lastStartTimestamp y lastStopTimestamp, ambos de GCE | Resta y rechazo de duración negativa | Z y offsets equivalentes son comparables; un valor sin zona mezclado con uno aware puede producir TypeError no codificado. Ruta heredada, sin corrección en este cierre. |
| cost_reconciliation.aware / interval_cost | creationTimestamp API y límites locales de conciliación | Exige zona, normaliza UTC, aplica max/min y resta | Maneja representaciones equivalentes y distintas precisiones. Sin arreglo adicional. |
| cost_reconciliation.reconcile | creationTimestamp de dos inventarios API para la misma identidad | Comparación textual dentro de metadatos | Puede rechazar representaciones distintas de un mismo instante; es rechazo conservador, no aceptación de otra identidad. No se observó esa variación en GCE real. Se reporta sin corregir. |
| host_runtime.ready_elapsed y configuración de startup | start_requested_utc y native_deadline_utc locales; maxRunDuration API es una duración | Elapsed aware; checked_vm compara segundos de duración, no una fecha local con una fecha API | No reproduce la igualdad problemática de deleteAtTime. Se conservan las guardas actuales. |

[VERIFICADO] Las regresiones prueban las reservas de 1/4/12 h con reloj no entero,
UTC y offset -05:00; dos recuperaciones UNKNOWN equivalentes; cuatro rechazos por
±1 ms; y rechazo de fecha sin zona. Son fixtures sintéticos, sin reservas reales.

[DECLARADO] Los hashes y timestamps de los archivos privados históricos no se
normalizan ni se reescriben. Esta revisión no declara compatibilidad con fechas
arbitrarias sin zona ni acredita tiempos o facturación de GCE en vivo.
