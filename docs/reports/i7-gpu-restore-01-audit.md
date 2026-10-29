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
