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
| i7.Operator.reconcile, UNKNOWN encontrado | delete_at_utc histórico con microsegundos frente a deleteAtTime API | Candidatos discretos por compatibilidad | REPORTE_REVISION3 corrigió el modelo de 6bfbec9: el SDK redondea half-up, conservando .999 al desbordar. Esta ronda acepta solo el truncado o el redondeado; .968532 admite .968/.969. Fuera del conjunto se rechaza. Las reservas nuevas de segundos enteros conservan igualdad exacta, incluso ante ±1 ms. |
| i7.Operator.start | deleteAtTime API menos reloj local | Resta de datetime aware y margen mínimo native_stop_s + 60 | Z y offsets equivalentes conservan el instante; no se compara texto ni se pierde precisión en el margen. Sin arreglo adicional. |
| i7.Operator.reconcile, reserva conocida | now frente a deleteAtTime; min(now, TTL) menos created_utc API | Orden, mínimo y resta de instantes aware | No requiere igualdad de representación. El TTL limita el cargo; permanece la precisión recibida. Sin arreglo adicional. |
| i7.Operator.guard | now menos created_utc de creationTimestamp | Resta de datetime.fromisoformat | Python 3.14 admite Z y offsets; no hay igualdad exacta con un instante local previo. El método no aplica aquí min(now, TTL): observación de contabilidad fuera del arreglo de precisión; no se modifica. |
| i7_costs.retained_gib_hours y lifecycle.reserve_cost / ip_release | creationTimestamp guardado como created_utc o ip_reserved_utc frente a reloj local | Restas y cotas de retención | Admiten instantes aware con distinta precisión; no necesitan redondeo. Algunos consumidores no codifican explícitamente el error de un timestamp sin zona; no se añade migración ni corrección transversal. |
| lifecycle.settle_alternate_creation | lastStartTimestamp y lastStopTimestamp, ambos de GCE | Resta y rechazo de duración negativa | Z y offsets equivalentes son comparables; un valor sin zona mezclado con uno aware puede producir TypeError no codificado. Ruta heredada, sin corrección en este cierre. |
| cost_reconciliation.aware / interval_cost | creationTimestamp API y límites locales de conciliación | Exige zona, normaliza UTC, aplica max/min y resta | Maneja representaciones equivalentes y distintas precisiones. Sin arreglo adicional. |
| cost_reconciliation.reconcile | creationTimestamp de dos inventarios API para la misma identidad | Comparación textual dentro de metadatos | Puede rechazar representaciones distintas de un mismo instante; es rechazo conservador, no aceptación de otra identidad. No se observó esa variación en GCE real. Se reporta sin corregir. |
| host_runtime.ready_elapsed y configuración de startup | start_requested_utc y native_deadline_utc locales; maxRunDuration API es una duración | Elapsed aware; checked_vm compara segundos de duración, no una fecha local con una fecha API | No reproduce la igualdad problemática de deleteAtTime. Se conservan las guardas actuales. |

[VERIFICADO] Las regresiones de esta ronda prueban reservas nuevas de 1/4/12 h,
UTC y offset -05:00; lecturas heredadas .968532 a .969, .968499 a .968 y .9997 a
.999; y rechazo fuera del conjunto permitido. Las nuevas reservas rechazan ±1 ms
y cambios submilisegundo. Un heredado .968532 también acepta .968 por la
compatibilidad acordada, aunque sea -1 ms respecto de .969; no se afirma que ese
caso deba rechazarse. Son fixtures sintéticos, sin reservas reales.

[DECLARADO] Los hashes y timestamps de los archivos privados históricos no se
normalizan ni se reescriben. Esta revisión no declara compatibilidad con fechas
arbitrarias sin zona ni acredita tiempos o facturación de GCE en vivo.

## Commit, build y GO de la fase humana

[VERIFICADO por código y siete pruebas sintéticas] verified_gate_go verifica el
hash del archivo privado, el hash lógico del proof, el veredicto GO y la igualdad
de proof.inventory.image.image_id y proof.inventory.source.commit con
installation.json. No lee HEAD ni consulta Git por sí mismo.

| Cambio | Resultado |
|---|---|
| Cambiar solo el commit de installation.json, manteniendo el proof anterior | VERIFIED_GATE_GO_REQUIRED, incluso si la imagen no cambió. |
| Cambiar solo la imagen de installation.json | VERIFIED_GATE_GO_REQUIRED, incluso si el commit no cambió. |
| Nueva identidad de build y nuevo proof/recibo coherente con esa imagen y commit, antes de abrir la fase | Puede abrir HUMAN_PHASE_OPEN; cambiar el commit antes del build no impide un GO futuro de la nueva identidad. Esto se prueba con fixtures, no con un build real. |
| Cambiar HEAD manteniendo installation.json y prueba antiguos | verified_gate_go aislado todavía verifica esa pareja antigua; verify_tool_source controla HEAD por separado. Si el diff incluye código o .harness, la CLI rechaza OPERATOR_SOURCE_COMMIT_DIFFERS. |
| Publicación posterior que cambia únicamente docs/ u output/audit/ | verify_tool_source conserva su excepción documental; no vuelve a etiquetar el commit de la imagen ni el proof. |
| Sustituir el GO después de abrir la fase humana | human_start_profile rechaza GATE_EVIDENCE_CHANGED_SINCE_OPEN; no se migra silenciosamente la fase abierta. |

[VERIFICADO] Antes de build, i7_cpu_pipeline.build exige HEAD igual al commit
instalado y al de la rama publicada (PUBLISHED_I7_SOURCE_REQUIRED). La prueba
descargada de Streamlit también liga imagen y commit. El recibo local de readiness
se liga a los bytes de fuente, incluidos scripts y tests: cambiar esos bytes
requiere nueva evidencia de verificación; init.sh local no sustituye sus recibos
Windows/Linux/I7 ni un GO real.

[DECLARADO] El nuevo código debe pertenecer a la identidad del build futuro y
de las pruebas y el GO posteriores. La instalación y el build históricos de
468eb25 no se reetiquetan para saltar las guardas. Aquí no se publica, reconstruye,
migra estado ni produce GO en GCP.

## Verificación de las correcciones

[HISTÓRICO, MODELO CORREGIDO] Antes del arreglo temporal hubo ocho regresiones fallidas: seis
reservas nuevas y dos recuperaciones equivalentes. Después, 133 tests de tiempo,
ledger, human y operator pasaron. Retirar la truncación del esperado histórico
causó dos fallos; aceptar todo TTL causó seis. Los archivos se restauraron byte
por byte. Esos conteos correspondían al fake truncador original y no demostraban
compatibilidad con el redondeo real del SDK; la validación actual figura abajo.

[VERIFICADO] El test de beta falló antes del arreglo al alcanzar admit_paid.
Retirar las guardas de reserve, CLI del dueño y CLI Fase B produjo respectivamente
1, 9 y 11 fallos. Retirar por separado las guardas de imagen y commit del GO
produjo un fallo por mutante. Las siete mutaciones se restauraron byte por byte.
Evidencia local: `C:/Users/enziz/AppData/Local/Temp/i7-incident-mutations-t9iydm64`.

[VERIFICADO] Suite focalizada final: todos los test_study_i7*.py más policy,
314 pasan y 8 omitidos (sin fijar el runtime Chromium en esa invocación).
El preflight nuevo, ejecutado contra el SDK local, devuelve GCLOUD_BETA_INSTALLED
con beta_version=2026.09.25. Los seis checks de documentación pasan.

[DECLARADO] init.sh completo se ejecuta después del sexto commit, con runtime
Chromium fijado y la máquina quieta; su salida y el hash congelado se entregan
juntos en el traspaso final. Esta declaración no anticipa un resultado.

## Ronda 2 sobre a32c9dc: precisión corregida

Fuente: `C:/CloudRAG/iteration7-revision-cruzada-20261027/REPORTE_REVISION3.md`,
SHA-256 `5616fa1373a565ffb29cd7573c20315c7e1b73957ba25b39382b462d503f8154`.

[VERIFICADO] Una sonda importó únicamente FormatDateTime del SDK instalado,
sin ejecutar gcloud: .968532 se convierte en .969, .968499 en .968 y .9997 en
.999, tanto en Z como en -05:00. Origen: command_lib/compute/reservations/util.py:325,
core/util/times.py:320 y :350-365. Evidencia local:
`C:/Users/enziz/AppData/Local/Temp/i7-r4-sdk-rounding-mznw74qu/sdk-rounding.json`.

[VERIFICADO] H-A: cuatro casos rojos sobre a32c9dc antes de corregir el
comparador. H-B: dos casos rojos antes de sustituir el fake truncador por un
modelo decimal independiente half-up, con saturación de 999. Tras ambos cambios,
104 pruebas de tiempo, SDK y GO pasan.

| Mutante | Cambio | Resultado |
|---|---|---|
| M1 | Igualdad estricta contra el esperado con microsegundos, sin candidatos | 20 fallos y 34 pasan; restaurado byte por byte. |
| MX | Solo redondeado SDK, retirando la compatibilidad truncada de H-A | 3 fallos y 51 pasan; restaurado byte por byte. |
| Control H-B | Volver a truncar la salida de MillisecondCloud | 2 fallos y 4 pasan; restaurado byte por byte. |

MX ya no confunde una lectura SDK correcta con un error del fake: sus tres
fallos actuales protegen la aceptación truncada que H-A exige para heredados.
Logs: `C:/Users/enziz/AppData/Local/Temp/i7-r4-mutations-3el4oyqb`.

[DECLARADO] El cierre completo de esta ronda se ejecuta tras sus cuatro commits
y se entrega con el hash exacto en el reporte externo. Los conteos y la cabecera
anteriores conservan su alcance histórico; no se confunden con ese cierre nuevo.
