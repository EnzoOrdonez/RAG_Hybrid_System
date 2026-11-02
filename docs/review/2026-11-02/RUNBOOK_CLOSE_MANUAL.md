# Cierre manual por inventario: casos F-01, B-3 y B-4 (base b470d3e)

Solo documentación. No sustituye a `close` cuando `close` funciona. Herramienta:
`docs/review/2026-11-02/closeout/` (`closeout.ps1` lanza `closeout.py`).
- [VERIFICADO] La lógica de `closeout.py` está probada con un gcloud falso:
  `tests/test_review_2026_11_02_closeout.py`, 13 tests y 4 mutantes muertos.
- [NO VERIFICADO] `closeout.ps1` no se ejecutó, porque en la auditoría no había PowerShell. Tampoco
  hubo ninguna llamada real a GCP. En el primer uso, lee el `.ps1` antes de lanzarlo.

## Reglas
1. **No editar la raíz** (ADR-0026). `installation.json`, `active.json`, `receipts/` y `runs/` quedan
   como están. La carpeta `-Out` del cierre va fuera de la raíz, por ejemplo
   `C:\CloudRAG\closeout-AAAAMMDD-<caso>`.
2. **Primero, siempre, lo reversible.** Ejecutar `reconcile` del runbook y después `closeout` en modo
   solo lectura. El borrado se hace en una segunda ejecución con `-Delete -Inventory`.
3. **Confirmación por recurso.** Hay que escribir la palabra exacta (`BORRAR-<tipo>-<6 últimos del id>`).
   Si no coincide, el recurso se omite.
4. **Sin `-IncludeRetained`, salvo decisión explícita.** Esa opción permite borrar la VM y el disco GPU
   (`cloudrag-i7-vm-*`) y el snapshot limpio (`cloudrag-i7-clean-*`). El runbook de la Fase B deja su
   retención a Enzo antes de 24 h.
5. **La herramienta nunca toca** `cloudrag-i5-i6cowork-iap`, nada sin marcador I7 o CPU-i7, ni recursos
   de otro marcador cuando se pasa `-Root`.
6. **Reanudable.** Se vuelve a lanzar con el mismo `inventory.json`. Lo que ya no existe se registra como
   `ALREADY_ABSENT`. El exit 1 indica que quedó algo abortado o sin verificar: revisa `closeout-log.jsonl`.

```powershell
$run = 'C:\CloudRAG\<raíz I7>'
$out = 'C:\CloudRAG\closeout-20261102-f01'
.\docs\review\2026-11-02\closeout\closeout.ps1 -Sdk $sdk -Project <proyecto> -Root $run -Out $out
# revisar la tabla y inventory.json; después:
.\docs\review\2026-11-02\closeout\closeout.ps1 -Sdk $sdk -Project <proyecto> -Root $run -Out $out -Delete -Inventory "$out\inventory.json"
```

Con `-Root`, el alcance usa `resource_marker` y el periodo **técnico** (`period_ids.technical`), no el
`period_id` vigente. Por eso funciona también tras F-01 [VERIFICADO:
`test_root_scope_uses_the_technical_period_even_after_smoke`].

## Cómo reconocer cada caso

Todas las señales se leen sin modificar nada (`Get-Content` / `ConvertFrom-Json`).

| Señal | F-01 | B-3 | B-4 |
|---|---|---|---|
| Error de `close` | `ValueError: CPU_CLEANUP_IDENTITY_CHANGED` | `OperatorError` («Google Cloud rechazó la operación») en el `describe` de `cloudrag-i5-restore-i7cpu-*` | `OperatorError` en `compute instances describe cloudrag-i7-unallocated` |
| `installation.json` | `purpose = "smoke"` y `period_id = period_ids.smoke` | `purpose = "technical"` | `primary_vm.id = "0"`, `primary_vm.name = "cloudrag-i7-unallocated"` |
| `active.json` → `reservation` | `deleted_verified: true` (`close` ya la cerró) | según el paso | presente **sin** `deleted_verified` |
| `active.json` → `cpu_resources` / `cpu_cleanup_verified` | presente / ausente | presente / ausente | presente / ausente |
| `active.json` → `creation_intents` | sin UNKNOWN | sin UNKNOWN | una fila `disks` (y quizá `instances`) con `status: UNKNOWN` |
| `runs/NNNN-receipt.json` | — | un `compute instances delete cloudrag-i5-restore-*` con `exit_code: 0` y, después, un `describe` del mismo nombre con `exit_code: 1` | `beta compute reservations create` con exit 0 y `compute disks create` (o `instances create`) con exit ≠ 0 o 124 |
| `receipts/` | — | — | `reservation/*` con `SPECIFIC_RESERVATION_HELD` y ningún `gpu-restore/*` |

Si las señales no encajan en una sola columna, no borres nada: guarda el inventario de solo lectura y
revísalo antes de decidir.

## F-01: tras el smoke, `close` no reconoce la CPU
1. `close` ya detuvo la GPU y borró la reserva. Compruébalo con `active.json`, `reservation.deleted_verified`.
2. `closeout` en solo lectura con `-Root`. Deben salir como `temporary`:
   - la VM `cloudrag-i5-restore-i7cpu-<8>`, su disco `-boot` y la regla `-iap`;
   - el disco `cloudrag-i7-service-<12>`;
   - el snapshot `cloudrag-i7-stage-<12>`.
3. `-Delete`: se confirma cada uno de esos cinco recursos. Orden: la VM (se le quita la protección contra borrado solo a ella justo antes de borrarla), los discos, el snapshot y la regla.
4. Lo retenido (VM/disco GPU y snapshot limpio) queda para la decisión de 24 h.

## B-3: `cleanup_cpu` falló a mitad
1. No relances `close` en bucle: siempre falla en el mismo `describe`.
2. `closeout` en solo lectura con `-Root`. Solo aparece lo que todavía existe. Lo ya borrado no sale, y
   en una ejecución reanudada consta como `ALREADY_ABSENT`.
3. `-Delete` sobre lo que quede. Un `DELETE_ERROR_RETRY_LATER` se vuelve a intentar con el mismo
   inventario [VERIFICADO: `test_resumes_after_partial_failure_and_logs_absent_resources`].

## B-4: reserva tomada y GPU no creada
1. Ejecuta primero `i7_phase_b ... --live reconcile`. Si la VM sí se creó (por ejemplo, la respuesta se
   perdió por timeout), `i7_reconcile.py:88-91` la adopta en `primary_vm`. En ese caso es la ruta
   normal: la VM detenida y después `close`.
2. Si sigue `primary_vm.id = "0"`: `closeout` en solo lectura. La reserva `cloudrag-i7-<MMDDhhmmss>-<8>`
   aparece como `temporary`. Si existe un disco `cloudrag-i7-vm-<12>-boot`, aparece como `retained`.
3. `-Delete`: confirma la reserva. La herramienta comprueba antes que ninguna VM la consuma
   (`ABORTED_RESERVATION_HAS_CONSUMER`) [VERIFICADO: `test_b4_case_reservation_without_any_vm_is_deleted`].
4. Disco GPU huérfano: es caro (180 GB). Si Enzo decide borrarlo, usa `-IncludeRetained` y confirma
   solo ese recurso; responde cualquier otra cosa a los demás.

## Después del cierre manual
- El estado de la raíz no cambia: `close` seguirá sin dar CLOSED. El registro del cierre es
  `closeout-log.jsonl` más `inventory.json`; consérvalos junto a los recibos de la fase.
- Contabilidad de B-4 [VERIFICADO en código, no ejecutado]. Con la reserva borrada a mano,
  `reconcile` hace esto (`i7.py:469-481`):
  - hasta su `deleteAtTime`, responde `DELETE_OUTCOME_UNKNOWN_UNTIL_NATIVE_DEADLINE`;
  - después, marca la reserva como cerrada y carga el gasto hasta `deleteAtTime`. Esa carga es
    conservadora: sobreestima frente a la factura real.
- Contrastar con la consola de facturación, como pide el runbook de la Fase B.
