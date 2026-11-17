# ADR-0026: Rechazo de capacidad al crear la reserva de GPU

[DECLARADO] Aceptado el 2026-11-01 para corrección local sobre
d1b09e3b98dedd87c2572ea4c442230322095f92, después del fallo de `gpu-restore`
del intento 6. Tarea F-020. Commit y push esperan un OK separado de Enzo
(ADR-0013). Alcance decidido por Enzo: solo `us-central1-a`, sin cambio de zona.

## Problema

[VERIFICADO en la nube y en recibos] El 2026-11-01 a las 17:25:06Z, `gpu-restore`
de la raíz `iteration7-faseB-20261101-d1b09e3` ejecutó
`beta compute reservations create` en `us-central1-a`.
- Resultado: exit 1 a los 14,9 s, con `ZONE_RESOURCE_POOL_EXHAUSTED` (`runs/0003-receipt.json`).
- `compute reservations list` salió vacío después: no hay ninguna reserva, ni VM ni disco de GPU.

[VERIFICADO en código] El operador trató ese rechazo como un resultado
desconocido:
- `reserve()` suma las horas y guarda `reservation_intent` con estado UNKNOWN
  antes del create. Cualquier excepción deja el intent como estaba.
- Con un intent abierto, `reserve()` falla con `UNKNOWN_RESERVATION_RECONCILE_FIRST`
  y `close` devuelve `BLOCKED_UNKNOWN_RESOURCE`.
- `reconcile()` no libera el intent antes de su `delete_at` (13,5 h después).
  A partir de ahí carga la exposición completa (USD 9,79) como gasto.
- `reservation_gpu_hours_used` quedó en 13,5 de 18,5 h. Una segunda reserva de
  13,5 h supera el tope acumulado.
- `cloud_client` registraba `cloud_error_code` solo para `compute instances
  start|create`. El create de reservas guardaba solo el hash de stderr, así que
  el intento 6 no tiene ninguna prueba persistida del código de error.

Consecuencia: una falta de capacidad, que no crea nada ni cuesta nada, agotaba
el intento completo y obligaba a otra raíz y a otro cpu-build.

## Decisión

[DECLARADO]
1. `policy.CapacityRejected(OperatorError)` lleva el código del proveedor en el
   atributo `code`. `cloud_client` la lanza ante `ZONE_RESOURCE_POOL_EXHAUSTED[_WITH_DETAILS]`
   o «does not have enough resources» (`NOT_ENOUGH_RESOURCES`). El mensaje para
   el usuario no cambia, y los `except OperatorError` existentes la siguen
   atrapando.
2. El recibo de `beta compute reservations create` gana `cloud_error_code`.
   `capacity_error_text` mantiene el filtro de privacidad.
3. Si el create lanza `CapacityRejected`, `reserve()` vuelve a listar las
   reservas. Libera la admisión solo si:
   - el listado sale bien, y
   - no aparece una reserva con el nombre del intent, ni ninguna con el marcador
     de la instalación.

   Liberar significa:
   - quitar el intent;
   - restaurar `reservation_gpu_hours_used`, `open_exposure_usd` y, en la fase
     humana, `reservation_blocks`;
   - escribir el recibo `RESERVATION_CAPACITY_REJECTED_ABSENT`;
   - volver a lanzar el error.
4. En los demás casos el intent sigue UNKNOWN, como antes: otro error del
   proveedor, un timeout, un `OperatorError` sin tipo, un listado fallido o una
   reserva propia visible. Si la reserva propia es visible, queda el recibo
   `RESERVATION_CAPACITY_REJECTED_PRESENT_KEPT_UNKNOWN`. `reconcile()` no cambia.
   Si falla el `persist()` de la liberación, el estado en memoria vuelve a la
   copia previa, así que nunca va por delante del intent UNKNOWN del disco.
5. Sin cambio de zona: la reserva, el disco y la VM siguen en `primary_vm`
   (`us-central1-a`). El reintento lo decide Enzo; el operador no reintenta solo.

[DECLARADO] La raíz del intento 6 no se edita a mano. Su intent sin prueba
persistida sigue UNKNOWN hasta 2026-11-02T06:55:06Z. Después, `reconcile` carga la
cota de USD 9,79, que es contable y no gasto real, y `close` limpia la CPU con
OK de Enzo.

## Consecuencias

[DECLARADO]
- Un rechazo de capacidad verificado deja de costar horas y dinero de la
  Fase B, y `gpu-restore` se puede reintentar en la misma raíz.
- El cambio toca `scripts/`, así que hacen falta:
  - un commit nuevo;
  - las suites Windows, Linux e I7;
  - un `local-readiness.json` nuevo;
  - una raíz nueva con `i7_config`;
  - un cpu-build desde el commit nuevo (intento 7).
- Los lanzadores del intento 7 deben usar una carpeta `live-tramo1-rN` por
  reintento, porque `New-LiveFolder` no reutiliza carpetas.
- [DECLARADO, decidido por Enzo el 2026-11-01] **Gate:** antes, `i7_phase_b.py`
  creaba `gate-cohort-intent.json` (`'x'`, `replay_allowed: False`) antes de
  `reserve(5)`. Un rechazo de capacidad liberado gastaba así el único intento
  del gate (L-2 de `docs/reports/i7-gpu-restore-01-audit.md`). Ahora:
  1. Si el intento ya existe, el gate sale con `GATE_ATTEMPT_EXISTS_NO_REPLAY` sin
     reservar, así que una repetición prohibida no paga 5 h.
  2. Si no hay una reserva abierta, se ejecuta `reserve(5)`.
  3. El intento se crea con `'x'` y `replay_allowed: False` recién con la reserva
     retenida, antes de `switch_profile` y de cualquier medición.

  Con un rechazo liberado no queda intento y el gate se puede repetir. Si el
  resultado es UNKNOWN tampoco queda intento: `reserve()` sigue bloqueado hasta
  `reconcile`, y si la reserva aparece, la repetición la reutiliza. En ningún
  caso se midió nada. La regla de no repetir un gate empezado no cambia: el
  intento sigue siendo único desde el primer paso medido. `launch_gate.ps1`
  mantiene su comprobación previa del archivo.
- Un listado que todavía muestre una reserva propia antigua (por ejemplo, en
  borrado) deja el intent UNKNOWN. Es conservador y vuelve al bloqueo de
  antes.

## Verificación

[VERIFICADO] `tests/test_study_i7_reservation_capacity.py` (15 tests):
- antes del cambio, `ImportError`; con solo la clase nueva, 7 en rojo por
  comportamiento;
- después del cambio, los 15 pasan y las suites focalizadas del operador e I7
  siguen en verde.

La revisión independiente (ecc:code-reviewer) no encontró nada CRITICAL ni HIGH.
Sus hallazgos LOW sobre el orden del `persist` y el recibo del caso visible se
aplicaron, con dos tests más que se escribieron después del código, no antes.

[VERIFICADO] `tests/test_study_i7_gate_capacity.py` (5 tests):
- con el orden anterior, 4 en rojo;
- el de la reserva abierta reutilizada describía el orden anterior y ya pasaba;
- con el cambio, los 5 pasan, y también los de `test_study_i7_reservation_budget.py`
  y `test_study_i7_phase_b.py`.

Mutantes con restauración byte a byte, todos detectados:
- M1: sin relectura;
- M2: sin restaurar horas ni exposición;
- M3: se ignora el marcador;
- M4: sin liberación;
- M5: el create de reservas sin `cloud_error_code`;
- M6: `reservation_blocks` sin restaurar;
- M7: sin volver la memoria atrás cuando falla `persist`;
- M8: sin recibo del caso visible;
- G1: el intento del gate otra vez antes de `reserve(5)`;
- G2: sin la comprobación previa del intento existente.
