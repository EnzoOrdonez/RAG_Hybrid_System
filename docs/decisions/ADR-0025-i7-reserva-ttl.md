# ADR-0025: Horas de reserva de la Fase B frente a la regla de TTL por arranque

[DECLARADO] Aceptado el 2026-10-31 para corrección local sobre
4bc1c424947f2d2e0c2eebb9aa81b83cc8fb104c, antes de lanzar el cpu-build del
intento 6. Tarea F-020. Commit y push esperan un OK separado de Enzo
(ADR-0013). Ningún cpu-build con la raíz de 4bc1c42.

## Problema

[VERIFICADO en código] `Operator.start()` rechaza un arranque si a la reserva le
quedan menos de `native_stop_s` + 60 s (`RESERVATION_TTL_TOO_SHORT_FOR_BOOT`,
`i7.py`). Con `native_stop_s` = 10800, el margen es 3 h 01 min.
- El estímulo arranca una vez por boot, 12 veces. El 12.º boot empieza tras
  11 boots: 4,06 h en la serie I6, 2,25 h en el caso ligero y 5,87 h en el peor
  caso medido.
- Con la reserva inicial de 12 h, el peor caso medido quedaba 1,21 h por debajo
  del límite aun sin anexo ni pausas.
- `reservation_intent_exposure` aceptaba solo `hours <= 12`. Lo usa `guard()`, así
  que una reserva más larga no se habría podido cerrar.

[VERIFICADO en código] En la compuerta solo hay dos arranques: el del smoke y el
del gate. `restore` y `delete_sessions` exigen la VM en RUNNING (`maintenance()`)
y no arrancan nada. El segundo arranque llega a unas 0,94 h + restore desde la
reserva de 5 h, y debe llegar antes de 1,98 h.

## Decisión

[DECLARADO] Constantes únicas en `i7.py`:
- `PHASE_B_RESERVATION_1_H` = 13,5;
- `PHASE_B_GATE_RESERVATION_H` = 5;
- `PHASE_B_RESERVATION_TOTAL_H` = 18,5.

Las usan el tope por reserva, el tope acumulado, la base de costo de `guard()`,
las cantidades del ledger (`i7_costs.default_quantities`), `gpu_restore` y el
gate.

[VERIFICADO con la lista archivada] Exposición de 24 h con cierre:
USD 19,75136847, menor que 20. Opciones rechazadas:
- 14 h + 5 h: USD 20,10;
- 14 h + 6,5 h: USD 21,17.

El máximo que cabe es 18,85 h de GPU en total.

## Consecuencias

[DECLARADO]
- Con 13,5 h, la holgura para el anexo y las pausas antes del 12.º boot es:
  5,01 h en el caso ligero, 3,20 h en la serie I6 y 0,29 h en el peor caso
  medido.
- El gate cabe si restore dura 1,04 h o menos. Se mide en el TRAMO 2a, antes de
  reservar.
- El cambio toca `scripts/`, así que hacen falta un commit nuevo, suites
  Windows/Linux/I7, un `local-readiness.json` nuevo, una raíz nueva con
  `i7_config` y un cpu-build desde el commit nuevo (`verify_tool_source`,
  `i7_readiness.source_sha256`, `i7_admission`).
- La raíz y la preparación de 4bc1c42 se conservan sin usar.

## Verificación

[VERIFICADO] `tests/test_study_i7_reservation_budget.py` (6 tests) y la
parametrización de 13,5 h en `tests/test_study_i7_reservation_time.py`:
- antes del cambio: `ImportError` (rojo);
- después del cambio: 171 tests focalizados pasan.

Mutantes con restauración byte a byte, todos detectados:
- M1: base de costo `<= 12`;
- M2: tope por reserva 12;
- M3: `<` en lugar de `<=`;
- M4: tope acumulado 17;
- M5: ledger con `l4=17`;
- M6: `reserve(12)` en `gpu_restore`;
- M7: `reserve(6)` en el gate;
- M8: reserva inicial de 14 h.
