# ADR-0032: Resultado del intento 12 y nombre de reserva ligado a la VM

[DECLARADO] Registrado el 2026-11-14 sobre la imagen de d2c2845 y el anexo publicado en d053c4b
(`fix/interview-readiness`). Tarea F-020. Claude ejecutó el intento con el control delegado por
Enzo ("Haz lo necesario para cumplir el objetivo... todo depende de ti"; "sigue hasta completar el
objetivo"), dentro del tope de USD 20 + 3 por intento. El cambio toca `scripts/`, así que exige
commit, suites, `local-readiness`, raíz y cpu-build nuevos (intento 13).

## Resultado del intento 12

[VERIFICADO en recibos privados y en la bitácora del intento]

| Paso | Resultado |
|---|---|
| 1 cpu-build | OK. |
| 2 gpu-restore | Capacidad L4 en la primera ronda. |
| 3, 4, 4b | ready, tls-three y contexts: exit 0. |
| Anexo | Publicado en d053c4b y anclado: `PUBLISHED_ANNEX_AND_CONTEXTS_VERIFIED`, `LOCAL_RUNBOOK_READY`. |
| 5a, 5b | smoke y restore: exit 0. |
| 6 privacy | exit 0 con estado **PASS**. |
| 7 withdraw-purge | exit 0. |
| 8a estímulo | **ACCEPTED 12/12**, el primero de la Fase B. Valida la ventana de acuse de ADR-0031. |
| 8b gate | Cinco rechazos de capacidad L4 liberados en `reserve(5)` (11:59Z a 14:06Z; ADR-0026, intento no consumido). Reserva 2 obtenida a las 14:37Z. Fallo terminal sin medición: `SPECIFIC_RESERVATION_NOT_BOUND`. |
| 9 close | `CLOSED`: VM detenida y verificada, reserva cerrada, temporales CPU ausentes. |

Gasto: USD 7,95 (exposición conservadora, no factura). La latencia del despliegue sigue sin medir:
no hay GO ni NO-GO.

## Problema

[VERIFICADO en recibos] El gate escribió `gate-cohort-intent.json` tras obtener la reserva 2. En
`switch_profile('smoke')`, `i7_smoke.run` llamó a `operator.start('smoke')`, que lanzó
`SPECIFIC_RESERVATION_NOT_BOUND` (`i7.py`, comprobación de `reservationAffinity`). El `finally` detuvo
la VM y `guard` borró la reserva 2 en un minuto. El intento único quedó consumido sin medir.

[VERIFICADO en código]
- La VM GPU se crea una sola vez con `--reservation-affinity=specific --reservation=<reserva 1>`
  (`i7_cpu_pipeline.gpu_restore`).
- `reserve()` daba a cada reserva un nombre nuevo con marca de tiempo. La reserva 2 del gate, y cada
  bloque de la fase humana, nunca coincidían con la afinidad de la VM.
- Los dobles de prueba devolvían una afinidad igual al nombre en estado, o ninguna. Por eso las suites
  no detectaron el defecto.

[VERIFICADO en documentación de Google Cloud] La afinidad de una VM existente solo puede pasar a
`ANY_RESERVATION` o `NO_RESERVATION`. No se puede re-ligar a otra reserva específica.

## Decisión

[DECLARADO] `Operator._consumable_reservation_name()` elige el nombre en `reserve()`:
- **Sin VM GPU** (`primary_vm.id == '0'`): nombre nuevo `cloudrag-i7-<MMDDHHMMSS>-<periodo>`, como
  antes.
- **Con VM GPU:** se reutiliza el nombre de la reserva anterior, que ya está cerrada y verificada
  (`PRIOR_RESERVATION_NOT_CLOSED` lo exige). Antes de cualquier efecto pagado se lee la VM y se exige
  `SPECIFIC_RESERVATION` con ese nombre y el mismo id. Si no, `SPECIFIC_RESERVATION_NOT_BOUND`.
- La lectura ocurre antes de mutar el estado o la admisión. En el gate ocurre antes de escribir
  `gate-cohort-intent.json`, así que una VM mal ligada ya no consume el intento único.
- No cambian `start()`, `guard()`, `reconcile()` ni la contabilidad. Todas distinguen las reservas por
  id además del nombre.

## Límites conocidos

[DECLARADO]
- Reutilizar un nombre exige que Google Cloud lo libere al borrar la reserva anterior. `guard()` y
  `reconcile()` solo marcan una reserva como cerrada tras una lista sin ella. No se ha probado en la
  nube.
- Si Google Cloud rechaza el nombre, el error no es de capacidad: la intención queda UNKNOWN para
  `reconcile` y el gate no escribe su intento.
- El arreglo del lanzador privado para reintentar la capacidad del gate (intents
  `IDENTITY_RESOLVED` en el tramo 1) vive fuera del repo, en la carpeta del intento.

## Verificación

[VERIFICADO]
- `tests/test_study_i7_reservation_binding.py`: 10 tests.
  - La reserva del gate reutiliza el nombre que consume la VM.
  - Ningún `create` ni cambio de estado si la VM tiene otro nombre u otro id, afinidad ANY, NONE o
    ninguna, o si falta la reserva registrada.
  - La primera reserva, sin VM, recibe un nombre nuevo y no lee la VM.
  - Los bloques de la fase humana reutilizan el nombre, y un rechazo de capacidad sobre ese nombre
    libera la admisión.
- RED: 2 fallos en la primera versión. GREEN: pasan junto con las suites de I7 y del operador.
- Revisión de código independiente: sin hallazgos CRITICAL ni HIGH. Sus casos MEDIUM se añadieron.
- Los dobles de `ReservationCloud` ahora devuelven la afinidad con la que nació la VM. Las aserciones
  de secuencia exacta incluyen la lectura previa.
