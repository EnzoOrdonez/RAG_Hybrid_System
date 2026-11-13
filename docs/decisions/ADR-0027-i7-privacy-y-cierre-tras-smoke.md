# ADR-0027: Privacy y cierre después de un arranque smoke

[DECLARADO] Aceptado el 2026-11-13 para corrección local sobre
42e10830 (`fix/interview-readiness`), después del intento 8 de la Fase B.
Tarea F-020. Commit y push esperan un OK separado de Enzo (ADR-0013).

## Problema

[VERIFICADO en recibos] El 2026-11-11, la raíz `iteration8-faseB-20261109-86143c7`
completó smoke (`BROWSER_EXPORT_VERIFIED`) y restore. Después:

- `privacy` terminó con exit 1: el invitado rechazó la operación con
  `I7_TECHNICAL_PRIVACY_ONLY`.
- `withdraw-purge` dio `FOUR_CONTROLS_VERIFIED`.
- `close` terminó con exit 1 y `CPU_CLEANUP_IDENTITY_CHANGED`.

[VERIFICADO en código] Dos defectos, ninguno cubierto por tests:

1. **privacy-08.** `i7_phase_b.dispatch('privacy')` arranca la VM con
   `operator.start('smoke')` (commit b58ec1c, 16/10), porque el recorrido usa el
   participante sintético del smoke y `withdraw_purge` exige el período smoke.
   `guest_bridge.disk_dispatch` exigía `purpose == 'technical'` (f26395c, 15/10).
   El `installation.json` del intento 8 tiene `iteration` 7 y `purpose` smoke:
   el paso 6 no podía pasar en ningún intento.
2. **close-08.** `cleanup_cpu` recalculaba el nombre esperado de la VM CPU con
   `cpu_label(operator.config)`, es decir, con `config['period_id']`. Cada
   `start(purpose)` cambia `period_id` al período de ese propósito
   (`lifecycle.py:160`). La VM CPU se nombra en el período technical, durante
   el cpu-build. Después de cualquier arranque smoke, el cierre rechaza su
   propia VM. Comprobado en el intento 8 sin publicar identificadores: el
   nombre registrado coincide con el período technical y no con el actual.

## Decisión

1. El invitado acepta `i7-privacy` en los períodos sintéticos `technical` y
   `smoke`, y solo en la iteración 7. Los períodos humanos (`study`, `pilot`,
   `rehearsal`) siguen rechazados, ahora con `I7_SYNTHETIC_PRIVACY_ONLY`. El
   protocolo de privacidad ya describía el recorrido con el smoke sintético
   (`docs/STUDY_I7_PRIVACY_PROTOCOL.md`).
2. `cleanup_cpu` compara con el nombre del período del build:
   `build_period(config)` devuelve `period_ids['technical']` y, si falta,
   `period_id`. Las demás comprobaciones de identidad (id, description y VM
   detenida) no cambian. Una VM nombrada para el período smoke se sigue
   rechazando.

No se cambia el orden de los pasos ni el propósito con que arranca privacy.

## Consecuencias

- El arreglo 1 corre dentro del invitado: el intento 9 necesita un cpu-build
  nuevo, con imagen y recibos nuevos, y una raíz nueva.
- El arreglo 2 corre en el host. No sirve para cerrar las raíces de los intentos
  7 y 8: sus recursos se borraron a mano el 2026-11-13 con OK de Enzo
  (`C:/CloudRAG/cleanup-20261113`), y esas raíces quedan abandonadas.
- El presupuesto de Fase B de USD 20 + 3 (ADR-0015) está agotado por los
  intentos 6 a 8. [DECLARADO] El 2026-11-13 Enzo aprobó para el intento 9 un
  tope nuevo de USD 20 + 3 de cierre, con la misma regla de corte. El
  lanzamiento espera su aviso: él debe estar disponible para privacy y el anexo.

## Verificación

- Tests nuevos: `tests/test_i7_privacy_smoke_period.py` (7) y dos en
  `tests/test_study_i7_period_names.py`. 7 rojos sobre 42e1083 por las causas
  esperadas; 17 verdes después del cambio.
- Familia I7 y estudio focalizada en verde; `bash init.sh` con exit 0 en las
  tres capas.
