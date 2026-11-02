# Seguimiento 2026-11-02: respuestas a las tareas 1, 2 y 5 de Enzo

Base b470d3e. Todo se verificó con código y tests de esta rama. No hubo ninguna llamada a GCP.

## 1. B-4: ruta exacta [VERIFICADO, reproducido]

**Cómo se llega a la VM provisional.**
- `scripts/study_operator/i7_config.py:43` crea la raíz con
  `primary_vm = {'name':'cloudrag-i7-unallocated','id':'0'}`.
- `scripts/study_operator/i7_cpu_pipeline.py:266` toma `reserve(PHASE_B_RESERVATION_1_H)`. La reserva
  queda creada y persistida en `state['reservation']` (`i7.py:304-308`).
- Entre `i7_cpu_pipeline.py:267` y `:297` no hay `try`. Cualquier excepción en ese tramo sale de
  `gpu_restore` sin tocar `primary_vm`:
  - `disks create` (`:272`);
  - `describe` (`:274`);
  - `observed` (`:275`);
  - la comprobación de origen (`:276-278`);
  - `write_startup` (`:283`);
  - `record_intent` (`:284`);
  - `instances create` (`:285-293`);
  - `describe` / `observed` / el marcador (`:294-297`).
- La única asignación de la VM real es `:298` (`operator.config.update(primary_vm=...)`), y está después
  de todo eso.

**Por qué `close` no puede borrar la reserva.**
- `i7_workflows.py:187` omite `stop()` porque el id es `'0'`.
- `:189` llama a `guard()`, porque hay reserva y no tiene `deleted_verified`.
- `i7.py:429` describe `self.selected()`, es decir `config['primary_vm']` (`lifecycle.py:41`), que es
  `cloudrag-i7-unallocated`.
- Un `describe` de algo que no existe sale con exit ≠ 0, y `cloud_client` lo convierte en `OperatorError`.
- El `reservations delete` (`i7.py:440`) nunca llega a enviarse.

**Excepción.** Si la VM sí se creó y solo se perdió la respuesta, `reconcile-all` la adopta
(`i7_reconcile.py:88-91`) y `close` vuelve a funcionar. B-4 vale cuando la VM no existe o cuando no se
ejecutó `reconcile`.

**Reproductor.** `tests/test_review_2026_11_02_recovery.py::test_b4_close_cannot_release_reservation_held_before_gpu_vm_exists`.
El fallo se inyecta en `disks create` (`QUOTA_EXCEEDED`).
- Salida: `close raised: NOT_FOUND instances/cloudrag-i7-unallocated`, sin ningún `reservations delete`.
- `reservation_gpu_hours_used` queda en 13,5.

B-4 se mantiene como S1 confirmado.

## 2. Archivos de un solo uso frente a un rechazo gratuito con R1 tomada

Tests: `tests/test_review_2026_11_02_single_use.py`. El rechazo se modela como `CapacityRejected` en
`compute instances start` (`cloud_client` lo lanza para cualquier comando cuyo stderr traiga el código de
capacidad). Los 3 tests pasan sobre la base.

| Archivo | Se escribe en | Comprobaciones previas al gasto que van DESPUÉS | ¿Lo consume un rechazo gratuito? | Recuperación documentada sin editar la raíz |
|---|---|---|---|---|
| `gate-cohort-intent.json` | `i7_phase_b.py:105-106` (`'x'`, `replay_allowed: False`), después de `reserve(5)` o de reutilizar una reserva abierta (`:102-104`) | `switch_profile` (`:111`, que llama a `admit_paid(0)`), `run(smoke)`, que llama a `start('smoke')` (`i7_smoke.py:28`), con admisión, TTL (`i7.py:355`), identidad (`:359`), afinidad (`:362`) y, por último, `instances start` (`i7.py:383`) | **Sí** [VERIFICADO]. Un `CapacityRejected` en el primer `instances start` deja el archivo, y el reintento da `GATE_ATTEMPT_EXISTS_NO_REPLAY` | **No.** `STUDY_I7_RUNBOOK_FASE_B.md:315-318` solo dice que el gate no se repite. ADR-0026 promete que el intento es único «desde el primer paso medido», pero un arranque rechazado no mide nada. Recuperar exige borrar el archivo de la raíz, lo que equivale a editarla a mano |
| `attempts/stimulus-NN.json` | `i7_workflows.py:82-83` (`'x'`, `INTENT`) | Todo `operator.start()` (`:84`): admisión, TTL, identidad, afinidad e `instances start` | **Sí** [VERIFICADO]. Queda `INTENT` sin `.complete.json`, y el reintento se bloquea en la ranura 1 con `TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW`. En el reintento ese error queda oculto tras el `RESERVATION_ALREADY_CLOSED` del `finally` (B-2b) y solo se ve en `__context__` | **No.** El código pide «revisión humana» (`:79-80`), pero ningún documento del repo dice en qué consiste; `grep` sin resultados fuera de `docs/review`. Además, el `finally` cierra R1 (`:113-114`), y con 13,5 h usadas una R1 nueva supera el tope de 18,5 h (`i7.py:272`) |
| `live-contexts.json` | `i7_phase_b.py:222`, **después** de `start()`, `wait_ready`, `maintenance` y el puente (`:215-221`) | Ninguna: las comprobaciones van antes | **No** [VERIFICADO]. Con el rechazo el archivo no existe y el reintento vuelve a llegar al arranque | No hace falta. El riesgo residual es otro (B-7, S3): un corte entre `:222` y `:224` hace que el reintento pague un boot y falle con `FileExistsError` |

Conclusión: los dos primeros archivos violan la regla «registrar justo antes del primer efecto pagado».
Ninguno de los dos tiene una recuperación documentada que no pase por editar la raíz.

## 5. R-15: cambio mínimo para tornado [VERIFICADO en local; imagen NO reconstruida]

**Situación [VERIFICADO con pip-audit 2026-11-02, solo tornado].**
- `requirements-lock.txt:124` fija `tornado==6.5.4`, con 8 avisos.
- CI (`requirements-app.txt:2177`) fija `6.5.8`, que todavía tiene 6.
- Los arreglos llegan en 6.5.7, 6.5.8, 6.5.9 y 6.6. **Solo 6.6 los cierra todos.**
- `streamlit==1.54.0` exige `tornado>=6.0.3,!=6.5.0,<7`. `uv pip compile` resuelve
  `streamlit==1.54.0` + `tornado==6.6` sin conflicto.

**Cambio mínimo** (`patches/004-r15-tornado-lock.patch`), dos archivos:
1. `requirements-lock.txt:124`: `tornado==6.5.4` pasa a `tornado==6.6`.
2. `config/i7-build-recipe-pin.json`: el sha256 de `requirements-lock.txt`. Si no se actualiza, fallan
   `test_prebuild_matches_owner_lock_and_recipe_without_image` y
   `test_cpu_preflight_allows_first_build_without_image` (`I7_BUILD_RECIPE_HASH_MISMATCH`). Con los dos
   archivos cambiados, la suite completa queda sin regresiones (en ese worktree solo falló R-39, que
   es de entorno).
- Coherencia recomendada, aunque no forma parte del mínimo: subir también `requirements-app.txt` a 6.6
  con sus hashes (`uv pip compile`), para que CI pruebe lo mismo que la imagen.

**¿Obliga a reconstruir la imagen del intento 8? Sí.**
- El `Dockerfile` instala desde el lock (`:8-17`), así que el paquete del servidor cambia y con él el
  `image_id`.
- Además, `config/` y `requirements-lock.txt` están dentro de:
  - `i7_readiness.source_sha256` (`i7_readiness.py:13`), que pide un `local-readiness.json` nuevo;
  - `verify_recipe` (`i7_streamlit_preflight.py:23-33`).
- Hacen falta un commit nuevo (aunque no toque `scripts/`), un cpu-build nuevo y una raíz nueva
  (`verify_tool_source` rechaza diferencias fuera de `docs/`).
- **No verificado:** compatibilidad en ejecución de Streamlit 1.54.0 con tornado 6.6 dentro de la imagen
  (no hay Docker), y el resto de avisos del lock (275 en total según el informe anterior, [DECLARADO]):
  en esta sesión pip-audit solo se ejecutó sobre tornado.
