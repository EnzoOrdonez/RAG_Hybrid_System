# Anexos de la auditoría del intento 8 (b470d3e..3bb9ca8)

Todo lo que se marca como «ejecutado» se corrió en Linux, con Python 3.14.6 (uv), `.venv-app`
instalado desde `requirements-app.txt` (CPU) y Playwright 1.58.0, sin `PLAYWRIGHT_BROWSERS_PATH`.

## A. §2.3 Redacción del volcado con Chromium real

- Script desechable en el scratchpad: lanza Chromium 1243 del contenedor y carga una página con:
  - un campo de contraseña que contiene una clave;
  - un textarea de varias líneas con el token;
  - un enlace a `https://<IP>.sslip.io`;
  - un párrafo que muestra el token;
  - un `console.log` con el token, la URL del host y una cookie.
- Después llama a `dump_failure` con el token como secreto privado.

| Archivo | Token | Contraseña | IP pública del host | Texto de cookie |
|---|---|---|---|---|
| aria.yaml | no | no | **sí** (`/url:`) | no |
| console.json | no | no | **sí** (`websockets`, texto de consola) | sí, solo porque la propia página lo imprime |
| controls.json | no | no | no (solo guarda `path`) | no |
| screenshot.png | no (bytes) | no (el campo se pinta enmascarado) | no | no |

- El volcado no pide cabeceras ni cookies a Playwright. La IP y el UA del canario están en `private` y se redactan.
- Lo que el código no puede redactar es el contenido visual de `screenshot.png`. El campo «Invitación» es
  `type='password'` (`src/ui/views/study_page.py:30`), así que hoy se pinta enmascarado. Si un futuro
  campo de texto mostrara secretos, aparecerían en la imagen.

## B. §2.4 Timeouts del navegador (`i7_browser.py`, producción / `local_test`)

Fuente: Playwright 1.58.0. Referencias: `_helper.py:306-335` (orden explícito → defecto de página → 30000)
y `_connection.py:648-656` (una llamada sin calculador no envía `timeout`).

| Línea | Operación | Producción | local_test | Nota |
|---|---|---|---|---|
| 200 | `chromium.launch` | 180000 | 180000 | defecto de `launch` |
| 208-209 | `new_context`, `new_page` | sin tope | sin tope | teórico |
| 227 | `set_default_timeout` | 60000 (antes 600000) | 20000 | |
| 262 | `page.goto` | 60000 (antes 600000) | 20000 | hereda la navegación; `wait_ready` va antes |
| 264, 108-113 | `fill`, `blur`, `press`, `input_value`, `evaluate` | 60000 | 20000 | |
| 99-100 | `option`: `click`, `is_checked` | 60000 | 20000 | |
| 121 | `visible()` | 90 s (`Probar consulta`: 900 s) | 25 s | sondeo con `is_visible` |
| 285, 291 | `Continuar`, `Guardar respuestas del bloque` | 90 s (antes `wait_for` 600000) | 25 s | aparecen en el mismo rerun |
| 297 | `groups.nth().wait_for` | 60000 (antes 600000) | 20000 | |
| 256-258 | `click('Probar consulta')` | `settled` 600 s + `visible` 900 s + `click` 900000 ms + `until` 900 s | ≈45 s | **3300 s en el peor caso** |
| 256-258 | `click()` normal | 600 + 90 + 60 + 600 = 1350 s | ≈70 s | |
| 243-250, 326-331 | `wait()` y bucle de exportación | 600 s + 1 `observe` (≈180 s) | 25 s | se comprueba el plazo antes de `observe` |
| 76, 121 | `idle()` → `count()`, `is_visible()` | sin tope | sin tope | sin espera, pero bloquean si el renderer se cuelga |
| 156-180 | volcado: `input_value`, `inner_text`, `is_disabled`, `get_attribute`, `screenshot`, `aria_snapshot` | 5000 c/u | — | `.all()` y `all_inner_texts` sin tope |
| 351 | `browser.close()` | sin tope | sin tope | en el `finally` |

- No queda ninguna operación de Playwright en 600000 ms.
- `expect()` no se usa.
- La cota superior teórica de un smoke completo en producción es ≈31 h. En la práctica las esperas duran
  milisegundos.

## C. §2.5 Rama y commit fijados

| Sitio | Qué fija | ¿Falla con HEAD 3bb9ca8 en fix/i7-attempt8 y el remoto en b470d3e? |
|---|---|---|
| `i7_cpu_pipeline.py:42-46` | `ls-remote origin refs/heads/fix/interview-readiness == HEAD == config.commit` | **Sí**: `PUBLISHED_I7_SOURCE_REQUIRED`, antes de pagar (ejecutado) |
| `i7_cpu_pipeline.py:90-91` | clona la rama **local** `fix/interview-readiness` después de `admit_paid` | Si se llega aquí: rama local obsoleta → construye el commit viejo; rama local ausente → rc 128 (I8-01, ejecutado) |
| `build_context.py:56-61,69-71` | `branch --show-current` y `ls-remote` sobre el **clon** | No compara con GitHub, solo con el clon local |
| `i7_workflows.py:35-45` (`verify_tool_source`) | `HEAD == config.commit`, o un diff limitado a `docs/` y `output/audit/`; árbol limpio en scripts, src y config | Falla si la raíz se creó en otro commit (ejecutado: `OPERATOR_SOURCE_COMMIT_DIFFERS`) |
| `i7_readiness.py:10-25` | `source_sha256` de los bytes (incluye `tests/`) | El recibo del 16/10 no sirve; hace falta uno nuevo |
| `i7_config.py:24,58` | `commit = git rev-parse HEAD` al crear la raíz | Requiere una raíz nueva en 3bb9ca8 |
| `i7_streamlit_preflight.py:55-58` | `proof.commit == installation.commit` | Solo después del build pagado |
| `i7_admission.py:62-73,195` | el remoto `fix/interview-readiness` es el anexo, y solo `docs/` y `output/audit/` cambian desde la imagen | Afecta al gate, no al cpu-build |
| `cpu_build.py:300-309`, `installation.py:90`, `installation_upgrade.py:137`, `unattended_diagnostic.py:154`, `manage_gate_window.ps1:328` | `fix/interview-readiness` | Fuera de la ruta del cpu-build de I7 |
| `config/i7-build-recipe-pin.json` | SHA de 5 archivos, entre ellos `build_context.py` | Coincide en 3bb9ca8 (sha256sum ejecutado) |
| tests/ | todos usan git falso | La capa 2 no detecta el problema de la rama |
| `.github/workflows/ci.yml:5-11` | solo `main` y `summer/*` | Ninguna rama del intento 8 pasa por CI |
| docs: `STUDY_I7_RUNBOOK_FASE_B.md:9,41-43`, `STUDY_I7_IMAGE_RECIPE.md:6`, `.harness/progress.md:1,101` | «solo fix/interview-readiness»; `$run` y `$evidence` antiguos | Contradicen fix/i7-attempt8 (I8-08) |

Simulación ejecutada con un remoto bare (escenarios):

| Escenario | Resultado |
|---|---|
| A: local fix/i7-attempt8, remoto en b470d3e | :46 corta antes de pagar |
| B: rama local fix/interview-readiness en 3bb9ca8 sin push | :46 corta antes de pagar |
| C: remoto avanzado, sin rama local | pasa :46; :90 da rc 128 después de pagar |
| D: remoto avanzado y checkout de fix/interview-readiness en 3bb9ca8 | pasan todas las guardas de git |
| E: remoto avanzado, rama local obsoleta | pasa :46; :90 clona b470d3e sin aviso |

## D. §2.6 Mutación (37 mutantes; tests de 3bb9ca8: `test_study_i7_attempt8.py` y `test_study_i7_smoke_code.py`)

- Cada mutante se aplicó en el checkout, se ejecutaron los dos archivos de tests y se restauró con `git checkout`.
- Al terminar, `git diff --quiet -- scripts src` confirmó que el árbol volvía a estar limpio.

| Mutante | Barrera rota | Resultado con los tests de 3bb9ca8 | Con `test_review_i8_mutants.py` |
|---|---|---|---|
| M01-M08, M10, M12, M14-M16 | código ≥ P900, validación en `checked_config`, regex/env de la app, `assignment`, env, seed, observe, stop en except, stop en tls, withdraw/restore configurables | mueren | — |
| M18, M21, M22, M24, M27, M29, M30, M33-M36 | redacción ARIA del valor, `scrub` de aria/console, botones visibles, 60000, tope de 90 s, error de socket, `dump-errors`, orden validar→stop, `participant=` | mueren | — |
| **M09** | `i7-smoke-status` exige el código configurado | **sobrevive** | sobrevive (necesita una sesión completa; no se añadió test) |
| **M11** | `except BaseException` → `Exception` en el smoke (KeyboardInterrupt) | **sobrevive** | muere |
| **M13** | lo mismo en privacy | **sobrevive** | muere |
| **M17** | redacción de las líneas hijas de un textbox | **sobrevive** | muere |
| **M19** | valores de los campos como secretos | **sobrevive** | muere |
| **M20** | variante normalizada en espacios | **sobrevive** | muere |
| **M25** | `mkdir(exist_ok=False)` del volcado | **sobrevive** | sobrevive (colisión en el mismo microsegundo; impacto nulo) |
| **M26** | `set_default_timeout(5000)` del volcado | **sobrevive** | muere |
| **M28** | token en los secretos de `run()` | **sobrevive** | muere |
| **M31** | `confirm(participant_id, participant)` | **sobrevive** | sobrevive (solo se ejerce en el recorrido real del navegador) |
| **M32** | espera a `Continuar` | **sobrevive** | sobrevive (ídem) |
| **M37** | roles combobox, spinbutton y otros en `VALUE_ROLE` | **sobrevive** | muere |
| M23 | `path` de controls | no aplicado (el patrón no coincidió) | — |

## E. §2.7 Regresiones

- La suite completa queda en exit 0 sobre 3bb9ca8.
- Los 73 tests de `origin/claude/review-2026-11-02` dan sobre 3bb9ca8 lo mismo que sobre b470d3e (58 pasan, 15 xfail): F-01, B-3 y B-4 no cambian.
- Cambios de comportamiento en smoke → restore → privacy → withdraw-purge:
  - (1) El smoke y privacy fallidos ahora paran la VM. Antes la dejaban RUNNING hasta el STOP nativo.
  - (2) `tls` hace `stop()` antes del primer `start()`. Arregla `STOPPED_VM_AND_SPECIFIC_RESERVATION_REQUIRED` tras `ready`.
  - (3) restore y withdraw usan `smoke_participant(config)`, que por defecto sigue siendo P999.
- En el gate, el `stop()` del except del smoke va seguido del `finally: stop(); guard()` (`i7_phase_b.py:151-153`). `stop()` es idempotente, así que no hay regresión.
