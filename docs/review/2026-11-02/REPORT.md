```
[Claude Code, auditor] | 2026-11-02 | rama `claude/review-2026-11-02` desde `fix/interview-readiness` @ `b470d3e` | tarea: auditoría del operador I7 (sin F-NNN nueva) | modo: sondeo
Capas (PLAYWRIGHT_BROWSERS_PATH sin definir): 1 secretos=exit 0 | 2 tests=exit 0 (2140 pasan, 0 fallan, 57 omitidas por entorno, 14 xfail estrictos de la auditoría) | 3 docs=exit 0
Capas (con PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers del contenedor): capa 2 = exit 1, 11 fallos de entorno (R-08)
Cambios: docs/review/2026-11-02/**, tests/test_review_2026_11_02_*.py (solo altas) | Sin verificar: Windows, Docker, GCP, GPU, PowerShell, Ollama real
```

# Auditoría del operador I7: 2026-11-02 (base b470d3e)

## 1. Acciones realizadas
- **Fase 0 [VERIFICADO].**
  - `git rev-parse origin/fix/interview-readiness` da `b470d3e76fe569fd63ee98b1000c6d75817cc964`, el HEAD esperado.
  - `git merge-base HEAD origin/fix/interview-readiness` da el mismo hash.
  - ADR-0025 y ADR-0026 están presentes.
  - `i7.py:279` contiene `.replace(microsecond=0)`.
  - Detalle en `PLAN.md`.
- **Pasadas.**
  - P1, mía: `period_id` tras el smoke e invariantes duplicadas.
  - P2, subagente A: dinero, TTL y presupuesto.
  - P3, subagente B: recuperación e idempotencia.
  - P4, subagente C: seguridad del operador y de `src/ui`, con bandit.
  - P5, mía: re-validación del informe anterior.
  - P6: tres refutadores independientes que no vieron mi razonamiento, uno para F-01, otro para A-2 y otro para B-1/2/3/4/6.
  - Además: ruff (F, E9, B) y los escáneres de la capa 1.
- **Archivos.**
  - En `docs/review/2026-11-02/`: `PLAN.md`, `DECISIONES.md`, `PROGRESO.md`, `findings.json`, este `REPORT.md` y tres parches en `patches/`.
  - Tests nuevos:
    - `tests/test_review_2026_11_02_period_id.py`: F-01.
    - `..._prior_findings.py`: R-01 y R-02.
    - `..._prior_open.py`: R-03, R-04, R-05, R-06 y R-09.
    - `..._money.py`: A-1 a A-4.
    - `..._recovery.py`: B-1 a B-6.
- **Zona congelada.** No se editó. Los parches se probaron en un worktree desechable del scratchpad.

## 2. Auditoría y trazabilidad
- **Rama base / HEAD inicial / HEAD final:** `fix/interview-readiness` / `b470d3e` / ver `git log` de `claude/review-2026-11-02`.
- **Árbol:** limpio tras el commit final. No hay push: espera el OK de Enzo (ADR-0013).
- **Pruebas:** 2211 recogidas: 2140 pasan, 0 fallan, 57 omitidas por entorno y 14 xfail estrictos.
  - Las omisiones se reparten así: 18 del SDK o PuTTY de Windows, 11 de Playwright, 5 de PowerShell, 4 de artefactos `data/indices`, 2 de modelos no descargados y el resto de contadores de Windows.
  - Tres ejecuciones repetidas de los tests nuevos dieron resultados idénticos.
- **Linters, secretos y dependencias:**
  - ruff F/E9/B sobre el alcance: 15 avisos y ningún defecto real (los B023 son cierres que se invocan en la misma iteración).
  - Capa 1: exit 0. Tiene los huecos de R-03.
  - `git diff --check`: limpio.
  - bandit: 123 avisos, todos falsos positivos o aceptados.
  - pip-audit: NO EJECUTADO en esta sesión. La cifra de R-15 viene del informe anterior y es [DECLARADO].
- **Entorno:** Python 3.14.6 instalado con uv en Linux, en lugar del 3.14.3 de Windows. `requirements-app.txt` se instaló con el índice CPU de PyTorch.

## 3. Hallazgos confirmados (por severidad)

| id | sev | archivo:línea | resumen | evidencia | etiqueta | confianza | corrección |
|---|---|---|---|---|---|---|---|
| F-01 | **S1** (refutador: S2) | `i7.py:368-371`, `i7_cpu_pipeline.py:319` | `start('smoke')` persiste el `period_id` del smoke y `cleanup_cpu` deriva de él el nombre esperado de la VM. `close` falla con `CPU_CLEANUP_IDENTITY_CHANGED` | `pytest --runxfail tests/test_review_2026_11_02_period_id.py` → `ValueError: CPU_CLEANUP_IDENTITY_CHANGED` | VERIFICADO | alta | `patches/001` |
| B-4 | **S1** | `i7_cpu_pipeline.py:266-295`, `i7.py:429-432` | Si `gpu_restore` falla después de `reserve(13.5)`, `guard()` describe la VM placeholder `id '0'` y `close` nunca borra la reserva | `close raised: NOT_FOUND instances/cloudrag-i7-unallocated`, sin ningún `reservations delete` | VERIFICADO | alta | `patches/002` |
| B-3 | **S1** | `i7_cpu_pipeline.py:317-348` | `cleanup_cpu` no se puede reanudar: el reintento falla en el `describe` de la VM ya borrada | `retry raised: NOT_FOUND instances/...`, y el disco de servicio y el snapshot de staging siguen existiendo | VERIFICADO | alta | `patches/003` (se aplica encima de 001) |
| A-1 | S2 | `i7.py:149-151`, `i7_cpu_pipeline.py:59,118` | La cota de CPU (3,95 USD) queda en `spent_usd`. Si la reserva 1 se cobra más de 13,105 h, el gate se bloquea con `I7_BUDGET_STOP` | `held 13.25 h -> gate I7_BUDGET_STOP` | VERIFICADO | alta | Documentar el límite y comprobarlo antes del gate |
| A-2 | S2 | `i7_costs.py:119-131`, `i7_phase_b.py:99-116` | No hay comprobación anticipada del corte de 24 h. ADR-0026 y el runbook prometen reintentar en la misma raíz sin mencionar ese límite | `gate first run: I7_RETENTION_DEADLINE...`, después `GATE_ATTEMPT_EXISTS_NO_REPLAY` | VERIFICADO | alta | Admisión anticipada en `reserve()`, gate y series, y una nota en ADR-0026 |
| A-3 | S2 | `i7.py:286,306,451` | `open_exposure_usd` es un escalar: la exposición de CPU se pierde si falla el STOP del `finally` de `clean_snapshot` | `gate ADMITTED although correct ledger total = 20.2792` | VERIFICADO | media | Llevar la exposición en un diccionario por origen |
| B-1 | S2 | `i7_workflows.py:82-84` | La INTENT se escribe antes de las comprobaciones de `start()` y un rechazo sin coste marca la ranura como terminal | `TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW` | VERIFICADO | alta | Escribir la INTENT justo antes de `add-metadata` |
| B-2 | S2 | `i7_workflows.py:112-114`, `i7_phase_b.py:151-153` | `finally: stop(); guard()` oculta la causa real (`KeyError`, `ALREADY_CLOSED`) y no llega a `guard()` si `stop()` falla | tests `b2a`, `b2b`, `b2c` | VERIFICADO | alta | `try`/`finally` anidado y `guard()` condicionado |
| B-5 | S2 | `i7.py:112-114` | `persist()` escribe los dos archivos sin atomicidad conjunta | test `b5` | VERIFICADO | alta | `persist_seq` compartido |
| B-6 | S2 | `i7_phase_b.py:102-106` | El gate reutiliza una reserva abierta sin comprobar su TTL y escribe el intento único antes de esa comprobación | `boots started: 0`, después `GATE_ATTEMPT_EXISTS_NO_REPLAY` | VERIFICADO | alta | Exigir una reserva nueva o comprobar el TTL antes del intento |
| C-1 (=R-10) | S2 | `backup_agent.py:55-61`, `service_gateway.py:18-25` | Los agentes root siguen symlinks del contenedor (uid 10001) | `agentC/test_c1_backup_symlink.py` | VERIFICADO | alta | `O_NOFOLLOW`/`O_EXCL`, o un directorio de root |
| C-2 (=R-14) | S2 | `service_gateway.py:82-91,195` | Un diferencial JSON entre Python y Go permite saltar las opciones congeladas de Ollama | decodificador Go con las etiquetas de Ollama: `temperature:1.5` | VERIFICADO (sin el Ollama real) | media | Reenviar `json.dumps(validado)` y rechazar claves duplicadas |
| R-03 | S2 | `scan_secrets.py:48-66,183-203` | La capa 1 no detecta `ya29.`, `AIza`, `hf_` ni `github_pat_`, y omite `.pem`, `.key` y `.env.*` | 7 xfail en `prior_open` | VERIFICADO | alta | Patrones e inspección con aviso |
| R-04 | S2 | `recompute_retrieval_stats.py:200-217,311` | Sobrescribe `retrieval_metrics.json` firmado sin `guard_write` | xfail | VERIFICADO | alta | `guard_write` |
| R-05 | S2 | `compute_exp17_powered.py:46,82` | El bootstrap depende de `PYTHONHASHSEED`: p = 0,236 con la semilla 42 y 0,2555 con la 7 (n_boot = 2000); el valor guardado es 0,2329 | xfail con subproceso | VERIFICADO | alta | `ensure_hashseed` o un orden fijo en un `_vN` nuevo |
| R-06 | S2 | 11 escritores de exp15 a exp19b | `guard_write` es opcional: 11 de 11 escritores no lo usan | xfail | VERIFICADO | alta | `guard_write` obligatorio |
| R-08 | S2 | `init.sh:19-26` | Con cualquier `PLAYWRIGHT_BROWSERS_PATH` se exige el navegador de I7 | 11 fallos en la línea base | VERIFICADO | alta | Exigirlo solo si existe el Chromium fijado |
| R-09 | S2 | 12 literales en 8 módulos | La rama no tiene una única fuente de verdad (hoy todas las copias coinciden) | xfail y control que pasa | VERIFICADO | alta | Una constante única |
| R-15 | S2 | `Dockerfile:8-17` | La imagen instala un lock de Windows sin hashes | 0 `sha256` en `requirements-lock.txt` | VERIFICADO | alta | Lock de Linux con `--require-hashes` (necesita OK) |
| R-16 | S2 | 43 archivos con el ID de proyecto, 24 con el usuario de Windows, 12 con el correo de la cuenta de servicio | Identificadores en el repo público | `git grep` | VERIFICADO | alta | Decide Enzo (ADR-0011) |
| A-4, A-5, B-7, C-3, C-4, M-1 | S3 | ver `findings.json` | `--hours 12` obsoleto, `fromisoformat` sin comprobar la zona, `live-contexts` con `'x'`, comillas Unicode en PowerShell, correo ACME, ruff | | mixta | | |

**Nota sobre F-01 (desacuerdo de severidad, no oculto).**
- **Argumento del refutador para S2:**
  - Falla de forma visible.
  - La reserva ya está borrada cuando falla.
  - El coste es de unos 1-1,5 USD al día [ESTIMADO].
  - Se recupera editando `period_id` a mano o borrando con gcloud.
- **Por qué lo mantengo en S1:**
  - La salida a mano contradice ADR-0026 («la raíz no se edita a mano»).
  - Ningún runbook documenta esa salida.
  - Se dispara justo en los caminos de fallo en los que `close` hace más falta.
- **Decisión pendiente:** Enzo elige la severidad.

## 4. Hallazgos refutados o descartados
- **R-01: REFUTADO en esta base.**
  - `reserve()` envía `delete_at` sin fracción de segundo.
  - Pasan 21 combinaciones de microsegundos (0 a 999 999) con desfases de 0, -7 y +5,5 h.
  - El mutante sin redondeo falla en 3 casos, así que el test sí detecta el defecto.
- **R-02: OBSOLETO** gracias a ADR-0026.
  - Un rechazo con listado vacío libera horas y exposición, y permite reintentar.
  - El residual está documentado en ADR-0026 §4: si el listado falla, la admisión queda UNKNOWN. Lo deja fijado `test_r02_residual_*`.
- **A-2, B-1, B-2 y B-6 como S1: rebajados a S2.**
  - A-2: el corte de 24 h es el tope documentado (runbook :261-263, `STUDY_I7_COST_LEDGER.md:26-27`).
  - B-1: el tope acumulado ya impide reanudar.
  - B-2c: `close` y `guard` lo recuperan.
  - B-6: necesita un fallo previo, y `reconcile` lo neutraliza.
- **Otras lecturas de `period_id` tras el smoke (`reserve`, `schedule`, `host_runtime`):** son nombres y rutas por periodo y se comportan como se pretende. El marcador de propiedad sigue ligado al periodo técnico.
- **`tls()` sin `stop()`:** en esta base `tls()` (`i7_phase_b.py:156-169`) llama a `stop()` en el `finally` de cada iteración. El arreglo local no publicado queda fuera de alcance.

## 5. Áreas verificadas sin hallazgos
- **Intent de reserva:** se persiste antes del create.
  - La reversión de ADR-0026 restaura el estado si falla `persist`.
  - `record_intent` va antes de cada create.
  - Los bordes `<=` 13,5 y 18,5, `>24` y `<` TTL coinciden con las ADR.
- **`reconcile`:** no cobra dos veces. `guard` no puede cobrar más allá de la vida de la VM, por el STOP nativo de 3 h y el TTL de más de 3 h 01 min.
- **Locking:** `i7.main` e `i7_phase_b.main` toman `i7.lock`. No hay mutación concurrente.
- **Inyección y rutas:**
  - Los `--command=` usan `shlex.quote` o son fijos.
  - Todos los IDs se validan con `re.fullmatch`.
  - Las invitaciones se guardan como hash SHA-256 y el token sale una sola vez por TTY.
  - Los recibos de nube no guardan salida privada.
- **Modo participante:** sin `unsafe_allow_html`.
- **Invariante P999:** las 20 apariciones coinciden y la política de `smoke` en `study_sessions.py:103` es coherente. No hay origen único (S3, sin defecto actual).

## 6. Crítica de diseño y resiliencia
- **El patrón dominante es «registro de un solo uso antes de las comprobaciones previas al efecto».** Aparece en `gate-cohort-intent`, `stimulus-NN` y `live-contexts`. Hace que un rechazo gratuito consuma un intento irrepetible (B-1, B-6, A-2). La regla debería ser «registrar justo antes del primer efecto pagado».
- **`close` no es una operación total.**
  - Depende del estado de `config` (F-01).
  - No se puede reanudar (B-3).
  - No contempla una reserva sin VM (B-4).
  - Para la última línea de defensa del gasto, conviene un cierre por inventario: listar por marcador y borrar por identidad.
- **El ledger mezcla cotas y gasto.** `spent_usd` contiene cotas conservadoras que nunca se reemplazan (A-1), y `open_exposure_usd` es un escalar (A-3). Hoy lo compensa el plan independiente de `i7_costs`, pero la doble barrera es más débil de lo que parece.
- **Relojes.** Todo es consciente de la zona salvo `i7.py:503` (A-5, inalcanzable hoy). No hay ningún test de un salto de reloj en Windows: NO EJECUTADO.

## 7. Pruebas y evidencia

| Test | Antes, sin parche | Después, con parche (worktree desechable) |
|---|---|---|
| `test_cleanup_cpu_after_smoke_still_recognises_the_cpu_of_this_root` | xfail; con `--runxfail` lanza `CPU_CLEANUP_IDENTITY_CHANGED` | Pasa con 001, sin el marcador |
| `test_b4_*` (caracterización) | Pasa: el defecto existe | Falla como se espera con 002. `close` devuelve `BLOCKED_UNKNOWN_RESOURCE` con `RESERVATION_DELETED_VERIFIED` |
| `test_b3_*` (caracterización) | Pasa: el defecto existe | Falla como se espera con 003 |
| Suite completa con 001, 003 y 002 aplicados | | 0 regresiones. Solo fallan los dos tests de caracterización invertidos y `test_operational_measurement::test_fresh_refuses_legacy_output_and_checkout`, que también falla sin parches en ese worktree (R-39, entorno) |

Los tests `money`/`recovery` que fijan el comportamiento actual se tienen que invertir al aplicar cada corrección. Sus docstrings lo indican.

## 8. Estado y veredicto
- **Veredicto: NO-GO condicionado** para una nueva raíz de la Fase B mientras no se apliquen los parches 002 (B-4) y 001 + 003 (F-01 y B-3), o al menos mientras el runbook no documente el cierre manual por identidad para esos tres caminos.
- Ningún hallazgo es S0. Ninguno permite superar el tope de 20 USD del plan.
- Los tres S1 dejan recursos facturando sin que la herramienta los cierre, en los caminos de fallo, que ya ocurrieron en el intento 6.
- Aplicar los parches toca `scripts/`, así que según ADR-0025/0026 hacen falta un commit nuevo, las suites de Windows, Linux e I7, una raíz nueva y un cpu-build nuevo. La decisión es de Enzo.

## 9. Alcance no ejecutado
- Windows: 18 tests del SDK/PuTTY y 5 de PowerShell omitidos.
- Docker, GCP y GPU.
- El binario real de Ollama (C-2 se verificó con un decodificador Go equivalente).
- PowerShell (C-3).
- pip-audit.
- Mutation testing sistemático: solo se hizo el mutante de R-01.
- El arreglo local de `tls()`.
- Fuera de alcance a propósito: el resto de `scripts/` (salvo R-03 a R-06) y `scripts/cloud_entrypoint.py`.
- El presupuesto de 3 h y 6 pasadas se cumplió: P1 a P6.
