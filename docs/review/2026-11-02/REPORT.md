```
[Claude Code, auditoría autónoma] | 2026-11-02 04:05 UTC | rama `claude/blissful-euler-0a3xkp` @ `<HEAD final, ver §2>` | tarea F-017 | modo: ejecución (solo docs/review y tests/review_2026_11_02)
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (1815 pasan [1813 + 2 controles nuevos], 0 fallan, 57 omitidas, 26 xfail estrictos nuevos) | 3 docs=exit 0   [con `env -u PLAYWRIGHT_BROWSERS_PATH`, ver R-08]
Cambios: docs/review/2026-11-02/** (PLAN, DECISIONES, PROGRESO, findings.json, REPORT, 4 parches), tests/review_2026_11_02/** (4 archivos), .harness/{feature_list.json,progress.md}
Sin verificar: GCP real (eco de deleteAtTime de la API), Windows/PowerShell nativo, Docker, GPU, Ollama, modelos HF, suite slow con artefactos de data/
```

# Auditoría 2026-11-02: informe final

Alcance: todo el repositorio, ordenado por riesgo ([ALCANCE] vacío). Presupuesto: 3 h y 6 pasadas (valores por defecto),
consumido en ~2 h 45 min. Base: `main` @ `8fdaa65`. El registro completo está en [findings.json](findings.json) (40 hallazgos).

## 1. Acciones Realizadas

- **Fase 1, baseline.** Ejecuté `init.sh` tres veces: sin dependencias, con el venv 3.14 y sin la variable de navegador del contenedor. Ejecuté también la suite completa. Hice un inventario de 94 kLOC (study_operator ~14,7 k; src ~17 k; scripts ~26,6 k). Ver [PLAN.md](PLAN.md).
- **Fase 2, plan y crítica previa.** Ver [PLAN.md](PLAN.md). Las decisiones de nivel 1 (D-01 a D-06) están en [DECISIONES.md](DECISIONES.md).
- **Fase 3.** Corrí cinco subagentes en paralelo, todos en solo lectura:
  - A: dinero, límites y recuperación en `study_operator`.
  - B: seguridad, con bandit.
  - C: lógica y reproducibilidad en `src/` y en los scripts que producen cifras.
  - D: invariantes duplicadas y coherencia entre docs y código.
  - E: calidad de los tests, ruff, mypy, pip-audit, cobertura, flakes (más de 200 corridas) y mutación (33 mutantes).
- **Fase 4.** Cinco subagentes adversariales independientes intentaron refutar los 11 S0/S1 propuestos (A-01, A-02, D-01, D-02, C-01, C-03, C-04, B-01, B-02, E-01). Ver §4.
- **Reproductores añadidos**, todos `xfail(strict=True)` según D-04: 26 xfail y 2 controles que pasan:
  - `tests/review_2026_11_02/test_i7_reservation_lifecycle.py`: R-01, R-02 y un control que pasa.
  - `tests/review_2026_11_02/test_scan_secrets_gaps.py`: R-03, 8 casos y un control que pasa.
  - `tests/review_2026_11_02/test_evidence_writers.py`: R-04, R-05 y R-06, 13 casos.
  - `tests/review_2026_11_02/test_operator_cli_receipt.py`: R-07, 3 casos.
- **Parches propuestos.** No los apliqué: tocan zona congelada o productores de evidencia. Cada uno se verificó en un `git worktree` temporal: sus xfail pasan a XPASS y no hay regresiones en la suite CPU.
  - [001](patches/001-i7-reservation-ttl-and-rejected-create.patch): R-01 y R-02.
  - [002](patches/002-evidence-writers-guard-and-exp17-order.patch): R-04, R-05 y parte de R-06.
  - [003](patches/003-scan-secrets-gaps.patch): R-03.
  - [004](patches/004-operator-cli-shadowed-run.patch): R-07.

## 2. Auditoría y Trazabilidad

- **Rama base, HEAD inicial y final:** `main` / `8fdaa65` / último commit `docs(review)` de esta rama (ver `git log`). No se publicó nada: el usuario eligió publicar solo al cierre (D-06, ADR-0013).
- **Estado del árbol:** limpio después del commit final.
- **Pruebas:**
  - Baseline con `env -u PLAYWRIGHT_BROWSERS_PATH`: `init.sh` da exit 0 en las tres capas (1813 pasan, 57 omitidas).
  - Con la variable del contenedor: capa 2 con exit 1, 11 fallos falsos de navegador (R-08).
  - Suite completa: 1 fallo de entorno (R-21) y 60 omitidas. Hay 33 omisiones solo-Windows o PowerShell, 2 por falta de Playwright y Chromium fijado, y 5 por modelos o índices de `data/`. Todas cuentan como **omitidas por entorno**, no como aprobadas.
  - Estado final: 1815 pasan (1813 existentes + 2 controles), 0 fallan y 57 omitidas, más 26 xfail estrictos nuevos.
  - La suite no depende del orden de ejecución, del hash seed (0/123) ni de la zona horaria; probé 8 variantes, también con pytest-randomly.
- **Linters:**
  - ruff: 150 avisos, informativo en CI (F-009). Ninguno es un bug real salvo R-07.
  - mypy: confirma R-07.
  - bandit: 140 resultados, todos falsos positivos (tabla de la pasada B, resumida en §5).
- **Escaneo de secretos:** PASS, 0 activos y 6 en baseline. Ese PASS hay que leerlo junto a R-03.
- **Dependencias:** pip-audit da **FAIL informativo**. Son 116 avisos en `requirements-app.txt` (CI) y 275 en `requirements-lock.txt` (la imagen del estudio). El único claramente alcanzable es tornado, el servidor de Streamlit (R-15).

## 3. Hallazgos Confirmados (ordenados por severidad)

| id | sev | archivo:línea | resumen | evidencia | etiqueta | confianza | corrección propuesta |
|---|---|---|---|---|---|---|---|
| R-01 | **S0** | `scripts/study_operator/i7.py:108,120,279` | `reserve()` fija `until` con microsegundos y lo compara por igualdad exacta con `deleteAtTime`; gcloud trunca `--delete-at-time` a ms (SDK 570, `reservations/util.py:322`, formato `%3f`). En el primer gpu-restore real la reserva L4 se crea y factura, pero queda fuera del estado y sin `guard` (KeyError) hasta su TTL de 12 h (~8,5 USD de lista). Después aplica R-02 | `test_reserve_records_reservation_when_clock_has_microseconds` falla con `RESERVATION_NATIVE_DELETION_NOT_VERIFIED`; el control con ms pasa | [VERIFICADO] SDK; [SUPUESTO] eco de la API | alta / media | Parche 001: `until` a segundos enteros y tolerancia < 1 s |
| R-02 | S1 | `i7.py:102-110,286-289` | Un create rechazado por stockout ya consumió 12 de las 17 h, queda UNKNOWN 12 h y luego suma 8,73 USD fantasma. Después `reserve(12)` da `I7_BUDGET_STOP`: gpu-restore queda bloqueado | `test_rejected_reservation_create_does_not_consume_plan` | [VERIFICADO] | alta | Parche 001: marcar el rechazo, cerrar tras un list vacío y devolver las horas |
| R-03 | S1 | `scripts/scan_secrets.py:48-66,184-203,335` | La capa 1 omite en silencio .pem, .key, .env.*, credentials.json, secrets.toml y UTF-16. No reconoce ya29., AIza, hf_, github_pat_, gho_ ni `db_password`. No tiene detector de IP ni de correo | 8 casos en `test_scan_secrets_gaps.py` | [VERIFICADO] | alta | Parche 003. El baseline por patrón queda para decisión del dueño |
| R-04 | S1 | `scripts/recompute_retrieval_stats.py:311`, `scripts/audit/smoke_recompute_retrieval_stats.py:66-70` | El smoke, sin variables, reescribe `exp8` y `exp8b/retrieval_metrics.json` firmados sin `guard_write`. Hoy solo lo impide que el modelo no está en caché | Test de inventario y passC/r03 (el sha cambia en la copia) | [VERIFICADO] | alta | Parche 002: guarda antes de cargar el oráculo; smoke de solo lectura por defecto |
| R-05 | S1 | `scripts/compute_exp17_powered.py:46,82` | El bootstrap de exp17 depende de PYTHONHASHSEED. El artefacto no se reproduce con ninguna semilla de 0 a 150; el ledger cita «HHEM 0,23» y con 42 sale 0,240 | `test_exp17_cluster_bootstrap_is_independent_of_hash_seed` | [VERIFICADO] | alta | Parche 002: `sorted()` y salida `_v2`. Corregir la cifra requiere OK |
| R-06 | S1 | 11 scripts `compute_exp1*/tierA/rescore_nli_v3` (ver findings.json) | La guarda es opt-in: 10 de 11 escritores sobrescriben evidencia congelada con la invocación por defecto, 8 de ellos solo con CPU | 12 tests parametrizados | [VERIFICADO] | alta | `guard_write` en cada escritura y un test de inventario |
| R-07 | S2 | `scripts/study_operator/cli.py:67,93,120` | `run` queda sombreado: stimulus-* ejecuta la acción remota y luego lanza TypeError, sin recibo | 3 tests | [VERIFICADO] | alta | Parche 004 (`import run as stimulus_run`) |
| R-08 | S2 | `init.sh:26` | Con cualquier `PLAYWRIGHT_BROWSERS_PATH` activa REQUIRE: rojo falso en máquinas o imágenes con Playwright | 11 FAILED con la variable y 0 sin ella | [VERIFICADO] | alta | Exportar solo si existe `chromium-<pin>` |
| R-09 | S2 | `build_context.py:58`, `i7_cpu_pipeline.py:30` (10 sitios) | La rama `fix/interview-readiness` es obligatoria por nombre, y `progress.md:16` dice que borrarla «es seguro» | grep y lectura | [VERIFICADO] | alta | Constante o SHA únicos; corregir progress |
| R-10 | S2 alto | `backup_agent.py:55-61` | Un agente root escribe siguiendo symlinks en un directorio RW del contenedor | passB/repro_backup_symlink | [VERIFICADO] | alta | `O_EXCL\|O_NOFOLLOW` |
| R-11 | S2 | `i7.py:123,240,267` | `open_exposure_usd` es un escalar que se sobrescribe | passA/r03 | [VERIFICADO] | alta | Diccionario por clave |
| R-12 | S2 | `run_checkpoint.py:45-51,146` | TOCTOU entre la admisión y la escritura de exposición | passA/r04 | [VERIFICADO] | alta (prob. baja) | Escribir dentro del lock |
| R-13 | S2 | `retention_cleanup.py:175-182` (+6 sitios) | Recibos «inmutables» no atómicos | passA/r05 | [VERIFICADO] | alta (prob. baja) | tmp + fsync + `os.link` |
| R-14 | S2 | `service_gateway.py:79-89` | Diferencial de parser JSON entre Python y Go | passB/repro_gateway + Go | [VERIFICADO]/[DECLARADO] | alta | Rechazar claves duplicadas o con otra capitalización; allowlist |
| R-15 | S2 | `Dockerfile:8-17`, `requirements-lock.txt` | La imagen se construye sin hashes y con dependencias distintas de las que prueba CI; tornado con CVE alcanzable | pip-audit | [VERIFICADO] | alta | Lock Linux con hashes (requiere OK) |
| R-16 | S2 | `titulacion/emails_lewis_y_gyt.md`, `cpu_build.py:23`, `deployment.py:112` | Correos de terceros e identificadores en el repo público | conteo de passB/D | [VERIFICADO] | alta | Decide el dueño (ADR-0011) |
| R-17 | S2 | `compute_faithfulness_metrics.py:213` | Los IC de exp12 dependen del orden de un set (con 42 coinciden 24/24) | passC/r02 | [VERIFICADO] | alta | `ensure_hashseed_at_startup(42)` |
| R-18 | S2 | `statistical_analysis.py:197-217` | Un NaN contamina BH en toda la familia | passC/r05 | [VERIFICADO] | alta | Rama zero_diff y rechazar p no finitos |
| R-19 | S2 | `bm25_index.py:111` | argsort no estable en BM25 | passC/r06 | [VERIFICADO] | media-alta | `kind="stable"`, solo en exp15+ |
| R-20 | S2 | `test_study_stimulus_supervisor.py:24-33` | Causa de F-016: presupuesto de 3 s para arrancar con spawn; 22 % de fallos con carga ~30 | passE/flake | [VERIFICADO] | alta | Ampliar presupuestos solo en el test |
| R-21 | S2 | `test_local_model_resolution.py:89-110` | Falla en vez de omitirse sin `data/indices`; con artefactos ejecuta modelos | Suite completa | [VERIFICADO] | alta | Skip por artefactos; comparar solo la recuperación |
| R-22 | S2 | `test_decline_rule.py:221`, `test_signed_evidence_guard.py:106` | Las guardas de evidencia se degradan a skip | passE (exp18 movido en una copia) | [VERIFICADO] | alta | `pytest.fail` o manifiesto |
| R-23 | S2 | `pricing.py:23,29,66,115`, `policy.py:26-86` | 11 mutantes no equivalentes sobreviven | passE/mutate.py | [VERIFICADO] | alta | Un test por superviviente |
| R-24 | S2 | `run_runtime_noise_probe.py:79-96` | `verdict()` contradice ADR-0012 en la rama c | passD | [VERIFICADO] | alta | Usar el IC en la regla |
| R-25 | S2 | `adr-log.md:113` frente a `run_exp19b_pipeline.py:12-17` | ADR-0010 está obsoleto | lectura | [VERIFICADO] | alta | ADR nuevo |
| R-26 | S2 | `README.md:203` frente a `compute_retrieval_metrics.py:714` | El README promete una sobrescritura que la guarda rechaza | passD | [VERIFICADO] | alta | Corregir el README y adelantar la guarda |
| R-27 | S2 | `i7.py:41`, `capacity_probe.py:19` | ~101 literales GCP sin fuente única (hoy coinciden) | grep | [VERIFICADO] | alta | Importar desde region_scope |
| R-28 a R-40 | S3 | ver findings.json | Bordes de presupuesto `<=`/`>=`, literales duplicados, docs obsoletos, huecos de borde de la guarda, cobertura del 58 %, un test sin aserciones, ruff, `unsafe_allow_html` en modo desarrollo, efecto lateral `not-created/` de un test | — | [VERIFICADO] | alta | ver findings.json |

**Cifra entregada afectada:** solo R-05. `paper/summer_ablation_log.md:497` («HHEM 0,23») no se reproduce: da 0,24 con
la semilla 42 y 0,2375 con el orden fijo. El veredicto «el bootstrap conservador no cruza» se mantiene bajo
cualquier orden. No toca A.3 ni LACCI. No corregí el ledger: necesito tu OK y la regla append-only.

## 4. Hallazgos Refutados o Descartados

- **Severidad corregida por la verificación adversarial:**
  - A-01 subió de S1 a **S0** (R-01). El revisor encontró en el SDK la causa que lo vuelve casi seguro, no hipotético. También corrigió «bloqueo para siempre» a «hasta el TTL de 12 h».
  - C-03 bajó de S0 a **S1** (R-04): requiere un modelo descargado y git lo restaura.
  - D-01 bajó de S1 a **S2** (R-08): CI no usa init.sh y es un rojo falso, no un verde falso.
  - D-02 bajó de S1 a **S2** (R-09): falla en cerrado antes de tocar la nube. Además, la afirmación de que «faltan correcciones de main (640fcab)» es cierta pero irrelevante para lo desplegado: el fix de F-014 solo actúa fuera de `.worktrees/`.
  - B-01 bajó de S1 a **S2 alto** (R-10): requiere RCE previa en un contenedor endurecido y el atacante no elige el contenido.
  - E-01 bajó de S1 a **S2** (R-07): solo afecta al operador de iteración 5; el guest bloquea un nuevo arranque y collect guarda la evidencia antes de fallar.
- **Refutado en parte:** `rescore_nli_v3` dentro de R-06. La invocación por defecto crea un archivo nuevo, que la guarda permite; solo sobrescribe con `--variant vb_agree`.
- **Descartados:**
  - Los 140 resultados de bandit: MD5 de GCS, sockets con modo 660, `urlopen` sobre literales, argv en lista sin `shell=True`.
  - Los 28 B023 de ruff.
  - La «fuga» por las IP globales: son el rango público de IAP de Google.
  - Mezclas de unidades en pricing y costos: no hay ninguna.
  - Relojes sin zona horaria: todos fallan en cerrado.
- **Defecto de mi propio proceso:** el primer parche del escáner, tomado tal cual de la pasada B, rompía 4 tests existentes. Lo corregí: estrecho `gh[orsu]_` para que `ghp_` no cuente dos veces y retiro el baseline por patrón, que contradice un contrato vigente. Solo después lo entregué como 003.

## 5. Áreas Verificadas Sin Hallazgos

- **Métricas de recuperación:** nDCG@5 binario y graduado (a mano frente a sklearn, 0,477624), Recall, P@k, MRR (sin relevantes da 0,0), AP, RRF (1/(60+r) con r desde 1, un ID compartido suma una vez por lista).
- **Estadística:** BH con el paso de monotonicidad; recomputar los 12 p_raw del CSV inmutable de exp8 da diferencia 0,0. Bootstrap emparejado, d_z, Wilcoxon `zero_method` y TOST (pasada C, r09: 17/17).
- **NLI:** orden de etiquetas [contradiction, entailment, neutral] con softmax ([DECLARADO], según las tarjetas de los modelos).
- **Reproducibilidad de exp18 powered:** idéntico bit a bit con las semillas 42 y 3.
- **Guarda `signed_evidence`:** rechaza `..` y symlinks hacia archivos firmados; mutantes muertos salvo uno equivalente.
- **Unidades monetarias:** USD/h, GiB-mes/730, `baseUnitConversionFactor`, escalón máximo. Relojes en UTC con zona. Intención persistida antes de cada create en 8 flujos. `--max-run-duration=3h` en todo `instances create`.
- **Seguridad:**
  - Sin `shell=True` ni `os.system`; el SSH usa comando fijo, datos por stdin y host keys fijadas.
  - Rutas validadas por regex más comprobación del padre; tar con filtro.
  - Sin pickle, `yaml.load` ni `torch.load` en el alcance.
  - Tokens de invitación: solo se persiste el hash.
  - Aislamiento del contenedor: `--network none`, `--read-only`, `cap-drop ALL`.
  - Sin `verify=False`.
- **Invariantes que coinciden:** experimentos protegidos, pin de Chromium (1.58.0, revisión 1208, sha), uv 0.11.21 y ruff 0.15.5, regex de participante `P[0-9]{2,6}`, TTL de invitación de 24 h, Granite según ADR-0002, ADR-0003 (expansión apagada) y ADR-0005.
- **Suite de tests:** independiente del orden, del hash seed y de la zona horaria (8 variantes); no hay mocks tautológicos ni `assert True`.

## 6. Crítica de Diseño y Resiliencia

- **El patrón «intención antes del efecto» está bien implantado, pero su cierre es binario.** R-01 y R-02 muestran que, cuando el efecto falla o se verifica mal, la intención solo se resuelve esperando el TTL completo. El TTL nativo acota el dinero, pero no el bloqueo operativo. Hace falta un tercer estado: «rechazado explícitamente + list vacío ⇒ resuelto ya».
- **Las comparaciones de igualdad exacta contra el eco de la nube son frágiles.** R-01 es un caso de libro: un valor local más preciso que el canal de ida y vuelta. Los tests no lo detectan porque el reloj fijo cae a medianoche.
- **La protección de evidencia es opt-in, aunque su documentación dice lo contrario.** El registro invertido de ADR-0009 resolvió que la lista quedara obsoleta, pero no que se llame a la guarda. Solo la aplican 5 de unos 16 escritores (R-04, R-06). Un test de inventario lo convertiría en una garantía real.
- **La reproducibilidad depende de una variable de entorno.** PYTHONHASHSEED=42 aparece en init.sh y en los runbooks, pero dos scripts de cifras no la imponen (R-05, R-17). El primero ni siquiera reproduce su propio artefacto con 42.
- **La capa 1 promete más de lo que mide** (R-03). Al ser la compuerta de un repo público, una omisión silenciosa es peor que un rojo falso.
- **Escrituras sin atomicidad y carreras** (R-12, R-13): las probabilidades son bajas, pero la recuperación pide intervención manual. Es incompatible con un operador desatendido.
- **Deriva entre lo probado y lo desplegado** (R-09, R-15): CI no cubre ni la rama ni el lock que se despliegan.

## 7. Pruebas y Evidencia

- **Antes:** `pytest tests/review_2026_11_02 --runxfail` falla los 26 por la razón documentada. Por ejemplo:
  - `RESERVATION_NATIVE_DELETION_NOT_VERIFIED` y `UNKNOWN_CREATE_OUTCOME_PRESERVE_INTENT`;
  - `TypeError: unsupported operand type(s) for /: 'function' and 'str'`;
  - el bootstrap de exp17 da 0,256 con PYTHONHASHSEED=0 frente a 0,236 con 42 (n_boot=2000);
  - `assert 'guard_write(' in ...`;
  - el escáner devuelve exit 0 ante una clave .pem.
- **Controles que pasan hoy** (descartan un fallo del arnés): reserva con reloj en ms y escáner ante `password =`.
- **Después** (cada parche en un worktree temporal desde HEAD): todos los xfail afectados pasan a XPASS(strict), es decir, «FAILED» por diseño, lo que obliga a retirar la marca. El resto de la suite CPU sigue en verde. La única excepción, `test_operational_measurement.py::test_fresh_refuses_legacy_output_and_checkout`, también falla **sin** parche en un worktree fuera de `.worktrees/`: es de entorno (R-39).
- **Reproductores de las pasadas:** scratchpad de la sesión, no versionado: `passA/r01..r06`, `passB/repro_*`, `passC/repro/r01..r09`, `passE/{repro_cli_bug,mutate,flake}` y `verify*/`. Se pierden con el contenedor. Los que importan quedaron convertidos en tests versionados.

## 8. Estado y Veredicto

**NO-GO para la fase B pagada de I7 (F-005)** hasta aplicar el parche 001 y ensayarlo:
- **R-01 (S0):** hace casi seguro que el primer `gpu-restore` real deje una reserva L4 facturada, inmanejable y ociosa.
- **R-02 (S1):** un stockout bloquea la fase hasta 12 h y consume el plan.

Ensayo propuesto, sin ejecutar (requiere tu OK de gasto): `reserve --hours 1` en la zona calificada, comprobar
que `state.reservation` existe y que `guard` la borra, y repetir simulando el stockout con una zona sin L4.

**GO condicionado para el resto:** no hay evidencia dañada (escaneo 0, `git status` limpio, ningún escritor ejecutado),
pero R-04 y R-06 deben cerrarse antes de volver a ejecutar cualquier script de cifras. R-05 requiere tu decisión sobre la cifra del ledger.

## 9. Alcance No Ejecutado

- **Eco real de `deleteAtTime` en la API de Compute:** no tengo red hacia GCP. Por eso la confianza de R-01 sobre la API es media, aunque el truncado del SDK está verificado.
- **Windows y PowerShell nativos:** 33 tests omitidos; los `.ps1` solo se leyeron.
- **Docker / imagen I7, GPU, Ollama, modelos HF:** no ejecutados.
- **Suite `slow` con artefactos de `data/`:** ausentes en este clon.
- **Navegador fijado de I7 (chromium-1208):** no instalado.
- **Historia git completa:** el clon es superficial (157 commits), así que el origen del artefacto de exp17 no se pudo cerrar del todo.
- **Mutación completa:** solo 33 mutantes dirigidos, en 3 módulos.
- **Fuera del alcance a propósito:** los parches en zona congelada, la corrección del ledger, el borrado o movimiento de archivos con datos personales (R-16) y el push. Esperan tu OK.
