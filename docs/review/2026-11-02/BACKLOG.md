# Backlog de la auditoría 2026-11-02 (nada aplicado)

| Id | Sev | Qué | Propuesta | Estado |
|---|---|---|---|---|
| R-05 | S2 | El bootstrap de exp17 depende de `PYTHONHASHSEED` | Parche 005 y test | Listo para revisión |
| R-03 | S2 | Huecos de la capa 1 de secretos | Patrones e inspección con aviso | Tests xfail ya en la rama |
| R-16 | S2 | Identificadores en el repo público | Decisión del dueño (ADR-0011) | Pendiente de Enzo |
| F-016 | S2 | Intermitencia de `test_short_calls_always_check_host_telemetry` | Solo en el test | Pendiente |
| CLAUDE-37 | S3 | La línea de `init.sh` «en Linux o CI» no es cierta | Corregir el texto | Pendiente: confirmar la línea |

## R-05: semilla determinista del bootstrap de exp17

**Causa [VERIFICADO].**
- `scripts/compute_exp17_powered.py:46` construye `counts` iterando `set(base) | set(bal)`.
- `:84` hace `qids = list(counts)`, y el remuestreo indexa por posición (`qids[i]` con
  `rng.integers`). La semilla del RNG ya es 42; lo que no es fijo es el orden de las queries.

**Evidencia.**
- Datos reales, HHEM, `n_boot` = 2000: p = 0,236 con `PYTHONHASHSEED=42` y p = 0,2555 con 7.
- El valor firmado es 0,2329.

**Arreglo.** `patches/005-r05-exp17-deterministic-bootstrap.patch`:
- `sorted()` en `load_counts` y en `cluster_bootstrap`;
- escribe `powered_reanalysis_v2.{json,md}` con `guard_write`, así que el artefacto firmado no se
  sobrescribe (ADR-0001 y ADR-0009).

**Test.** `tests/test_review_2026_11_02_backlog.py::test_cluster_bootstrap_is_invariant_to_query_insertion_order`:
- sobre la base: xfail estricto;
- con el parche: XPASS(strict), verificado en un worktree.

**Cifras con el parche** [VERIFICADO]. Se calcularon llamando a `cluster_bootstrap` con `n_boot` = 10000,
sin ejecutar `main()` y sin escribir nada. Salen idénticas con `PYTHONHASHSEED` 42, 7 y 123.

| Verificador | Firmado: IC95 | Firmado: p | Con el parche: IC95 | Con el parche: p |
|---|---|---|---|---|
| small | [-0,0954, 0,1647] | 0,302 | [-0,1004, 0,1656] | 0,2906 |
| base | [-0,0389, 0,1195] | 0,1428 | [-0,0385, 0,1178] | 0,1418 |
| hhem | [-0,0609, 0,1326] | 0,2329 | [-0,0622, 0,1314] | 0,2375 |

Ninguno cruza p < 0,05: el veredicto «no cruza» se mantiene. Cambiar la cifra citada en
`paper/summer_ablation_log.md` («HHEM 0,23») requiere el OK frase por frase y una entrada en el
ledger. El parche no toca la prosa.

## R-03: huecos de la capa 1
- Tokens que no se detectan: `ya29.`, `AIza`, `hf_` y `github_pat_`.
- `.pem`, `.key` y `.env.*` se omiten sin aviso.
- Tests: 7 xfail estrictos en `tests/test_review_2026_11_02_prior_open.py`.
- Propuesta: añadir los cuatro patrones e inspeccionar esos archivos en vez de omitirlos. Si existe
  alguno, la capa debe dar exit ≠ 0 con un aviso que no imprima el contenido.
- Esto toca `scripts/scan_secrets.py`. No está en la zona congelada del operador, pero sí en `scripts/`,
  y la decisión vigente lo pospone.

## R-16: identificadores en el repo público [VERIFICADO con `git grep`]
- El ID del proyecto aparece 75 veces en 43 archivos.
- El usuario local de Windows aparece en 24 archivos.
- Hay 12 archivos con correo de cuenta de servicio.
- El correo ACME está en `scripts/study_operator/deployment.py:112`.
- Hay correos de terceros en `titulacion/emails_lewis_y_gyt.md`.
- No son secretos. Según ADR-0011 no se borra nada: Enzo decide si redacta o lo acepta como riesgo.

## F-016: intermitencia del supervisor
- [DECLARADO en `.harness/feature_list.json`] Tres apariciones de `TimeoutError 'Independent worker
  deadline reached'` (`scripts/study_gate_supervisor.py:123`) en
  `tests/test_study_stimulus_supervisor.py::test_short_calls_always_check_host_telemetry`.
- [DECLARADO, informe anterior R-20] La causa probable es un presupuesto de 3 s para el arranque
  `spawn` bajo carga. La propuesta es tocar solo el test: `preparation_seconds=60`, `call_seconds=10`
  y una aserción que no exija igualdad exacta.
- [VERIFICADO en esta sesión] No falló en ninguna de las ejecuciones completas de la suite (línea base
  y finales). Eso no la descarta.

## CLAUDE-37: línea desactualizada de CLAUDE.md (pendiente de confirmar)
- **Mi candidata** es `CLAUDE.md:37`: «Ejecutar `bash init.sh` desde Git Bash en Windows (no WSL), o en
  Linux o CI».
- **[VERIFICADO] CI no ejecuta `init.sh`.** `.github/workflows/ci.yml:48-54` lanza las tres capas por
  separado.
- **[VERIFICADO] En Linux, `init.sh` da 11 fallos falsos** cuando `PLAYWRIGHT_BROWSERS_PATH` está
  definida sin el Chromium fijado de I7 (R-08). Pasa en este contenedor.
- **Texto propuesto:** «… o en Linux; CI ejecuta las tres capas por separado. Si
  `PLAYWRIGHT_BROWSERS_PATH` apunta a otro Chromium, desdefinirla o la capa 2 da rojos falsos (R-08).»
- **Por qué no está aplicado:** el cambio toca `CLAUDE.md`, fuera del alcance de esta tarea. Si te
  referías a otra línea, indícalo.
