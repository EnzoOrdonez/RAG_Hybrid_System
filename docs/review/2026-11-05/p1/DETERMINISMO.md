# P1: determinismo de las estadísticas reportadas (base 3bb9ca8)

Rama `claude/p1-determinismo`. Arnés: [determinism_harness.py](determinism_harness.py). Resumen bruto: [harness_summary_3bb9ca8.json](harness_summary_3bb9ca8.json).

## a) Inventario [EJECUTADO]

- **Búsqueda.** `git grep` de bootstrap, permutaciones, `np.random`, `random.*`, `default_rng`, `RandomState`, splits y tests estadísticos (wilcoxon, BH, kappa, McNemar…) en `scripts/` y `src/`, excluyendo `study_operator/`. Salen 41 archivos.
  - 4 son módulos de librería (`src/evaluation/statistical_analysis.py`, `study_analysis.py`, `study_statistics.py` y `src/utils/reproducibility.py`). Se ejercitan a través de los scripts y de la capa 2.
  - De los 37 scripts, se excluyeron 6 que llaman a LLM, red o telemetría: `run_exp15_ablation`, `nli_batch_experiment`, `run_runtime_noise_probe`, `probe_fingerprint_drift`, `unattended_diagnostic` y `measure_survey_config_latency`. Quedan como **no verificado**.
- **Análisis estático** (iteración de `set` sin `sorted`, `hash()`, RNG sin semilla). Único caso en una estadística reportada: `scripts/compute_exp17_powered.py`.
  - `:46`: `for q in set(base) | set(bal)` construye el dict en orden de hash.
  - `:84`: `qids = list(counts)`; el bootstrap indexa por posición.
  - Los demás ya ordenan: `compute_exp18_powered.py:66`, `compute_tierA_arm_stats.py:80`, `compute_descriptive_cis.py:149`.
  - `scripts/audit/smoke_seeds.py:34,44` usa `np.random.rand` a propósito: es un smoke de semillas y fija la semilla antes.

## b) Arnés [EJECUTADO]

- **Montaje.** Cada semilla (`PYTHONHASHSEED` = 0, 7 y 42) usa su propio worktree desechable de 3bb9ca8, así que las salidas firmadas del checkout nunca se escriben.
- **Qué se compara:** `sha256(stdout)` y el `sha256` de cada archivo creado o modificado. La ruta del worktree se normaliza a `<WT>`. Sin esa normalización salían 6 falsos «no deterministas» por las rutas absolutas en stdout o JSON.
- **Resultado:** de 31 scripts ejecutados, **16 son deterministas**, **1 no lo es** (exp17) y **14 quedan no verificados**.

| script | determinista (antes) | nota |
|---|---|---|
| `_analyze_h5_replicas`, `_analyze_rescore_v2`, `analyze_gold_v4`, `analyze_user_sessions`, `audit/_write_phase2_5_csv`, `audit/smoke_bh_fdr`, `audit/smoke_compute_retrieval_stats`, `audit/smoke_seeds`, `compute_descriptive_cis`, `compute_exp15_hhem_analysis`, `compute_exp18_powered`, `compute_exp19b_stats`, `compute_tierA_arm_stats`, `lexical_diagnostic`, `run_gold_sensitivity`, `verify_summer_offline` | sí (16) | bit a bit con las 3 semillas |
| **`compute_exp17_powered`** | **NO** | difieren `powered_reanalysis.json` y `.md` |
| `_subsample_azure`, `analyze_exp15_disagreement`, `build_gold_v4`, `compute_exp19a_selector_probe`, `rescore_grounding_exp15`, `rescore_nli_v2` | no verificado | falta una entrada no versionada (FileNotFoundError) |
| `analyze_taxonomy_calibration`, `compute_exp18_unsupported_taxonomy`, `compute_faithfulness_metrics` | no verificado | la guarda de evidencia rechaza reescribir (es correcto); habría que pasar a `_vN` |
| `audit/smoke_recompute_retrieval_stats`, `compute_retrieval_metrics`, `recompute_retrieval_stats --all` | no verificado | necesitan modelos de HF que no están descargados |
| `contrast_interview_observer --output` | no verificado | precheck de telemetría y AC de Windows |
| `compute_exp15_nli_sweep` | no verificado a 120 s | se reintenta con 900 s; el resultado va en PROGRESO.md |

## c) Arreglo (cambio mínimo, a partir del parche 005)

- **Cambios en `scripts/compute_exp17_powered.py`:**
  - `sorted(...)` en `:48` y en `:86`.
  - La salida va a `powered_reanalysis_v2.{json,md}` mediante `guard_write`.
  - El artefacto firmado no se toca (ADR-0001/0009).
- **Tests:** `tests/test_p1_determinism.py`.
  - Hay 4 tests: idéntico con semillas 0, 7 y 42 (subproceso); independiente del orden de inserción; `load_counts` ordenado; nunca apunta al artefacto firmado.
  - Con el arreglo pasan los 4. **Sin el arreglo fallan los 4** (ejecutado).
- **No se han escrito ni cometido** salidas `_v2` en `experiments/`. Publicar el recálculo es decisión de Enzo, porque toca una cifra ya entregada.

## d) Efecto sobre las cifras (exp17, `n_boot` = 10000, valores firmados de `powered_reanalysis.json`) [EJECUTADO en worktree]

| verificador | estadística | firmado | original con semilla 0 / 7 / 42 | arreglado con semilla 0 = 7 | Δ arreglado − firmado |
|---|---|---|---|---|---|
| small | p bootstrap (una cola) | 0,302 | 0,2848 / 0,3003 / 0,2951 | 0,2906 | −0,0114 |
| small | IC 95 % | [−0,0954, 0,1647] | — | [−0,1004, 0,1656] | ≈0,005 |
| base | p bootstrap | 0,1428 | 0,1424 / 0,1496 / 0,1440 | 0,1418 | −0,0010 |
| hhem | p bootstrap | 0,2329 | 0,2385 / 0,2479 / 0,2401 | 0,2375 | +0,0046 |
| todos | GLMM: p, OR y z | 0,1517 / 0,05591 / 0,02075 | iguales | iguales | 0 |

- **El valor firmado no se reproduce con ninguna de las tres semillas.** La semilla de hash de la corrida original es desconocida.
- Ningún p del bootstrap cruza 0,05 en ninguna variante, y la GLMM es determinista. **Cualitativamente, la conclusión de exp17 no cambia.**
- Lo que sí cambia es el tercer decimal citado, si alguna prosa cita el bootstrap. Ver P2.

## No verificado
- Los 14 scripts de la tabla, con su motivo.
- Los 6 excluidos por LLM, red o telemetría.
- Windows y Python 3.14.3: aquí se usó 3.14.6 en Linux.
