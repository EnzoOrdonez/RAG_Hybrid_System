# REPRODUCE — regenerar las cifras desde limpio

Cómo volver a obtener cada número citable del proyecto, ordenado de lo más barato a lo más
caro. Hay dos niveles de reproducción, declarados por separado a propósito:

1. **Rederivación offline** (niveles 1-4): recomputa cada cifra publicada desde los
   resultados y probabilidades **versionados** en `experiments/results/`. Corre sin GPU,
   sin LLM y sin red.
2. **Repetición completa** (nivel 5 y regeneración): requiere descargar los modelos y
   reconstruir corpus e índices. **No están versionados**: `data/models/`, `data/indices/`,
   los chunks y el corpus procesado existen solo localmente (gitignored); un clon público
   del repo no los contiene. Ver §0 para cómo obtenerlos.

Última verificación registrada: **2026-08-30**, niveles 1 y 2 en verde y suite
**341/341**. Esta fecha identifica la corrida que respalda la afirmación; no implica que
una edición posterior haya vuelto a ejecutar la suite.

---

## 0. Entorno (obligatorio)

El intérprete del PATH es 3.11 y **no** tiene el stack ML. Usar Python **3.14** con las
dependencias de `requirements-lock.txt` instaladas; apunta `$PY` a ese intérprete
(la ruta mostrada es la del entorno original del autor, conservada como procedencia
histórica — ajusta a tu instalación):

```powershell
$PY = "C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$env:HF_HUB_OFFLINE = 1      # ningún script debe salir a la red
$env:TRANSFORMERS_OFFLINE = 1
$env:PYTHONHASHSEED = 42     # determinismo de hashing (semilla global = 42)
$env:PYTHONUTF8 = 1          # OBLIGATORIA en consola cp1252 (ver nota abajo)
```

> **`PYTHONUTF8=1` — depende de la consola, y por eso conviene fijarla siempre.**
> `verify_v4_offline.py` imprime caracteres no-ASCII (`≈`) en su reporte. Si la consola está en
> `cp1252` —Windows PowerShell 5.1 o `conhost` heredado— el verificador muere con
> `UnicodeEncodeError` **después** de haber recomputado todo, así que parece un fallo de las
> cifras cuando es de codificación de salida. Con `PYTHONUTF8=1` pasa siempre.
>
> **Medido el 2026-08-21 por Claude Code:** en PowerShell **7.6.5** con `chcp 65001` (UTF-8, el
> defecto de PS7) el verificador da **exit 0 sin** la variable — `sys.stdout.encoding` ya es
> `utf-8`. O sea que el fallo histórico documentado en `CLAUDE.md` es real pero **condicional a
> la consola**, no universal. Fijar la variable hace la receta determinista en cualquiera de las
> dos, que es justo lo que una receta de reproducibilidad tiene que garantizar.
>
> Distinción deliberada: este `$PY` 3.14 es el **entorno reproducible**, no el soporte del
> paquete. `setup.py` y el README declaran **3.11+** como soporte de instalación; las dos cosas
> se declaran por separado a propósito.

Comprobación rápida: `& $PY -c "import torch,transformers,scipy,statsmodels;print(torch.__version__, torch.cuda.is_available())"`

Modelos: todo se resuelve desde `data/models/` (bge-large-en-v1.5, bge-reranker-large,
ms-marco-MiniLM-L-12-v2, nli-deberta-v3-{small,base,large}, hhem-2.1). Manifiestos con
sha256 en `data/models/*_manifest.json`. Índice único construido:
`data/indices/*_bge-large_adaptive_500.*`.

> **Aviso de artefactos no versionados (gitignored):** `data/models/`, `data/indices/`,
> los chunks y el corpus procesado **no viajan con un clon**. Para obtenerlos:
> los modelos se descargan de Hugging Face con los IDs exactos de arriba (verificar
> contra los sha256 de los manifiestos); los índices se reconstruyen con los scripts de
> `src/embedding/` sobre los chunks; el corpus se regenera con los crawlers de
> `src/ingestion/` más la curación documentada en `data/evaluation/README.md`.

---

## 1. Suite de tests (segundos, sin GPU)

```powershell
& $PY -m pytest -m "not slow and not gpu"   # ~1.5 s — la de cada edición
& $PY -m pytest                             # ~25 s — incluye carga de modelos NLI
```

Marcadores en `pytest.ini`: `slow` (carga un transformer), `gpu` (requiere CUDA),
`needs_artifacts` (necesita `experiments/results/*`).

Qué fija cada archivo:

| Test | Qué protege |
|---|---|
| `test_nli_calibration.py` | el umbral 0,7 opera sobre **probabilidades softmax**, no logits (Flag 135) |
| `test_benchmark_parity.py` | la ruta de benchmark de `llm_manager` es byte-idéntica a la pre-demo |
| `test_arm_stats.py` | la **familia BH declarada == nº real de contrastes** (regresión de la entrada 9/15 del ledger); pareo decline-aware |
| `test_coverage_balancer.py` | contrato de `balance()`; resolución de proveedores; **aceptación: reproduce los `balanced_ids` de exp17 en 25/25** |

## 2. Verificación offline de las cifras (minutos, sin GPU)

```powershell
& $PY scripts\verify_v4_offline.py        # cifras v4 / N9 (Nota 3, firmadas)
& $PY scripts\verify_summer_offline.py    # descubre por forma los artefactos de verano (Tier A + exp15-exp19b)
```

Ambos salen con código 0 solo si **todo** cuadra, y escriben un reporte en `output/audit/`.
`verify_summer_offline.py` re-deriva cada celda de fidelidad desde
`nli_probs__*.json.gz` / `grounding_probs__hhem.json.gz` y recomputa los contrastes pareados
(Wilcoxon, d_z, bootstrap, BH), además de dos guardas:

- **carga del instrumento**: el nivel HHEM del ancla debe caer en 0,40-0,55. Un HHEM mal
  cargado puntuaba ~0,04 y aun así "corría" (ledger entrada 6).
- **familia BH declarada** == nº de contrastes del propio artefacto.

## 3. Re-análisis (CPU, minutos a horas)

Reconstruyen artefactos derivados sin volver a generar con el LLM.

```powershell
# tabla pareada brazo-vs-ancla (los 3 experimentos, los 3 verificadores)
& $PY scripts\compute_tierA_arm_stats.py --verifier hhem `
      --exp-dir experiments\results\exp17_crosscloud_balanced --baseline-arm baseline

& $PY scripts\compute_exp16_guards.py          # guardas anti-gaming
& $PY scripts\compute_exp17_powered.py         # GLMM + bootstrap de cluster
& $PY scripts\compute_exp15_ensemble_sweep.py  # barrido de ensembles + control negativo
```

`--exp-dir` / `--baseline-arm` son obligatorios fuera de Tier A: sin ellos el artefacto sale
con la metadata de Tier A (era el defecto D1, ya corregido y con test).

## 4. Referencia humana piloto (COMPLETADA 2026-08-29 — reanálisis)

```powershell
& $PY scripts\analyze_gold_v4.py --simulate 0.15   # smoke test, no escribe nada
& $PY scripts\analyze_gold_v4.py                # análisis real (el CSV ya está relleno y adjudicado)
& $PY scripts\run_gold_sensitivity.py           # sensibilidad: preadjudicación / reconciliada / sin 9
& $PY scripts\analyze_taxonomy_calibration.py   # calibración de la taxonomía (40 ítems)
```

> **PELIGRO — `build_gold_v4.py` no forma parte del flujo normal.** Su implementación
> actual escribe directamente `claim_audit_sample_v4.csv`, su etapa B y el meta; volver a
> ejecutarlo sobre `output/audit/` sobrescribiría los CSV adjudicados. La muestra final se
> considera entrada congelada del reanálisis. Una regeneración excepcional requeriría antes
> un backup verificable y modificar el runner para escribir en un directorio temporal con
> sufijo nuevo; mientras no exista esa salida parametrizada, **no ejecutar el script**.

Salidas finales versionadas: `output/audit/gold_v4_analysis.{json,md}`,
`output/audit/gold_v4_sensitivity.md`, `output/audit/taxonomy_calibration_report.md`,
`output/audit/triple_judge_agreement.md` y `docs/SECCION_VALIDACION_HUMANA.md`.

El muestreo original fue determinista (seed 42): la misma receta selecciona los mismos
150 claims, pero eso no autoriza a sobrescribir sus juicios.
El análisis pondera por Horvitz-Thompson re-ejecutando el muestreador real; reporta el
**n efectivo de Kish final de 38,5** (`output/audit/gold_v4_analysis.json`), bastante
menor que 150 porque el diseño sobre-muestrea a propósito las celdas de desacuerdo.

## 5. Regeneración con GPU (horas — solo con motivo)

Vuelve a puntuar o a generar. **No hace falta para reproducir ninguna cifra publicada**;
los artefactos ya están committeados.

```powershell
# re-puntuar (GPU, sin LLM): ~1-3 h por verificador sobre exp12
& $PY scripts\rescore_nli_exp15.py --verifier base        # reanuda desde el .partial
& $PY scripts\rescore_grounding_exp15.py --tau 0.5

# re-generar un brazo (GPU + Ollama): horas
& $PY scripts\run_exp15_ablation.py --pass G --exp-id exp15_ablation_tierA
& $PY scripts\run_exp15_ablation.py --pass N --verifier small --exp-id exp15_ablation_tierA
```

**Aviso H5 (determinismo dependiente del entorno):** Ollama 0.22.1 está congelado durante la
fase. Incluso a temp=0 y seed=42, la primera generación tras un arranque en frío puede diferir
de las siguientes (caché cold-vs-warm). Por eso todo contraste es **pareado dentro de la misma
sesión** y cada brazo lleva una sonda de determinismo 3× registrada en `probe_report`. Nunca
compares un brazo nuevo contra números de una sesión anterior.

**Aviso de caché LLM:** `data/llm_cache/{model}_cache.json` se indexa por
`config_name ‖ prompt ‖ system ‖ temperature ‖ seed ‖ max_tokens` — **no** por exp-id. Un brazo
con el mismo `config_name` y prompt que otro experimento se sirve del caché de aquél
(le pasó a exp16). Usar `--no-cache` para brazos co-temporales.

---

## Reglas que la reproducción no debe romper

- `experiments/results/exp3..exp19b` (+`exp8b`) son **solo lectura** — evidencia firmada y cerrada, tags
  `nota3-evidencia-2026-06-11` / `nota3-N9-cierre-2026-07-02`. Comprobar con
  `git diff --name-status nota3-evidencia-2026-06-11 -- experiments/results`: solo debe haber
  altas (`A`). Todo recálculo va a archivos `_vN` nuevos; no hay experimentos nuevos previstos.
- `paper/audit_findings.md` y `paper/audit_outputs/exp8_stats_corrected.csv` son **inmutables**.
- `scripts/compute_retrieval_metrics.py` y `compute_faithfulness_metrics.py` escriben
  **in-place** en el directorio que se les pasa. Para verificar, usar los `verify_*_offline.py`,
  que importan funciones y **jamás** ejecutan sus `main()`.
- `config/config.yaml` secciones `retrieval:`/`reranking:` y `config/deprecated/` **no se
  consumen** y contradicen al sistema real. Las perillas vivas están en
  `src/pipeline/pipeline_config.py`; el inventario trazado, en `docs/KNOB_MAP_summer.md`.

## Config de despliegue (encuestas SUS/Likert)

`SURVEY_DEPLOY` en `src/pipeline/pipeline_config.py` = el sistema medido con **exactamente dos**
perillas cambiadas, ambas justificadas por la fase:

- `prompt_routing=True` — sin esto `RAGPipeline` tipa todo como `default` y usa otra plantilla
  que la ruta medida en **115/194** queries.
- `balance_cross_cloud_providers=True` — la única palanca positiva de la fase (exp17, cobertura
  **condicional** histórica 7/25→25/25, HHEM +0,081; la auditoría **estricta** canónica del
  2026-08-29 es 2/25→20/25, es decir 8 %→80 % — ver `output/audit/provider_coverage_probe.md`).
  Piloto n=25, no significativo: es una decisión de despliegue, no una afirmación de la tesis.

Está **fuera** de `PIPELINE_CONFIGS` a propósito, para que `get_config("hybrid")` siga
devolviendo el sistema medido. Fijado por `test_coverage_balancer.py`.

---

## Actualización exp19b (2026-08-22)

Esta nota reemplaza operativamente el aviso H5 anterior para exp19b. La huella del warmup
discriminó tres modos de carga de Ollama, pero dos drafts completos separados por 4,7 h
resultaron **194/194 bit-idénticos**. Por eso la huella queda como log informativo y la
compuerta real es `draft_replay_check`: antes de regen reproduce cinco respuestas archivadas
por el mismo camino de generación y exige 5/5 identidades. Un fallo termina con
`RUNTIME_STATE_CHANGED` (exit 3) antes de puntuar.

En Windows PowerShell 5.1 el lanzador completo se invoca con `powershell`, no con `pwsh`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\launch_exp19b_full.ps1
```

El orden protegido es `draft -> extract/select en CPU -> draft_replay_check -> regen`.
`select_exp19b_evidence.py` fija el CrossEncoder en `device="cpu"` y el lanzador oculta CUDA
a extract/select; la GPU no se toca entre draft y el final de regen. Draft y select arrancan
con `--no-resume`, y draft/regen mantienen la caché LLM desactivada.

**Resultado cerrado (2026-08-22):** la corrida completa pareada terminó con replay 5/5 y
Δ HHEM **+0,0451** (IC95 [0,0076; 0,0817], p=0,018), TOST dentro de la banda ±0,081.
Veredicto: mejora local alineada al verificador HHEM, **no** una mejora de fidelidad
independiente del verificador. El lanzador de arriba queda como registro operativo:
**no** debe re-ejecutarse sobre la evidencia final, que está cerrada y es de solo lectura.
