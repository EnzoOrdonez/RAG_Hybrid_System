# REPRODUCE — regenerar las cifras desde limpio

Cómo volver a obtener cada número citable del proyecto, ordenado de lo más barato a lo más
caro. **Todo lo del nivel 1 y 2 corre sin GPU, sin LLM y sin red**: la fase de verano se
diseñó para que el pase GPU se pague una sola vez por verificador y todo lo demás sea
re-agregación en CPU de las probabilidades persistidas.

Estado actual: **todas las verificaciones de nivel 1 y 2 pasan** (2026-07-30).

---

## 0. Entorno (obligatorio)

El intérprete del PATH es 3.11 y **no** tiene el stack ML. Usar el 3.14:

```powershell
$PY = "C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$env:HF_HUB_OFFLINE = 1      # ningún script debe salir a la red
$env:TRANSFORMERS_OFFLINE = 1
$env:PYTHONHASHSEED = 42     # determinismo de hashing (semilla global = 42)
$env:PYTHONUTF8 = 1          # OBLIGATORIA en consola cp1252 (ver nota abajo)
```

> **`PYTHONUTF8=1` no es opcional en Windows.** La consola por defecto es `cp1252` y
> `verify_v4_offline.py` imprime `≈` en su reporte: sin esta variable el verificador muere con
> `UnicodeEncodeError` **después** de haber recomputado todo, así que parece un fallo de las
> cifras cuando es de codificación de salida. Con la variable pasa completo. Esta receta la
> omitía; añadida por Claude Code el 2026-08-21 07:35 (hora local).
>
> Distinción deliberada: este `$PY` 3.14 es el **entorno reproducible**, no el soporte del
> paquete. `setup.py` y el README declaran **3.11+** como soporte de instalación; las dos cosas
> se declaran por separado a propósito.

Comprobación rápida: `& $PY -c "import torch,transformers,scipy,statsmodels;print(torch.__version__, torch.cuda.is_available())"`

Modelos: todo se resuelve desde `data/models/` (bge-large-en-v1.5, bge-reranker-large,
ms-marco-MiniLM-L-12-v2, nli-deberta-v3-{small,base,large}, hhem-2.1). Manifiestos con
sha256 en `data/models/*_manifest.json`. Índice único construido:
`data/indices/*_bge-large_adaptive_500.*`.

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
& $PY scripts\verify_summer_offline.py    # Tier A + exp16 + exp17 (fase de verano)
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

## 4. Gold humano (dependencia humana + CPU)

```powershell
& $PY scripts\build_gold_v4.py                  # regenera etapa A (150) + etapa B (50)
& $PY scripts\analyze_gold_v4.py --simulate 0.15   # smoke test, no escribe nada
& $PY scripts\analyze_gold_v4.py                # análisis real (requiere el CSV relleno)
```

El muestreo es determinista (seed 42): regenerar **no** cambia qué 150 claims salen.
El análisis pondera por Horvitz-Thompson re-ejecutando el muestreador real; reporta el
**n efectivo de Kish**, que es bastante menor que 150 porque el diseño sobre-muestrea a
propósito las celdas de desacuerdo.

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

- `experiments/results/exp3..exp14` (+`exp8b`) son **solo lectura** — evidencia firmada, tags
  `nota3-evidencia-2026-06-11` / `nota3-N9-cierre-2026-07-02`. Comprobar con
  `git diff --name-status nota3-evidencia-2026-06-11 -- experiments/results`: solo debe haber
  altas (`A`). Todo recálculo va a archivos `_vN` nuevos o a IDs `exp15+`.
- `paper/audit_findings.md` y `exp8_stats_corrected.csv` son **inmutables**.
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
- `balance_cross_cloud_providers=True` — la única palanca positiva de la fase (exp17: cobertura
  7/25→25/25, HHEM +0,081). Piloto n=25, no significativo: es una decisión de despliegue, no una
  afirmación de la tesis.

Está **fuera** de `PIPELINE_CONFIGS` a propósito, para que `get_config("hybrid")` siga
devolviendo el sistema medido. Fijado por `test_coverage_balancer.py`.
