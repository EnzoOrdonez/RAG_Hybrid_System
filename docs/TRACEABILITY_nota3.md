# Trazabilidad informe ↔ repositorio — Nota 3

Mapa verificado **resultado del informe → experimento → script → archivo de salida → tabla/figura**.
Cada ruta de este documento existe en el repo (verificada 2026-06-30). Responde la observación de
trazabilidad del asesor (20/06/2026).

## Corpus experimental vs corpus del repo

- **Corpus experimental (exp9-13, lo que cita el informe):** subconjunto AWS/Azure/GCP =
  **2.697 documentos / 24.481 chunks** (`data/chunks/adaptive/size_500/*.json`).
- El README reporta el corpus **general** del repo (3.951 docs / 46.318 chunks, incluye
  Kubernetes + CNCF); esos dos proveedores **no** entraron en las corridas del informe.

## Experimentos: histórico vs evidencia final

| Carpeta | Estado | Rol |
|---|---|---|
| `experiments/results/exp3..exp8`, `exp8b` | **HISTÓRICO** | corpus/oráculo pre-rebuild; conservados como historia, **no citar** |
| `experiments/results/exp9_llm_only_no_rag` | FINAL | Control 0 (LLM sin RAG) |
| `experiments/results/exp10_retrieval194`, `exp11_retrieval194_fullrerank` | FINAL | retrieval, set 194, multi-oráculo (D12) |
| `experiments/results/exp12_matrix` | FINAL | matriz fidelidad 4×4 (194) |
| `experiments/results/exp13_expansion` | FINAL | expansión ON vs OFF (D11) — veredicto N4 |

Evidencia firmada por el tag **`nota3-evidencia-2026-06-11`**.

## Matriz de trazabilidad

Preámbulo de entorno para TODO comando: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42`.

| Resultado del informe | Experimento | Script(s) | Salida cruda | Tabla/figura citada |
|---|---|---|---|---|
| **Tabla 4** — retrieval por sistema y oráculo (P@1/R@5/MRR/NDCG@5) | exp11 | `run_retrieval_only.py` → `compute_retrieval_metrics.py` (×2 oráculos) → `_export_paper_tables_nota3.py` | `exp11…/results.json`, `…/retrieval_metrics__{bge-reranker-indep,ms-marco-circular}.json` | `output/tables/nota3/tabla4_retrieval__exp11_retrieval194_fullrerank.{md,csv}` |
| **Estabilidad de oráculos** (NDCG@5, d_z, p_BH) | exp11 | `_oracle_stability.py` | mismos `retrieval_metrics__*.json` | `output/tables/nota3/oracle_stability__exp11_retrieval194_fullrerank.{md,csv}` |
| **Tabla 6 v2** — fidelidad answered por escenario×modelo | exp12 | `run_generation_matrix.py` → `compute_faithfulness_metrics.py` → `_export_paper_tables_nota3.py` | `exp12_matrix/results.json`, `…/faithfulness_metrics_v2.json` | `output/tables/nota3/tabla6_fidelidad_v2__exp12_matrix.{md,csv}` (+ `tabla6_sensibilidad_denominador`, `tabla6c_clasificacion_v2`, `tabla5_modelo_principal_v2`, `tabla_claims_desglose`) |
| **Tabla 6b** — tasa de declinación | exp12 | idem | `faithfulness_metrics.json` (v1, vía `--write-v1`) | `output/tables/nota3/tabla6_declinacion__exp12_matrix.{md,csv}` |
| **Latencias p50/p95** | exp12 | `_latency_p50p95.py` | `exp12_matrix/results.json` | `output/tables/nota3/latency__exp12_matrix.{md,csv}` |
| **Figuras f1–f4** | exp11/exp12 | `_make_figures_nota3.py` | tablas/jsons anteriores | `output/figures/nota3/f{1,2,3,4}_*.png` |
| **Expansión cross-cloud OFF≈ON (N4)** | exp13 | `run_exp13_expansion.py` → `compute_retrieval_metrics.py` / `compute_faithfulness_metrics.py` | `exp13_expansion/results.json`, `retrieval_metrics__bge-indep.json`, `faithfulness_metrics_v2.json` | (veredicto en `RESULTADOS_RESUMEN.md` §2/§8) |
| **Control 0 (LLM sin RAG)** | exp9 | `run_llm_only_benchmark.py` | `exp9_llm_only_no_rag/results.json` | (RESULTADOS §3, "Sin RAG = 0 por construcción", N3) |

## Recetas mínimas (regenerar SOLO lo citado, sin correr las 13 experimentaciones)

Las salidas crudas (`results.json`) ya están versionadas; estos comandos re-derivan las
tablas/figuras a partir de ellas (no re-corren generación LLM salvo el primer bloque):

```bash
# Tabla 4 + estabilidad de oráculos (retrieval ya guardado en exp11):
python scripts/compute_retrieval_metrics.py --experiment exp11_retrieval194_fullrerank \
    --oracle-model BAAI/bge-reranker-large --oracle-label bge-reranker-indep
python scripts/compute_retrieval_metrics.py --experiment exp11_retrieval194_fullrerank   # ms-marco (circular, referencia)
python scripts/_oracle_stability.py --experiment exp11_retrieval194_fullrerank

# Tabla 6 (fidelidad) + Tabla 5 + latencias, desde exp12 ya guardado:
python scripts/compute_faithfulness_metrics.py --experiment exp12_matrix --write-v1
python scripts/_export_paper_tables_nota3.py exp12_matrix granite4.1-8b
python scripts/_latency_p50p95.py exp12_matrix

# Figuras f1-f4:
python scripts/_make_figures_nota3.py
```

## Ronda N8 / v3 (auditoría del instrumento NLI, 2026-06-30)

Corrige H1 (artefactos de formato excluidos del denominador) y H2 (guarda de contradicción).
**Archivos nuevos `_v3`; nada firmado se sobrescribe.** Requiere GPU/Ollama-stack (máquina de Enzo):

```bash
# 0) (una vez) elegir la variante de guarda H2 con el trade-off:
python scripts/_h2_variant_eval.py --verifier base        # -> output/audit/h2_variant_eval.json  (SUB-GATE: elegir variante)

# 1) re-score v3 con extracción corregida + variante elegida, 2 verificadores:
python scripts/rescore_nli_v3.py --verifier base  --variant <v> [--margin d]
python scripts/rescore_nli_v3.py --verifier small --variant <v> [--margin d]

# 2) métricas v3 (reusa la maquinaria estadística v2):
python scripts/compute_faithfulness_metrics.py --experiment exp12_matrix \
    --faithfulness-source experiments/results/exp12_matrix/faithfulness_rescore_v3__base__<v>.json
#    -> experiments/results/exp12_matrix/faithfulness_metrics_v3.json

# 3) muestra de auditoría v3 (no sobrescribe la firmada del 06-11):
python scripts/build_claim_audit_sample.py --out-suffix _v3
```

Tablas v3: `python scripts/_export_tabla6_v3.py exp12_matrix faithfulness_metrics_v3_small.json`
→ `output/tables/nota3/tabla6_fidelidad_v3__exp12_matrix.{md,csv}`.

**Resultado v3 (small, comparable con la Tabla 6 publicada):** retrieval n.s. en fidelidad (0/12,
robusto bajo AMBOS verificadores base+small); entre-modelos **2/18 sig bajo small** (6/18 base) —
corrige el "todo n.s." de N5; gemma/qwen **suben +0,05..0,11** al excluir artefactos. Veredictos y
deltas: ledger **N8** en `paper/audit_findings_cc_addenda.md`; cifras en `RESULTADOS_RESUMEN.md` §3.

### v4 (N9, 2026-07-02): exclusión de filas vacuas — TABLA CITABLE

Las respuestas cuyos claims extraídos son TODOS artefactos (`genuine==0`; 59/1798, concentradas
en gemma/qwen) entraban al denominador v3 como faithfulness=1,0 vacuo. v4 las excluye (mismo
trato que method='none', Flag 137; decisión de Enzo 01/07, ledger N9):

```bash
python scripts/compute_faithfulness_metrics.py --experiment exp12_matrix \
    --faithfulness-source experiments/results/exp12_matrix/faithfulness_rescore_v3__base__vb_agree.json \
    --out-tag v4 --exclude-vacuous
python scripts/compute_faithfulness_metrics.py --experiment exp12_matrix \
    --faithfulness-source experiments/results/exp12_matrix/faithfulness_rescore_v3__small__vb_agree.json \
    --out-tag v4_small --exclude-vacuous
python scripts/_export_tabla6_v4.py exp12_matrix faithfulness_metrics_v4_small.json
```

**Tabla citable = `output/tables/nota3/tabla6_fidelidad_v4__exp12_matrix.md`.** Veredictos v4:
retrieval n.s. 0/12 (robusto bajo ambos verificadores); entre-modelos 1/18 (small) y 1/18 (base),
con pares supervivientes DISTINTOS entre verificadores — frágil. Ledger **N9**.

Notas de reproducibilidad (N9): (a) la receta de Tabla 4 sobrescribe `retrieval_metrics__*.json`
in-place en exp11, y sus oráculos (bge-reranker-large, ms-marco) pueden no estar en la caché local
(re-descarga necesaria); (b) el veredicto de exp13 (expansión OFF≈ON) permanece bajo métrica v2 —
comparación intra-modelo con el mismo confound en ambos brazos; si el paper lo cita junto a cifras
v4, añadir nota de instrumento; (c) los rescores NLI usan `data/models/nli-deberta-v3-{base,small}`
locales, no el hub de HF.
