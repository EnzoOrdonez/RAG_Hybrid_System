# P2: trazabilidad de cifras reportadas (base 3bb9ca8)

Rama `claude/p2-trazabilidad`. Este informe no edita ningún texto: las correcciones se proponen en §3 para revisarlas frase por frase.

**Documentos analizados:**
- el camera-ready LACCI (`docs/2026305869.pdf`, extraído con `pdftotext -layout`);
- `README.md`;
- `docs/reports/results-summary.md` y `docs/reports/summer-benchmarks.md`;
- el borrador `paper/overleaf_ready/main.tex`.

## 1. Barrido automático [EJECUTADO]

- **Script:** [trace_figures.py](trace_figures.py).
- **Método:** para cada decimal con al menos 2 cifras, busca algún JSON o CSV de `experiments/results/**` u `output/**` que contenga un número que, redondeado igual, coincida.
- **Salida bruta:** [trace_auto.json](trace_auto.json).
- **Advertencia:** coincidir en valor demuestra que el valor existe, no que sea la misma magnitud (0,25 aparece en muchos archivos). Por eso las cifras clave se verificaron a mano en §2.

| Documento | Cifras | Con fuente por valor | Sin fuente |
|---|---|---|---|
| camera-ready (PDF) | 85 | 69 | 16. Corpus: verificado a mano en §2. SLA de AWS 99.99/99.95: hecho externo. Identificadores arXiv |
| README.md | 56 | 47 | 9. Corpus: verificado a mano. Latencias del gate 65.75/45.09 s: **sin fuente en el repo** |
| results-summary.md | 113 | 110 | 3. «+16,75 %», ya retirado. Dos identificadores arXiv |
| summer-benchmarks.md | 141 | 141 | 0 |
| main.tex (borrador) | 8 | 8 | 0, pero ver D1 |

## 2. Cifras clave verificadas a mano [EJECUTADO]

| Cifra | Dónde se afirma | Fuente | Estado |
|---|---|---|---|
| Tabla de retrieval: P@5, R@5, MRR y NDCG@5 de BM25, denso, solo RRF y reranked (16 valores; 0.740, 0.624…) | camera-ready, tabla de retrieval | `experiments/results/exp11_retrieval194_fullrerank/retrieval_metrics__bge-reranker-indep.json`, n=194 | **verificada 16/16** |
| NDCG@5 circular: 0.552 / 0.649 / 0.668 / 0.995 | camera-ready, `results-summary.md` §1 | `…/retrieval_metrics__ms-marco-circular.json` | verificada |
| d_z: +0.45 (p_BH 5,2e-8), +1.38, −0.08 (p 0.240) y +0.52 | camera-ready | `statistical_tests.ndcg@5` de los dos JSON de exp11 | verificada; los signos siguen el orden del par en el JSON |
| Tabla de fidelidad, `faithfulness_answered` (n): 12 celdas RAG × 4 modelos | camera-ready | `experiments/results/exp12_matrix/faithfulness_metrics_v4_small.json`, `primary_answered` | **verificada 12/12** |
| Corpus: 24,481 chunks; AWS 6,366, Azure 9,606, GCP 8,509; EC2 4,215; Functions 10,681→2,500; Blob 3,841→1,500 | camera-ready y README | `data/corpus_stats.json`; `exp10/11 …/results.json` | verificada |
| P@1 0.930, MRR 0.942, NDCG@5 0.736, BM25 0.785, denso 0.860 | `main.tex:65-67` | `experiments/results/exp8/retrieval_metrics.json`, n=200 | verificada para exp8; ver D1 |
| d = 0.626, p < 0.0001 | `main.tex:68` | `paper/audit_outputs/exp8_stats_corrected.README.md:68`: es d_pooled de BM25 frente al híbrido. El mayor d_z de exp8 es 0,5023. El p del híbrido frente al denso es 0,000874 (NDCG) y 0,015 (MRR) | **discrepante (D1)** |
| p del bootstrap de exp17: 0,302 / 0,1428 / 0,2329 | `experiments/…/powered_reanalysis.md:7-9` y `summer-benchmarks.md:149` («HHEM 0.23») | `compute_exp17_powered.py`, no determinista (P1) | **discrepante (D4)** |
| Latencia p95 del gate: 51.57 / 65.75 / 45.09 s | `README.md:150` | ningún archivo del repo; registro fuera del repo (`C:\CloudRAG`) | **sin fuente (D3)**: no verificable aquí |

## 3. Discrepancias por gravedad y corrección propuesta (sin aplicar)

- **D1 (S1). `paper/overleaf_ready/main.tex` es un borrador anterior a la auditoría.** Tiene 28 `PLACEHOLDER`, y su abstract afirma:
  - `:69-70`: «16.8 percent gain in answer faithfulness». La ganancia está **retirada** (`results-summary.md` §2: exp7 era un no-op y exp13 la refuta), y el camera-ready no la contiene.
  - `:68`: «Cohen's d = 0.626» sin decir que es BM25 frente al híbrido (Flag 86, `paper/audit_findings.md:586-589`), con «p < 0.0001», que no se cumple frente al denso.
  - Usa exp8 (200 q) en lugar de exp11 (194 q, oráculo independiente).
  - **Propuesta:** marcar el archivo como borrador histórico obsoleto, o retirarlo de `overleaf_ready/`. El texto citable es el camera-ready.
  - Lo decide Enzo frase por frase, porque es prosa del paper.
- **D2 (S2). `results-summary.md:3-4` dice que sus cifras «siguen siendo las citables del paper».** Pero su Tabla 6 v2 (§3) es `faithfulness_metrics_v2.json` y no coincide con la del camera-ready (v4_small). Ejemplo: Gemma léxico 0,381 (60) frente a **0.413 (57)**; Qwen léxico 0,246 (42) frente a 0.255 (36).
  - **Propuesta:** añadir al aviso de cierre que la tabla de fidelidad citable es la v4_small del camera-ready.
- **D3 (S3). `README.md:150`:** las latencias del gate no tienen fuente versionada.
  - **Propuesta:** citar el recibo (ruta o hash) o indicar «registro fuera del repo».
- **D4 (S3). `summer-benchmarks.md:149` («HHEM 0.23») y `powered_reanalysis.md`:** dependen de la semilla de hash. Con el arreglo de P1, el HHEM pasa a 0,2375, que se redondea a 0,24.
  - **Propuesta:** si Enzo publica `powered_reanalysis_v2`, citar «≈0,24 (semilla de hash fijada)». La conclusión (no significativo) no cambia.

## No verificado
- La prosa de la tesis A.3 y la ficha: no están en el repo.
- Los registros fuera del repo (`C:\CloudRAG`, `D:\CloudRAG_archivo`).
- `docs/` completo: solo se barrieron los 4 documentos de resultados y el camera-ready. Los demás `docs/*.md` (80 archivos) quedan sin revisar por falta de tiempo.
