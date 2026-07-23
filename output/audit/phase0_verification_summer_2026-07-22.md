# Verificación Fase 0 (verano) — cifras v4 desde JSONs firmados
Fecha: 2026-07-22 · script: `scripts/verify_v4_offline.py` (solo lectura sobre experiments/)

## Entorno verificado (2026-07-22)
| Ítem | Valor | Estado |
|---|---|---|
| Intérprete | py 3.14.3 (`...\pythoncore-3.14-64\python.exe`) | OK |
| torch | 2.10.0+cu126, CUDA ok (RTX 3060 Laptop) | OK |
| pydantic | 2.12.5 (`model_copy(update=...)` disponible) | OK |
| Ollama | **0.22.1** (registrado para H5; update 0.32.1 disponible — NO actualizar en fase de verano) | OK |
| Modelos Ollama | granite4.1:8b (5.3 GB) + gemma4:e4b, mistral:7b-instruct, qwen3.5:9b presentes | OK |
| NLI locales | `data/models/nli-deberta-v3-{small,base}` | OK |
| Índice | `bge-large_adaptive_500` (faiss + bm25 + chunk_map) — único construido | OK |
| Caché HF hub | **solo** nli-deberta-v3-small. FALTAN: bge-large-en-v1.5 (~1.3 GB), ms-marco-MiniLM-L-12-v2 (~134 MB), bge-reranker-large (~2.2 GB) → descarga aprobada a `data\models\` | HALLAZGO |
| Disco C: | 77.5 GB libres | OK |
| Vars | HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, PYTHONHASHSEED=42, seed=42 | OK |

Consecuencia del hallazgo de caché: la re-ejecución del oráculo de exp11 es imposible offline
hoy; ablaciones de retrieval nuevas (Tier B) quedan gateadas por la descarga aprobada.
Nota: la purga de caché implica que la corrida firmada de exp11 no es re-ejecutable en el
estado actual de la máquina (los JSONs firmados permanecen como evidencia).

## pytest (py 3.14, offline)
`7 passed, 1 skipped in 71.90s` — exacto a lo esperado (skip = `test_nli_output_is_softmax_probabilities`,
condicional a carga del cross-encoder). Warning cosmético: `torch.jit.script` deprecado en 3.14.

## Integridad de evidencia firmada
`git diff --stat nota3-evidencia-2026-06-11 HEAD -- experiments/results` →
8 archivos, **44 702 inserciones, 0 deleciones** (solo aditivos: v3/v4 en exp12 + exp14 nuevo).
Ningún directorio firmado modificado.

## Fidelidad v4_small (rescore: faithfulness_rescore_v3__small__vb_agree.json)
- [OK ] v4_small: celdas primary_answered — 16/16 exactas (tol 1e-09)
- [OK ] v4_small/between_scenario: mismo conjunto de pares — 24 pares
- [OK ] v4_small/between_scenario: pares exactos (d_z, p_bh, n, sig_bh) — 24/24
- [OK ] v4_small/between_model: mismo conjunto de pares — 18 pares
- [OK ] v4_small/between_model: pares exactos (d_z, p_bh, n, sig_bh) — 18/18
- [OK ] v4_small: 0/12 RAG-vs-RAG significativos — 0/12 sig
  - entre-modelos sig (v4_small): ['denso | granite4.1-8b vs denso | mistral-7b-instruct'] de 18 pares

## Fidelidad v4 (rescore: faithfulness_rescore_v3__base__vb_agree.json)
- [OK ] v4: celdas primary_answered — 16/16 exactas (tol 1e-09)
- [OK ] v4/between_scenario: mismo conjunto de pares — 24 pares
- [OK ] v4/between_scenario: pares exactos (d_z, p_bh, n, sig_bh) — 24/24
- [OK ] v4/between_model: mismo conjunto de pares — 18 pares
- [OK ] v4/between_model: pares exactos (d_z, p_bh, n, sig_bh) — 18/18
- [OK ] v4: 0/12 RAG-vs-RAG significativos — 0/12 sig
  - entre-modelos sig (v4): ['lexico | gemma4-e4b vs lexico | granite4.1-8b'] de 18 pares

- [OK ] 0/18 robusto entre verificadores (pares sig disjuntos) — small=['denso | granite4.1-8b vs denso | mistral-7b-instruct'] base=['lexico | gemma4-e4b vs lexico | granite4.1-8b']
- [OK ] Granite 0.235(75)/0.247(85)/0.299(87) (RESULTADOS_RESUMEN Tabla 6 v4) — lexico=0.235(75); denso=0.247(85); hibrido=0.299(87)
- [OK ] CSV tabla6_v4 == JSON v4_small (16 celdas) — 16/16

## Retrieval exp11 (consistencia; recomputación completa imposible offline)
- [OK ] exp11/bge-reranker-indep: total_queries=194
- [OK ] exp11/bge-reranker-indep: oracle_is_circular=False — oracle=BAAI/bge-reranker-large
- [OK ] exp11/bge-reranker-indep: 4 NDCG@5 vs RESULTADOS_RESUMEN — RAG Lexico=0.4421; RAG Semantico=0.6237; RAG Hibrido Propuesto=0.7405; RAG Hibrido=0.6026
- [OK ] exp11/ms-marco-circular: total_queries=194
- [OK ] exp11/ms-marco-circular: oracle_is_circular=True — oracle=cross-encoder/ms-marco-MiniLM-L-12-v2
- [OK ] exp11/ms-marco-circular: 4 NDCG@5 vs RESULTADOS_RESUMEN — RAG Lexico=0.5516; RAG Semantico=0.6494; RAG Hibrido Propuesto=0.9948; RAG Hibrido=0.6681
- Nota: los scores por par (query, chunk) del oráculo NO están persistidos en exp11;
  la recomputación completa del NDCG queda bloqueada hasta la descarga de modelos (decisión
  tomada: snapshot a data\models). Cerrable después con recompute a dir NUEVO, nunca in-place.

## exp13 expansión (25 q cross-cloud)
- [OK ] exp13: 25 queries
  - NDCG@5 (retrieval_metrics__bge-indep.json): exp_off | granite4.1-8b=0.852; exp_on | granite4.1-8b=0.820
- [OK ] exp13: NDCG off≈0.852 / on≈0.820 — off=[0.8521075013923834] on=[0.8202662249446669]
  - fidelidad primary (v2): exp_off | granite4.1-8b=0.285; exp_on | granite4.1-8b=0.324
- [OK ] exp13: fidelidad v2 off≈0.285 / on≈0.324 (RESUMEN §8, N7) — off=[0.2850636363636363] on=[0.32376]

## Veredicto
**TODO CUADRA** — cifras v4 reproducidas en memoria desde JSONs firmados; ningún archivo de experiments/ modificado.
