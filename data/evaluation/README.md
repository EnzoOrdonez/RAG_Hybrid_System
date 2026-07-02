# Conjunto de evaluación — curación 200 → 194 (Nota 3)

Este directorio contiene el conjunto de consultas del benchmark de la Nota 3 y su bitácora
de curación (pedido del asesor, 20/06/2026 — punto 5 de las oportunidades de mejora).

## Archivos

| Archivo | Contenido |
|---|---|
| `test_queries.json` | **194 consultas curadas** (set experimental de exp10–exp13). Campos: `query_id`, `question`, `answer`, `relevant_chunk_ids`, `cloud_providers`, `query_type`, `difficulty`, `category`. |
| `test_queries.backup_pre_phase3.json` | Backup del set original de **200** consultas, previo a la depuración (Fase 3, junio 2026). Verificado: backup = 194 actuales + 6 removidas, sets exactos. |
| `test_queries_removed_log.json` | Bitácora de las **6 consultas removidas** en la depuración 200→194, con el motivo por consulta (dominios excluidos del corpus experimental / duplicidad / calidad). |
| `cross_cloud_subset.json` | Subconjunto de **25 consultas multi-proveedor** (`len(cloud_providers) > 1`) usado por exp13 (expansión ON vs OFF). Es un subconjunto exacto de `test_queries.json`. Nota: versiones tempranas del A.3 decían "30"; la cifra correcta es 25 (ledger N6). |
| `ground_truth.json` | **Placeholder histórico** (feb. 2026, `judgments: []`). Ningún código lo consume: el ground truth efectivo de retrieval es el oráculo dinámico por cross-encoder (circular ms-marco como referencia + `BAAI/bge-reranker-large` independiente — ledger N2/Flag 17). Se poblará solo si se completa la anotación humana pendiente (H3). |

## Notas

- `relevant_chunk_ids` está vacío en las 194 consultas **por diseño**: la relevancia se
  computa con el pool multi-sistema puntuado por el oráculo (ver
  `scripts/compute_retrieval_metrics.py` y `docs/TRACEABILITY_nota3.md`).
- exp9 (control LLM-only, sin RAG) corrió ANTES de la depuración, sobre las 200 consultas
  originales — por eso su cifra publicada es "195/200". exp10–exp13 usan las 194.
- Conteos verificados contra disco en la auditoría N9 (2026-07-01):
  `output/audit/N9_auditoria_integral_fase1_2026-07-01.md`.
