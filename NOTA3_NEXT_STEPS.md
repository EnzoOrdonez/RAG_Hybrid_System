# Nota 3 — estado y próximos pasos (2026-06-07)

> **ACTUALIZADO 2026-06-11: RONDA COMPLETA.** exp11 + exp12 (4 modelos × 4 escenarios × 194)
> + exp13 (expansión) terminados; Phase 6 generada. Resumen autoritativo: **`RESULTADOS_RESUMEN.md`**.
> Lo de abajo es el runbook original (histórico).
> **ACTUALIZADO 2026-07-02 (N9):** figuras f1-f4 ya generadas (f2/f3 en métrica v2; regenerar con
> v4 antes del A.3). Queda: reescritura A.3/A.1/paper con cifras **v4** (Tabla 6 v4, "0/18
> entre-modelos robusto bajo ambos verificadores" — framing B, cierre N9), revisión humana de `claim_audit_sample_v3.csv` (0/50). H5 ABORTADO en el cierre N9 (ruido de entorno; parcial 140/300 = evidencia en
> exp14_h5_replicas/). Nota: exp13 usó 25 q cross-cloud (no 30, N6).

## exp12 matrix — COMPLETO (histórico: instrucciones de resume/run)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 \
python scripts/run_generation_matrix.py --exp-id exp12_matrix \
  --models granite4.1:8b,gemma4:e4b,mistral:7b-instruct,qwen3.5:9b
```

- 4 modelos × 4 escenarios × 194 q, **temp=0**, num_predict=1024. ETA ≈ **55 h**.
- Orden: granite4.1 (headline) → gemma4:e4b → mistral:7b → **qwen3.5 (cuello, último)**.
- **Checkpoint/resume** por (modelo, escenario) cada 10 q:
  `experiments/results/exp12_matrix/checkpoint__<modelo>__<escenario>.json`.
- **RESUME**: re-ejecutar el MISMO comando — salta lo ya hecho vía checkpoint.
- **Monitor**: `ls experiments/results/exp12_matrix/checkpoint__*.json`; `tail` del log.
- Caché LLM **limpiada** antes de arrancar (latencias limpias). NO re-limpiar entre resumes.
- Al terminar, el runner consolida todo en `experiments/results/exp12_matrix/results.json`.

## Determinismo por modelo (smoke, respuesta a Fase 1d)
granite4.1 ✅ y qwen3.5 ✅ son bit-idénticos ×3 @ temp=0; **gemma4:e4b ❌ y mistral:7b ❌
NO** (no-determinismo a nivel de kernel Ollama/llama.cpp). La caché congela una muestra por
query (la corrida es reproducible-desde-caché). **Headline = granite4.1** (determinista +
RAG-tuned + entra en 6 GB). Limitación documentada en MODELS.md.

## Tras completar exp12 — Phase 6

1. **Fidelidad pareada (LISTO):**
   `python scripts/compute_faithfulness_metrics.py --experiment exp12_matrix`
   → `faithfulness_metrics.json` (familias BH por RQ: entre-escenarios / entre-modelos;
   McNemar para honest_decline). Ya validado en exp8.
2. **Latencias p50/p95** por escenario×modelo — TODO `scripts/_latency_p50p95.py`
   (excluir `from_cache=True`; campos `latency.{generation_ms,hallucination_check_ms,total_ms}`,
   `cost_proxy.tok_per_s`).
3. **exp13 expansión ON vs OFF** (30 q cross-cloud = `len(cloud_providers)>1`,
   materializar el subset) con el modelo de mejor fidelidad de exp12 — TODO runner
   (hybrid + `use_expansion` ON/OFF vía D11, generación determinista).
4. **Tablas profe nota3** — TODO `scripts/_export_paper_tables_nota3.py`:
   Tabla 5 (por escenario, modelo principal), Tabla 6 (matriz fidelidad 4×4 con "sin RAG"
   anotado `answered_rate`/`decline_rate`), latencias; coma decimal "0,923".
   Retrieval (Tabla 4 + estabilidad de oráculos) **YA** en `output/tables/nota3/`.
5. **RESULTADOS_RESUMEN.md** — hallazgos, deltas vs ronda previa, veredicto exp7,
   delta reranker (exp11 vs exp10 por oráculo), mejor modelo de fidelidad, puntos atacables.

## Ya hecho (retrieval + fixes; brazo de retrieval cerrado)
- **exp11 retrieval + multi-oráculo**: híbrido post-rerank NDCG@5 0,995 (ms-marco circular)
  → **0,740 (bge-reranker independiente)**; > denso **real** (d_z=+0,45, p_BH<0,001); fusión
  RRF sola ≈ denso (n.s.). Orden estable entre oráculos.
  `output/tables/nota3/oracle_stability__exp11_retrieval194_fullrerank.{md,csv}`.
- Fixes: D12 (reranker full-text), D11 (expansión real en RRF, default OFF), determinismo
  (temp=0 + hashseed re-exec), test NLI, multi-oráculo + percentil, stats de fidelidad,
  np.bool_, ledgers (N1 exp7 retirado / N2 circularidad / N3 sin-RAG), README, MODELS.md.
