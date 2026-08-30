# KNOB_MAP — Fase de verano (ablación exp15+)

Inventario verificado (2026-07-22; re-verificado 2026-07-30) de cada perilla efectiva del pipeline
y su consumidor real. Los valores vivos están hardcodeados en `src/pipeline/pipeline_config.py`
y en los runners de `scripts/`.

**Config muerta vs viva (re-verificado 2026-07-30 por grep sobre todo el repo):**
- `config/evaluation_config.yaml` — **cero consumidores** (Python, yaml, sh, ps1). Es documentación.
- `config/config.yaml` — **secciones retrieval/reranking muertas** en runtime de experimentos, pero
  el archivo **SÍ se consume** en el lado corpus: `ingestion_pipeline.py:21`, `deduplicator.py:313`,
  `text_cleaner.py:199`, `build_index.py:39`. **No retirarlo**; retirar solo sus secciones muertas.
- `config/cloud_services.yaml`, `config/terminology_mappings.yaml` — vivos
  (`ingestion_pipeline.py:28`, `terminology_normalizer.py:25`, `query_processor.py:83`).

## Perillas efectivas (valores vigentes)

| Perilla | Definida en | Consumida en | Valor |
|---|---|---|---|
| `fusion_method` | `src/pipeline/pipeline_config.py:73` | `rag_pipeline.py:167` → `hybrid_index.py:128-133` | `rrf` |
| `rrf_k` | `pipeline_config.py:75` | `hybrid_retriever.py:77` → `hybrid_index.py:171-174` (**solo constructor** de `HybridRetriever`) | 60 |
| `alpha` (solo linear) | `pipeline_config.py:74` | `hybrid_index.py:154` | 0.5 |
| `retrieval_top_k` (candidatos) | `pipeline_config.py:28` | `rag_pipeline.py:228-229` | 50 |
| `final_top_k` (fragmentos al LLM) | `pipeline_config.py:30` | `rag_pipeline.py:248-251` | 5 |
| `reranker_top_k` | `pipeline_config.py:29` | declarado; la ruta de rerank usa `final_top_k` | 20 |
| Modelo reranker | `pipeline_config.py:77` | `rag_pipeline.py:182-188` → `src/reranking/cross_encoder_reranker.py` | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Modelo embedding | `embedding_manager.py:29` (mapa `bge-large`) | `load_hybrid_index` (`rag_pipeline.py:499-512`) | `BAAI/bge-large-en-v1.5` (1024d) |
| Chunking | `pipeline_config.py:34-35` | claves de archivo de índice (`data/indices/`) | adaptive / 500 (overlap 50 al construir) |
| `query_expansion` (D11) | `pipeline_config.py:26,84` | `rag_pipeline.py:218,231` → `hybrid_retriever.py:57-68` (bajo rrf solo pierna BM25) | **False** (N4/exp13) |
| `terminology_normalization` | `pipeline_config.py:27,85` | solo lado corpus (`run.py:63-77`); sin efecto query-time | True |
| Escenarios de generación | `scripts/run_generation_matrix.py:57-62` | mapea a configs de exp11 | sin_rag / lexico / denso / hibrido |
| Decodificación LLM | `run_generation_matrix.py:211`; `llm_manager.py:337-350` | Ollama | granite4.1:8b, temp 0.0, seed 42, num_predict 1024 |
| Verificador NLI | `src/generation/hallucination_detector.py:187` | runtime | `cross-encoder/nli-deberta-v3-small` (base vía `scripts/rescore_nli_v3.py --verifier base`) |
| Umbrales NLI | `hallucination_detector.py:188-189` | `decide_nli_status` (:133-181) | ent 0.7 / contr 0.7 |
| Variante NLI | `hallucination_detector.py:203` | `decide_nli_status` | `vb_agree` (contradicted exige ≥2 chunks >0.7) |
| Denominador fidelidad | `scripts/compute_faithfulness_metrics.py:224-229` | familias v2/v3/v4 | `primary_answered` + 3 sensibilidades |
| Umbral oráculo retrieval | `scripts/compute_retrieval_metrics.py:44-59` | métricas binarias | p50 primario (t0 legacy); NDCG graded = headline |

## Estándar de verificación (formalizado 2026-08-04, ledger entrada 22)

**Tres verificadores, DOS familias: NLI-small + NLI-base (entailment) + HHEM-2.1 (grounding).**
La ortogonalidad de familia es lo que da valor a la triangulación; dos NLI de la misma familia dan
votos correlacionados y HHEM aporta la señal independiente. **No existe ni existió un "trío NLI"**.

`deberta-large` queda **RETIRADO**: se detuvo en 11/12 configs, el runner solo promueve
`nli_probs__large.json.gz` al completar las 12, así que el artefacto nunca se escribió y **ninguna
cifra reportada lo consumió**. Su `.partial` se conserva committeado como registro del intento.
`compute_exp15_ensemble_sweep.py` declara `NLI_MEMBERS_EXPECTED = ("small", "base")` y ancla la
etiqueta de `E2_vote` en `len(members) >= 3` — con dos miembros el voto es **unanimidad**, no mayoría,
y el nombre emitido lo dice (`E2_vote[2m=unanimity]`).

**Clases de declinación (defecto #7, misma entrada).** El valor serializado `pure_decline` mide un
**prefijo** (marcador en los primeros 300 chars), **no** un rechazo: la mayoría de esas filas afirman
claims igual. Se lee como `decline_prefix` (`DISPLAY_LABELS`) y se acompaña de `asserts_content`
(¿hay claims genuinos?). Para cualquier argumento de abstención o usabilidad, la cifra correcta es
`asserts_nothing_rate` de `guards.json`. El token de serialización **no se renombra**: vive en 7
archivos de evidencia firmada de los que `verify_v4_offline.py` re-deriva cifras publicadas.

## Rutas canónicas (no negociables para exp15+)

- **Prompt canónico** = `scripts/run_generation_matrix.py` (`build_prompt`, routing por query_type;
  115/194 no-default) + `src/generation/prompt_templates.py`.
  **Corregido 2026-07-30:** la afirmación previa («`RAGPipeline.query()` **NO** replica esta ruta»)
  era imprecisa. Verificado en `rag_pipeline.py:270-291`: la construcción del prompt es la MISMA
  (`build_context` → `get_template` → rama `cross_cloud` con `context_by_provider` → `SYSTEM_PROMPT`);
  `rgm.build_prompt` está documentado como réplica de ella. La diferencia real es el **origen del
  contexto**: `rgm` recibe `retrieved_ids` firmados de exp11, `RAGPipeline` recupera en vivo. Sigue
  vigente la regla operativa: para **brazos comparables** usar `rgm` (contexto congelado); `RAGPipeline`
  es la ruta de **despliegue**. La paridad de prompt debe quedar cubierta por test antes de empaquetar.
  **[TEXTO CANÓNICO — fijado 2026-08-21 07:40 por Claude Code, idéntico en
  `docs/TRACEABILITY_nota3.md`]** Esta entrada y la de `TRACEABILITY_nota3.md` parecían
  contradecirse; la re-auditoría del 2026-08-04 (entrada [Kimi Code] en `CLAUDE.md`) mostró que
  **ambas son correctas en su contexto y que la contradicción era de redacción, no de código**. La
  **construcción** del prompt es la misma en las dos rutas. Lo que difiere es el **origen del
  `query_type`**: el ruteo es una perilla **opt-in**, `prompt_routing`
  (`src/pipeline/pipeline_config.py:49`, por defecto `False`), y sin ella el pipeline asigna
  `query_type = "default"` a todo (`rag_pipeline.py:251-254` en `query()` y `417-420` en
  `query_stream()`). `TRACEABILITY_nota3.md` describe **la config legacy que se evaluó** (perilla
  apagada); esta entrada describe **la ruta con `prompt_routing=True`**, que es la que lleva
  `SURVEY_DEPLOY`. La segunda diferencia —el **origen del contexto**— sigue vigente en ambos
  contextos y es la que manda la regla operativa de arriba.
- **Contextos exp12** = `retrieved_ids` de exp11 (top-5 post-rerank, orden exp11); el retrieval no
  se re-ejecuta en generación. exp11 también guarda el orden pre-rerank RRF (top-5) por query.
- **Caché LLM** `data/llm_cache/{model}_cache.json`, key = sha256(config_name ‖ prompt ‖ system ‖
  temperature ‖ seed ‖ max_tokens) → segregación por brazo de ablación = `config_name` único
  (convención exp15: `"<arm> | <model>"`). Hits de caché reportan `latency_ms=0` (`from_cache=True`):
  excluir de agregados de latencia.
- **Checkpoint/resume**: `checkpoint__{label}__{scenario}.json` cada 10 queries, resume por
  `completed_ids` (`run_generation_matrix.py:199-203,244-248`).

## Advertencias operativas

- `scripts/compute_retrieval_metrics.py` y `scripts/compute_faithfulness_metrics.py` escriben
  **in-place en el dir del experimento** que se les pasa; `scripts/rescore_nli_v3.py` hardcodea
  `exp12_matrix`. Para verificación offline usar `scripts/verify_v4_offline.py` (importa funciones,
  jamás ejecuta los main()). Scripts exp15 deben parametrizar el dir de salida.
- Único índice construido: `bge-large_adaptive_500`. Ablaciones de chunking/embedding exigen
  reconstruir índice (costo alto, fuera de Tier 0/A/B inicial).
- Modelos ausentes de todo caché local (2026-07-22): bge-large-en-v1.5, ms-marco-MiniLM-L-12-v2,
  bge-reranker-large → descarga aprobada a `data\models\` antes de Tier B.
- Ollama 0.22.1 congelado durante la fase (H5: determinismo dependiente del entorno; sonda 3× por
  brazo obligatoria). Gemma/mistral no deterministas a temp=0 incluso en frío.
