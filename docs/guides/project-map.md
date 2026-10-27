# Mapa del proyecto

Guía de orientación para revisar el repositorio sin conocerlo. Fecha: 2026-10-27 (rama `docs/f020-despliegue`, base `8fdaa65`).
Etiquetas: [VERIFICADO] = comprobado con el comando citado en esa fecha; [DECLARADO] = tomado de un documento del repositorio.

Convención: las rutas entre comillas invertidas están versionadas en git y `scripts/check_project_map.py` comprueba que existen
(y que el número de línea citado no excede el archivo). Las rutas locales ignoradas por git se escriben sin comillas invertidas.

## 1. Árbol anotado

| Ruta | ¿En git? | Contenido |
|---|---|---|
| `src/ingestion/` | sí | Crawlers de AWS, Azure, GCP, Kubernetes y CNCF; `doc_parser` e `ingestion_pipeline` |
| `src/preprocessing/` | sí | Limpieza, deduplicación, metadatos y normalización de terminología |
| `src/chunking/` | sí | Chunkers fijo, recursivo, semántico, jerárquico y adaptativo (el adaptativo es el usado) |
| `src/embedding/` | sí | Embeddings BGE e índices BM25, FAISS e híbrido (`src/embedding/index/`) |
| `src/retrieval/` | sí | Procesador de consultas, recuperadores BM25, denso e híbrido, balanceo de proveedores |
| `src/reranking/` | sí | Cross-encoder y alternativas sin reranking |
| `src/generation/` | sí | Ollama, plantillas de prompt, detector de alucinaciones (NLI) y formateo con citas |
| `src/pipeline/` | sí | `RAGPipeline`, que orquesta todas las etapas |
| `src/evaluation/` | sí | Métricas de recuperación y fidelidad, estadística y análisis del estudio |
| `src/ui/` | sí | App Streamlit y estudio con usuarios |
| `src/utils/` | sí | Guarda de evidencia firmada, reproducibilidad y modelos locales |
| `config/` | sí | `config/config.yaml`, `config/terminology_mappings.yaml`, cuestionarios SUS y UEQ-S |
| `data/evaluation/` | sí | Consultas de evaluación y el marcador de juicios de relevancia |
| `data/corpus_stats.json` | sí | Estadística del corpus |
| data/raw/, data/processed/, data/chunks/, data/indices/, data/embeddings/, data/models/ | no (local, `.gitignore` líneas 16-21 y 26) | Corpus, índices y modelos, unos 700 MB |
| `experiments/results/` | sí | Evidencia experimental exp3 a exp19b (inmutable, sección 5) |
| `scripts/` | sí | Ejecutores, cálculo de métricas, verificadores y operador del estudio (`scripts/study_operator/`) |
| `tests/` | sí | Suite pytest (capa 2 de `init.sh`) |
| `paper/` | sí | Paper LACCI, auditoría y ledgers |
| `docs/` | sí | Documentación ([índice](../README.md)) |
| `.harness/` | sí | Estado de tareas (`.harness/feature_list.json`) y traspaso (`.harness/progress.md`) |
| .venv-app/, .worktrees/, .env, showcase/ | no (local) | Entorno, worktrees, credenciales y material de demostración |
| C:/CloudRAG/ y D:/CloudRAG_archivo/ | no (fuera del repo) | Registros de ejecución del despliegue; índice en C:/CloudRAG/LEEME_CLOUDRAG.md |

## 2. Corpus

Los documentos están en data/processed/{aws,azure,gcp}/ (un JSON por documento) y los chunks en
data/chunks/adaptive/size_500/ (un JSON por chunk; adaptativo, 500 tokens, solapamiento 50). Los índices están en data/indices/
(BM25, FAISS bge-large y mapa de chunks) y los embeddings en data/embeddings/. Todo es local.

| Proveedor | Documentos procesados | Chunks |
|---|---:|---:|
| AWS | 996 | 6 366 |
| Azure | 1 461 | 9 606 |
| GCP | 240 | 8 509 |
| Total | 2 697 | 24 481 |

[VERIFICADO 2026-10-27] Comandos ejecutados en la raíz del checkout principal:

```bash
cd data && for d in raw processed; do for p in aws azure gcp; do echo "$d/$p: $(find $d/$p -type f | wc -l)"; done; done
python -c "import json,glob,collections; c=collections.Counter(json.load(open(f,encoding='utf-8')).get('cloud_provider','?') for f in glob.glob('data/chunks/adaptive/size_500/*.json')); print(dict(c), sum(c.values()))"
```

Las cifras coinciden con `data/corpus_stats.json`. El índice contiene 2 644 `doc_id`: el submuestreo estratificado de Azure de la
Fase 2.5 (seed 42) dejó 53 documentos sin chunks.

## 3. Claves de los datos (solo nombres de campo)

- **Documento** (data/processed/): `doc_id`, `title`, `content`, `sections`, `code_blocks`, `tables`, `cloud_provider`, `service_name`,
  `service_category`, `doc_type`, `url_source`, `file_path`, `last_updated`, `word_count`, `char_count`, `has_code`, `has_tables`, `heading_count`.
- **Chunk** (data/chunks/): `chunk_id`, `doc_id`, `text`, `token_count`, `cloud_provider`, `service_name`, `doc_type`, `url_source`,
  `section_hierarchy`, `heading_path`, `chunk_index`, `total_chunks`, `has_code`, `has_table`, `content_type`, `normalized_terms`,
  `detected_siglas`, `cross_cloud_equivalences`, `chunking_strategy`, `chunk_size_config`, `embedding`, `bm25_tokens`.
- **`data/evaluation/test_queries.json`**: lista de 194 consultas con `query_id`, `question`, `answer`, `relevant_chunk_ids`,
  `cloud_providers`, `query_type`, `difficulty` y `category`. Tipos: 66 comparativas, 65 procedimentales y 63 factuales.
  `relevant_chunk_ids` está vacío en las 194.
- **`data/evaluation/ground_truth.json`**: marcador con `version`, `description` y `judgments` vacío. No hay juicios humanos de
  relevancia; por eso la recuperación se evalúa con un oráculo independiente (sección 6).
- **Resultados de un experimento** (`experiments/results/exp12_matrix/results.json`): `experiment_id`, `seed`, `temperature`,
  `context_source`, `metrics_schema_version`, `models`, `scenarios` y `configs`. Cada configuración «escenario | modelo» tiene
  `results` con `query_id`, `answer`, `retrieved_ids`, `hallucination_metrics` (incluye `faithfulness` y `is_honest_decline`),
  `tokens`, `latency` y `cost_proxy`.

## 4. Pipeline: etapa, módulo y función

| Etapa | Módulo y función (archivo:línea) |
|---|---|
| Procesamiento y expansión de la consulta | `src/retrieval/query_processor.py:106` (`process`), `src/retrieval/query_processor.py:187` (`_expand_terms`) |
| BM25 (k1 = 1,2; b = 0,75) | `src/retrieval/bm25_retriever.py:35`, `src/embedding/index/bm25_index.py:93` |
| Denso BGE | `src/retrieval/dense_retriever.py:22`, `src/embedding/embedding_manager.py:140`, `src/embedding/index/faiss_index.py:82` |
| Fusión RRF (k = 60) | `src/retrieval/hybrid_retriever.py:34`, `src/embedding/index/hybrid_index.py:159` (`_rrf_fusion`) |
| Balanceo de proveedores (exp17) | `src/retrieval/coverage_balancer.py:93` |
| Reranking con cross-encoder | `src/reranking/cross_encoder_reranker.py:83` (`rerank`) |
| Prompt | `src/generation/prompt_templates.py:15` (`SYSTEM_PROMPT`), `src/generation/prompt_templates.py:112` (`build_context`) |
| Generación con Ollama | `src/generation/llm_manager.py:228` (`generate`), `src/generation/llm_manager.py:348` (`_generate_ollama`) |
| Verificación NLI | `src/generation/hallucination_detector.py:244` (`check`), `src/generation/hallucination_detector.py:478` (`_nli_matching`) |
| Verificación HHEM (solo fase de verano) | `scripts/rescore_grounding_exp15.py:50` (`load_hhem`) |
| Citas y declinación honesta | `src/generation/response_formatter.py:91` (`format`), `src/generation/response_formatter.py:165` (`_parse_citations`) |
| Orquestación | `src/pipeline/rag_pipeline.py:237` (`query`), `src/pipeline/rag_pipeline.py:584` (`create_pipeline`) |
| Interfaz | `src/ui/app.py:43` (páginas); CLI en `run.py` |
| Métricas | `src/evaluation/retrieval_metrics.py:68` (`ndcg_at_k`), `src/evaluation/statistical_analysis.py:160` (`paired_comparison`), `src/evaluation/statistical_analysis.py:273` (corrección BH) |
| Guarda de evidencia | `src/utils/signed_evidence.py:92` (`guard_write`) |

## 5. Evidencia inmutable y ledgers

- **Evidencia experimental:** `experiments/results/` (exp3 a exp14, exp8b y la fase de verano exp15 a exp19b). La guarda en código
  es `src/utils/signed_evidence.py:46` (registro invertido: todo protegido salvo lo declarado vivo) y `src/utils/signed_evidence.py:92-109`
  (aborta si se sobrescribe un archivo; los recálculos van en archivos `_vN` nuevos). Pruebas: `tests/test_signed_evidence_guard.py`.
- **Auditoría y ledgers:** `paper/audit_findings.md`, `paper/audit_outputs/exp8_stats_corrected.csv`, `paper/audit_findings_cc_addenda.md`,
  `paper/correction_log.md` y `paper/summer_ablation_log.md`. Su inmutabilidad es una regla escrita, no una guarda en código
  ([guardrails.md](guardrails.md), [ADR-0001](../decisions/adr-log.md#adr-0001)).

## 6. Preguntas probables y respuestas cortas

Cifras [DECLARADO] de [results-summary.md](../reports/results-summary.md) y [summer-benchmarks.md](../reports/summer-benchmarks.md).

1. **¿Qué es el sistema?** Un RAG híbrido sobre documentación de AWS, Azure y GCP: BM25 más BGE, fusión RRF, cross-encoder y generación local con Ollama.
   La respuesta se verifica con NLI por afirmación y se cita la fuente.
2. **¿De qué tamaño es el corpus?** 2 697 documentos procesados y 24 481 chunks (sección 2).
   El conteo es reproducible con los dos comandos citados.
3. **¿Cómo se evaluó la recuperación sin juicios humanos?** Con un oráculo independiente (bge) sobre 194 consultas, porque `data/evaluation/ground_truth.json` está vacío.
   El oráculo anterior era circular y su 0,995 se retiró (`docs/reports/results-summary.md:25-28`).
4. **¿Cuánto rinde cada método?** NDCG@5: BM25 0,442; denso 0,624; RRF 0,603; híbrido con reranking 0,740.
   Fuente: `docs/reports/results-summary.md:25-28`.
5. **¿De dónde viene la ventaja del híbrido?** Del reranking, no de la fusión: RRF sin reranking no difiere del denso.
   Híbrido con reranking frente a denso: d_z = +0,45, p_BH < 0,001 (`docs/reports/results-summary.md:30-34`).
6. **¿Ayuda la expansión de consultas?** No: el +16,8 % anterior se retiró y en exp13 la expansión es algo peor (0,852 frente a 0,820, no significativo).
   Fuente: `docs/reports/results-summary.md:44`.
7. **¿Cuál es la fidelidad citable?** La tabla 6 v4: con el híbrido, granite 0,299, gemma 0,317, mistral 0,288 y qwen 0,294.
   Fuente: `docs/reports/results-summary.md:103-110`.
8. **¿El método de recuperación cambia la fidelidad?** No: 0 de 12 pares RAG contra RAG son significativos con los dos verificadores.
   Entre modelos, 0 de 18 (`docs/reports/results-summary.md:118`, `docs/reports/results-summary.md:136`).
9. **¿RAG mejora frente a no usar RAG?** Sí, para granite, gemma y mistral: d_z entre 0,86 y 1,09, p_BH < 0,001.
   Fuente: `docs/reports/results-summary.md:124`.
10. **¿Por qué granite parece peor en fidelidad?** Declina en el 62 a 66 % de las consultas, frente al 20 a 25 % de mistral.
    Las declinaciones honestas se tratan con denominadores aparte (`docs/reports/results-summary.md:194`).
11. **¿Qué latencia tiene?** Mediana de unos 38 s con gemma y unos 239 s con qwen en la matriz exp12.
    Fuente: `docs/reports/results-summary.md:198-199`.
12. **¿Qué aportó la fase de verano?** Solo la selección de evidencia movió la fidelidad: exp17 elevó la cobertura de 7/25 a 25/25 y HHEM de 0,477 a 0,558.
    Es direccional y no significativo (n = 25, p = 0,24; `docs/reports/summer-benchmarks.md:129-140`).
13. **¿Cómo se evita alterar resultados?** Con la guarda `src/utils/signed_evidence.py:92`: no se sobrescribe evidencia, los recálculos van en `_vN`.
    Además rige la regla [ADR-0001](../decisions/adr-log.md#adr-0001).
14. **¿Dónde están las correcciones y la auditoría?** En `paper/audit_findings.md` y `paper/correction_log.md`.
    Las adendas N1 a N9 están en `paper/audit_findings_cc_addenda.md`.
15. **¿Cómo se reproduce?** Con [reproduce.md](reproduce.md) y `init.sh`, que verifica secretos, tests y documentación.
    Usa Python 3.14, seed 42 y los modelos en modo offline.
16. **¿Qué modelos se usan?** Granite 4.1 8B como principal; gemma, mistral y qwen en la matriz.
    Detalle en [models.md](../architecture/models.md) y [ADR-0002](../decisions/adr-log.md#adr-0002).
17. **¿Cómo está diseñada la arquitectura?** Hay una descripción ISO 42010 con vistas en [system-design.md](../architecture/system-design.md).
    El código de cada etapa está en la sección 4.
18. **¿Qué es el estudio con usuarios?** Una comparación cegada de dos condiciones con SUS, UEQ-S y Likert, desplegada en GCP.
    No admite participantes antes de la aprobación ética y del GO de la compuerta de latencia (`docs/STUDY_I7_RUNBOOK_FASE_B.md`).
19. **¿Por qué hay tantas decisiones registradas?** Cada cambio de método queda en un ADR que no se edita.
    El registro está en [adr-log.md](../decisions/adr-log.md).
20. **¿Qué limitaciones reconoce el trabajo?** No hay juicios humanos de relevancia, n = 25 en la fase de verano y la fidelidad se mide con verificadores automáticos.
    Las diferencias de fidelidad entre métodos no son significativas.
