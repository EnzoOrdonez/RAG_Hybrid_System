# Descripción de arquitectura de CloudRAG (ISO/IEC/IEEE 42010)

**Sistema de interés:** CloudRAG (`hybrid-rag-system`), un sistema RAG híbrido que responde preguntas
sobre documentación oficial de AWS, Azure y GCP, más la evidencia experimental de la tesis y del paper
IEEE LACCI 2026 que lo evalúan.
**Versión de la descripción:** 2026-10-25, sobre la rama `chore/docs-harness`. Cada afirmación cita el
archivo que la respalda. Si el código y este documento discrepan, manda el código y este documento se
corrige.

## 1. Partes interesadas y preocupaciones

| Parte interesada | Preocupaciones principales |
|---|---|
| Autor (Enzo Ordoñez Flores) | C1 reproducibilidad de cada cifra citada; C2 integridad de la evidencia firmada; C3 costo y hardware (GPU de 6 GB) |
| Coautor y asesor (Winston Lewis Fuentes, `CITATION.cff`) | C1, C2 y C4: validez metodológica (oráculos, verificadores, estadística pareada) |
| Revisores y jurado (LACCI, Seminario) | C4 y C5: trazabilidad de tablas y figuras hasta experimento, script y salida |
| Participantes del estudio con usuarios | C6: privacidad, sesiones y latencia aceptable. El estudio está bloqueado hasta la aprobación ética |
| Agentes de IA que mantienen el repo (Claude, Codex, Kimi) | C7: arranque en frío sin ambigüedad, límites de escritura claros y verificación con exit code |
| Lectores del repositorio público | C8: no exponer secretos, IPs ni datos personales |

## 2. Puntos de vista

| Punto de vista | Preocupaciones que cubre | Convenciones |
|---|---|---|
| PV-1 Funcional (pipeline) | C1, C4, C5 | Etapas y configuraciones con nombre, según `src/pipeline/` |
| PV-2 Datos y evidencia | C1, C2, C5 | Directorios y su regla de escritura |
| PV-3 Despliegue y ejecución | C3, C6, C8 | Nodos, procesos y variables de entorno |
| PV-4 Verificación | C1, C2, C7, C8 | Comandos con exit code y quién los ejecuta |

## 3. Vista funcional (PV-1)

Docstring de `src/pipeline/rag_pipeline.py`:

```
query -> processing -> retrieval -> reranking -> generation -> hallucination check
```

| Etapa | Módulo | Tecnología (README «Architecture», `pipeline_config.py`) |
|---|---|---|
| Ingesta | `src/ingestion/` | Crawlers de AWS, Azure, GCP, Kubernetes y CNCF (los dos últimos excluidos de la evidencia de Nota 3) |
| Preprocesamiento | `src/preprocessing/` | Limpieza, normalización terminológica, deduplicación |
| Fragmentación | `src/chunking/` | 5 estrategias; la medida es adaptativa de 500 tokens |
| Indexado | `src/embedding/` | `BAAI/bge-large-en-v1.5` (1024 dim) en FAISS IndexFlatIP; BM25 con k1=1,2 y b=0,75 (`config/config.yaml`) |
| Recuperación | `src/retrieval/` | BM25, denso e híbrido. Fusión RRF con k=60: puntuación 1/(rrf_k+rank+1) (`hybrid_index.py`) |
| Reordenamiento | `src/reranking/` | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Generación | `src/generation/llm_manager.py` | Ollama local, temperatura 0 |
| Verificación de fidelidad | `src/generation/hallucination_detector.py` | NLI `nli-deberta-v3-small` (runtime), `-base` como segundo verificador, HHEM en la fase de verano |
| Interfaz | `src/ui/` | Streamlit. Modo `participant` por defecto (`CLOUDRAG_MODE`) y cinco vistas solo en `development` |

Configuraciones con nombre (`src/pipeline/pipeline_config.py`):

- `BASELINE_LEXICAL`, `BASELINE_SEMANTIC`, `PROPOSED_HYBRID` y `LLM_ONLY_NO_RAG` forman el registro experimental (`get_config`).
- `SURVEY_DEPLOY` es el despliegue de encuestas, fuera del registro: ver [ADR-0005](../decisions/adr-log.md#adr-0005).
- `PROPOSED_HYBRID`: híbrido RRF, top-50 de recuperación, top-20 al reordenador, top-5 final, expansión apagada ([ADR-0003](../decisions/adr-log.md#adr-0003)).

Modelos evaluados y su selección: [models.md](models.md).

## 4. Vista de datos y evidencia (PV-2)

| Ubicación | Contenido | Regla de escritura |
|---|---|---|
| `data/raw`, `data/processed`, `data/chunks` | Corpus. Nota 3 usa el subconjunto AWS/Azure/GCP: 24 481 fragmentos | Regenerable; partes ignoradas por git |
| `data/indices`, `data/models` | Índices FAISS/BM25 y snapshots de modelos | Ignorados por git; procedencia en [models.md](models.md) |
| `experiments/results/exp3..exp14`, `exp8b` | Evidencia firmada de Nota 3 | Inmutable ([ADR-0001](../decisions/adr-log.md#adr-0001)); la hace cumplir `src/utils/signed_evidence.py` |
| `experiments/results/exp15..exp19b` | Evidencia de la fase de verano | Congelada; `LIVE_EXPERIMENTS = frozenset()` ([ADR-0009](../decisions/adr-log.md#adr-0009)) |
| `experiments/probes/` | Sondas declaradas no-evidencia | Se puede escribir; fuera del descubrimiento ([ADR-0011](../decisions/adr-log.md#adr-0011)) |
| `output/tables`, `output/figures`, `output/audit` | Tablas, figuras y auditorías derivadas | Recálculos con sufijo `_vN` |
| `paper/` | Ledgers append-only y prosa del paper | Inmutables o append-only ([guardrails.md](../guides/guardrails.md)) |

La correspondencia entre tabla citada, experimento, script y salida está en `docs/TRACEABILITY_nota3.md`.

## 5. Vista de despliegue y ejecución (PV-3)

| Nodo | Procesos | Notas |
|---|---|---|
| Estación del autor (Windows, RTX 3060 Laptop de 6 GB) | Python 3.14 con stack ML, Ollama en `localhost:11434`, Streamlit | VRAM medida de los modelos evaluados: 5,2 a 5,5 GB ([models.md](models.md)) |
| Imagen de contenedor (`Dockerfile`) | `python:3.14.3-slim-bookworm` fijada por digest, `scripts/cloud_entrypoint.py` | `CLOUDRAG_MODE=participant`, `HF_HUB_OFFLINE=1`, `PYTHONHASHSEED=42`, `PYTHONUTF8=1`. Construcción en `docs/CLOUD_DEPLOYMENT.md` |
| Despliegue del estudio en la nube | Operador y runbooks en `docs/STUDY_*` | Registros de ejecución fuera del repo, en `C:\CloudRAG\` (índice `LEEME_CLOUDRAG.md`). Las direcciones y credenciales no se versionan (C8) |

Variables obligatorias para reproducir: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONHASHSEED=42`,
`PYTHONUTF8=1`, semilla 42 ([reproduce.md](../guides/reproduce.md) §0).

## 6. Vista de verificación (PV-4)

| Capa | Comando | Dónde corre | Qué garantiza |
|---|---|---|---|
| 1 Secretos | `python scripts/scan_secrets.py --baseline secrets_baseline.json` | `init.sh`, CI | C8 |
| 2 Tests CPU | `python -m pytest tests/ -q -m "not slow and not gpu"` | `init.sh`, CI | C1, C2 (incluye las guardas de evidencia) |
| 3 Documentación y harness | `python scripts/check_docs.py` | `init.sh`, CI | C7: enlaces resuelven, esquema de `feature_list.json`, `CLAUDE.md` < 100 líneas, ninguna línea del `CLAUDE.md` anterior perdida |
| Offline (manual) | `scripts/verify_v4_offline.py`, `scripts/verify_summer_offline.py` | Estación del autor | C1: rederivan las cifras desde artefactos versionados |
| Informativa | `ruff check .` | CI, `continue-on-error` | Calidad de código; no bloquea |

## 7. Decisiones y justificación

Las decisiones de arquitectura y proceso están en [../decisions/adr-log.md](../decisions/adr-log.md).

## 8. Inconsistencias conocidas

ISO/IEC/IEEE 42010 pide registrarlas en lugar de ocultarlas.

1. `config/config.yaml` sección `reranking`: dice `ms-marco-mini-6` y no se consume en runtime. El sistema vivo usa `ms-marco-MiniLM-L-12-v2` (aviso dentro del propio YAML y en `docs/KNOB_MAP_summer.md`). **Documentada como inerte/reservada el 2026-10-25 (F-008):** la sección se conserva sin implementarla ni borrarla, y el comentario del YAML lo declara. Ningún valor de esta sección altera el sistema.
2. `multidimensional_scoring=True` y `terminology_normalization=True` en `PROPOSED_HYBRID`: las banderas existen (campos de `PipelineConfig` en `src/pipeline/pipeline_config.py`), pero el pipeline no las implementa (`docs/CLOUD_DEPLOYMENT.md`). **Documentadas como inertes/reservadas el 2026-10-25 (F-008):** `PROPOSED_HYBRID` está congelado por ADR-0005 y no se modifica; el registro vive aquí y en el comentario de `config/config.yaml`. Activarlas no cambia el comportamiento.
3. Modelo de la UI. La tabla de componentes del README dice «Llama 3.1 8B q4 (demo / UI only)», mientras que «Key Features» y «Configuration» dicen que la UI usa Granite 4.1 por defecto. El código (`SURVEY_DEPLOY`) usa Granite. Llama 3.1 sigue en `PROPOSED_HYBRID` como registro del sistema enviado a LACCI. **Resuelto el 2026-10-25 (F-007):** se corrigió la fila de la tabla del README; la UI usa `SURVEY_DEPLOY.llm_model = granite4.1:8b` (comprobado en ejecución), y Llama 3.1 solo queda como registro `PROPOSED_HYBRID` y línea base histórica de exp8.
4. Corpus. El README da 3 951 documentos y 46 318 fragmentos para el repo completo; la evidencia de Nota 3 usa 2 697 documentos y 24 481 fragmentos. No es un error, pero cualquier cifra citada debe decir a qué corpus se refiere.
