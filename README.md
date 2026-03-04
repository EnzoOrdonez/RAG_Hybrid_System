<h1 align="center">
  ☁️ CloudRAG — Hybrid RAG System for Cloud Documentation
</h1>

<p align="center">
  <strong>Diseño y Validación de un Modelo Semántico Híbrido para Optimizar Sistemas RAG sobre Documentación Técnica Cloud en AWS, Azure y GCP</strong>
</p>

<p align="center">
  <em>Tesis de Grado — Ingeniería de Sistemas, Universidad de Lima</em><br>
  <strong>Enzo Fabrizio Ordoñez Flores</strong> · 8vo Semestre · 2026
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?logo=python" alt="Python 3.14">
  <img src="https://img.shields.io/badge/PyTorch-GPU-red?logo=pytorch" alt="PyTorch GPU">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-orange" alt="FAISS">
  <img src="https://img.shields.io/badge/Ollama-LLM-green" alt="Ollama">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-GPL_3.0-purple" alt="License">
</p>

---

## 📋 Resumen

Este proyecto implementa y evalúa un **sistema RAG (Retrieval-Augmented Generation) híbrido** que combina búsqueda léxica (BM25) y semántica (dense embeddings) para responder consultas sobre documentación técnica de los principales proveedores cloud: **AWS, Azure, GCP, Kubernetes y CNCF**.

El sistema propuesto integra normalización terminológica cross-cloud, chunking adaptativo, fusión híbrida de resultados y re-ranking multidimensional, logrando una mejora significativa sobre los baselines puramente léxicos o semánticos.

### Contribuciones principales

- **Chunking Adaptativo**: Estrategia que combina segmentación jerárquica con subdivisión semántica, preservando bloques de código y tablas como unidades atómicas.
- **Normalización Terminológica Cross-Cloud**: Mapeo automático de terminología equivalente entre proveedores (ej: VPC ↔ Virtual Network ↔ VPC Network).
- **Fusión Híbrida con Re-ranking Multidimensional**: Combinación de BM25 + Dense embeddings con Reciprocal Rank Fusion, seguido de scoring que integra relevancia, recencia, calidad de fuente y diversidad.
- **Detección de Alucinaciones con NLI**: Verificación automática de claims usando Natural Language Inference para medir faithfulness.

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
│                "Compare S3 vs Blob Storage"                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   QUERY     │  Normalización terminológica
                    │  PROCESSOR  │  Expansión de siglas (BM25)
                    │             │  Clasificación de tipo
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │   BM25   │ │  Dense   │ │   Hybrid     │
        │ Retriever│ │Retriever │ │  Retriever   │
        │(Control 1)│ │(Control 2)│ │(Experimental)│
        └────┬─────┘ └────┬─────┘ └──────┬───────┘
             │             │              │
             │             │       ┌──────▼──────┐
             │             │       │  RRF/Linear  │
             │             │       │   FUSION     │
             │             │       └──────┬───────┘
             │             │              │
             └─────────────┼──────────────┘
                           │
                    ┌──────▼──────┐
                    │  RE-RANKER  │  Cross-encoder + Scoring
                    │             │  multidimensional (relevancia,
                    │             │  recencia, fuente, diversidad)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │     LLM     │  Ollama (Llama 3.1, Mistral,
                    │  GENERATOR  │  Qwen 2.5) + APIs opcionales
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │HALLUCINATION│  NLI (DeBERTa v3)
                    │  DETECTOR   │  Claim extraction + Evidence
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  RESPONSE   │  Citaciones, confidence,
                    │  FORMATTER  │  latency breakdown
                    └─────────────┘
```

---

## 📦 Estructura del Proyecto

```
hybrid-rag-system/
├── config/
│   ├── config.yaml                 # Configuración principal
│   ├── cloud_services.yaml         # 5 fuentes documentales
│   └── terminology_mappings.yaml   # Mapeos terminológicos cross-cloud
├── src/
│   ├── ingestion/                  # Crawlers para AWS, Azure, GCP, K8s, CNCF
│   ├── preprocessing/              # Limpieza, normalización, deduplicación
│   ├── chunking/                   # 5 estrategias (fixed, recursive, semantic,
│   │                               #   hierarchical, adaptive)
│   ├── embedding/                  # 4 modelos de embedding + FAISS/BM25 indices
│   ├── retrieval/                  # 3 retrievers (BM25, Dense, Hybrid)
│   ├── reranking/                  # Cross-encoder + scoring multidimensional
│   ├── generation/                 # LLM manager, prompts, hallucination detector
│   ├── pipeline/                   # RAG pipeline end-to-end + 3 configs
│   ├── evaluation/                 # Métricas, benchmark runner, análisis estadístico
│   └── ui/                         # Streamlit app (5 páginas)
├── scripts/
│   ├── download_docs.py            # Descargar documentación
│   ├── build_index.py              # Construir índices FAISS + BM25
│   ├── run_benchmark.py            # Ejecutar experimentos
│   ├── generate_test_queries.py    # Generar dataset de evaluación
│   ├── export_results.py           # Exportar figuras y tablas LaTeX
│   └── analyze_user_sessions.py    # Analizar sesiones de evaluación con usuarios
├── data/
│   ├── raw/                        # Documentación descargada (5 providers)
│   ├── processed/                  # Documentos parseados con metadata
│   ├── chunks/                     # 5 estrategias × 3 tamaños = 15 sets
│   ├── indices/                    # FAISS + BM25 indices
│   ├── embeddings/                 # Embeddings cacheados (.npy)
│   └── evaluation/                 # Test queries + sesiones de usuarios
├── experiments/
│   └── results/                    # Resultados de los 8 experimentos
├── output/
│   ├── figures/                    # Gráficos PNG 300 DPI
│   ├── tables/                     # Tablas LaTeX (booktabs)
│   └── csv/                        # Datos exportados
├── notebooks/                      # Jupyter notebooks exploratorios
├── .streamlit/config.toml          # Tema de la UI
├── run.py                          # Entry point CLI
├── setup.py
└── requirements.txt
```

---

## 🚀 Instalación y Uso

### Requisitos previos

- Python 3.10+
- GPU con CUDA (recomendado: NVIDIA RTX 3060+ con 6GB+ VRAM)
- [Ollama](https://ollama.com/download) instalado (para LLMs locales)

### 1. Clonar e instalar

```bash
git clone https://github.com/EnzoOrdonez/RAG_Hybrid_System.git
cd RAG_Hybrid_System
pip install -r requirements.txt
```

### 2. Descargar modelos LLM

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct
ollama pull mistral:7b-instruct
```

### 3. Descargar documentación y construir índices

```bash
python scripts/download_docs.py --services 3
python scripts/build_index.py --embedding bge-large --chunker adaptive --size 500
```

### 4. Ejecutar la UI

```bash
streamlit run src/ui/app.py
```

### 5. Ejecutar desde CLI

```bash
# Query interactiva
python run.py --interactive --config hybrid

# Comparar los 3 sistemas
python run.py --compare "What is the difference between Lambda and Azure Functions?"

# Health check
python run.py --health-check
```

---

## 🧪 Diseño Experimental

### 3 sistemas comparados (within-subjects)

| Sistema | Retrieval | Reranking | Query Expansion |
|---------|-----------|-----------|-----------------|
| **Control 1** — RAG Léxico | BM25 | No | No |
| **Control 2** — RAG Semántico | Dense (BGE-large) | No | No |
| **Experimental** — RAG Híbrido | BM25 + Dense (RRF) | Cross-encoder + multidimensional | Sí |

### 8 experimentos

| # | Experimento | Configuraciones | Descripción |
|---|-------------|-----------------|-------------|
| 1 | Chunking Comparison | 5 estrategias × 3 tamaños | Evalúa fixed, recursive, semantic, hierarchical, adaptive en 300/500/700 tokens |
| 2 | Embedding Comparison | 4 modelos | MiniLM, BGE-large, E5-large, Instructor |
| 3 | Retrieval Strategy | 6 configuraciones | BM25, Dense, Hybrid-Linear (α=0.3/0.5/0.7), Hybrid-RRF |
| 4 | Re-ranking Impact | 4 opciones | Sin reranking, 3 cross-encoders |
| 5 | LLM Comparison | 3-5 modelos | Llama 3.1, Qwen 2.5, Mistral + APIs opcionales |
| 6 | Ablation Study | 7 configs | Quitar cada componente del sistema completo |
| 7 | Cross-Cloud | 4 sistemas × 30 queries | Evaluación específica de queries comparativas |
| 8 | End-to-End | 3 sistemas × 200 queries | Evaluación final con tests estadísticos completos |

### Ejecutar benchmarks

```bash
# Quick test (20 queries)
python scripts/run_benchmark.py --experiment exp8 --quick

# Experimento completo
python scripts/run_benchmark.py --experiment exp8

# Todos los experimentos
python scripts/run_benchmark.py --all

# Solo exportar resultados (figuras + LaTeX)
python scripts/run_benchmark.py --export-only
```

---

## 📊 Métricas

### Retrieval
- Recall@K (K=5, 10, 20)
- Precision@K
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)

### Generación
- Exact Match (EM)
- F1 Token Overlap
- ROUGE-L
- BERTScore

### Alucinación
- Faithfulness Score (NLI-based)
- Hallucination Rate
- Rúbrica 1-5 (para evaluación humana)

### Latencia
- p50, p95, p99 por etapa (retrieval, reranking, generation, hallucination check)

### Análisis estadístico
- Shapiro-Wilk (test de normalidad)
- Test T pareado o Wilcoxon signed-rank
- Cohen's d (tamaño de efecto, mínimo 0.5)
- Bootstrap CI 95% (1000 iteraciones)

---

## 👥 Evaluación con Usuarios

La UI incluye un **Evaluation Mode** que implementa el protocolo experimental completo:

1. **Login** con consentimiento informado
2. **Entrenamiento** (3 queries de ejemplo, 5 min)
3. **Evaluación** de 3 sistemas (10 queries cada uno, orden aleatorizado)
4. **Descansos** de 5 min entre sistemas (timer en pantalla)
5. **Cuestionario SUS** (System Usability Scale, 10 ítems)
6. **Preguntas abiertas** (3 preguntas cualitativas)

Diseño within-subjects con contrabalanceo: cada participante prueba los 3 sistemas en orden diferente.

```bash
# Lanzar UI para evaluación
streamlit run src/ui/app.py

# Analizar resultados después de las sesiones
python scripts/analyze_user_sessions.py
```

---

## 📁 Corpus Documental

| Fuente | Documentos | Chunks (adaptive-500) | Servicios principales |
|--------|-----------|----------------------|----------------------|
| AWS | 996 | ~11,000 | EC2, ECS, Lambda, S3, VPC, DynamoDB |
| Azure | 1,461 | ~17,000 | Functions, Blob Storage, Virtual Network |
| GCP | 240 | ~7,000 | Compute Engine, Cloud Storage, GKE |
| Kubernetes | 1,162 | ~12,000 | Pods, Deployments, Services, Networking |
| CNCF | 92 | ~200 | Cloud-native glossary (200 terms) |
| **Total** | **3,951** | **~46,300** | — |

Parámetros de chunking: tamaños 300, 500, 700 tokens · overlap 50 tokens · 5 estrategias × 3 tamaños = 462,326 chunks totales.

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| Embeddings | BAAI/bge-large-en-v1.5 (1024d) + 3 alternativos |
| Vector Search | FAISS (IndexFlatIP) |
| Lexical Search | BM25 (rank_bm25, k1=1.2, b=0.75) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-12-v2 |
| LLMs locales | Ollama (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B) |
| NLI | cross-encoder/nli-deberta-v3-small |
| UI | Streamlit + Plotly |
| GPU | NVIDIA RTX 3060 (6GB VRAM, CUDA) |
| Análisis | scipy, numpy, matplotlib, pandas |

---

## 📄 Criterios de Éxito

| Métrica | Criterio | vs Baseline |
|---------|----------|-------------|
| Recall@5 | ≥ 15% mejora | sobre léxico |
| MRR | ≥ 20% mejora | sobre ambos baselines |
| NDCG@10 | ≥ 12% mejora | sobre ambos baselines |
| F1 | ≥ 15% mejora | sobre ambos baselines |
| Tasa de alucinación | ≤ 50% del baseline | reducción ≥ 50% |
| Latencia p95 | ≤ 2x baseline más rápido | — |
| Latencia promedio | ≤ 3 segundos | — |
| SUS | ≥ 70 | (percentil 68+) |
| Likert utilidad | ≥ 4.0 / 5.0 | — |
| Cohen's d | ≥ 0.5 | efecto mediano mínimo |

---

## 📝 Licencia

Este proyecto está bajo la licencia [GPL-3.0](LICENSE).

---

<p align="center">
  <em>Desarrollado como parte de la tesis de grado en Ingeniería de Sistemas</em><br>
  <strong>Universidad de Lima · 2026</strong>
</p>
