<h1 align="center">☁️ CloudRAG</h1>

<p align="center">
  <strong>Hybrid RAG System for Cloud Documentation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-orange" alt="FAISS">
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-green" alt="Ollama">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit" alt="Streamlit">
</p>

---

## What is CloudRAG?

A hybrid Retrieval-Augmented Generation system that answers questions about cloud documentation from **AWS, Azure, GCP, Kubernetes, and CNCF**. It combines lexical search (BM25) with semantic search (dense embeddings) using Reciprocal Rank Fusion, cross-encoder re-ranking, and local LLMs via Ollama.

**Why hybrid?** Pure keyword search misses semantically related content. Pure embedding search misses exact technical terms. CloudRAG fuses both to get the best of each approach — then re-ranks with a cross-encoder for precision.

---

## Key Features

- **Hybrid retrieval**: BM25 + BGE-large embeddings with RRF fusion
- **Cross-cloud terminology**: Automatic mapping between providers (VPC ↔ Virtual Network ↔ VPC Network)
- **Adaptive chunking**: Preserves code blocks and tables as atomic units
- **Cross-encoder re-ranking**: ms-marco-MiniLM-L-12-v2 for precision refinement
- **Hallucination detection**: NLI-based faithfulness scoring with DeBERTa v3
- **Local LLMs**: Runs entirely on your machine with Ollama (Llama 3.1, Mistral, Qwen)
- **Streamlit UI**: 5-page web interface with chat, metrics dashboard, and evaluation tools
- **Benchmarking suite**: 9 experiments with statistical analysis (Wilcoxon, Cohen's d, Bootstrap CI)

---

## Performance

Evaluated on 200 queries across 5 cloud documentation sources:

| System | Precision@1 | Recall@5 | MRR | NDCG@5 | Faithfulness |
|--------|------------|----------|-----|--------|-------------|
| BM25 (lexical) | 0.785 | 0.368 | 0.828 | 0.554 | 0.496 |
| Dense (semantic) | 0.860 | 0.424 | 0.894 | 0.661 | 0.509 |
| **Hybrid (ours)** | **0.930** | **0.472** | **0.942** | **0.736** | **0.514** |

Hybrid outperforms both baselines with statistical significance (p < 0.0001, Cohen's d = 0.626).

---

## Architecture

```
Query → Normalization + Expansion
      → BM25         ─┐
      → Dense (BGE)   ─┤→ RRF Fusion → Cross-encoder Re-ranking
      → Hybrid        ─┘
      → LLM Generation (Ollama)
      → Hallucination Check (NLI)
      → Response with citations
```

| Component | Technology |
|-----------|-----------|
| Embeddings | BAAI/bge-large-en-v1.5 (1024 dim) |
| Vector DB | FAISS IndexFlatIP |
| Lexical | BM25 (rank_bm25) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-12-v2 |
| LLMs | Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B |
| NLI | cross-encoder/nli-deberta-v3-small |
| UI | Streamlit + Plotly |

---

## Corpus

3,951 documents, 46,318 indexed chunks (adaptive chunking, 500 tokens):

| Source | Documents | Description |
|--------|-----------|-------------|
| AWS | 996 | EC2, ECS, Lambda, S3, VPC, DynamoDB, CloudWatch |
| Azure | 1,461 | Functions, Blob Storage, Virtual Network, AKS, Cosmos DB |
| GCP | 240 | Compute Engine, Cloud Storage, GKE, BigQuery |
| Kubernetes | 1,162 | Pods, Deployments, Services, Networking, Storage |
| CNCF | 92 | Cloud-native glossary (200+ terms) |

---

## Quick Start

### Requirements
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM
- [Ollama](https://ollama.com/download)

### Install

```bash
git clone https://github.com/EnzoOrdonez/RAG_Hybrid_System.git
cd RAG_Hybrid_System
pip install -r requirements.txt

# Download LLM models
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct

# Verify
python run.py --health-check
```

### Run

```bash
# Interactive chat
python run.py --interactive --config hybrid

# Compare all 3 systems
python run.py --compare "What is the difference between Lambda and Azure Functions?"

# Web UI
python -m streamlit run src/ui/app.py

# Run benchmarks
python scripts/run_benchmark.py --experiment exp8 --quick
```

---

## Project Structure

```
cloudrag/
├── src/
│   ├── ingestion/          # Crawlers for 5 documentation sources
│   ├── preprocessing/      # Text cleaning, normalization, deduplication
│   ├── chunking/           # 5 strategies: fixed, recursive, semantic, hierarchical, adaptive
│   ├── embedding/          # BGE-large + FAISS + BM25 index management
│   ├── retrieval/          # BM25, Dense, Hybrid (RRF + Linear fusion)
│   ├── reranking/          # Cross-encoder re-ranker
│   ├── generation/         # LLM manager (Ollama) + hallucination detector (NLI)
│   ├── pipeline/           # End-to-end RAG pipeline (7 stages)
│   ├── evaluation/         # Metrics, benchmark runner, statistical analysis
│   └── ui/                 # Streamlit app (5 pages)
├── scripts/                # CLI: benchmark, export, analyze
├── experiments/results/    # JSON results per experiment
├── data/                   # Corpus + chunks + indices (~341 MB)
├── output/                 # Figures (PNG) + tables (LaTeX) + CSV
└── run.py                  # Main entry point
```

---

## Experiments

9 experiments covering retrieval strategies, re-ranking, LLM comparison, ablation, and cross-cloud evaluation:

| Experiment | What it tests | Key finding |
|------------|--------------|-------------|
| exp3 | Retrieval strategies (BM25, Dense, Hybrid, alpha grid) | RRF outperforms linear fusion |
| exp4 | Re-ranking impact | Cross-encoder improves precision |
| exp5 | LLM comparison (Llama, Qwen, Mistral) | Mistral best faithfulness (0.504) |
| exp6 | Ablation (remove each component) | Re-ranker most impactful (+3.4 pts) |
| exp7 | Cross-cloud normalization | +16.8% faithfulness with normalization |
| exp8/8b | End-to-end (Llama / Mistral) | Hybrid > Dense > BM25 consistently |

All comparisons validated with Wilcoxon signed-rank test (p < 0.0001) and Cohen's d effect sizes.

---

## Streamlit UI

5 pages: **Chat** (interactive Q&A with system selector), **Metrics Dashboard** (Plotly charts), **Document Explorer** (search corpus), **Evaluation Mode** (controlled user studies), **Experiment Runner** (run benchmarks from UI).

```bash
python -m streamlit run src/ui/app.py
# Open http://localhost:8501
```

---

## Configuration

| Setting | Default | Options |
|---------|---------|---------|
| Embedding model | bge-large | MiniLM, bge-large, e5-large, instructor |
| Chunk size | 500 tokens | 300, 500, 700 |
| Chunk strategy | adaptive | fixed, recursive, semantic, hierarchical, adaptive |
| Fusion method | RRF (k=60) | RRF, Linear (alpha 0.0-1.0) |
| Re-ranker | ms-marco-L-12 | ms-marco-L-6, ms-marco-L-12, bge-reranker |
| LLM | llama3.1:8b | llama3.1, mistral, qwen2.5 |
| Top-K | 5 | 1-20 |

---

## License

[GPL-3.0](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/EnzoOrdonez">Enzo Ordoñez</a> · Universidad de Lima · 2026
</p>
