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

A hybrid Retrieval-Augmented Generation system that answers questions about cloud documentation from **AWS, Azure, and GCP** (the experimental corpus; the crawler also supports Kubernetes/CNCF sources, which are excluded from the Nota 3 evidence). It combines lexical search (BM25) with semantic search (dense embeddings) using Reciprocal Rank Fusion, cross-encoder re-ranking, and local LLMs via Ollama.

**Why hybrid?** Pure keyword search misses semantically related content. Pure embedding search misses exact technical terms. CloudRAG fuses both to get the best of each approach — then re-ranks with a cross-encoder for precision.

---

## Key Features

- **Hybrid retrieval**: BM25 + BGE-large embeddings with RRF fusion
- **Cross-cloud terminology**: Automatic mapping between providers (VPC ↔ Virtual Network ↔ VPC Network)
- **Adaptive chunking**: Preserves code blocks and tables as atomic units
- **Cross-encoder re-ranking**: ms-marco-MiniLM-L-12-v2 for precision refinement
- **Hallucination detection**: NLI-based faithfulness scoring with DeBERTa v3
- **Local LLMs**: Runs entirely on your machine with Ollama (demo: Llama 3.1; evaluated set: Granite 4.1, Gemma 4, Mistral 7B, Qwen 3.5 — see MODELS.md)
- **Streamlit UI**: 5-page web interface with chat, metrics dashboard, and evaluation tools
- **Benchmarking suite**: 12 versioned experiments (exp3-exp13 + exp8b) with paired statistics (Wilcoxon, Cohen's d_z, Bootstrap CI, BH/Holm)

---

## Performance

Evaluated on the curated **194-query** set (depuration 200→194 logged in
`data/evaluation/test_queries_removed_log.json`) over the rebuilt corpus, under an
**independent relevance oracle** (bge-reranker-large — the pipeline's own reranker is
ms-marco, so scoring with it is circular; both reported in
`output/tables/nota3/tabla4_retrieval__exp11_retrieval194_fullrerank.md`):

| System | P@1 | P@5 | R@5 | MRR | NDCG@5 |
|--------|-----|-----|-----|-----|--------|
| BM25 (lexical) | 0.531 | 0.443 | 0.299 | 0.603 | 0.442 |
| Dense (BGE) | 0.686 | 0.546 | 0.390 | 0.742 | 0.624 |
| Hybrid pre-rerank (RRF) | 0.613 | 0.543 | 0.377 | 0.699 | 0.603 |
| **Hybrid post-rerank (ours)** | **0.716** | **0.637** | **0.468** | **0.770** | **0.740** |

Hybrid(post) > Dense is significant under the independent oracle (d_z = +0.45,
p_BH < 0.001); the advantage comes from the **reranking stage**, not the RRF fusion
(pre-rerank ≈ Dense, n.s.). Under the circular oracle the hybrid scores NDCG@5 = 0.995
by construction — reported only as a circularity reference (ledger N2).

Generation faithfulness (4 LLMs × 4 scenarios × 194, NLI verifier): RAG ≫ no-RAG for
every testable model, but the **retrieval method does not significantly move generation
faithfulness** (n.s. under 2 NLI verifiers × 4 denominators; ledger N5). Decline-aware
v2 metric and instrument audit: `output/tables/nota3/` + `RESULTADOS_RESUMEN.md`.

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
| LLMs (evaluated, Nota 3) | Granite 4.1 8B · Gemma 4 E4B · Mistral 7B · Qwen 3.5 9B |
| LLM (demo / UI only) | Llama 3.1 8B q4 — see [MODELS.md](MODELS.md) |
| NLI | cross-encoder/nli-deberta-v3-small (runtime) + nli-deberta-v3-base (2nd verifier) |
| UI | Streamlit + Plotly |

---

## Corpus

3,951 documents, 46,318 indexed chunks (adaptive chunking, 500 tokens):

> **Experimental corpus (Nota 3, exp9-13):** the AWS/Azure/GCP subset only —
> **2,697 documents / 24,481 chunks**. Kubernetes + CNCF are indexed in the repo
> but were not part of the report's experimental runs.

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

# Download the DEMO model (UI / sustentación).
# The 4 models EVALUATED in the Nota 3 report (granite4.1:8b, gemma4:e4b,
# mistral:7b-instruct, qwen3.5:9b) are documented in MODELS.md — pull those
# to reproduce exp12.
ollama pull llama3.1:8b-instruct-q4_K_M

# Verify
python run.py --health-check
```

### Environment & reproducibility (required for experiments)

The raw-evidence snapshot for the Nota 3 round (exp9-13) is published as the annotated
tag **`nota3-evidencia-2026-06-11`** (v2-era faithfulness metric, ledger N1-N7). The
citable faithfulness figures were since corrected **offline** — v3 (N8: format-artifact
exclusion) and **v4 (N9: vacuous-row exclusion; the citable Tabla 6)** — without touching
the signed raw outputs. See `RESULTADOS_RESUMEN.md` and ledger entries N8/N9; a post-N9
tag will mark the documentation-ready state.

**Traceability + minimal repro recipes:** [docs/TRACEABILITY_nota3.md](docs/TRACEABILITY_nota3.md)
maps every cited table/figure to its experiment → script → output path, and lists the commands
to regenerate only the report's artifacts (without re-running the full experiment suite).

### Reproducing the Nota 3 report (evidence -> tables)

The raw outputs (`experiments/results/exp9..13`) are versioned; every cited number is
re-derivable offline from them. Citable artifacts and estimated runtimes:

| What | Command (see TRACEABILITY for flags) | Est. time / hardware |
|---|---|---|
| Tabla 6 **v4** (faithfulness, citable) | `compute_faithfulness_metrics.py --exclude-vacuous` + `_export_tabla6_v4.py` | ~1 min CPU (re-aggregation only) |
| Tabla 4 (retrieval, both oracles) | `compute_retrieval_metrics.py --oracle-model ...` | ~10-20 min (GPU helps; downloads oracle models if not cached; **overwrites** `retrieval_metrics__*.json` in place) |
| NLI re-score v3 (base+small) | `rescore_nli_v3.py --verifier base|small` | ~1-2 h GPU 6 GB (models in `data/models/`, ~3.3 GB, gitignored) |
| Figures f1-f4 (+f2 v4) | `_make_figures_nota3.py` | ~1 min CPU |
| Full exp12 matrix (NOT needed to verify the report) | `run_generation_matrix.py` | ~30 h GPU + Ollama, new LLM generation |

Query-set curation (200 -> 194) is documented in `data/evaluation/README.md`.

All experiment/benchmark runs must set these environment variables **before**
launching Python, so they are read at interpreter startup:

```bash
# Linux / macOS / Git-Bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
python scripts/run_retrieval_only.py --exp-id exp11_retrieval194_fullrerank
```

```powershell
# Windows PowerShell
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1; $env:PYTHONHASHSEED=42
python scripts/run_retrieval_only.py --exp-id exp11_retrieval194_fullrerank
```

- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` — the embedding/reranker/NLI
  models are already cached; offline mode avoids a known Hugging Face client bug
  when several models are loaded consecutively. (One-time exception: downloading
  a new relevance-oracle model.)
- `PYTHONHASHSEED=42` — fixes `hash()` of strings so set/dict iteration in the
  RRF fusion is bit-reproducible. Runners also call
  `reproducibility.ensure_hashseed_at_startup(42)`, which re-execs once if the
  var is unset, but setting it explicitly is preferred.
- `seed=42` everywhere; generation uses `temperature=0` (greedy decoding) so LLM
  output is reproducible independent of sampling seed.

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

### Cómo lanzar la demo (video de sustentación / sesiones B.4)

```bash
# 1) Ollama corriendo y modelo de demo descargado
ollama serve            # si no está ya corriendo
ollama pull llama3.1:8b-instruct-q4_K_M

# 2) Lanzar la UI (recomendado para el video y las sesiones SUS)
python -m streamlit run src/ui/app.py
# abre http://localhost:8501 → página "Chat"
```

Notas operativas (fixes 2026-06-11, ver commits F6):

- **Modelo de demo:** `llama3.1:8b-instruct-q4_K_M` (default del selector). Es el único
  que cabe entero en la GPU de 6 GB; gemma4:e4b (9,6 GB) cae a CPU y es más lento aquí.
- **GPU para Ollama:** la UI fija `CUDA_VISIBLE_DEVICES=""` para sus propios modelos
  (embedder/reranker/NLI van a CPU) y deja la GPU completa al LLM. Con GPU ≥12 GB:
  `CLOUDRAG_DEMO_GPU=1` para revertir.
- **keep_alive:** la UI envía `keep_alive=30m` y precalienta el modelo al primer uso de
  la sesión → sin carga fría entre consultas ni tras pausas (sesiones de ~30 min).
- **Streaming:** primeros tokens ~4-5 s tras enviar la consulta; respuesta completa
  ~40-50 s (512 tokens). La verificación NLI corre DESPUÉS de mostrar la respuesta
  (toggle en el sidebar).
- **Primera consulta del proceso:** paga ~25 s extra de cargas perezosas (reranker en
  CPU). Para el video: hacer una consulta de calentamiento antes de grabar.
- Los benchmarks NO usan este camino (paridad cubierta por
  `tests/test_benchmark_parity.py`).

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

12 versioned experiments (exp3-exp13 + exp8b) covering retrieval strategies, re-ranking,
LLM comparison, ablation, and cross-cloud evaluation. exp3-8/8b ran on the pre-rebuild
corpus/oracle and are kept as history; the paper's evidence is the final round (exp10-13
on the curated 194-query set + exp9 control on the pre-curation 200-query set):

| Experiment | What it tests | Key finding |
|------------|--------------|-------------|
| exp9 | LLM-only control (no RAG), pre-curation 200-query set | Fabricates in 195/200; RAG's floor baseline |
| exp10-11 | Retrieval, multi-oracle (D12 fix) | Hybrid>Dense real (d_z +0.45) but inflated under circular oracle (0.995 vs 0.740); edge lives in the rerank stage |
| exp12 | Faithfulness matrix (4 LLMs × 4 scenarios × 194) | RAG ≫ no-RAG; retrieval method n.s. on faithfulness — robust under metrics v2/v3/v4 (ledgers N5/N8/N9), 2 verifiers × 4 denominators |
| exp13 | Cross-cloud expansion ON vs OFF (D11 fix) | Expansion does NOT help; the earlier exp7 "+16.8%" claim is **retired** (its arms ran identical retrieval — N1/N4) |

Paired stats throughout: Wilcoxon signed-rank + Cohen's d_z + bootstrap CI, BH/Holm
corrected per research-question family.

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
| LLM (demo/UI) | llama3.1:8b (demo) | evaluated set in MODELS.md: granite4.1, gemma4:e4b, mistral, qwen3.5 |
| Top-K | 5 | 1-20 |

---

## License

[GPL-3.0](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/EnzoOrdonez">Enzo Ordoñez</a> · Universidad de Lima · 2026
</p>
