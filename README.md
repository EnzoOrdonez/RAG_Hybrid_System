[![CI](https://github.com/EnzoOrdonez/RAG_Hybrid_System/actions/workflows/ci.yml/badge.svg)](https://github.com/EnzoOrdonez/RAG_Hybrid_System/actions/workflows/ci.yml)
<h1 align="center">☁️ CloudRAG</h1>
<p align="center">
  <strong>Hybrid RAG System for Cloud Documentation</strong>
</p>

<p align="center">
  <!-- Package support, not the experimental environment. Corrected by Claude Code, 2026-08-21
       07:30 local: the badge read 3.14, which is the REPRODUCIBLE ENVIRONMENT (REPRODUCE.md),
       not what the package supports. setup.py has always said >=3.11. -->
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-orange" alt="FAISS">
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-green" alt="Ollama">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit" alt="Streamlit">
</p>

---

## Publication

This system and its evaluation are described in the paper **"Hybrid Retrieval-Augmented
Generation for Multi-Cloud Documentation: A Comparative Evaluation of Lexical, Semantic,
and Hybrid Pipelines Against an LLM-Only Baseline"** (E. Ordonez Flores, W. L. Fuentes),
**accepted at IEEE LACCI 2026** (Latin American Conference on Computational Intelligence,
Lima, Peru, November 3-6, 2026). Camera-ready source: `docs/Paper_IEEE_RAG_Hibrido_LACCI_v9.tex`;
IEEE Xplore-compatible certified PDF: `docs/2026305869.pdf`. See `CITATION.cff`.

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
- **Local LLMs**: Runs with Ollama (UI default: Granite 4.1; evaluated set: Granite 4.1, Gemma 4, Mistral 7B, Qwen 3.5 — see MODELS.md)
- **Streamlit UI**: invitation-only participant evaluation; five views in private development mode
- **Benchmarking suite**: 19 versioned experiments (exp3-exp19b + exp8b) with paired statistics (Wilcoxon, Cohen's d_z, Bootstrap CI, BH/Holm)

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
> **2,697 processed documents (2,644 represented in the index after the stratified
> subsampling; the delta is 53 Azure docs whose chunks were all removed) / 24,481
> chunks**. Kubernetes + CNCF are indexed in the repo
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
- Python 3.11+ — what the **package** supports (`setup.py: python_requires=">=3.11"`)
- NVIDIA GPU with 6GB+ VRAM
- [Ollama](https://ollama.com/download)

> **Running the experiments is a different question.** The signed evidence was produced on a
> specific interpreter, **Python 3.14** at
> `C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe`, which is the only one on
> that machine carrying the ML stack. Package support and reproducible environment are separate
> declarations and are kept separate on purpose — see `REPRODUCE.md §0`.

### Install

For the participant application, use the Python 3.14 environment, hashed dependency
lock and artifact provisioning steps in [Interview readiness](docs/INTERVIEW_READINESS.md).
Installing packages alone does not provision the corpus, indices or model snapshots.
Interview gate (2026-09-09): **NO-GO**. The controlled hybrid pilot completed 40
responses, but cold p95 was 83.24 s and the real P900 session remains incomplete.
See the runbook for evidence, memory diagnosis and the blocked cloud preparation.
The recipe below is the historical CLI environment, not the interview deployment.

```bash
git clone https://github.com/EnzoOrdonez/RAG_Hybrid_System.git
cd RAG_Hybrid_System
pip install -r requirements.txt

# Historical CLI example model.
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

- [Guía de anotación del gold](docs/GUIA_ANOTACION_GOLD_V4.md)
- [Validación con gold humano](docs/SECCION_VALIDACION_HUMANA.md)
- [Reproducción de experimentos](REPRODUCE.md)
- [Ledger de ablaciones de verano](paper/summer_ablation_log.md)

### Reproducing the Nota 3 report (evidence -> tables)

The raw outputs (`experiments/results/exp9..19b`) are versioned; every cited number is
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

### Aplicación para participantes y demo privada

La aplicación abre **Evaluation Mode** por defecto y requiere una invitación del
operador. Antes de iniciarla, seguir [la guía de instalación y operación](docs/INTERVIEW_READINESS.md)
para fijar almacenamiento, manifiesto de artefactos, digest del modelo y endpoint.
Las entrevistas permanecen bloqueadas hasta validar el despliegue y su latencia real.

El Chat de desarrollo se habilita explícitamente en un entorno privado:

```bash
# Con el entorno y los modelos ya provisionados (Linux/macOS):
export CLOUDRAG_MODE=development
.venv-app/bin/python -m streamlit run src/ui/app.py
# abre http://localhost:8501 → página "Chat"
```

En PowerShell: `$env:CLOUDRAG_MODE='development'` y usar `.venv-app/Scripts/python.exe`.
Configuración actual y diferencias históricas:

- **Modelo por defecto:** `granite4.1:8b`, seed 42, caché desactivada, salida máxima
  1024 y contexto 4096. El Chat permite overrides; Evaluation conserva la receta fijada.
- **GPU para Ollama:** el entorno de entrevistas fija auxiliares en CPU
  (embedder/reranker/NLI); Ollama administra la GPU disponible. El override
  `CLOUDRAG_DEMO_GPU=1` corresponde a una demo privada con PyTorch CUDA compatible,
  no al entorno CPU fijado y medido para entrevistas.
- **keep_alive y streaming:** corresponden al Chat. Evaluation utiliza `query()` y
  espera generación y verificación antes de mostrar la respuesta y habilitar ratings.
- **Latencia:** las cifras de junio (4–5 s a primeros tokens, 40–50 s completos con
  512 tokens, ~25 s de carga inicial) pertenecen a la demo histórica. No verifican
  Granite a 1024 tokens ni el p95 ≤60 s exigido para el despliegue de entrevistas.
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
│   └── ui/                 # Participant entry; 5 private development views
├── scripts/                # CLI: benchmark, export, analyze
├── experiments/results/    # JSON results per experiment
├── data/                   # Corpus + chunks + indices (~341 MB)
├── output/                 # Figures (PNG) + tables (LaTeX) + CSV
└── run.py                  # Main entry point
```

---

## Experiments

19 versioned experiments (exp3-exp19b + exp8b) covering retrieval strategies, re-ranking,
LLM comparison, ablation, and cross-cloud evaluation. exp3-8/8b ran on the pre-rebuild
corpus/oracle and are kept as history; the paper's evidence is the final round (exp10-13
on the curated 194-query set + exp9 control on the pre-curation 200-query set); exp14-19b are
post-paper audit and validation experiments (runtime-noise floor, verifier ablations, evidence
ceiling, anchored selection), complemented by the claim-level human gold (v4):

| Experiment | What it tests | Key finding |
|------------|--------------|-------------|
| exp9 | LLM-only control (no RAG), pre-curation 200-query set | Fabricates in 195/200; RAG's floor baseline |
| exp10-11 | Retrieval, multi-oracle (D12 fix) | Hybrid>Dense real (d_z +0.45) but inflated under circular oracle (0.995 vs 0.740); edge lives in the rerank stage |
| exp12 | Faithfulness matrix (4 LLMs × 4 scenarios × 194) | RAG ≫ no-RAG; retrieval method n.s. on faithfulness — robust under metrics v2/v3/v4 (ledgers N5/N8/N9), 2 verifiers × 4 denominators |
| exp13 | Cross-cloud expansion ON vs OFF (D11 fix) | Expansion does NOT help; the earlier exp7 "+16.8%" claim is **retired** (its arms ran identical retrieval — N1/N4) |
| exp14 | H5 replicas, runtime-noise floor | Re-scoring 140 replicas under the same runtime: \|Δ\| mean 0.0616, p90 0.2005, 23.3% beyond the ±0.081 band — noise quantified, not hidden |
| exp15 | Verifier ablations (NLI variants, tier-A) | Sensitivity of faithfulness to verifier choice and claim pool |
| exp16 | Anchored decoding probe | Determinism probe: 3× back-to-back is bit-identical; replicas separated by other generations are not |
| exp17 | Cross-cloud balanced arm | Provider-coverage probe (descriptive): strict coverage 8% baseline vs 80% balanced, 0/125 foreign-provider chunks |
| exp18 | Evidence ceiling + unsupported-claim taxonomy | 759 unsupported claims taxonomized; stratified sample with Horvitz-Thompson weights, Kish n_eff = 27.4 |
| exp19b | Anchored evidence selector, paired + replay-gated | HHEM Δ=+0.0451 (CI95 [0.0076; 0.0817], p=0.018), TOST within ±0.081; verdict: verifier-aligned local improvement, not verifier-independent |
| gold v4 | Claim-level human validation (150 claims, 200 claim-condition judgments, blinded LLM judges) | Best verifier κ=0.30 (weighted, CI crosses 0) vs pilot human reference; LLM judges κ₂=0.754 between them but 0.17-0.20 vs human; labels = pilot reference, not ground truth |

Paired stats throughout: Wilcoxon signed-rank + Cohen's d_z + bootstrap CI, BH/Holm
corrected per research-question family.

---

## Streamlit UI

Participant mode registers only **Evaluation Mode**, with invitation login and no
operator routes. The five views under `src/ui/views/` (**Chat**, **Metrics Dashboard**,
**Document Explorer**, **Evaluation Mode**, **Experiment Runner**) are available only
in the separate private `CLOUDRAG_MODE=development` environment. Installation and
required deployment variables are documented in
[INTERVIEW_READINESS.md](docs/INTERVIEW_READINESS.md).

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
| LLM (UI default) | granite4.1:8b | evaluated set in MODELS.md: granite4.1, gemma4:e4b, mistral, qwen3.5 |
| Top-K | 5 | 1-20 |

---

## License

[GPL-3.0](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/EnzoOrdonez">Enzo Ordoñez</a> · Universidad de Lima · 2026
</p>
