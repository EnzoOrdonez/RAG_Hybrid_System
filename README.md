HEAD
# Hybrid RAG System for Cloud Documentation

**Thesis:** *Diseno y Validacion de un Modelo Semantico Hibrido para Optimizar Sistemas RAG sobre Documentacion Tecnica Cloud en AWS, Azure y GCP*

**Author:** Enzo Ordonez Flores - Universidad de Lima, Ingenieria de Sistemas

## Project Structure

```
hybrid-rag-system/
  config/              # Configuration files (YAML)
  src/                 # Source code
    ingestion/         # Crawlers and document parser
    preprocessing/     # Text cleaning, normalization, deduplication
    chunking/          # 5 chunking strategies
    embedding/         # (Phase 2)
    retrieval/         # (Phase 3)
    reranking/         # (Phase 3)
    generation/        # (Phase 4)
    evaluation/        # (Phase 5)
  data/                # Raw, processed, and chunked data
  scripts/             # CLI utilities
  notebooks/           # Jupyter notebooks for analysis
  experiments/         # Experiment configs and results
  output/              # Figures, tables, reports
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Phase 1: Data Pipeline

### 1. Download Documentation

```bash
# Preview what will be downloaded
python scripts/download_docs.py --dry-run

# Download a small subset for testing (3 services per provider)
python scripts/download_docs.py --services 3

# Download a single provider
python scripts/download_docs.py --provider aws

# Download everything
python scripts/download_docs.py
```

### 2. Preprocess Documents

```bash
# Run full preprocessing pipeline
python run.py preprocess --input data/processed
```

### 3. Generate Chunks

```bash
# All strategies, all sizes
python run.py chunk --input data/processed --output data/chunks

# Single strategy
python run.py chunk --strategy adaptive --sizes 500

# Compare strategies
python run.py chunk --strategy fixed recursive semantic hierarchical adaptive
```

### 4. View Statistics

```bash
python scripts/download_docs.py --stats
```

## Corpus Sources (5)

| Source | Type | Method |
|--------|------|--------|
| AWS | GitHub repos (awsdocs) | git clone |
| Azure | GitHub (MicrosoftDocs/azure-docs) | sparse checkout |
| GCP | Web scraping | BeautifulSoup |
| Kubernetes | GitHub (kubernetes/website) | git clone |
| CNCF Glossary | GitHub (cncf/glossary) | git clone |

## Chunking Strategies (5)

| Strategy | Description |
|----------|-------------|
| Fixed | Baseline - fixed token size |
| Recursive | Hierarchical separators |
| Semantic | Split at semantic boundaries |
| Hierarchical | Respects heading structure |
| **Adaptive** | **Thesis proposal: hierarchical + semantic + domain rules** |

## Experimental Design

- **Within-subjects** with 3 systems (BM25, Semantic, Hybrid)
- Chunk sizes: 300, 500, 700 tokens
- Overlap: 50 tokens
- Random seed: 42

# RAG_Hybrid_System
636d642feb203874d00b9fa450461ff8989e7f95
