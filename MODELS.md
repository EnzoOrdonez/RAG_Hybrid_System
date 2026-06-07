# Model selection — Nota 3 generation matrix (exp12)

**Selection criterion (§3.6):** open, current (2025–2026) models from *distinct labs*,
runnable on the same consumer hardware (≤6 GB VRAM, RTX 3060 Laptop) at q4
quantization.

**Status (2026-06-07):** candidate tags **web-verified to exist**; **not yet
pulled or measured**. Measured VRAM / tokens-per-second and exact digests are
filled in by the Phase 4 smoke (gate before the matrix). Currently installed
(old preliminary round): `llama3.1:8b-instruct-q4_K_M`, `mistral:7b-instruct`,
`qwen2.5:7b-instruct`.

## Candidates (to pull + smoke)

| # | Tag | Lab | Released | Params | Exists | q4 VRAM vs 6 GB | Notes |
|---|-----|-----|----------|--------|--------|------------------|-------|
| 1 | `granite4.1:8b` | IBM | 2026-04-29 | 8B dense | ✅ Ollama | ~4.7–5 GB → **fits** | Apache-2.0, explicitly RAG-oriented; **replaces Llama** in the matrix |
| 2 | `mistral3:8b` (Ministral 3 8B) | Mistral AI | 2025-12-02 | 8B dense | ✅ (confirm exact tag) | ~4.7 GB → **fits** | last small dense Mistral; vision-capable. If `mistral3:8b` tag absent at pull → use the published Ministral-3-8B tag and document |
| 3a | `gemma4:e4b` | Google | 2026-03-31 | E4B (Per-Layer Embeddings) | ✅ Ollama | **fits** | edge variant; likely Gemma pick |
| 3b | `gemma4:12b` | Google | 2026-06-03 | 12B | ✅ Ollama | needs ~16 GB → **does NOT fit** | smoke will confirm it fails the "100% GPU @ ≥5 tok/s" rule |
| 4 | `qwen3.5:9b` | Alibaba | ~2026-05 | 9B | ✅ Ollama (6.6 GB) | ⚠️ **>6 GB → partial CPU offload** | borderline; smoke decides if it runs 100% on GPU at ≥5 tok/s |

**Gemma rule (Phase 3):** smoke both `e4b` and `12b`; keep the largest that runs
100% on GPU at ≥5 tok/s. On 6 GB, `12b` is expected to fail → `e4b`.

**qwen3.5:9b risk:** the default tag is 6.6 GB on disk; with the KV cache for
RAG contexts it will likely exceed 6 GB VRAM and offload to CPU (slow). If the
smoke shows <5 tok/s or CPU offload, document and either drop it or accept it
with a measured-latency caveat.

## Rejected (evidence for the selection, §3.6)

- **Llama** — no open Llama in the ≤13B dense class after Llama 3.x: Llama 4 is
  MoE (109B/400B), and there is no open Llama 5 small dense. `granite4.1:8b`
  takes the IBM/Llama slot. (`llama3.1:8b` may be kept ONLY as an optional 5th
  legacy baseline for continuity if GPU budget allows — last priority.)
- **Kimi K2.6** — open-weight but ~1T-parameter MoE; requires 8×H100. Out of the
  study's hardware class (≤6 GB VRAM).
- **DeepSeek-R1 distills (7/8B)** — distilled on Qwen/Llama bases (not an
  independent lab family) and are reasoning models whose "thinking" tokens
  distort latency and the claim extraction the NLI verifier depends on.

## To complete in Phase 4 (smoke)

For each pulled model: exact tag + digest + pull date + on-disk size + **measured
VRAM** + **tokens/s** (smoke) + params (total/effective) + cost-proxy per query.

**Sources (verified 2026-06-07):**
Qwen3.5 — ollama.com/library/qwen3.5:9b; Gemma 4 — blog.google / ollama.com/library/gemma4:e4b ;
Granite 4.1 — ollama.com/library/granite4.1:8b , huggingface.co/ibm-granite/granite-4.1-8b ;
Mistral/Ministral 3 — mistral.ai/news/mistral-3 (Ministral 3 8B, 2025-12-02).
