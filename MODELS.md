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

---

## Phase 4 smoke results (2026-06-07; 3 q × 4 scenarios, determinism 3×)

**Pull outcome:** `granite4.1:8b` ✅ (5.3 GB), `gemma4:e4b` ✅ (9.6 GB), `qwen3.5:9b` ✅ (6.6 GB);
`gemma4:12b` ✗ (pull failed — moot, would not fit 6 GB); `mistral3:8b` / Ministral-3-8B ✗
(no Ollama library tag yet → **fallback `mistral:7b-instruct`**, documented per §3.6 as the
last small dense Mistral available on Ollama).

| Model (tag) | VRAM used | tok/s (warm, greedy) | determ. 3× @ temp=0 | Verdict |
|---|---|---|---|---|
| `granite4.1:8b` (IBM) | 5351 MB ✅ | 7.1 | **YES** | **primary** — RAG-tuned, deterministic, fits |
| `qwen3.5:9b` (Alibaba) | 5443 MB ✅ | 5.2 | **YES** | fits (disk 6.6 GB, VRAM OK); slow + verbose → ETA bottleneck |
| `gemma4:e4b` (Google) | 5487 MB ✅ | 22.7 | **NO** | fast, fits, but non-deterministic at temp=0 |
| `mistral:7b-instruct` (Mistral) | 5189 MB ✅ | 8.0 | **NO** | fallback; non-deterministic at temp=0 |

All four **fit in 6 GB** (qwen3.5:9b VRAM concern resolved). Gemma 12B excluded.

**Determinism (Phase 1d):** granite4.1 and qwen3.5 are bit-identical across 3 runs at
temp=0; gemma4:e4b and mistral:7b are NOT (Ollama/llama.cpp kernel-level non-determinism on
this GPU). The LLM cache freezes one sample per query, so a single matrix run is
reproducible-from-cache, but re-generation of gemma/mistral may differ → documented
limitation; the deterministic **granite4.1** is the recommended headline model.

**ETA (full matrix 4×4×194 ≈ 3104 generations):** ≈ **50–58 h**, dominated by qwen3.5
(~26 h alone: 5.2 tok/s + ~900-token answers). Above the 28–52 h estimate. Levers:
reduce `num_predict` (1024→512) for verbose models, drop qwen3.5, or stage with resume.

**Latency-measurement note:** cached generations report `latency_ms=0` (→ spurious tok/s);
the Phase 6 latency aggregator must exclude `from_cache=True` rows, or clear the LLM cache
before the timed matrix run.

## NLI verifiers (faithfulness instrument, ledger N5 — added 2026-06-11)

| Role | Model | Notes |
|---|---|---|
| Runtime verifier (exp1-13) | `cross-encoder/nli-deberta-v3-small` | HF cache; softmax + TRUE rule, ENT/CONTR thresholds 0.7, max over 5 chunks |
| Second verifier (F3b audit) | `cross-encoder/nli-deberta-v3-base` | **fp16** (max prob drift 1.8e-4 vs fp32); local snapshot `data/models/nli-deberta-v3-base/` (gitignored, ~700 MB) because HF downloads are blocked by TLS interception on this machine — re-download with `curl --ssl-no-revoke` |

Agreement (N5): claim-level kappa **0.411** on the 50-claim human sample; config-mean
Spearman 0.825 (published metric) / 0.559 n.s. (v2 primary); per-model scenario ordering
flips in 3/4 models → NLI faithfulness is reported as instrument-relative (contrasts only).
The claim-format ablation (small verifier, no "Header:" prefix) was cleanly negative
(kappa 0.87). Details: `output/audit/rescore_v2_summary.md`.

## Demo model (UI, not benchmarks — F6, 2026-06-11)

Demo default = `llama3.1:8b-instruct-q4_K_M`: the only model that fits the 6 GB GPU
whole once the UI frees CUDA for Ollama (`CUDA_VISIBLE_DEVICES=""` for aux models).
gemma4:e4b (9.6 GB on disk) always partially offloads to CPU on this machine and is
SLOWER in the demo despite its exp12 throughput (exp12 did no live retrieval, so
Ollama had the GPU nearly to itself). See README "Cómo lanzar la demo".
