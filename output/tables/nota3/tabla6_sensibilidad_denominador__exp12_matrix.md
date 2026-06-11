# Tabla — Sensibilidad del denominador (4 definiciones) — exp12_matrix

Las conclusiones deben leerse bajo las 4 definiciones (espejo del multi-oráculo).

| Config | Primaria (sin pure_decline) | Sens. A (flag v1) | Sens. B (estricta) | Sens. C (publicada v1) | n primaria | n publicada |
|---|---|---|---|---|---|---|
| Sin RAG — Granite 4.1 8B | 0,000 | 0,000 | 0,000 | 0,000 | 189 | 194 |
| Sin RAG — Gemma 4 E4B | 0,000 | 0,000 | 0,000 | 0,000 | 191 | 194 |
| Sin RAG — Mistral 7B | 0,000 | 0,000 | 0,000 | 0,000 | 193 | 194 |
| Sin RAG — Qwen 3.5 9B | 0,000 | 0,000 | 0,000 | 0,000 | 17 | 17 |
| RAG léxico (BM25) — Granite 4.1 8B | 0,234 | 0,243 | 0,261 | 0,170 | 75 | 189 |
| RAG léxico (BM25) — Gemma 4 E4B | 0,381 | 0,400 | 0,375 | 0,331 | 60 | 104 |
| RAG léxico (BM25) — Mistral 7B | 0,245 | 0,231 | 0,261 | 0,222 | 114 | 168 |
| RAG léxico (BM25) — Qwen 3.5 9B | 0,246 | 0,268 | 0,260 | 0,306 | 42 | 130 |
| RAG denso (BGE) — Granite 4.1 8B | 0,255 | 0,267 | 0,272 | 0,193 | 85 | 189 |
| RAG denso (BGE) — Gemma 4 E4B | 0,356 | 0,331 | 0,345 | 0,322 | 65 | 101 |
| RAG denso (BGE) — Mistral 7B | 0,275 | 0,266 | 0,281 | 0,258 | 132 | 172 |
| RAG denso (BGE) — Qwen 3.5 9B | 0,246 | 0,238 | 0,250 | 0,254 | 51 | 141 |
| RAG híbrido — Granite 4.1 8B | 0,293 | 0,316 | 0,330 | 0,202 | 87 | 190 |
| RAG híbrido — Gemma 4 E4B | 0,312 | 0,285 | 0,298 | 0,268 | 60 | 98 |
| RAG híbrido — Mistral 7B | 0,282 | 0,274 | 0,290 | 0,256 | 122 | 174 |
| RAG híbrido — Qwen 3.5 9B | 0,293 | 0,257 | 0,319 | 0,278 | 47 | 142 |