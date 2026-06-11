# Tabla 6c — Clasificación v2 de respuestas (census) — exp12_matrix

pure_decline: rechazo en apertura (300c). hedged_partial: marcador de rechazo tardío con contenido sustantivo. answered: sin marcador. % vacías sobre n total (qwen sin_rag: 91,2 % vacías).

| Config | % pure_decline | % hedged_partial | % answered | % vacías | n |
|---|---|---|---|---|---|
| Sin RAG — Granite 4.1 8B | 2,6 | 3,6 | 93,8 | 0,0 | 194 |
| Sin RAG — Gemma 4 E4B | 1,5 | 9,3 | 89,2 | 0,0 | 194 |
| Sin RAG — Mistral 7B | 0,5 | 2,6 | 96,9 | 0,0 | 194 |
| Sin RAG — Qwen 3.5 9B | 0,0 | 0,0 | 100,0 | 91,2 | 194 |
| RAG léxico (BM25) — Granite 4.1 8B | 61,3 | 8,2 | 30,4 | 0,0 | 194 |
| RAG léxico (BM25) — Gemma 4 E4B | 63,5 | 1,7 | 34,8 | 8,2 | 194 |
| RAG léxico (BM25) — Mistral 7B | 39,2 | 6,2 | 54,6 | 0,0 | 194 |
| RAG léxico (BM25) — Qwen 3.5 9B | 67,7 | 13,1 | 19,2 | 33,0 | 194 |
| RAG denso (BGE) — Granite 4.1 8B | 56,2 | 10,3 | 33,5 | 0,0 | 194 |
| RAG denso (BGE) — Gemma 4 E4B | 61,1 | 1,7 | 37,1 | 9,8 | 194 |
| RAG denso (BGE) — Mistral 7B | 30,9 | 10,3 | 58,8 | 0,0 | 194 |
| RAG denso (BGE) — Qwen 3.5 9B | 63,4 | 10,6 | 26,1 | 26,8 | 194 |
| RAG híbrido — Granite 4.1 8B | 55,2 | 9,8 | 35,1 | 0,0 | 194 |
| RAG híbrido — Gemma 4 E4B | 60,6 | 2,3 | 37,1 | 9,8 | 194 |
| RAG híbrido — Mistral 7B | 37,1 | 9,3 | 53,6 | 0,0 | 194 |
| RAG híbrido — Qwen 3.5 9B | 66,9 | 7,7 | 25,4 | 26,8 | 194 |