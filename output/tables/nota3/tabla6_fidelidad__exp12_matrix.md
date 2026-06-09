# Tabla 6 — Fidelidad (faithfulness NLI) por escenario y modelo — exp12_matrix

Media sobre respuestas con claims verificables (excluye method none/error). `*` Sin RAG = 0 por construcción (sin contexto no hay claim verificable, N3): léase junto a la tabla de declinación, no como fidelidad comparable.

| Escenario | Granite 4.1 8B | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| Sin RAG | 0,000* | 0,000* | 0,000* | 0,000* |
| RAG léxico (BM25) | 0,170 | 0,331 | 0,222 | — |
| RAG denso (BGE) | 0,193 | 0,322 | 0,258 | — |
| RAG híbrido | 0,202 | 0,268 | 0,256 | — |