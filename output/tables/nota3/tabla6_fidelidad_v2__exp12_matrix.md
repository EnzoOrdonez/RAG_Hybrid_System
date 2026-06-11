# Tabla 6 v2 — Fidelidad en respuestas contestadas (faithfulness_answered) — exp12_matrix

Métrica primaria v2 (N5): media de fidelidad NLI sobre respuestas NO declinadas (excluye pure_decline = marcador de rechazo en los primeros 300 caracteres; incluye respuestas con hedge tardío). n_answered entre paréntesis. `*` Sin RAG = 0 por construcción (N3).

| Escenario | Granite 4.1 8B | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| Sin RAG | 0,000* (189) | 0,000* (191) | 0,000* (193) | 0,000* (17) |
| RAG léxico (BM25) | 0,234 (75) | 0,381 (60) | 0,245 (114) | 0,246 (42) |
| RAG denso (BGE) | 0,255 (85) | 0,356 (65) | 0,275 (132) | 0,246 (51) |
| RAG híbrido | 0,293 (87) | 0,312 (60) | 0,282 (122) | 0,293 (47) |