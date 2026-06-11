# Tabla 5 v2 — Modelo principal (Granite 4.1 8B) — exp12_matrix

Métrica primaria v2 (N5): media de fidelidad NLI sobre respuestas NO declinadas (excluye pure_decline = marcador de rechazo en los primeros 300 caracteres; incluye respuestas con hedge tardío). n_answered entre paréntesis. `*` Sin RAG = 0 por construcción (N3).

| Escenario | Fidelidad (primaria) | n_answered | Sens. A | Sens. B | Publicada v1 | % pure_decline | % hedged |
|---|---|---|---|---|---|---|---|
| Sin RAG | 0,000* | 189 | 0,000 | 0,000 | 0,000 | 2,6 | 3,6 |
| RAG léxico (BM25) | 0,234 | 75 | 0,243 | 0,261 | 0,170 | 61,3 | 8,2 |
| RAG denso (BGE) | 0,255 | 85 | 0,267 | 0,272 | 0,193 | 56,2 | 10,3 |
| RAG híbrido | 0,293 | 87 | 0,316 | 0,330 | 0,202 | 55,2 | 9,8 |