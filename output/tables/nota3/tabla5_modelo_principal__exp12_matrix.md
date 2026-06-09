# Tabla 5 — Modelo principal (Granite 4.1 8B) por escenario — exp12_matrix

`*` Sin RAG fidelidad = 0 por construcción (N3).

| Escenario | Fidelidad | Alucinación (1−fid) | Declinación % | No-evidencia % | n_efectivo |
|---|---|---|---|---|---|
| Sin RAG | 0,000* | 1,000 | 0,0% | 100,0% | 194 |
| RAG léxico (BM25) | 0,170 | 0,830 | 65,5% | 0,0% | 189 |
| RAG denso (BGE) | 0,193 | 0,807 | 61,9% | 0,0% | 189 |
| RAG híbrido | 0,202 | 0,798 | 61,9% | 0,0% | 190 |