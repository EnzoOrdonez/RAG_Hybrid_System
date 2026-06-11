# Tabla 6b — Tasa de declinación honesta por escenario y modelo — exp12_matrix

Fracción de respuestas marcadas como declinación honesta ("no hay información suficiente"). Confunde la comparación de fidelidad entre modelos: granite declina mucho más que mistral.

| Escenario | Granite 4.1 8B | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| Sin RAG | 0,0% | 0,0% | 0,0% | 0,0% |
| RAG léxico (BM25) | 65,5% | 55,2% | 25,3% | 43,8% |
| RAG denso (BGE) | 61,9% | 51,0% | 20,6% | 47,9% |
| RAG híbrido | 61,9% | 52,6% | 24,2% | 46,4% |