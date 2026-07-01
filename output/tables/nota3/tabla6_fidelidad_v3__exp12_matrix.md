# Tabla 6 v3 — Fidelidad corregida (N8) — exp12_matrix

Métrica v3 (N8, verificador experiments/results/exp12_matrix/faithfulness_rescore_v3__small__vb_agree.json): faithfulness_answered corregida = artefactos de formato excluidos del denominador (H1) + guarda de contradicción vb_agree (H2). `*` Sin RAG = 0 por construcción (N3). v1/v2 se conservan como superseded. **Veredictos:** retrieval n.s. en fidelidad (robusto, 0/12); entre-modelos 2/18 sig bajo small (6/18 base) — corrige el 'todo n.s.' de N5.

| Escenario | Granite 4.1 8B | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| Sin RAG | 0,000* (189) | 0,000* (191) | 0,000* (193) | 0,000* (17) |
| RAG léxico (BM25) | 0,235 (75) | 0,443 (60) | 0,246 (114) | 0,361 (42) |
| RAG denso (BGE) | 0,247 (85) | 0,425 (65) | 0,281 (132) | 0,351 (51) |
| RAG híbrido | 0,299 (87) | 0,363 (60) | 0,288 (122) | 0,354 (47) |