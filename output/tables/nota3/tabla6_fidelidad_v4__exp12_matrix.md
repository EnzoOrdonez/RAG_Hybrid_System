# Tabla 6 v4 — Fidelidad sin vacuas (N9) — exp12_matrix

Métrica v4 (N9, verificador experiments/results/exp12_matrix/faithfulness_rescore_v3__small__vb_agree.json): faithfulness_answered = v3 (artefactos de formato excluidos, H1; guarda vb_agree, H2) + exclusión de respuestas 100%-artefactos (genuine==0, faithfulness=1,0 vacuo en v3; 59/1798, mismo trato que method='none' por Flag 137). `*` Sin RAG = 0 por construcción (N3). v1/v2/v3 se conservan como superseded. **Veredictos:** retrieval n.s. en fidelidad (robusto, 0/12 bajo AMBOS verificadores); entre-modelos 1/18 sig bajo small (denso granite-vs-mistral, d_z=+0,42, p_bh=0,014) y 1/18 bajo base (léxico gemma-vs-granite, d_z=−0,53, p_bh=0,038) — corrige el '2/18' de N8, que dependía de filas vacuas.

| Escenario | Granite 4.1 8B | Gemma 4 E4B | Mistral 7B | Qwen 3.5 9B |
|---|---|---|---|---|
| Sin RAG | 0,000* (189) | 0,000* (191) | 0,000* (193) | 0,000* (17) |
| RAG léxico (BM25) | 0,235 (75) | 0,413 (57) | 0,246 (114) | 0,255 (36) |
| RAG denso (BGE) | 0,247 (85) | 0,377 (60) | 0,281 (132) | 0,265 (45) |
| RAG híbrido | 0,299 (87) | 0,317 (56) | 0,288 (122) | 0,294 (43) |