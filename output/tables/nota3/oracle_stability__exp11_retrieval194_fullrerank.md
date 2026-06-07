# Estabilidad del orden entre oráculos — exp11_retrieval194_fullrerank

Mitigación de la circularidad del oráculo. El reranker del pipeline es `ms-marco-MiniLM-L-12-v2`; usar ese mismo modelo como oráculo de relevancia es **circular** (tras el fix D12 el híbrido post-rerank alcanza NDCG@5 ≈ 1,0 por construcción). Se reporta además un oráculo **independiente** (`BAAI/bge-reranker-large`) y el ranking **pre-rerank** del híbrido (solo fusión RRF, sin el cross-encoder compartido).

## Tabla A — NDCG@5 graded por sistema y oráculo

| Sistema | ms-marco-circular (circular) | bge-reranker-indep (independiente) |
|---|---|---|
| Léxico (BM25) | 0,552 | 0,442 |
| Denso (BGE) | 0,649 | 0,624 |
| Híbrido (pre-rerank RRF) | 0,668 | 0,603 |
| Híbrido (post-rerank) | 0,995 | 0,740 |

## Tabla B — Contrastes clave NDCG@5 (d_z pareado, p BH-corregido)

| Contraste (NDCG@5) | Oráculo | d_z | p_BH | sig_BH |
|---|---|---|---|---|
| Híbrido (post-rerank) vs Denso (BGE) | ms-marco-circular (circular) | +1,38 | <0,001 | sí |
| Híbrido (post-rerank) vs Denso (BGE) | bge-reranker-indep (independiente) | +0,45 | <0,001 | sí |
| Híbrido (pre-rerank RRF) vs Denso (BGE) | ms-marco-circular (circular) | +0,07 | 0,344 | no |
| Híbrido (pre-rerank RRF) vs Denso (BGE) | bge-reranker-indep (independiente) | -0,08 | 0,240 | no |
| Híbrido (post-rerank) vs Híbrido (pre-rerank RRF) | ms-marco-circular (circular) | +1,24 | <0,001 | sí |
| Híbrido (post-rerank) vs Híbrido (pre-rerank RRF) | bge-reranker-indep (independiente) | +0,52 | <0,001 | sí |

**Lectura.** El orden Híbrido(post) > Denso > Híbrido(pre) ≈ Denso > Léxico se mantiene entre oráculos. La ventaja del híbrido sobre el denso es **real y significativa con el oráculo independiente** pero su magnitud estaba **inflada por la circularidad**; la fusión RRF sola (pre-rerank) **no** supera al denso. El aporte proviene de la etapa de reranking.
