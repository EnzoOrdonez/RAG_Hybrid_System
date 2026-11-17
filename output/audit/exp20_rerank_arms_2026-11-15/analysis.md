# exp20: brazos de control con el mismo reranker

Corrida del 2026-11-15T23:33:48, 194 consultas, 3997.7 s. NO FIRMADO. Corrida de auditoría en output/audit; no es evidencia de experiments/results.
Modelos cargados: BAAI/bge-large-en-v1.5 (encoder denso); cross-encoder/ms-marco-MiniLM-L-12-v2 (reranker). No cargados: Ollama; NLI; HHEM; BAAI/bge-reranker-large (oráculo independiente; no está en disco).

**Cómo leerlo:** las secciones 4 y 5 usan el puntaje del propio reranker, que es circular: sirve para comparar qué pool le da al reranker mejores candidatos según su propio criterio, no para decir qué brazo recupera mejor. La calidad se lee en la sección 6 (oráculo independiente), si está presente.

## 1. Reproducción de exp11 con modelos reales

| Brazo | Top-5 idéntico al firmado (ids y orden) | Mismo conjunto |
|---|---|---|
| lexico | 194/194 | 194/194 |
| denso | 194/194 | 194/194 |
| hibrido_pre | 193/194 | 194/194 |
| hibrido_rerank | 194/194 | 194/194 |

## 2. Pools que ve el reranker

| Pool | Tamaño medio | Mínimo | Consultas con menos de 50 | Con menos de 25 |
|---|---|---|---|---|
| lexico | 49.9 | 38 | 3 | 0 |
| denso | 50.0 | 50 | 0 | 0 |
| hibrido | 48.1 | 25 | 47 | 0 |

- Consultas con filtro de proveedor: 79/194. En ellas el pool híbrido tiene 45.3 candidatos de media y el denso 50.0. Los brazos base piden 100 candidatos antes de filtrar y el híbrido fusiona 50 y filtra después, sin reponer.
- Consultas en las que la consulta densa difiere del texto original: 0/194.
- Desviación máxima al puntuar dos veces el mismo par: 2.86e-06.

## 3. Híbrido con reranker frente a denso con el mismo reranker

| Fragmentos en común en el top-5 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Consultas | 3 | 2 | 14 | 46 | 74 | 55 |

- Solapamiento medio: 3.81 de 5. Listas idénticas: 55/194.
- Del top-5 del híbrido, el 81.2% ya estaba en el pool denso de 50. Del top-5 del denso con reranker, el 87.8% estaba en el pool híbrido.

## 4. Qué pool le sirve mejor al reranker (medida circular, ver límite)

| Brazo | Puntaje medio del reranker en el top-5 (logit) | Longitud media del fragmento (caracteres) | Consultas con texto repetido en el top-5 |
|---|---|---|---|
| lexico | +2.209 | 1311 | 31/194 |
| denso | +2.599 | 1506 | 26/194 |
| hibrido_pre | +2.967 | 1404 | 34/194 |
| lexico_rerank | +4.005 | 1648 | 30/194 |
| denso_rerank | +4.216 | 1701 | 23/194 |
| hibrido_rerank | +4.286 | 1652 | 25/194 |
| union_rerank | +4.390 | 1714 | 25/194 |

| A − B | Diferencia media | d_z | p (Wilcoxon) | A gana / empate / A pierde |
|---|---|---|---|---|
| hibrido_rerank − denso_rerank | +0.070 | +0.25 | 0.0019 | 85/56/53 |
| hibrido_rerank − lexico_rerank | +0.280 | +0.41 | 4.6e-13 | 116/43/35 |
| union_rerank − hibrido_rerank | +0.104 | +0.64 | 6e-20 | 111/83/0 |
| union_rerank − denso_rerank | +0.174 | +0.61 | 4.2e-21 | 118/76/0 |

## 5. Lo que la fusión le quita al reranker

- En 111/194 consultas, al menos un fragmento que el reranker pondría en su top-5 sobre la unión de pools no llegó al pool híbrido. Son 151 fragmentos de 970: 81 estaban solo en el pool denso, 68 solo en el léxico y 2 en ambos.
- Composición del pool híbrido: 23.7% en ambos pools base, 38.7% solo en el denso, 37.6% solo en el léxico, 0.0% en ninguno.

## 6. NDCG@5 con el oráculo independiente (BAAI/bge-reranker-large, copia local verificada por SHA-256)

- [OK] reproduce el NDCG@5 independiente firmado de exp11 con el pool de cuatro sistemas: desviación máxima 1.9e-07.

| Brazo | NDCG@5, pool de 4 (firmado) | NDCG@5, pool de 7 | Probabilidad media de relevancia del top-5 |
|---|---|---|---|
| lexico | 0.4421 | 0.4276 | 0.2550 |
| denso | 0.6237 | 0.6035 | 0.3586 |
| hibrido_pre | 0.6026 | 0.5817 | 0.3482 |
| lexico_rerank | no existe | 0.6461 | 0.3699 |
| denso_rerank | no existe | 0.6985 | 0.4146 |
| hibrido_rerank | 0.7405 | 0.7142 | 0.4232 |
| union_rerank | no existe | 0.7062 | 0.4238 |

| A − B (NDCG@5, pool de 7) | Diferencia media | d_z | p (Wilcoxon) | p agrupando por plantilla | A gana / empate / A pierde |
|---|---|---|---|---|---|
| hibrido_rerank − denso_rerank | +0.0157 | +0.11 | 0.075 | 0.16 | 79/53/62 |
| hibrido_rerank − lexico_rerank | +0.0682 | +0.33 | 1.6e-05 | 0.0032 | 95/42/57 |
| denso_rerank − denso | +0.0951 | +0.37 | 2e-06 | 3.8e-06 | 128/0/66 |
| lexico_rerank − lexico | +0.2185 | +0.82 | 3.6e-20 | 7.2e-12 | 159/0/35 |
| hibrido_rerank − hibrido_pre | +0.1325 | +0.53 | 9.4e-11 | 8e-08 | 134/0/60 |
| hibrido_pre − denso | -0.0217 | -0.08 | 0.19 | 0.37 | 95/0/99 |
| union_rerank − hibrido_rerank | -0.0080 | -0.07 | 0.25 | 0.49 | 49/83/62 |

| Consultas | n | híbrido+reranker − denso+reranker | p (Wilcoxon) | gana / empate / pierde |
|---|---|---|---|---|
| con filtro de proveedor | 79 | +0.0002 | 0.66 | 23/24/32 |
| sin filtro | 115 | +0.0264 | 0.007 | 56/29/30 |

- La fusión deja fuera de su top-5 el primer resultado denso en 82/194 consultas. Según el oráculo independiente, ese fragmento tiene una probabilidad media de relevancia de 0.434 y el primero de la lista fusionada, 0.413 (diferencia +0.021, p = 0.71). De esos 82, 82 seguían en el pool de 50 y el reranker devolvió 29 al top-5 final.

**Brazo `hibrido_justo_rerank`** (derivado sin modelos: fusiona los pools ya filtrados por proveedor, 50 + 50, y reordena con los puntajes guardados del reranker). Coincide con `hibrido_rerank` en 115/115 consultas sin filtro y cambia el top-5 en 6/79 consultas con filtro.

| A − B (NDCG@5, pool de 8) | Consultas | n | Diferencia media | d_z | p (Wilcoxon) | p agrupando por plantilla | gana / empate / pierde |
|---|---|---|---|---|---|---|---|
| hibrido_justo_rerank − denso_rerank | todas | 194 | +0.0147 | +0.10 | 0.13 | 0.21 | 77/54/63 |
| hibrido_justo_rerank − denso_rerank | con filtro de proveedor | 79 | -0.0023 | -0.01 | 0.41 | 0.89 | 21/25/33 |
| hibrido_justo_rerank − denso_rerank | sin filtro | 115 | +0.0264 | +0.21 | 0.007 | 0.08 | 56/29/30 |
| hibrido_rerank − denso_rerank | todas | 194 | +0.0157 | +0.11 | 0.075 | 0.16 | 79/53/62 |
| hibrido_rerank − denso_rerank | con filtro de proveedor | 79 | +0.0002 | +0.00 | 0.66 | 0.99 | 23/24/32 |
| hibrido_rerank − denso_rerank | sin filtro | 115 | +0.0264 | +0.21 | 0.007 | 0.08 | 56/29/30 |
| hibrido_justo_rerank − hibrido_rerank | todas | 194 | -0.0010 | -0.04 | 0.46 | 0.57 | 2/188/4 |
| hibrido_justo_rerank − hibrido_rerank | con filtro de proveedor | 79 | -0.0025 | -0.06 | 0.46 | 0.57 | 2/73/4 |
| hibrido_justo_rerank − hibrido_rerank | sin filtro | 115 | +0.0000 | +0.00 | 1 | nan | 0/115/0 |

- Acuerdo entre jueces (Spearman dentro de cada consulta, ms-marco frente a bge-reranker): todos los brazos: media +0.472, mediana +0.538, por debajo de 0,5 en 86/194 consultas; solo brazos sin reranker: media +0.547, mediana +0.625, por debajo de 0,5 en 64/194 consultas. Es una cota inferior: solo se puntuaron fragmentos que algún brazo puso en su top-5 (rango restringido).

Afinidad de cada juez con cada pierna (puesto percentil según ese juez, explicado por el puesto según el otro juez, efectos fijos de consulta y dos indicadores de pertenencia). **No identifica un sesgo:** cuatro de los siete brazos eligen su top-5 con ms-marco y ninguno con bge-reranker, así que la muestra completa anula por construcción los coeficientes de ms-marco. En la muestra sin esa selección la asimetría desaparece.

| Muestra | Pares | Juez | β «estaba en el top-50 denso» | β «estaba en el top-50 léxico» |
|---|---|---|---|---|
| todos los brazos | 2957 | bge-reranker | +0.134 (p = 4.3e-22) | +0.060 (p = 4.5e-05) |
| todos los brazos | 2957 | ms-marco | +0.007 (p = 0.69) | -0.023 (p = 0.14) |
| solo brazos sin reranker | 2140 | bge-reranker | +0.096 (p = 2.7e-07) | +0.019 (p = 0.22) |
| solo brazos sin reranker | 2140 | ms-marco | +0.121 (p = 8.6e-12) | +0.093 (p = 1.6e-07) |

## Comprobaciones

- Fallos: 0
