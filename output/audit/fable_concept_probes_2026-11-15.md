# Sondas de concepto sobre el diseño de evaluación (2026-11-15)

Sin pesos de modelos: solo vocabularios (`tokenizer.json`), la corrida de auditoría exp20 (no firmada) y la evidencia firmada en modo lectura.

## 1. Truncado silencioso a 512 tokens (Flags 33, 130 y 144: conocidos, nunca medidos)

**Embedder (bge-large, 510 tokens útiles).** Fragmentos del índice: 24481. Longitud en tokens p50/p95/máx: 299/734/11187. Fragmentos que no caben: 7072 (28.9 %). Tokens del corpus que el vector no representa: 13.5 %. BM25 sí indexa el texto completo, así que las dos piernas no ven el mismo documento.

**Reranker (ms-marco, par consulta + fragmento).**

| Fragmentos de | Pares | Pares truncados | Tokens de fragmento no leídos |
|---|---|---|---|
| pool_hibrido | 9330 | 2466 (26.4 %) | 11.3 % |
| lexico | 970 | 213 (22.0 %) | 8.3 % |
| denso | 970 | 280 (28.9 %) | 12.2 % |
| hibrido_pre | 970 | 251 (25.9 %) | 9.2 % |
| lexico_rerank | 970 | 376 (38.8 %) | 12.7 % |
| denso_rerank | 970 | 405 (41.8 %) | 14.2 % |
| hibrido_rerank | 970 | 391 (40.3 %) | 14.5 % |
| union_rerank | 970 | 415 (42.8 %) | 14.7 % |

**Verificador NLI (DeBERTa-v3, par fragmento + afirmación).** Cada celda: fragmentos del top-5 truncados / tokens de contexto que el verificador no lee. El LLM sí recibe el fragmento completo, así que una afirmación apoyada en la cola de un fragmento largo cuenta como no apoyada.

| Contexto del escenario | afirmación de 20 tokens | afirmación de 40 tokens | afirmación de 60 tokens | Consultas con algún fragmento truncado (40 tokens) |
|---|---|---|---|---|
| lexico | 19.0 % / 6.1 % | 21.1 % / 7.4 % | 23.6 % / 9.0 % | 109/194 |
| denso | 26.2 % / 9.8 % | 29.4 % / 11.4 % | 30.8 % / 13.1 % | 142/194 |
| hibrido_rerank | 37.0 % / 11.5 % | 39.9 % / 13.5 % | 42.7 % / 15.7 % | 159/194 |

**¿Se nota en la fidelidad?** Regresión dentro de cada par (modelo, consulta): la única variación es el escenario de recuperación. `x1` = fracción del contexto que el NLI no lee; `x2` = tokens de contexto en miles. Errores agrupados por consulta.

| Verificador | Filas | Pares (modelo, consulta) | β fracción no leída | β miles de tokens |
|---|---|---|---|---|
| small | 912 | 453 | +0.521 (p = 0.174) | -0.041 (p = 0.446) |
| base | 912 | 453 | +0.016 (p = 0.959) | -0.083 (p = 0.077) |

## 2. El NDCG es relativo al pool: el mismo ranking cambia de nota al añadir brazos

- [OK] exp20 reproduce el NDCG@5 circular firmado de exp11 con el pool original de cuatro sistemas: desviación máxima 8.4e-08

| Brazo | NDCG@5 circular, pool de 4 (firmado) | NDCG@5 circular, pool de 7 | Cambio |
|---|---|---|---|
| lexico | 0.5516 | 0.5434 | -0.0083 |
| denso | 0.6494 | 0.6356 | -0.0138 |
| hibrido_pre | 0.6681 | 0.6576 | -0.0104 |
| lexico_rerank | no existe | 0.8916 |  |
| denso_rerank | no existe | 0.9543 |  |
| hibrido_rerank | 0.9948 | 0.9695 | -0.0253 |
| union_rerank | no existe | 0.9948 |  |

Ningún ranking cambió entre las dos columnas: solo cambió quién más aportó fragmentos al pool. Oráculo circular (el propio reranker): la tabla mide la dependencia del pool, no la calidad.

## 3. Asimetría de sobre-recuperación cuando hay filtro de proveedor

| Consultas | n | Pool híbrido medio | Pools con menos de 50 | NDCG@5 circular: híbrido+reranker − denso+reranker | Pierden algún fragmento del top-5 de la unión |
|---|---|---|---|---|---|
| con filtro | 79 | 45.3 | 47 | +0.0323 (p = 0.013) | 44/79 |
| sin filtro | 115 | 50.0 | 0 | +0.0034 (p = 0.244) | 67/115 |

## 4. El reranker prefiere fragmentos largos

- Correlación de Spearman entre puntaje del reranker y longitud en tokens, dentro de cada consulta (unión de pools): media +0.150, positiva en 144/194 consultas, p = 2.2e-15 (Wilcoxon).
- Tokens medios por fragmento: pool 367; lexico 317; denso 376; hibrido_pre 340; lexico_rerank 401; denso_rerank 417; hibrido_rerank 408; union_rerank 420.
- Consecuencia: los escenarios con reranker entregan más texto al LLM. La comparación de generación «híbrido frente a base» mezcla qué fragmentos se eligen con cuánto texto se entrega.

## 5. Las 194 consultas no son independientes: salen de plantillas

- Plantillas (proveedor enmascarado, primeras 4 palabras): 74 para 194 consultas. Mayor grupo: 15. Grupos de una sola consulta: 41.

| Contraste (NDCG@5 circular, pool de 7) | Diferencia media | ICC por plantilla | Efecto de diseño | n efectivo | p con consultas independientes | p agrupando por plantilla |
|---|---|---|---|---|---|---|
| híbrido+reranker − denso (firmado, sin reranker) | +0.3339 | +0.420 | 2.83 | 69 | 5.8e-47 | 7e-22 |
| híbrido+reranker − léxico (firmado, sin reranker) | +0.4262 | +0.426 | 2.85 | 68 | 8.5e-52 | 1.6e-20 |
| híbrido+reranker − denso+reranker (control exp20) | +0.0152 | -0.054 | 0.76 | 254 | 0.0085 | 0.018 |
| híbrido+reranker − léxico+reranker (control exp20) | +0.0780 | +0.387 | 2.68 | 72 | 3.9e-08 | 0.00018 |

Lo mismo sobre la generación (exp12): brecha de fidelidad entre modelos con la pregunta y el contexto fijos, juntando los tres escenarios. Diferencias pareadas por (escenario, consulta). Holm sobre los 6 pares.

Verificador `small`:

| A − B | Brecha | Celdas | ICC por plantilla | p Holm (unidad: consulta) | p Holm (unidad: plantilla) |
|---|---|---|---|---|---|
| gemma4-e4b − granite4.1-8b | +0.057 | 127 | +0.125 | 0.295 | 0.355 |
| gemma4-e4b − mistral-7b-instruct | +0.039 | 148 | +0.164 | 0.932 | 1 |
| gemma4-e4b − qwen3.5-9b | -0.010 | 77 | +0.064 | 1 | 1 |
| granite4.1-8b − mistral-7b-instruct | -0.061 | 209 | -0.037 | 0.00126 | 0.000782 |
| granite4.1-8b − qwen3.5-9b | -0.008 | 96 | +0.185 | 1 | 1 |
| mistral-7b-instruct − qwen3.5-9b | +0.030 | 110 | +0.096 | 0.932 | 1 |

Verificador `base`:

| A − B | Brecha | Celdas | ICC por plantilla | p Holm (unidad: consulta) | p Holm (unidad: plantilla) |
|---|---|---|---|---|---|
| gemma4-e4b − granite4.1-8b | +0.078 | 127 | +0.137 | 0.066 | 0.144 |
| gemma4-e4b − mistral-7b-instruct | +0.069 | 148 | +0.126 | 0.134 | 0.24 |
| gemma4-e4b − qwen3.5-9b | -0.044 | 77 | +0.241 | 0.3 | 0.437 |
| granite4.1-8b − mistral-7b-instruct | -0.025 | 209 | +0.214 | 0.267 | 0.437 |
| granite4.1-8b − qwen3.5-9b | -0.057 | 96 | +0.133 | 0.134 | 0.24 |
| mistral-7b-instruct − qwen3.5-9b | -0.060 | 110 | +0.002 | 0.0576 | 0.0198 |

## 6. Texto duplicado que ocupa huecos del top-5

- Fragmentos del índice con texto idéntico a otro: 1951 de 24481 (8.0 %), en 806 grupos (el mayor, 69).

| Brazo | Consultas con texto repetido en el top-5 | Huecos del top-5 gastados en repeticiones |
|---|---|---|
| lexico | 31/194 | 61/970 (6.3 %) |
| denso | 26/194 | 40/970 (4.1 %) |
| hibrido_pre | 34/194 | 59/970 (6.1 %) |
| lexico_rerank | 30/194 | 49/970 (5.1 %) |
| denso_rerank | 23/194 | 37/970 (3.8 %) |
| hibrido_rerank | 25/194 | 43/970 (4.4 %) |
| union_rerank | 25/194 | 42/970 (4.3 %) |

## 7. Qué dicen las métricas en términos absolutos

- El oráculo independiente da probabilidades en (0, 1). La mediana del pool es 0.131: «relevante» en P@k, R@k y MRR significa «por encima de la mediana del pool», no «relevante». Por construcción la mitad del pool es relevante, sea cual sea su calidad real.

| Sistema | NDCG@5 (relativo al pool) | Probabilidad media de relevancia del top-5 según el oráculo | P@5 con umbral 0 |
|---|---|---|---|
| RAG Lexico (BM25) | 0.4421 | 0.2550 | 1.00 |
| RAG Semantico (Dense) | 0.6237 | 0.3586 | 1.00 |
| RAG Hibrido Propuesto | 0.7405 | 0.4232 | 1.00 |
| RAG Hibrido (pre-rerank RRF) | 0.6026 | 0.3482 | 1.00 |

- Un NDCG@5 de 0,74 no significa que el 74 % del top-5 sea relevante: el propio oráculo asigna al top-5 del mejor sistema una probabilidad media de relevancia de 0,42. La columna de umbral 0 vale 1,00 para todos con este oráculo (una probabilidad nunca es negativa): esa variante no discrimina.

## 8. Tipado de consultas: «cross_cloud» no significa «varios proveedores»

- El procesador tipa 51 consultas como `cross_cloud`. De ellas, 26 comparan dos servicios del MISMO proveedor (por ejemplo, dos servicios de AWS): reciben la plantilla de comparación entre nubes y se recuperan sin filtro de proveedor.

Fragmentos del top-5 que son de un proveedor que la pregunta no nombra:

| Grupo | n | lexico | denso | hibrido_pre | hibrido_rerank | denso_rerank |
|---|---|---|---|---|---|---|
| un proveedor, con filtro | 79 | 0/395 (0.0 %) | 0/395 (0.0 %) | 0/395 (0.0 %) | 0/395 (0.0 %) | 0/395 (0.0 %) |
| un proveedor, otro tipo, sin filtro | 64 | 32/320 (10.0 %) | 14/320 (4.4 %) | 7/320 (2.2 %) | 10/320 (3.1 %) | 11/320 (3.4 %) |
| un proveedor, tipada cross_cloud, sin filtro | 26 | 3/130 (2.3 %) | 2/130 (1.5 %) | 0/130 (0.0 %) | 0/130 (0.0 %) | 0/130 (0.0 %) |
| varios proveedores (anotado) | 25 | 1/125 (0.8 %) | 0/125 (0.0 %) | 0/125 (0.0 %) | 0/125 (0.0 %) | 0/125 (0.0 %) |

## 9. Comprobaciones

- Fallos: 0
