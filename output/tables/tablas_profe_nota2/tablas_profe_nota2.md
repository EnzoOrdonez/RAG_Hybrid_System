# Métricas de recuperación — corpus reconstruido (194 consultas)

_Fuente: `experiments/results/exp10_retrieval194/` — 194 consultas, corpus 2697 docs / 24481 fragmentos, seed=42._


## Sin RAG

| Modelo | Precision@1 | Precision@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| Mistral 7B | — | — | — | — | — |
| Qwen 2.5 7B | — | — | — | — | — |
| Llama 3.1 8B | — | — | — | — | — |

> Nota. Las métricas de recuperación son independientes del modelo generador, pues la etapa de recuperación ocurre antes de la generación; por ello los valores coinciden entre modelos. El escenario sin RAG no recupera documentos, por lo que estas métricas no aplican. Resultados sobre el corpus reconstruido (2 697 documentos / 24 481 fragmentos) con las 194 consultas del conjunto de evaluación depurado.


## RAG léxico (BM25)

| Modelo | Precision@1 | Precision@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| Mistral 7B | 0,820 | 0,769 | 0,386 | 0,861 | 0,584 |
| Qwen 2.5 7B | 0,820 | 0,769 | 0,386 | 0,861 | 0,584 |
| Llama 3.1 8B | 0,820 | 0,769 | 0,386 | 0,861 | 0,584 |

> Nota. Las métricas de recuperación son independientes del modelo generador, pues la etapa de recuperación ocurre antes de la generación; por ello los valores coinciden entre modelos. El escenario sin RAG no recupera documentos, por lo que estas métricas no aplican. Resultados sobre el corpus reconstruido (2 697 documentos / 24 481 fragmentos) con las 194 consultas del conjunto de evaluación depurado.


## RAG denso (BGE)

| Modelo | Precision@1 | Precision@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| Mistral 7B | 0,902 | 0,788 | 0,428 | 0,931 | 0,692 |
| Qwen 2.5 7B | 0,902 | 0,788 | 0,428 | 0,931 | 0,692 |
| Llama 3.1 8B | 0,902 | 0,788 | 0,428 | 0,931 | 0,692 |

> Nota. Las métricas de recuperación son independientes del modelo generador, pues la etapa de recuperación ocurre antes de la generación; por ello los valores coinciden entre modelos. El escenario sin RAG no recupera documentos, por lo que estas métricas no aplican. Resultados sobre el corpus reconstruido (2 697 documentos / 24 481 fragmentos) con las 194 consultas del conjunto de evaluación depurado.


## RAG híbrido (propuesto)

| Modelo | Precision@1 | Precision@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| Mistral 7B | 0,923 | 0,864 | 0,456 | 0,939 | 0,752 |
| Qwen 2.5 7B | 0,923 | 0,864 | 0,456 | 0,939 | 0,752 |
| Llama 3.1 8B | 0,923 | 0,864 | 0,456 | 0,939 | 0,752 |

> Nota. Las métricas de recuperación son independientes del modelo generador, pues la etapa de recuperación ocurre antes de la generación; por ello los valores coinciden entre modelos. El escenario sin RAG no recupera documentos, por lo que estas métricas no aplican. Resultados sobre el corpus reconstruido (2 697 documentos / 24 481 fragmentos) con las 194 consultas del conjunto de evaluación depurado.
