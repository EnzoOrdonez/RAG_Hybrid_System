# Latencias por etapa (p50/p95) — exp12_matrix

Excluye respuestas cacheadas (`from_cache=True`, latency=0) y errores. Generación y total en segundos; verificación NLI en ms.

| Config (escenario | modelo) | n | gen p50 (s) | gen p95 (s) | NLI p50 (ms) | NLI p95 (ms) | total p50 (s) | total p95 (s) | tok/s p50 |
|---|---|---|---|---|---|---|---|---|
| denso | gemma4-e4b | 194 | 38,9 | 39,9 | 86,5 | 724,0 | 39,1 | 40,4 | 25,8 |
| denso | granite4.1-8b | 194 | 88,8 | 254,5 | 1908,0 | 7277,4 | 91,2 | 261,8 | 4,3 |
| denso | mistral-7b-instruct | 194 | 44,9 | 109,7 | 953,4 | 2510,9 | 45,8 | 111,5 | 9,5 |
| hibrido | gemma4-e4b | 194 | 36,1 | 40,0 | 56,8 | 750,9 | 36,6 | 40,5 | 25,8 |
| hibrido | granite4.1-8b | 194 | 66,3 | 159,1 | 1220,8 | 2986,8 | 67,6 | 161,3 | 7,1 |
| hibrido | mistral-7b-instruct | 194 | 41,8 | 90,3 | 1077,8 | 2698,4 | 43,0 | 92,6 | 9,8 |
| lexico | gemma4-e4b | 194 | 38,6 | 39,8 | 85,7 | 741,6 | 38,9 | 40,4 | 25,9 |
| lexico | granite4.1-8b | 194 | 65,4 | 159,6 | 1194,1 | 4298,0 | 67,0 | 161,9 | 7,1 |
| lexico | mistral-7b-instruct | 194 | 33,6 | 77,6 | 639,6 | 1715,3 | 34,7 | 78,9 | 11,6 |
| sin_rag | gemma4-e4b | 194 | 38,7 | 39,3 | 0,0 | 0,0 | 38,7 | 39,3 | 26,4 |
| sin_rag | granite4.1-8b | 194 | 65,7 | 105,0 | 0,0 | 0,0 | 65,7 | 105,0 | 9,2 |
| sin_rag | mistral-7b-instruct | 194 | 26,6 | 43,3 | 0,0 | 0,0 | 26,6 | 43,3 | 14,4 |
| sin_rag | qwen3.5-9b | 110 | 238,8 | 240,5 | 0,0 | 0,0 | 238,8 | 240,5 | 4,3 |