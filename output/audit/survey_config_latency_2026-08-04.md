# Config de encuestas — latencia en la RUTA DE DESPLIEGUE (2026-08-04)

exp18's k=10 arm used FROZEN contexts and no provider balancing, and reports TOTAL generation time; the UI streams, so the survey-relevant latency is TTFT. These numbers are the deployment-path counterpart.

| config | n | retrieval p50 | **TTFT p50** | TTFT p90 | total p50 | total p90 | palabras | chunks |
|---|---|---|---|---|---|---|---|---|
| k=5 | 12 | 1044.4 ms | **25985.0 ms** | 33942.8 ms | 94075.1 ms | 114906.2 ms | 258 | 5 |
| k=10 | 12 | 1343.4 ms | **47197.8 ms** | 50803.6 ms | 113257.9 ms | 137057.6 ms | 234.7 | 10 |

Estratos muestreados (seed 42): `cross_cloud|no_trunc` n=2, `cross_cloud|trunc` n=2, `procedural|no_trunc` n=2, `procedural|trunc` n=2, `single_provider|no_trunc` n=2, `single_provider|trunc` n=2

Fidelidad NO se mide aqui (faithfulness — that stays with the offline scorers on exp18).