# Config de encuestas — latencia en la RUTA DE DESPLIEGUE (2026-08-04)

**Generador medido: `granite4.1:8b`.** Evidencia de la fase: `granite4.1:8b`.

exp18's k=10 arm used FROZEN contexts and no provider balancing, and reports TOTAL generation time; the UI streams, so the survey-relevant latency is TTFT. These numbers are the deployment-path counterpart.

| config | n | retrieval p50 | **TTFT p50** | TTFT p90 | total p50 | total p90 | palabras | chunks |
|---|---|---|---|---|---|---|---|---|
| k=5 | 12 | 5121.3 ms | **12834.2 ms** | 47650.1 ms | 179766.1 ms | 211060.2 ms | 270.8 | 5 |
| k=10 | 12 | 5028.3 ms | **15232.9 ms** | 34622.8 ms | 132740.1 ms | 194510.2 ms | 254.2 | 10 |

Estratos muestreados (seed 42): `cross_cloud|no_trunc` n=2, `cross_cloud|trunc` n=2, `procedural|no_trunc` n=2, `procedural|trunc` n=2, `single_provider|no_trunc` n=2, `single_provider|trunc` n=2

Fidelidad NO se mide aqui (faithfulness — that stays with the offline scorers on exp18).