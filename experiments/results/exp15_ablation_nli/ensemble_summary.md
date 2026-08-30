# Tier 3 · Bloque B — ensembles + control negativo

**Criterio de selección (pre-registrado, anti-p-hacking):** tasa de falso-contradicted (NLI) / falso-grounded (HHEM) en el control negativo (400 pares aleatorios). Menor = mejor. El downstream es DESCRIPTIVO, NO criterio.

| candidato | control neg (↓) | sig RAG /12 | granite p_bh |
|---|---|---|---|
| E5_base_and_hhem | 0.0025 | 0 | 0.288492 |
| hhem | 0.0325 | 1 | 0.020441 |
| E1_mean[2m] | 0.09 | 0 | 0.488688 |
| E2_vote[2m=unanimity] | 0.1025 | 0 | 0.544029 |
| base | 0.215 | 0 | 0.332385 |
| E4_sym_base | 0.215 | 0 | 0.6 |
| small | 0.2375 | 0 | 0.084936 |
| E3_conservative[2m] | 0.4125 | 0 | 0.62913 |
| agg:small:noisy_or | 0.5475 | 1 | 0.009353 |
| agg:base:noisy_or | 0.6 | 0 | 0.222036 |

**Front-runner provisional (por control negativo):** `E5_base_and_hhem`

Nota: la selección definitiva del instrumento espera el gold humano (Bloque D). El downstream NO se usa para elegir.
