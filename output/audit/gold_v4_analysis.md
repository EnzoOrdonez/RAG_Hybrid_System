# Gold v4 — arbitraje humano del verificador

n anotado 150/150 · usados 132 · dudoso 18 (regla: exclude). Familia BH = 5 candidate-vs-human kappa tests (fdr_bh).
Pesos Horvitz-Thompson sobre un pool de 14409 claims; **n efectivo de Kish = 38.5** — la κ ponderada estima la poblacion real pero con varianza alta por diseño; la columna `κ random_anchor` es la lectura limpia sin supuestos y `κ` por estrato es la que discrimina verificadores.

| Candidato | κ ponderado | IC95 | p_BH | κ random_anchor | acc anchor | prec | rec | ECE |
|---|---|---|---|---|---|---|---|---|
| hhem | 0.315 | [0.0137, 0.5712] | 0.212 | 0.3966 | 0.7037 | 0.6585 | 0.6136 | 0.3686 |
| E5_base_and_hhem | 0.1601 | [-0.0827, 0.3969] | 0.4545 | 0.1702 | 0.5185 | 0.6444 | 0.3295 | 0.3686 |
| small | 0.0916 | [-0.1844, 0.3548] | 0.6545 | 0.0156 | 0.4815 | 0.7907 | 0.3864 | 0.27 |
| base | 0.0839 | [-0.1887, 0.3405] | 0.6545 | 0.078 | 0.4815 | 0.6441 | 0.4318 | 0.4657 |
| E1_mean | -0.0288 | [-0.2844, 0.1861] | 0.8622 | -0.08 | 0.2963 | 0.6333 | 0.2159 | 0.4657 |

**Sesgo de evidencia (etapa B):** 18/50 juicios cambian al ver los 5 chunks (tasa 0.36); 12 hacia 'correcto'.

flips toward 'correcto' are the confound: evidence that was present in the context but hidden from stage A. A high rate means the stage-A kappa understates every verifier that reads all 5 chunks (HHEM, NLI vb_agree) and the weighted kappa above should be read as a lower bound for them.

Promotion to primary verifier is NOT decided here: the pre-registered blind criterion is the negative control in ensemble_sweep_results.json. This table is the human-arbitration evidence that goes to Enzo alongside it.