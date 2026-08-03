# exp18 — reanalisis claim-level (3 contrastes vs baseline_repro)

GLMM = `supported ~ arm + (1|query)` (binomial VB, conserva el pareo). Bootstrap = remuestreo de cluster por query del diff micro-promediado, seed 42. **Tests BILATERALES** (pre-registro, entrada 19).

| Verificador | Brazo | n_q | n_claims | micro base | micro brazo | diff | boot95 | boot p | GLMM OR | GLMM p |
|---|---|---|---|---|---|---|---|---|---|---|
| small | oracle_evidence | 194 | 4136 | 0.2655 | 0.2727 | 0.0072 | [-0.0524, 0.068] | 0.8068 | 0.9511 | 0.37668 |
| small | evidence_swapped | 60 | 1005 | 0.2923 | 0.1034 | -0.1889 | [-0.3431, -0.0217] | 0.0262 | 0.2718 | 0.0 |
| small | final_top_k_10 | 194 | 4960 | 0.2655 | 0.2738 | 0.0084 | [-0.0438, 0.0638] | 0.7574 | 1.0413 | 0.4275 |
| base | oracle_evidence | 194 | 4136 | 0.1481 | 0.1992 | 0.0512 | [0.007, 0.0973] | 0.0246 | 1.4505 | 0.0 |
| base | evidence_swapped | 60 | 1005 | 0.1776 | 0.031 | -0.1466 | [-0.2186, -0.0759] | 0.0 | 0.1606 | 0.0 |
| base | final_top_k_10 | 194 | 4960 | 0.1481 | 0.1386 | -0.0094 | [-0.0447, 0.027] | 0.592 | 0.9625 | 0.52746 |
| hhem | oracle_evidence | 194 | 4136 | 0.4929 | 0.5305 | 0.0375 | [-0.0088, 0.0823] | 0.1138 | 1.1724 | 0.0009 |
| hhem | evidence_swapped | 60 | 1005 | 0.4615 | 0.1069 | -0.3546 | [-0.4358, -0.272] | 0.0 | 0.1205 | 0.0 |
| hhem | final_top_k_10 | 194 | 4960 | 0.4929 | 0.4723 | -0.0206 | [-0.0596, 0.017] | 0.2934 | 1.0221 | 0.58978 |

claim-level is conditional on a genuine claim, so declines and vacuous answers drop out. evidence_swapped declines in 88% of its queries, so its conditional result answers only 'among the claims it did assert, were they grounded?' -- read with arm_stats and guards.

genuine claims per response differ sharply across arms (baseline 10.6, oracle 10.7, swapped 4.8, top-10 15.0), which is why the claim-level view is a first-line reading here.