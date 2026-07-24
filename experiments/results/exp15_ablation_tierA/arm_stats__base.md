# Tier A — arm vs baseline_repro (NLI base, vb_agree τ0.7)

Baseline_repro mean faithfulness: **0.2038** (n=60/60 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 4 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| reranker_off | True | 58 | 0.2038 | 0.1435 | -0.05 | [-0.1374, 0.0312] | 0.82138 | -0.1495 (negligible) | 0.82138 | no |
| final_top_k_3 | False | 58 | 0.2038 | 0.222 | 0.0285 | [-0.0545, 0.1129] | 0.38875 | 0.0883 (negligible) | 0.53282 | no |
| context_reversed | True | 55 | 0.2038 | 0.2144 | 0.0467 | [-0.0146, 0.1165] | 0.36067 | 0.1886 (negligible) | 0.53282 | no |
| context_lost_middle | False | 59 | 0.2038 | 0.2019 | -0.0054 | [-0.0755, 0.0613] | 0.39961 | -0.0198 (negligible) | 0.53282 | no |

**0/4 arm-vs-baseline contrasts significant (BH).**
det3x=False (final_top_k_3, context_lost_middle) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.