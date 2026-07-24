# Tier A — arm vs baseline_repro (NLI small, vb_agree τ0.7)

Baseline_repro mean faithfulness: **0.3077** (n=60/60 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 4 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| reranker_off | True | 58 | 0.3077 | 0.2433 | -0.0579 | [-0.1637, 0.0445] | 0.47979 | -0.1409 (negligible) | 0.59945 | no |
| final_top_k_3 | False | 58 | 0.3077 | 0.2821 | -0.019 | [-0.1012, 0.0656] | 0.29986 | -0.0594 (negligible) | 0.59945 | no |
| context_reversed | True | 55 | 0.3077 | 0.3147 | 0.0426 | [-0.012, 0.104] | 0.48808 | 0.1932 (negligible) | 0.59945 | no |
| context_lost_middle | False | 59 | 0.3077 | 0.3315 | 0.0271 | [-0.042, 0.0949] | 0.59945 | 0.1005 (negligible) | 0.59945 | no |

**0/4 arm-vs-baseline contrasts significant (BH).**
det3x=False (final_top_k_3, context_lost_middle) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.