# Tier A — arm vs baseline_repro (HHEM grounding, max_chunk τ0.5)

Baseline_repro mean faithfulness: **0.4499** (n=60/60 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 4 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| reranker_off | True | 58 | 0.4499 | 0.4381 | 0.0072 | [-0.0737, 0.0849] | 0.95766 | 0.0229 (negligible) | 0.95766 | no |
| final_top_k_3 | False | 58 | 0.4499 | 0.4378 | -0.0056 | [-0.1047, 0.0932] | 0.77135 | -0.0149 (negligible) | 0.95766 | no |
| context_reversed | True | 55 | 0.4499 | 0.4693 | 0.0603 | [-0.016, 0.1405] | 0.26901 | 0.2058 (small) | 0.95766 | no |
| context_lost_middle | False | 59 | 0.4499 | 0.4242 | -0.0249 | [-0.0982, 0.0456] | 0.62948 | -0.0876 (negligible) | 0.95766 | no |

**0/4 arm-vs-baseline contrasts significant (BH).**
det3x=False (final_top_k_3, context_lost_middle) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.