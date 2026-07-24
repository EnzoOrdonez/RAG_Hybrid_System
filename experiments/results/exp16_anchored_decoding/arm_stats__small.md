# exp16_anchored_decoding — arm vs baseline_repro (NLI small, vb_agree τ0.7)

baseline_repro mean faithfulness: **0.2961** (n=60 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 2 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| anchored_cite | True | 50 | 0.2961 | 0.2535 | -0.0342 | [-0.1003, 0.0256] | 0.34986 | -0.1482 (negligible) | 0.59065 | no |
| strict_abstain | True | 55 | 0.2961 | 0.2833 | -0.0022 | [-0.1175, 0.1158] | 0.59065 | -0.0049 (negligible) | 0.59065 | no |

**0/2 arm-vs-baseline contrasts significant (BH).**