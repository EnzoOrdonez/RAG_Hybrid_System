# exp16_anchored_decoding — arm vs baseline_repro (HHEM grounding, max_chunk τ0.5)

baseline_repro mean faithfulness: **0.4983** (n=60 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 2 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| anchored_cite | True | 50 | 0.4983 | 0.4142 | -0.0561 | [-0.1405, 0.0245] | 0.24902 | -0.1882 (negligible) | 0.49804 | no |
| strict_abstain | True | 55 | 0.4983 | 0.5319 | 0.0377 | [-0.0724, 0.1469] | 0.54384 | 0.0909 (negligible) | 0.54384 | no |

**0/2 arm-vs-baseline contrasts significant (BH).**