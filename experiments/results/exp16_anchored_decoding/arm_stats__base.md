# exp16_anchored_decoding — arm vs baseline_repro (NLI base, vb_agree τ0.7)

baseline_repro mean faithfulness: **0.2258** (n=60 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 2 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| anchored_cite | True | 50 | 0.2258 | 0.1397 | -0.0396 | [-0.1045, 0.018] | 0.30716 | -0.1752 (negligible) | 0.53355 | no |
| strict_abstain | True | 55 | 0.2258 | 0.2458 | 0.0308 | [-0.0874, 0.1485] | 0.53355 | 0.0687 (negligible) | 0.53355 | no |

**0/2 arm-vs-baseline contrasts significant (BH).**