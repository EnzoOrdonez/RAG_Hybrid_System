# exp19b_anchored_selector — arm vs baseline_repro (HHEM grounding, max_chunk τ0.5)

baseline_repro mean faithfulness: **0.4572** (n=190 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 1 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| claim_selected | True | 186 | 0.4572 | 0.5014 | 0.0451 | [0.0065, 0.0819] | 0.01798 | 0.1727 (negligible) | 0.01798 | YES |

**1/1 arm-vs-baseline contrasts significant (BH).**