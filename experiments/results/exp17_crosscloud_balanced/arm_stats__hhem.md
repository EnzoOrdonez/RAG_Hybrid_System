# exp17_crosscloud_balanced — arm vs baseline (HHEM grounding, max_chunk τ0.5)

baseline mean faithfulness: **0.4772** (n=25 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 1 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced | False | 25 | 0.4772 | 0.5583 | 0.081 | [-0.0373, 0.2032] | 0.23552 | 0.2564 (small) | 0.23552 | no |

**0/1 arm-vs-baseline contrasts significant (BH).**
det3x=False (balanced) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.