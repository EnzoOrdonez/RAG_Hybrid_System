# exp17_crosscloud_balanced — arm vs baseline_repro (NLI small, vb_agree τ0.7)

baseline_repro mean faithfulness: **0.199** (n=25 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 1 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced | False | 25 | 0.199 | 0.2359 | 0.0368 | [-0.0721, 0.145] | 0.39806 | 0.1306 (negligible) | 0.39806 | no |

**0/1 arm-vs-baseline contrasts significant (BH).**
det3x=False (balanced) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.