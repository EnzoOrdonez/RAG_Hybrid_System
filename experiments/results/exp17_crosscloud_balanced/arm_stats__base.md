# exp17_crosscloud_balanced — arm vs baseline_repro (NLI base, vb_agree τ0.7)

baseline_repro mean faithfulness: **0.1514** (n=25 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 1 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced | False | 25 | 0.1514 | 0.1959 | 0.0445 | [-0.038, 0.1244] | 0.13536 | 0.211 (small) | 0.13536 | no |

**0/1 arm-vs-baseline contrasts significant (BH).**
det3x=False (balanced) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.