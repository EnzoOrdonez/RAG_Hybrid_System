# exp18_evidence_ceiling — arm vs baseline_repro (NLI base, vb_agree τ0.7)

baseline_repro mean faithfulness: **0.1418** (n=191 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 3 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle_evidence | False | 190 | 0.1418 | 0.1775 | 0.0368 | [-0.0046, 0.0807] | 0.04335 | 0.121 (negligible) | 0.06503 | no |
| evidence_swapped | False | 58 | 0.1418 | 0.0641 | -0.1132 | [-0.2033, -0.0238] | 0.0033 | -0.3255 (small) | 0.0099 | YES |
| final_top_k_10 | True | 190 | 0.1418 | 0.125 | -0.0167 | [-0.0473, 0.0125] | 0.61292 | -0.078 (negligible) | 0.61292 | no |

**1/3 arm-vs-baseline contrasts significant (BH).**
det3x=False (oracle_evidence, evidence_swapped) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.