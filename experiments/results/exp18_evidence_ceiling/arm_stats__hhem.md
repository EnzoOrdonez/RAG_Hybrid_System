# exp18_evidence_ceiling — arm vs baseline_repro (HHEM grounding, max_chunk τ0.5)

baseline_repro mean faithfulness: **0.4638** (n=191 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 3 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle_evidence | False | 190 | 0.4638 | 0.4829 | 0.0217 | [-0.0292, 0.072] | 0.19175 | 0.0604 (negligible) | 0.28763 | no |
| evidence_swapped | False | 58 | 0.4638 | 0.1073 | -0.3185 | [-0.4099, -0.2214] | 0.0 | -0.8568 (large) | 0.0 | YES |
| final_top_k_10 | True | 190 | 0.4638 | 0.4835 | 0.0161 | [-0.0296, 0.0627] | 0.47652 | 0.0502 (negligible) | 0.47652 | no |

**1/3 arm-vs-baseline contrasts significant (BH).**
det3x=False (oracle_evidence, evidence_swapped) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.