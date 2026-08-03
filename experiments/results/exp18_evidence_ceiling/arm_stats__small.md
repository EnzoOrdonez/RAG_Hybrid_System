# exp18_evidence_ceiling — arm vs baseline_repro (NLI small, vb_agree τ0.7)

baseline_repro mean faithfulness: **0.2415** (n=191 scored). Decline-aware: None pairs dropped, vacuous=1.0. BH family = 3 contrasts.

| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle_evidence | False | 190 | 0.2415 | 0.241 | 0.0008 | [-0.0494, 0.0525] | 0.89516 | 0.0022 (negligible) | 0.89516 | no |
| evidence_swapped | False | 58 | 0.2415 | 0.072 | -0.2221 | [-0.3306, -0.1086] | 0.00018 | -0.5105 (medium) | 0.00054 | YES |
| final_top_k_10 | True | 190 | 0.2415 | 0.2511 | 0.0123 | [-0.0331, 0.0602] | 0.8335 | 0.0381 (negligible) | 0.89516 | no |

**1/3 arm-vs-baseline contrasts significant (BH).**
det3x=False (oracle_evidence, evidence_swapped) => answers carry H5 cold/warm-cache noise; paired Δ still valid (same session/queries) but weigh softly.