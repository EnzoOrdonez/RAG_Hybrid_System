# exp16_anchored_decoding — anti-gaming guards (rows: small)

| Arm | n | decline | words | genuine claims | verbatim 5gram overlap |
|---|---|---|---|---|---|
| baseline_repro | 60 | 0.5167 (31) | 351.7 | 11.95 | 0.1218 |
| anchored_cite | 60 | 0.5833 (35) | 222.9 | 7.55 | 0.0613 |
| strict_abstain | 60 | 0.6 (36) | 190.8 | 5.267 | 0.233 |

Read WITH arm_stats faithfulness: a real gain raises faithfulness while decline/
overlap stay near baseline and words/claims don't collapse. High overlap or high
decline alongside a faithfulness gain = instrument gaming, not improvement.