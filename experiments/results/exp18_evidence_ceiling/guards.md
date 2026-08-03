# exp18_evidence_ceiling — anti-gaming guards (rows: small)

Regla de declinación = `classify_response` de `compute_faithfulness_metrics.py`, la MISMA que usa la métrica de fidelidad (28 patrones, case-insensitive).

| Arm | n | pure_decline | hedged | answered | any refusal | words | genuine claims | overlap 5gram |
|---|---|---|---|---|---|---|---|---|
| baseline_repro | 194 | 0.4588 (89) | 0.1546 (30) | 0.3866 (75) | 0.6134 | 332.6 | 10.582 | 0.1076 |
| oracle_evidence | 194 | 0.4381 (85) | 0.1392 (27) | 0.4227 (82) | 0.5773 | 339.8 | 10.737 | 0.1054 |
| evidence_swapped | 60 | 0.8833 (53) | 0.1 (6) | 0.0167 (1) | 0.9833 | 159.7 | 4.833 | 0.0215 |
| final_top_k_10 | 194 | 0.2268 (44) | 0.1031 (20) | 0.6701 (130) | 0.3299 | 397.3 | 14.985 | 0.0935 |

Leer JUNTO a la fidelidad de arm_stats: una mejora real sube la fidelidad sin disparar el solape ni colapsar palabras/claims. Solape alto o declinación alta junto a una ganancia de fidelidad = gaming del instrumento, no mejora.

`hedged_partial` no es abstención: esas respuestas llevan una frase de rechazo y aun así afirman claims, y se puntúan normal. Un brazo que sube `pure_decline` sí está callándose; uno que sube solo `hedged` está hedgeando mientras responde.