# exp17_crosscloud_balanced — anti-gaming guards (rows: small)

Regla de declinación = `classify_response` de `compute_faithfulness_metrics.py`, la MISMA que usa la métrica de fidelidad (28 patrones, case-insensitive).

| Arm | n | pure_decline | hedged | answered | any refusal | words | genuine claims | overlap 5gram |
|---|---|---|---|---|---|---|---|---|
| baseline | 25 | 0.56 (14) | 0.24 (6) | 0.2 (5) | 0.8 | 349.4 | 11.48 | 0.1094 |
| balanced | 25 | 0.36 (9) | 0.16 (4) | 0.48 (12) | 0.52 | 427.4 | 14.88 | 0.129 |

Leer JUNTO a la fidelidad de arm_stats: una mejora real sube la fidelidad sin disparar el solape ni colapsar palabras/claims. Solape alto o declinación alta junto a una ganancia de fidelidad = gaming del instrumento, no mejora.

`hedged_partial` no es abstención: esas respuestas llevan una frase de rechazo y aun así afirman claims, y se puntúan normal. Un brazo que sube `pure_decline` sí está callándose; uno que sube solo `hedged` está hedgeando mientras responde.