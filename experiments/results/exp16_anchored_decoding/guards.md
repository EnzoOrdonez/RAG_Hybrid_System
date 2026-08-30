# exp16_anchored_decoding — anti-gaming guards (rows: small)

Regla de declinación = `classify_response` de `compute_faithfulness_metrics.py`, la MISMA que usa la métrica de fidelidad (28 patrones, case-insensitive).

**`decline_prefix` (antes `pure_decline`) mide un PREFIJO, no un rechazo.** La mayoría de esas filas sí afirman claims (memoria paramétrica tras el prefijo). Para cualquier argumento de usabilidad usar **`no afirma nada`**. Defecto #7, entrada 22.

| Arm | n | decline_prefix | hedged | answered | any refusal | **no afirma nada** | prefijo pero contesta | words | genuine claims | overlap 5gram |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_repro | 60 | 0.4667 (28) | 0.1333 (8) | 0.4 (24) | 0.6 | **0.0667** (4) | 24 | 351.7 | 11.95 | 0.1218 |
| anchored_cite | 60 | 0.65 (39) | 0.0667 (4) | 0.2833 (17) | 0.7167 | **0.1667** (10) | 29 | 222.9 | 7.55 | 0.0613 |
| strict_abstain | 60 | 0.6833 (41) | 0.0833 (5) | 0.2333 (14) | 0.7667 | **0.15** (9) | 32 | 190.8 | 5.267 | 0.233 |

Leer JUNTO a la fidelidad de arm_stats: una mejora real sube la fidelidad sin disparar el solape ni colapsar palabras/claims. Solape alto o declinación alta junto a una ganancia de fidelidad = gaming del instrumento, no mejora.

**Ni `hedged_partial` ni `decline_prefix` son abstención.** Ambas clases llevan una frase de rechazo y aun así afirman claims, y se puntúan normal. La afirmación previa de este mismo archivo —«un brazo que sube `pure_decline` sí está callándose»— era **falsa** y queda retirada (defecto #7): la mayoría de esas filas contestan de memoria paramétrica tras el prefijo. El brazo que de verdad se calla es el que sube **`no afirma nada`**, que es la columna medida por contenido.