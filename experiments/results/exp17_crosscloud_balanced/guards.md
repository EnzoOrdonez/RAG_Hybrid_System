# exp17_crosscloud_balanced — anti-gaming guards (rows: small)

Regla de declinación = `classify_response` de `compute_faithfulness_metrics.py`, la MISMA que usa la métrica de fidelidad (28 patrones, case-insensitive).

**`decline_prefix` (antes `pure_decline`) mide un PREFIJO, no un rechazo.** La mayoría de esas filas sí afirman claims (memoria paramétrica tras el prefijo). Para cualquier argumento de usabilidad usar **`no afirma nada`**. Defecto #7, entrada 22.

| Arm | n | decline_prefix | hedged | answered | any refusal | **no afirma nada** | prefijo pero contesta | words | genuine claims | overlap 5gram |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 25 | 0.56 (14) | 0.24 (6) | 0.2 (5) | 0.8 | **0.0** (0) | 14 | 349.4 | 11.48 | 0.1094 |
| balanced | 25 | 0.36 (9) | 0.16 (4) | 0.48 (12) | 0.52 | **0.0** (0) | 9 | 427.4 | 14.88 | 0.129 |

Leer JUNTO a la fidelidad de arm_stats: una mejora real sube la fidelidad sin disparar el solape ni colapsar palabras/claims. Solape alto o declinación alta junto a una ganancia de fidelidad = gaming del instrumento, no mejora.

**Ni `hedged_partial` ni `decline_prefix` son abstención.** Ambas clases llevan una frase de rechazo y aun así afirman claims, y se puntúan normal. La afirmación previa de este mismo archivo —«un brazo que sube `pure_decline` sí está callándose»— era **falsa** y queda retirada (defecto #7): la mayoría de esas filas contestan de memoria paramétrica tras el prefijo. El brazo que de verdad se calla es el que sube **`no afirma nada`**, que es la columna medida por contenido.