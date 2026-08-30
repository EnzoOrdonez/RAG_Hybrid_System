# exp19b_anchored_selector — anti-gaming guards (rows: small)

Regla de declinación = `classify_response` de `compute_faithfulness_metrics.py`, la MISMA que usa la métrica de fidelidad (28 patrones, case-insensitive).

**`decline_prefix` (antes `pure_decline`) mide un PREFIJO, no un rechazo.** La mayoría de esas filas sí afirman claims (memoria paramétrica tras el prefijo). Para cualquier argumento de usabilidad usar **`no afirma nada`**. Defecto #7, entrada 22.

| Arm | n | decline_prefix | hedged | answered | any refusal | **no afirma nada** | prefijo pero contesta | words | genuine claims | overlap 5gram |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_repro | 194 | 0.4536 (88) | 0.1649 (32) | 0.3814 (74) | 0.6186 | **0.0361** (7) | 82 | 329.7 | 10.515 | 0.1058 |
| claim_selected | 194 | 0.3969 (77) | 0.1186 (23) | 0.4845 (94) | 0.5155 | **0.0619** (12) | 66 | 357.4 | 11.603 | 0.1279 |

Leer JUNTO a la fidelidad de arm_stats: una mejora real sube la fidelidad sin disparar el solape ni colapsar palabras/claims. Solape alto o declinación alta junto a una ganancia de fidelidad = gaming del instrumento, no mejora.

**Ni `hedged_partial` ni `decline_prefix` son abstención.** Ambas clases llevan una frase de rechazo y aun así afirman claims, y se puntúan normal. La afirmación previa de este mismo archivo —«un brazo que sube `pure_decline` sí está callándose»— era **falsa** y queda retirada (defecto #7): la mayoría de esas filas contestan de memoria paramétrica tras el prefijo. El brazo que de verdad se calla es el que sube **`no afirma nada`**, que es la columna medida por contenido.