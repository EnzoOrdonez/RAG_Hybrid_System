# exp18_evidence_ceiling — anti-gaming guards (rows: small)

Regla de declinación = `classify_response` de `compute_faithfulness_metrics.py`, la MISMA que usa la métrica de fidelidad (28 patrones, case-insensitive).

**`decline_prefix` (antes `pure_decline`) mide un PREFIJO, no un rechazo.** La mayoría de esas filas sí afirman claims (memoria paramétrica tras el prefijo). Para cualquier argumento de usabilidad usar **`no afirma nada`**. Defecto #7, entrada 22.

| Arm | n | decline_prefix | hedged | answered | any refusal | **no afirma nada** | prefijo pero contesta | words | genuine claims | overlap 5gram |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_repro | 194 | 0.4588 (89) | 0.1546 (30) | 0.3866 (75) | 0.6134 | **0.0309** (6) | 84 | 332.6 | 10.582 | 0.1076 |
| oracle_evidence | 194 | 0.4381 (85) | 0.1392 (27) | 0.4227 (82) | 0.5773 | **0.0361** (7) | 80 | 339.8 | 10.737 | 0.1054 |
| evidence_swapped | 60 | 0.8833 (53) | 0.1 (6) | 0.0167 (1) | 0.9833 | **0.05** (3) | 50 | 159.7 | 4.833 | 0.0215 |
| final_top_k_10 | 194 | 0.2268 (44) | 0.1031 (20) | 0.6701 (130) | 0.3299 | **0.0052** (1) | 43 | 397.3 | 14.985 | 0.0935 |

Leer JUNTO a la fidelidad de arm_stats: una mejora real sube la fidelidad sin disparar el solape ni colapsar palabras/claims. Solape alto o declinación alta junto a una ganancia de fidelidad = gaming del instrumento, no mejora.

**Ni `hedged_partial` ni `decline_prefix` son abstención.** Ambas clases llevan una frase de rechazo y aun así afirman claims, y se puntúan normal. La afirmación previa de este mismo archivo —«un brazo que sube `pure_decline` sí está callándose»— era **falsa** y queda retirada (defecto #7): la mayoría de esas filas contestan de memoria paramétrica tras el prefijo. El brazo que de verdad se calla es el que sube **`no afirma nada`**, que es la columna medida por contenido.