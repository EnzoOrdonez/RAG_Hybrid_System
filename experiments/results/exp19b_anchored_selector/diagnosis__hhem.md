# exp18 — diagnóstico (hhem)

## Divergencia de respuesta vs baseline_repro

| Brazo | n | jaccard 5-gram | jaccard tokens | idénticas | claims del baseline que reaparecen |
|---|---|---|---|---|---|
| claim_selected | 194 | 0.1869 | 0.4792 | 0.0722 | 0.0763 |

**Lectura de `evidence_swapped`:** solape ALTO con el baseline ⇒ el generador no sigue la evidencia (el techo es del generador, y el nulo de recuperación queda explicado). Solape BAJO ⇒ sí la sigue, y el techo hay que buscarlo en su capacidad de anclar evidencia buena.
