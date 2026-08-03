# exp18 — diagnóstico (hhem)

## Divergencia de respuesta vs baseline_repro

| Brazo | n | jaccard 5-gram | jaccard tokens | idénticas | claims del baseline que reaparecen |
|---|---|---|---|---|---|
| oracle_evidence | 194 | 0.0806 | 0.3858 | 0.0 | 0.0132 |
| evidence_swapped | 60 | 0.0358 | 0.2512 | 0.0 | 0.0009 |
| final_top_k_10 | 194 | 0.0599 | 0.3535 | 0.0 | 0.0179 |

**Lectura de `evidence_swapped`:** solape ALTO con el baseline ⇒ el generador no sigue la evidencia (el techo es del generador, y el nulo de recuperación queda explicado). Solape BAJO ⇒ sí la sigue, y el techo hay que buscarlo en su capacidad de anclar evidencia buena.

## Equivalencia pre-registrada — `oracle_evidence` (TOST, banda ±0.081)

n=190 · Δ=0.0217 · IC90=[-0.0214, 0.0649] · p_TOST=0.01215 · **EQUIVALENTE**

EQUIVALENTE dentro de ±0.081: seleccionar con un oraculo de RELEVANCIA TOPICA no compra ni lo que compro balancear la cobertura. Esto descarta el margen alcanzable por ranking topico, NO el margen de seleccion en general -> leer junto a selection_bound.json (matriz, ledger entrada 20).

Banda anclada en el efecto HHEM de exp17 (+0,081): preexistente y ciega a este contraste, no un umbral ajustado hasta que algo pase.

## `final_top_k_10` partido por truncamiento observado

74 queries alcanzan el límite de 4096 tokens.

| Grupo | n | baseline | top-10 | Δ |
|---|---|---|---|---|
| untruncated | 117 | 0.5256 | 0.5514 | 0.0259 |
| truncated | 73 | 0.3692 | 0.3697 | 0.0005 |

Only `untruncated` is a clean read on whether more evidence helps. The truncated group mixes more evidence with cut-off evidence and must not be averaged into a single number.