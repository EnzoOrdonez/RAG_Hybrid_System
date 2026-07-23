# Tier 3 · Bloque A — anatomía del desacuerdo NLI small-vs-base

## 1. Matriz de confusión small(filas) × base(cols)

n=14469, acuerdo=0.673, κ=0.3232

| small\base | supported | contradicted | unsupported |
|---|---|---|---|
| supported | 1708 | 54 | 1968 |
| contradicted | 132 | 501 | 1063 |
| unsupported | 1118 | 401 | 7524 |

## 2. Tasa de desacuerdo por slice (pooled = 0.327)

| slice | tasa | n |
|---|---|---|
| pooled | 0.3273 | 14469 |
| model:gemma4-e4b | 0.3477 | 1044 |
| model:granite4.1-8b | 0.3102 | 5731 |
| model:mistral-7b-instruct | 0.3216 | 4415 |
| model:qwen3.5-9b | 0.3583 | 3279 |
| scen:denso | 0.3175 | 4888 |
| scen:hibrido | 0.3358 | 4991 |
| scen:lexico | 0.3285 | 4590 |
| len:<10 | 0.3751 | 3330 |
| len:10-20 | 0.3162 | 6683 |
| len:20-35 | 0.3052 | 4001 |
| len:>=35 | 0.3363 | 455 |

## 3. Fragilidad de umbral (best_ent∈[0.65,0.75] → flip supported al mover ent_t ±0.05)

| verif | near_ent | flips | % del total | near_contr |
|---|---|---|---|---|
| small | 613 | 440 | 3.04% | 403 |
| base | 218 | 172 | 1.19% | 173 |

## 4. Falso-contradicted (small=contradicted conf≥0.9 ∧ base=supported)

**128 claims** → `false_contradicted_candidates.csv`
por modelo: {'granite4.1-8b': 52, 'qwen3.5-9b': 43, 'gemma4-e4b': 6, 'mistral-7b-instruct': 27}
por escenario: {'lexico': 55, 'denso': 36, 'hibrido': 37}

## 5. Sensibilidad de agregación (gate v0 fijo, verificador small)

Cambio de etiqueta vs max: mean_top2=0.3227, noisy_or=0.0549

| agregador | sig RAG /12 | granite d_z | granite p_bh |
|---|---|---|---|
| max | 0 | -0.3094 | 0.084936 |
| mean_top2 | 1 | -0.2862 | 0.191724 |
| noisy_or | 1 | -0.4215 | 0.009353 |

## Gates

- A-G1: agregación mueve >15% etiquetas o el 0/12 → decisión Enzo del agregador
- A-G2: 128 falso-contradicted → tamaño del estrato contradicted en el gold (Bloque D)
