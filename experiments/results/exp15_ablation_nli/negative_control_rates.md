# Tier 3 · control negativo — validez de constructo

400 pares aleatorios (claim × 5 chunks NO relacionados). Un verificador NO debería etiquetar texto no relacionado como contradicted (NLI) / grounded (HHEM).

## NLI: tasa falso-contradicted (↓ mejor)

| verificador | v0 | vb_agree |
|---|---|---|
| small | 0.54 | 0.2375 |
| base | 0.5975 | 0.215 |

## HHEM: tasa falso-grounded por τ (↓ mejor)

| τ | tau_0.5 | tau_0.8 | tau_0.9 | tau_0.95 | tau_0.99 |
|---|---|---|---|---|---|
| rate | 0.01 | 0.0075 | 0.0075 | 0.005 | 0.0025 |

HHEM score en pares aleatorios: mean 0.0021, median 0.0