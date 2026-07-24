# Tier 3 · control negativo — validez de constructo

400 pares aleatorios (claim × 5 chunks NO relacionados). Un verificador NO debería etiquetar texto no relacionado como contradicted (NLI) / grounded (HHEM).

## NLI: tasa falso-contradicted (menor=mejor)

| verificador | v0 | vb_agree |
|---|---|---|
| small | 0.54 | 0.2375 |
| base | 0.5975 | 0.215 |

## HHEM: tasa falso-grounded por τ (menor=mejor)

| τ | tau_0.5 | tau_0.8 | tau_0.9 | tau_0.95 | tau_0.99 |
|---|---|---|---|---|---|
| rate | 0.0325 | 0.005 | 0.0 | 0.0 | 0.0 |

HHEM score en pares aleatorios: mean 0.0545, median 0.028