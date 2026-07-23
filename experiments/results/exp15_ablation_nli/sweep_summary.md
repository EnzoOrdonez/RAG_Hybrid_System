# Tier 0 — mapa de robustez del instrumento NLI (exp15)

Grid: 64 puntos = 4 variantes x 4^2 umbrales; ancla (vb_agree, 0.7, 0.7) == v4 firmado.

## Significativos RAG-vs-RAG /12 — verificador small

| variante | ent 0.5 / contr 0.5 | ent 0.5 / contr 0.6 | ent 0.5 / contr 0.7 | ent 0.5 / contr 0.8 | ent 0.6 / contr 0.5 | ent 0.6 / contr 0.6 | ent 0.6 / contr 0.7 | ent 0.6 / contr 0.8 | ent 0.7 / contr 0.5 | ent 0.7 / contr 0.6 | ent 0.7 / contr 0.7 | ent 0.7 / contr 0.8 | ent 0.8 / contr 0.5 | ent 0.8 / contr 0.6 | ent 0.8 / contr 0.7 | ent 0.8 / contr 0.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vb_agree | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| va_margin_d0.1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| va_margin_d0.2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Significativos RAG-vs-RAG /12 — verificador base

| variante | ent 0.5 / contr 0.5 | ent 0.5 / contr 0.6 | ent 0.5 / contr 0.7 | ent 0.5 / contr 0.8 | ent 0.6 / contr 0.5 | ent 0.6 / contr 0.6 | ent 0.6 / contr 0.7 | ent 0.6 / contr 0.8 | ent 0.7 / contr 0.5 | ent 0.7 / contr 0.6 | ent 0.7 / contr 0.7 | ent 0.7 / contr 0.8 | ent 0.8 / contr 0.5 | ent 0.8 / contr 0.6 | ent 0.8 / contr 0.7 | ent 0.8 / contr 0.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vb_agree | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| va_margin_d0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| va_margin_d0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Kappa small-vs-base (nivel claim, 3 clases)

| variante | ent 0.5 / contr 0.5 | ent 0.5 / contr 0.6 | ent 0.5 / contr 0.7 | ent 0.5 / contr 0.8 | ent 0.6 / contr 0.5 | ent 0.6 / contr 0.6 | ent 0.6 / contr 0.7 | ent 0.6 / contr 0.8 | ent 0.7 / contr 0.5 | ent 0.7 / contr 0.6 | ent 0.7 / contr 0.7 | ent 0.7 / contr 0.8 | ent 0.8 / contr 0.5 | ent 0.8 / contr 0.6 | ent 0.8 / contr 0.7 | ent 0.8 / contr 0.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0.3391 | 0.3404 | 0.3419 | 0.3442 | 0.3432 | 0.3451 | 0.3469 | 0.3491 | 0.3446 | 0.3468 | 0.3489 | 0.3514 | 0.3489 | 0.351 | 0.3538 | 0.3565 |
| vb_agree | 0.3125 | 0.3149 | 0.3172 | 0.3198 | 0.3166 | 0.3195 | 0.3218 | 0.3245 | 0.3173 | 0.3206 | 0.3232 | 0.3259 | 0.3216 | 0.3249 | 0.3275 | 0.33 |
| va_margin_d0.1 | 0.3086 | 0.3119 | 0.3141 | 0.3154 | 0.3128 | 0.3166 | 0.319 | 0.3202 | 0.3143 | 0.3182 | 0.3208 | 0.3221 | 0.3194 | 0.3235 | 0.3259 | 0.327 |
| va_margin_d0.2 | 0.3034 | 0.3052 | 0.3071 | 0.3079 | 0.3074 | 0.3097 | 0.3119 | 0.3125 | 0.3089 | 0.3111 | 0.3134 | 0.3142 | 0.3143 | 0.3165 | 0.3186 | 0.3193 |
