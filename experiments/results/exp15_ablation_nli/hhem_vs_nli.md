# Tier 3 — HHEM (grounding) vs NLI: nivel + robustez del nulo 0/12

HHEM cargado correctamente (fix de load); τ=0.5. Especificidad negativa: falso-grounded 0.033.

## 1. Nivel de fidelidad por config (HHEM vs NLI small v4)

| config | HHEM | NLI small | gap |
|---|---|---|---|
| lexico | granite4.1-8b | 0.4006 | 0.2347 | +0.166 |
| denso | granite4.1-8b | 0.4299 | 0.2465 | +0.183 |
| hibrido | granite4.1-8b | 0.4428 | 0.2988 | +0.144 |
| lexico | gemma4-e4b | 0.8037 | 0.4132 | +0.391 |
| denso | gemma4-e4b | 0.7933 | 0.3774 | +0.416 |
| hibrido | gemma4-e4b | 0.737 | 0.3173 | +0.420 |
| lexico | mistral-7b-instruct | 0.491 | 0.2462 | +0.245 |
| denso | mistral-7b-instruct | 0.5658 | 0.2806 | +0.285 |
| hibrido | mistral-7b-instruct | 0.5782 | 0.288 | +0.290 |
| lexico | qwen3.5-9b | 0.6371 | 0.255 | +0.382 |
| denso | qwen3.5-9b | 0.6919 | 0.2646 | +0.427 |
| hibrido | qwen3.5-9b | 0.6256 | 0.2936 | +0.332 |

**Gap HHEM−NLI: mean +0.307 (rango +0.144..+0.4273).** NLI sub-acredita la fidelidad de forma sistemática.

## 2. Contraste entre escenarios bajo HHEM (RAG-vs-RAG, familia BH v4-consistente 24)

| par | n | d_z | p_bh | sig |
|---|---|---|---|---|
| denso | gemma4-e4b vs hibrido | gemma4-e4b | 30 | -0.05 | 0.6399 | no |
| denso | gemma4-e4b vs lexico | gemma4-e4b | 30 | +0.33 | 0.2159 | no |
| hibrido | gemma4-e4b vs lexico | gemma4-e4b | 30 | +0.12 | 0.6198 | no |
| denso | granite4.1-8b vs hibrido | granite4. | 63 | -0.01 | 0.5809 | no |
| denso | granite4.1-8b vs lexico | granite4.1 | 51 | -0.17 | 0.3348 | no |
| hibrido | granite4.1-8b vs lexico | granite4 | 53 | -0.35 | 0.0204 | SÍ |
| denso | mistral-7b-instruct vs hibrido | mis | 109 | +0.05 | 0.5809 | no |
| denso | mistral-7b-instruct vs lexico | mist | 99 | -0.12 | 0.3568 | no |
| hibrido | mistral-7b-instruct vs lexico | mi | 93 | -0.22 | 0.0673 | no |
| denso | qwen3.5-9b vs hibrido | qwen3.5-9b | 29 | -0.13 | 0.7358 | no |
| denso | qwen3.5-9b vs lexico | qwen3.5-9b | 18 | -0.44 | 0.2 | no |
| hibrido | qwen3.5-9b vs lexico | qwen3.5-9b | 18 | -0.0 | 0.9794 | no |

**HHEM: 1/12 RAG-vs-RAG significativos** (NLI small/base dan 0/12). Significativo: hibrido | granite4.1-8b vs lexico | granite4.1-8b.

## Veredicto

HHEM (v4-consistent BH family): 1/12 RAG-vs-RAG significativos (hibrido | granite4.1-8b vs lexico | granite4.1-8b). NLI small/base dan 0/12. Bajo el instrumento de grounding limpio, granite hibrido-vs-lexico CRUZA significancia (p_bh 0.020, d_z -0.35) donde el NLI ruidoso no (p_bh 0.085): el efecto retrieval->fidelidad existe para el modelo determinista pero solo es detectable con un instrumento menos ruidoso. HHEM tambien sub-acredita el NIVEL (gap medio +0.307 sobre NLI). Efecto pequeno, tau-dependiente, solo granite (1/12): matizar, pendiente gold humano.

El par granite hibrido-vs-lexico es direccionalmente consistente (hib>lex) en los tres instrumentos; bajo el NLI ruidoso NO cruza BH (p_bh 0.085) pero bajo HHEM (grounding limpio, familia v4-consistente) SÍ (p_bh 0.020). Es 1/12, d_z pequeño (-0.35), tau-dependiente. Pendiente: gold humano para validar HHEM; deberta-large (8/12) 3.er voto.
