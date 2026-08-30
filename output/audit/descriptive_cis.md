# Intervalos y pruebas descriptivas

Todos los resultados de este documento son **descriptivos y están fuera de las familias BH**. Se agregan artefactos ya persistidos; no se reejecutó generación ni verificación.

## Sensibilidad al conjunto de evidencia (etapas A → B)

- Flips: 18/50 (36.0%); IC95 Wilson [24.1%, 49.9%].

| original \ segunda | correcto | incorrecto | dudoso |
|---|---:|---:|---:|
| correcto | 27 | 1 | 2 |
| incorrecto | 9 | 3 | 3 |
| dudoso | 3 | 0 | 2 |

Al reducir `correcto` frente a las demás categorías, la tabla pareada es:

| | B correcto | B otro |
|---|---:|---:|
| A correcto | 27 | 3 |
| A otro | 12 | 8 |

McNemar exacto correcto/resto (discordantes 12 vs 3): p=0.035156.
Como resumen direccional de **todos** los flips, 12 fueron hacia `correcto` y 6 tuvieron otra dirección; binomial exacta p=0.237885. Esta última no es McNemar ni una prueba confirmatoria.

## Confiabilidad intra-anotador

- Acuerdo: 11/20 (55.0%); IC95 Wilson [34.2%, 74.2%].
- κ de Cohen: 0.2683.

| original \ segunda | correcto | incorrecto | dudoso |
|---|---:|---:|---:|
| correcto | 7 | 3 | 1 |
| incorrecto | 1 | 2 | 1 |
| dudoso | 2 | 1 | 2 |

La matriz se reconstruyó con la tanda C y el archivo derivado de discordancias; no se abrió el archivo completo de juicios humanos.

## Taxonomía de claims `unsupported@0.5`

| veredicto | conteo no ponderado | tasa HT |
|---|---:|---:|
| correcto | 22 | 53.0% |
| incorrecto | 17 | 45.4% |
| dudoso | 1 | 1.6% |

n=40; población=759; n efectivo de Kish=27.409. Con n efectivo de Kish reducido, una tasa HT pequeña puede depender de muy pocos casos observados; se reportan también los conteos no ponderados.

## Sensibilidad descriptiva a τ

Se leyeron 2053 claims de 188 queries con la regla inclusiva `best_over_pool <= tau`.

| τ | claims `unsupported@τ` |
|---:|---:|
| 0.4 | 636 |
| 0.5 | 759 |
| 0.6 | 880 |

Estos conteos son una sensibilidad de umbral, no una nueva familia inferencial.
