# Calibración humana de la taxonomía exp18

Vocabulario controlado: `correcto`, `incorrecto`, `dudoso`.

## Cobertura y ponderación

- Filas: 40
- Juicios completos: 40
- Juicios vacíos: 0 (0.0%)
- Población representada: 759
- n efectivo de Kish: 27.409
- Masa HT sin juicio: 0.000

Los totales usan peso Horvitz–Thompson `1 / inclusion_prob`. No se imputan ni renormalizan juicios vacíos.

## Valores encontrados

- `correcto`: 22
- `incorrecto`: 17
- `dudoso`: 1

## Tasas HT por estrato

| Estrato | N | correcto | incorrecto | dudoso |
|---|---:|---:|---:|---:|
| a_synthesis_cand | 58 | 70.000% | 30.000% | 0.000% |
| b_parametric_cand | 178 | 70.000% | 30.000% | 0.000% |
| c_unattributed_cand | 400 | 50.000% | 50.000% | 0.000% |
| d_threshold_artifact | 123 | 30.000% | 60.000% | 10.000% |

## Extrapolación HT a los 759

| Veredicto | Total estimado | Tasa poblacional |
|---|---:|---:|
| correcto | 402.100 | 52.978% |
| incorrecto | 344.600 | 45.402% |
| dudoso | 12.300 | 1.621% |

## Test-retest

No existe la columna `human_verdict_retest`; no se calcula acuerdo.
