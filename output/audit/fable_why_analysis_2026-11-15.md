# Por qué el híbrido no siempre gana y por qué unos modelos puntúan más

Generado por `scripts/audit/fable_why_analysis.py` desde la evidencia firmada (solo lectura) y los índices en disco. No se ejecutó ningún modelo. Cada línea `[OK ]` es una comprobación que el script hace y que falla con exit 1 si deja de cumplirse.

## 1. Integridad de los índices (sin embeber ninguna consulta)

- [OK ] FAISS, BM25 y mapa tienen los mismos identificadores: 24481 / 24481 / 24481
- [OK ] sin identificadores repetidos
- [OK ] FAISS: un vector por identificador: ntotal=24481, dimensión=1024
- [OK ] FAISS usa producto interno (coseno sobre vectores normalizados)
- [OK ] vectores normalizados (muestra de 512, semilla 42): norma mín 1.00000, máx 1.00000
- [OK ] cada vector de la muestra se recupera a sí mismo en la posición 1: 512/512
- [OK ] solo aws, azure y gcp en el corpus: aws=6366, azure=9606, gcp=8509
- [OK ] BM25: ningún documento sin tokens: 0 vacíos

## 2. Recuperación: qué hace la fusión RRF (exp11, 194 consultas)

### 2.1 Propiedad demostrada: con k=60 y 50 candidatos por pierna, RRF es un filtro de consenso

- Peor documento presente en las dos listas (puesto 50 en ambas): 2/(60+50) = 0.018182.
- Mejor documento presente en una sola lista (puesto 1): 1/(60+1) = 0.016393.
- [OK ] 2/(k+N) > 1/(k+1) con k=60, N=50: todo documento de consenso queda por encima de todo documento de una sola pierna
- La propiedad vale mientras N < k + 2 = 62; deja de valer en N = 62.
- Consecuencia: el top-5 fusionado se llena primero con la intersección de las dos listas de 50. El primer resultado denso entra solo si BM25 también lo tiene entre sus 50, o si la intersección (tras el filtro de proveedor) tiene menos de 5 documentos.

### 2.2 Prueba de extremo a extremo de la pierna léxica con el índice real

- [OK ] el top-5 léxico recalculado hoy coincide con el firmado en exp11: 194/194 consultas idénticas (ids y orden)
- Consultas con filtro de proveedor activo: 79/194. En ellas el top-5 denso firmado es el top-5 después del filtro, así que su puesto en la pierna cruda no se conoce; las comprobaciones de puntaje exacto usan solo las consultas sin filtro.

### 2.3 Las listas firmadas obedecen la aritmética de RRF

- [OK ] ningún documento solo-denso precede a un documento de consenso conocido: 0 violaciones en 194 consultas
- [OK ] los documentos con puntaje RRF exacto conocido aparecen en orden no creciente: 0 violaciones sobre 210 documentos de 115 consultas sin filtro

### 2.4 Qué hace la fusión con las listas firmadas

Solapamiento entre el top-5 léxico y el top-5 denso (cuántos fragmentos comparten):

| Fragmentos en común | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Consultas | 104 | 47 | 32 | 9 | 2 | 0 |

- Media: 0.75 de 5. Las dos piernas casi nunca proponen los mismos fragmentos arriba, así que la intersección que RRF premia vive sobre todo en puestos profundos.
- Top-5 fusionado (pre-rerank): 37.4% de sus fragmentos estaban en el top-5 denso y 42.0% en el top-5 léxico.
- Retención del top-5 denso dentro del top-5 fusionado: 334/417 (80.1%) si BM25 también lo tenía entre sus 50; 29/553 (5.2%) si no.
- Primer resultado denso: se conserva en el top-5 fusionado en 112/194 consultas y se pierde en 82/194. De las pérdidas, 79 son fragmentos que BM25 no tenía entre sus 50 (los expulsa el filtro de consenso) y 3 sí los tenía pero fueron superados por otros de consenso.

Fragmentos solo-densos (fuera del top-50 de BM25) dentro del top-5 fusionado:

| Solo-densos en el top-5 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Consultas | 174 | 11 | 9 | 0 | 0 | 0 |

- Tras el reranker: 35.3% del top-5 final ya estaba en el top-5 fusionado; el 64.7% restante sube desde los puestos 6 a 50 del pool. Esa es la parte de la ventaja del híbrido que depende de la profundidad del pool y del reranker, no de la fusión.
- Top-5 final: 31.6% en el top-5 denso, 25.5% en el top-5 léxico, 31.1% fuera del top-50 de BM25.

## 3. Generación: por qué unos modelos puntúan más (exp12, 4 modelos x 3 escenarios RAG)

### 3.1 Hechos de diseño comprobados sobre los datos firmados

- [OK ] los cuatro modelos recibieron los mismos fragmentos, en el mismo orden, en cada consulta: 582/582 celdas (escenario, consulta)
- Por tanto, ninguna diferencia entre modelos dentro de un escenario puede venir de la recuperación. Solo quedan tres fuentes: qué consultas decide contestar cada modelo, cuántas afirmaciones emite y cuántas de ellas apoya el verificador.
- [OK ] la fidelidad firmada de cada fila es apoyadas / genuinas redondeada a 4 decimales: desvío máximo 5.0e-05
- [OK ] las 24 celdas recalculadas coinciden con la tabla citable v4 (n exacto, media a 1e-4): 24/24 celdas, dos verificadores
- Pares (modelo, consulta) en los que dos escenarios entregaron exactamente el mismo contexto: 0. Sin pares así, la evidencia firmada no permite separar el ruido propio de la generación del efecto del contexto dentro de exp12.

### 3.2 (small) Descomposición exacta de cada diferencia entre modelos

`total = dentro + selección`, y `dentro = apoyadas + emitidas`. *Selección*: parte de la brecha que existe porque cada modelo contesta consultas distintas. *Apoyadas*: brecha por número de afirmaciones que el verificador apoya. *Emitidas*: brecha por número de afirmaciones emitidas (el denominador). Las dos últimas se miden solo sobre las consultas que ambos contestaron.

| Escenario | A − B | Total | Selección | Dentro | · apoyadas | · emitidas | n A/B/común | Apoyadas A vs B | Emitidas A vs B | d_z | p_BH | |d_z| mín. detectable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lexico | gemma4-e4b − granite4.1-8b | +0.178 | +0.108 | +0.071 | -0.626 | +0.696 | 57/75/42 | 1.45 vs 4.05 | 3.88 vs 14.00 | +0.24 | 0.582 | 0.43 |
| lexico | gemma4-e4b − mistral-7b-instruct | +0.167 | +0.066 | +0.101 | -0.424 | +0.525 | 57/114/46 | 1.54 vs 3.35 | 3.65 vs 9.30 | +0.36 | 0.214 | 0.41 |
| lexico | gemma4-e4b − qwen3.5-9b | +0.158 | +0.043 | +0.115 | -0.473 | +0.588 | 57/36/21 | 1.52 vs 3.57 | 3.67 vs 10.10 | +0.43 | 0.333 | 0.61 |
| lexico | granite4.1-8b − mistral-7b-instruct | -0.011 | +0.037 | -0.049 | +0.093 | -0.142 | 75/114/62 | 3.66 vs 2.87 | 15.26 vs 9.84 | -0.23 | 0.333 | 0.36 |
| lexico | granite4.1-8b − qwen3.5-9b | -0.020 | -0.031 | +0.011 | +0.099 | -0.088 | 75/36/28 | 3.96 vs 2.86 | 15.07 vs 10.61 | +0.04 | 0.905 | 0.53 |
| lexico | mistral-7b-instruct − qwen3.5-9b | -0.009 | -0.064 | +0.055 | -0.080 | +0.134 | 114/36/32 | 2.66 vs 2.62 | 9.91 vs 10.03 | +0.17 | 0.582 | 0.50 |
| denso | gemma4-e4b − granite4.1-8b | +0.131 | +0.057 | +0.074 | -0.744 | +0.818 | 60/85/42 | 1.38 vs 4.60 | 4.12 vs 17.62 | +0.21 | 0.582 | 0.43 |
| denso | gemma4-e4b − mistral-7b-instruct | +0.097 | +0.071 | +0.025 | -0.544 | +0.570 | 60/132/55 | 1.29 vs 3.09 | 3.93 vs 10.02 | +0.06 | 0.950 | 0.38 |
| denso | gemma4-e4b − qwen3.5-9b | +0.113 | +0.160 | -0.048 | -0.455 | +0.407 | 60/45/26 | 1.04 vs 2.73 | 3.88 vs 10.19 | -0.18 | 0.608 | 0.55 |
| denso | granite4.1-8b − mistral-7b-instruct | -0.034 | +0.053 | -0.087 | +0.137 | -0.224 | 85/132/75 | 3.88 vs 3.55 | 16.56 vs 10.07 | -0.42 | 0.013 | 0.32 |
| denso | granite4.1-8b − qwen3.5-9b | -0.018 | +0.010 | -0.028 | +0.110 | -0.138 | 85/45/33 | 4.33 vs 3.76 | 19.45 vs 11.88 | -0.15 | 0.681 | 0.49 |
| denso | mistral-7b-instruct − qwen3.5-9b | +0.016 | -0.001 | +0.017 | -0.023 | +0.040 | 132/45/40 | 3.23 vs 3.45 | 11.03 vs 11.07 | +0.07 | 0.950 | 0.44 |
| hibrido | gemma4-e4b − granite4.1-8b | +0.018 | -0.010 | +0.028 | -0.851 | +0.880 | 56/87/43 | 1.42 vs 4.40 | 4.16 vs 15.79 | +0.08 | 0.681 | 0.43 |
| hibrido | gemma4-e4b − mistral-7b-instruct | +0.029 | +0.036 | -0.007 | -0.390 | +0.383 | 56/122/47 | 1.38 vs 2.66 | 3.87 vs 9.30 | -0.02 | 0.950 | 0.41 |
| hibrido | gemma4-e4b − qwen3.5-9b | +0.024 | +0.088 | -0.064 | -0.687 | +0.624 | 56/43/30 | 1.20 vs 3.50 | 3.97 vs 11.43 | -0.16 | 0.681 | 0.51 |
| hibrido | granite4.1-8b − mistral-7b-instruct | +0.011 | +0.054 | -0.043 | +0.082 | -0.125 | 87/122/72 | 4.29 vs 3.31 | 15.35 vs 10.42 | -0.18 | 0.582 | 0.33 |
| hibrido | granite4.1-8b − qwen3.5-9b | +0.005 | +0.011 | -0.006 | +0.240 | -0.246 | 87/43/35 | 4.46 vs 2.83 | 16.29 vs 10.80 | -0.02 | 0.905 | 0.47 |
| hibrido | mistral-7b-instruct − qwen3.5-9b | -0.006 | -0.028 | +0.022 | +0.053 | -0.031 | 122/43/38 | 3.55 vs 2.74 | 11.45 vs 11.08 | +0.08 | 0.814 | 0.45 |

- [OK ] (small) las 18 descomposiciones cierran exactamente
- Peso medio de la selección en la brecha: 50.2% (mediana 54.7%).
- Pares significativos tras BH (familia de 18): 1/18. Pares cuyo |d_z| observado queda por debajo del mínimo detectable con su n: 17/18.

### 3.2 (base) Descomposición exacta de cada diferencia entre modelos

`total = dentro + selección`, y `dentro = apoyadas + emitidas`. *Selección*: parte de la brecha que existe porque cada modelo contesta consultas distintas. *Apoyadas*: brecha por número de afirmaciones que el verificador apoya. *Emitidas*: brecha por número de afirmaciones emitidas (el denominador). Las dos últimas se miden solo sobre las consultas que ambos contestaron.

| Escenario | A − B | Total | Selección | Dentro | · apoyadas | · emitidas | n A/B/común | Apoyadas A vs B | Emitidas A vs B | d_z | p_BH | |d_z| mín. detectable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lexico | gemma4-e4b − granite4.1-8b | +0.219 | +0.041 | +0.178 | -0.251 | +0.429 | 57/75/42 | 1.67 vs 2.79 | 3.88 vs 14.00 | +0.53 | 0.038 | 0.43 |
| lexico | gemma4-e4b − mistral-7b-instruct | +0.232 | +0.061 | +0.171 | -0.201 | +0.372 | 57/114/46 | 1.41 vs 2.11 | 3.65 vs 9.30 | +0.46 | 0.062 | 0.41 |
| lexico | gemma4-e4b − qwen3.5-9b | +0.159 | +0.073 | +0.087 | -0.414 | +0.500 | 57/36/21 | 1.29 vs 3.19 | 3.67 vs 10.10 | +0.36 | 0.280 | 0.61 |
| lexico | granite4.1-8b − mistral-7b-instruct | +0.013 | +0.052 | -0.039 | +0.056 | -0.095 | 75/114/62 | 2.77 vs 2.18 | 15.26 vs 9.84 | -0.20 | 0.306 | 0.36 |
| lexico | granite4.1-8b − qwen3.5-9b | -0.060 | -0.005 | -0.055 | +0.079 | -0.134 | 75/36/28 | 3.14 vs 2.64 | 15.07 vs 10.61 | -0.21 | 0.306 | 0.53 |
| lexico | mistral-7b-instruct − qwen3.5-9b | -0.073 | -0.023 | -0.049 | -0.133 | +0.084 | 114/36/32 | 2.16 vs 2.50 | 9.91 vs 10.03 | -0.27 | 0.306 | 0.50 |
| denso | gemma4-e4b − granite4.1-8b | +0.059 | +0.001 | +0.058 | -0.495 | +0.553 | 60/85/42 | 1.21 vs 3.55 | 4.12 vs 17.62 | +0.16 | 0.437 | 0.43 |
| denso | gemma4-e4b − mistral-7b-instruct | +0.065 | +0.058 | +0.007 | -0.360 | +0.367 | 60/132/55 | 1.05 vs 2.11 | 3.93 vs 10.02 | +0.02 | 0.969 | 0.38 |
| denso | gemma4-e4b − qwen3.5-9b | -0.027 | +0.132 | -0.160 | -0.677 | +0.517 | 60/45/26 | 0.85 vs 3.23 | 3.88 vs 10.19 | -0.40 | 0.084 | 0.55 |
| denso | granite4.1-8b − mistral-7b-instruct | +0.006 | +0.064 | -0.058 | +0.234 | -0.292 | 85/132/75 | 3.21 vs 2.32 | 16.56 vs 10.07 | -0.26 | 0.084 | 0.32 |
| denso | granite4.1-8b − qwen3.5-9b | -0.087 | +0.002 | -0.089 | +0.038 | -0.127 | 85/45/33 | 3.97 vs 3.61 | 19.45 vs 11.88 | -0.42 | 0.084 | 0.49 |
| denso | mistral-7b-instruct − qwen3.5-9b | -0.092 | -0.018 | -0.074 | -0.152 | +0.078 | 132/45/40 | 2.35 vs 3.30 | 11.03 vs 11.07 | -0.31 | 0.279 | 0.44 |
| hibrido | gemma4-e4b − granite4.1-8b | +0.070 | +0.071 | -0.001 | -0.666 | +0.665 | 56/87/43 | 1.12 vs 3.28 | 4.16 vs 15.79 | -0.00 | 0.969 | 0.43 |
| hibrido | gemma4-e4b − mistral-7b-instruct | +0.106 | +0.064 | +0.042 | -0.323 | +0.365 | 56/122/47 | 1.13 vs 2.09 | 3.87 vs 9.30 | +0.13 | 0.473 | 0.41 |
| hibrido | gemma4-e4b − qwen3.5-9b | +0.009 | +0.044 | -0.035 | -0.697 | +0.663 | 56/43/30 | 1.07 vs 3.27 | 3.97 vs 11.43 | -0.13 | 0.319 | 0.51 |
| hibrido | granite4.1-8b − mistral-7b-instruct | +0.036 | +0.014 | +0.022 | +0.095 | -0.073 | 87/122/72 | 3.17 vs 2.24 | 15.35 vs 10.42 | +0.11 | 0.437 | 0.33 |
| hibrido | granite4.1-8b − qwen3.5-9b | -0.061 | -0.034 | -0.027 | +0.191 | -0.218 | 87/43/35 | 3.29 vs 2.49 | 16.29 vs 10.80 | -0.11 | 0.565 | 0.47 |
| hibrido | mistral-7b-instruct − qwen3.5-9b | -0.097 | -0.043 | -0.054 | -0.047 | -0.007 | 122/43/38 | 2.71 vs 2.76 | 11.45 vs 11.08 | -0.23 | 0.319 | 0.45 |

- [OK ] (base) las 18 descomposiciones cierran exactamente
- Peso medio de la selección en la brecha: 41.9% (mediana 44.8%).
- Pares significativos tras BH (familia de 18): 1/18. Pares cuyo |d_z| observado queda por debajo del mínimo detectable con su n: 16/18.

### 3.3 (small) Contraste: con la misma pregunta y el mismo contexto, ¿emitir más afirmaciones baja la fidelidad?

Modelo: `f = efecto fijo de (escenario, consulta) + beta·ln(emitidas) + efecto del modelo`. El efecto fijo absorbe la dificultad de la pregunta y el contexto recuperado, que son idénticos para los modelos comparados. Errores agrupados por consulta. n = 763 respuestas en 273 grupos de 128 consultas.

- beta = -0.0273 (EE 0.0203, p = 0.18). Duplicar el número de afirmaciones cambia la fidelidad en -0.019. La hipótesis NO se sostiene con este verificador: el efecto no se distingue de cero.
- Tasa media de apoyo: 0.282 apoyadas por afirmación. Tasa marginal dentro de la misma consulta: 0.233 (EE 0.029). Diferencia: +0.048, 1.7 errores estándar.

Brecha de fidelidad entre modelos con la pregunta y el contexto fijos, juntando los tres escenarios (A − B), antes y después de descontar ln(emitidas). Holm sobre los 6 pares. Es un análisis exploratorio nuevo: no sustituye a la familia BH por escenario de la tabla citable.

| A − B | Brecha | p | p Holm | Brecha ajustada | p | p Holm |
|---|---|---|---|---|---|---|
| granite4.1-8b − gemma4-e4b | -0.083 | 0.0032 | 0.016 | -0.046 | 0.23 | 1 |
| mistral-7b-instruct − gemma4-e4b | -0.034 | 0.23 | 0.59 | -0.010 | 0.78 | 1 |
| qwen3.5-9b − gemma4-e4b | -0.052 | 0.084 | 0.34 | -0.027 | 0.47 | 1 |
| mistral-7b-instruct − granite4.1-8b | +0.049 | 0.0026 | 0.016 | +0.037 | 0.025 | 0.15 |
| qwen3.5-9b − granite4.1-8b | +0.032 | 0.2 | 0.59 | +0.020 | 0.43 | 1 |
| qwen3.5-9b − mistral-7b-instruct | -0.017 | 0.5 | 0.59 | -0.017 | 0.51 | 1 |

### 3.3 (base) Contraste: con la misma pregunta y el mismo contexto, ¿emitir más afirmaciones baja la fidelidad?

Modelo: `f = efecto fijo de (escenario, consulta) + beta·ln(emitidas) + efecto del modelo`. El efecto fijo absorbe la dificultad de la pregunta y el contexto recuperado, que son idénticos para los modelos comparados. Errores agrupados por consulta. n = 763 respuestas en 273 grupos de 128 consultas.

- beta = +0.0007 (EE 0.0175, p = 0.97). Duplicar el número de afirmaciones cambia la fidelidad en +0.000. La hipótesis NO se sostiene con este verificador: el efecto no se distingue de cero.
- Tasa media de apoyo: 0.223 apoyadas por afirmación. Tasa marginal dentro de la misma consulta: 0.179 (EE 0.022). Diferencia: +0.044, 2.0 errores estándar.

Brecha de fidelidad entre modelos con la pregunta y el contexto fijos, juntando los tres escenarios (A − B), antes y después de descontar ln(emitidas). Holm sobre los 6 pares. Es un análisis exploratorio nuevo: no sustituye a la familia BH por escenario de la tabla citable.

| A − B | Brecha | p | p Holm | Brecha ajustada | p | p Holm |
|---|---|---|---|---|---|---|
| granite4.1-8b − gemma4-e4b | -0.081 | 0.0055 | 0.022 | -0.082 | 0.0074 | 0.03 |
| mistral-7b-instruct − gemma4-e4b | -0.068 | 0.019 | 0.058 | -0.068 | 0.017 | 0.05 |
| qwen3.5-9b − gemma4-e4b | -0.002 | 0.95 | 0.95 | -0.003 | 0.93 | 0.93 |
| mistral-7b-instruct − granite4.1-8b | +0.013 | 0.41 | 0.81 | +0.014 | 0.43 | 0.85 |
| qwen3.5-9b − granite4.1-8b | +0.079 | 0.00063 | 0.0038 | +0.079 | 0.0012 | 0.0073 |
| qwen3.5-9b − mistral-7b-instruct | +0.066 | 0.0021 | 0.011 | +0.065 | 0.0021 | 0.011 |

### 3.4 (small) Por qué el híbrido no siempre gana en fidelidad

| Modelo | A − B | Total | Selección | Dentro | · apoyadas | · emitidas | n A/B/común | d_z | p_BH | |d_z| mín. detectable | n para 80 % de potencia | Fragmentos distintos (media) | rho(|Δf|, fragmentos distintos) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-e4b | hibrido − lexico | -0.096 | +0.049 | -0.145 | -0.210 | +0.065 | 56/57/30 | -0.30 | 0.261 | 0.51 | 90 | 3.37 | +0.32 (p=0.08) |
| gemma4-e4b | hibrido − denso | -0.060 | +0.007 | -0.067 | -0.150 | +0.083 | 56/60/30 | -0.15 | 0.827 | 0.51 | 332 | 3.30 | -0.22 (p=0.24) |
| gemma4-e4b | denso − lexico | -0.036 | +0.090 | -0.126 | -0.245 | +0.119 | 60/57/30 | -0.25 | 0.387 | 0.51 | 130 | 4.30 | +0.44 (p=0.01) |
| granite4.1-8b | hibrido − lexico | +0.064 | -0.041 | +0.105 | +0.114 | -0.009 | 87/75/53 | +0.31 | 0.261 | 0.38 | 82 | 3.79 | +0.14 (p=0.30) |
| granite4.1-8b | hibrido − denso | +0.052 | -0.013 | +0.065 | -0.001 | +0.066 | 87/85/63 | +0.18 | 0.387 | 0.35 | 246 | 3.11 | +0.04 (p=0.76) |
| granite4.1-8b | denso − lexico | +0.012 | +0.012 | -0.000 | +0.055 | -0.055 | 85/75/51 | -0.00 | 0.827 | 0.39 | > 10 000 | 4.12 | +0.27 (p=0.06) |
| mistral-7b-instruct | hibrido − lexico | +0.042 | +0.039 | +0.003 | +0.124 | -0.122 | 122/114/93 | +0.01 | 0.942 | 0.29 | > 10 000 | 3.74 | +0.17 (p=0.11) |
| mistral-7b-instruct | hibrido − denso | +0.007 | +0.031 | -0.024 | -0.024 | +0.000 | 122/132/109 | -0.07 | 0.557 | 0.27 | 1520 | 3.32 | +0.26 (p=0.01) |
| mistral-7b-instruct | denso − lexico | +0.034 | +0.001 | +0.033 | +0.092 | -0.059 | 132/114/99 | +0.09 | 0.557 | 0.28 | 964 | 4.18 | +0.19 (p=0.05) |
| qwen3.5-9b | hibrido − lexico | +0.039 | +0.012 | +0.026 | -0.113 | +0.140 | 43/36/18 | +0.09 | 0.848 | 0.66 | 976 | 3.44 | +0.12 (p=0.63) |
| qwen3.5-9b | hibrido − denso | +0.029 | +0.143 | -0.114 | -0.261 | +0.147 | 43/45/29 | -0.34 | 0.261 | 0.52 | 67 | 3.38 | +0.13 (p=0.50) |
| qwen3.5-9b | denso − lexico | +0.010 | -0.087 | +0.097 | +0.101 | -0.004 | 45/36/18 | +0.25 | 0.557 | 0.66 | 121 | 4.17 | +0.10 (p=0.70) |

- Contrastes significativos tras BH (familia de 12): 0/12. Contrastes cuyo |d_z| observado alcanza el mínimo detectable con su n: 0/12.
- Contrastes del híbrido en los que la brecha total y la brecha dentro de las mismas consultas tienen signo opuesto: 2/8. En esos casos el orden de la tabla publicada lo decide qué consultas se contestan, no cómo se contestan.

### 3.4 (base) Por qué el híbrido no siempre gana en fidelidad

| Modelo | A − B | Total | Selección | Dentro | · apoyadas | · emitidas | n A/B/común | d_z | p_BH | |d_z| mín. detectable | n para 80 % de potencia | Fragmentos distintos (media) | rho(|Δf|, fragmentos distintos) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4-e4b | hibrido − lexico | -0.131 | +0.009 | -0.140 | -0.233 | +0.094 | 56/57/30 | -0.35 | 0.466 | 0.51 | 66 | 3.37 | +0.28 (p=0.14) |
| gemma4-e4b | hibrido − denso | +0.017 | +0.048 | -0.030 | -0.110 | +0.079 | 56/60/30 | -0.09 | 0.794 | 0.51 | 1041 | 3.30 | -0.22 (p=0.25) |
| gemma4-e4b | denso − lexico | -0.148 | +0.026 | -0.174 | -0.300 | +0.126 | 60/57/30 | -0.34 | 0.466 | 0.51 | 67 | 4.30 | +0.28 (p=0.13) |
| granite4.1-8b | hibrido − lexico | +0.018 | -0.037 | +0.055 | +0.054 | +0.001 | 87/75/53 | +0.20 | 0.497 | 0.38 | 204 | 3.79 | -0.06 (p=0.65) |
| granite4.1-8b | hibrido − denso | +0.006 | -0.013 | +0.019 | -0.035 | +0.054 | 87/85/63 | +0.07 | 0.497 | 0.35 | 1816 | 3.11 | +0.02 (p=0.89) |
| granite4.1-8b | denso − lexico | +0.012 | +0.007 | +0.005 | +0.093 | -0.088 | 85/75/51 | +0.02 | 0.748 | 0.39 | > 10 000 | 4.12 | -0.06 (p=0.68) |
| mistral-7b-instruct | hibrido − lexico | -0.004 | +0.015 | -0.019 | +0.042 | -0.062 | 122/114/93 | -0.07 | 0.497 | 0.29 | 1455 | 3.74 | +0.08 (p=0.47) |
| mistral-7b-instruct | hibrido − denso | -0.024 | +0.016 | -0.040 | -0.051 | +0.012 | 122/132/109 | -0.14 | 0.497 | 0.27 | 394 | 3.32 | +0.06 (p=0.51) |
| mistral-7b-instruct | denso − lexico | +0.019 | +0.027 | -0.008 | +0.027 | -0.035 | 132/114/99 | -0.03 | 0.654 | 0.28 | > 10 000 | 4.18 | +0.23 (p=0.02) |
| qwen3.5-9b | hibrido − lexico | +0.020 | -0.089 | +0.108 | -0.020 | +0.129 | 43/36/18 | +0.31 | 0.497 | 0.66 | 84 | 3.44 | +0.12 (p=0.64) |
| qwen3.5-9b | hibrido − denso | -0.019 | +0.058 | -0.077 | -0.277 | +0.200 | 43/45/29 | -0.21 | 0.497 | 0.52 | 174 | 3.38 | -0.04 (p=0.84) |
| qwen3.5-9b | denso − lexico | +0.039 | -0.003 | +0.041 | +0.083 | -0.042 | 45/36/18 | +0.09 | 0.654 | 0.66 | 909 | 4.17 | +0.39 (p=0.11) |

- Contrastes significativos tras BH (familia de 12): 0/12. Contrastes cuyo |d_z| observado alcanza el mínimo detectable con su n: 0/12.
- Contrastes del híbrido en los que la brecha total y la brecha dentro de las mismas consultas tienen signo opuesto: 1/8. En esos casos el orden de la tabla publicada lo decide qué consultas se contestan, no cómo se contestan.

## 4. Resultado de las comprobaciones

- Fallos: 0
