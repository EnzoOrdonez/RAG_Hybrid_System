# Validación con referencia humana piloto

> Borrador para adaptar al informe. La versión previa a la adjudicación es el resultado
> principal. Las cifras proceden de las entradas 27, 28, 28b y 28c de
> `paper/summer_ablation_log.md` y de los artefactos de auditoría citados al final.

## Qué se midió

Esta evaluación separa tres variables que no son intercambiables:

- `unsupported@τ`: decisión automática; el mejor score de soporte entre los chunks del
  pool es menor o igual que el umbral τ.
- `human-judged unsupported`: juicio humano sobre si la evidencia visible respalda el
  claim.
- `externally correct` / `externally incorrect`: juicio de corrección factual con
  conocimiento externo, sin inferir de dónde obtuvo el generador la información.

Por tanto, los 759 de 2053 claims de exp18 son `unsupported@0.5` según el verificador, no
759 falsedades ni 759 alucinaciones verificadas por una persona.

La referencia piloto consta de tres subconjuntos con propósitos y condiciones distintas;
no se suman como si fueran una muestra homogénea:

| Subconjunto | Propósito | Población objetivo | Evidencia visible | Muestreo | n |
|---|---|---:|---|---|---:|
| A | Concordancia entre verificadores y juicio humano de soporte | 14 409 claims del pool de arbitraje | Mejor chunk | Estratificado por celdas de desacuerdo y ancla aleatoria | 150 (132 binarios usados) |
| B | Sensibilidad del juicio al conjunto de evidencia | Los 150 claims de A | Cinco chunks de la misma fila | Submuestra pareada de A | 50 |
| Taxonomía exp18 | Corrección factual de casos `unsupported@0.5` | 759 claims | Pregunta y claim, sin chunks | 10 por cada uno de cuatro estratos candidatos | 40 |

La anotación se hizo mediante interfaces HTML offline, con vocabulario controlado y
comentario obligatorio en los casos dudosos. Los juicios humanos y los de los jueces LLM
permanecieron cegados entre sí durante la anotación. El muestreo de A y de la taxonomía
usa pesos Horvitz–Thompson; se informa el n efectivo de Kish para hacer visible la pérdida
de precisión por pesos desiguales. El protocolo está en
`docs/GUIA_ANOTACION_GOLD_V4.md`.

## Concordancia con la referencia piloto: resultado principal preadjudicación

En A hubo 132 juicios binarios utilizables; los 18 `dudoso` se excluyeron de κ según la
regla declarada. El n efectivo de Kish fue 38,8. κ cuantifica concordancia con esta
referencia bajo este diseño; no estima el nivel absoluto o «real» de fidelidad.

| Verificador | κ ponderada | IC95 | p ajustada BH | κ `random_anchor` |
|---|---:|---:|---:|---:|
| HHEM | 0,3033 | [-0,0206; 0,5622] | 0,3220 | 0,3491 |
| E5 (`base AND HHEM`) | 0,1583 | [-0,0849; 0,3851] | 0,4955 | 0,1748 |
| NLI small | 0,0860 | [-0,1964; 0,3518] | 0,6670 | 0,0397 |
| NLI base | 0,0829 | [-0,1961; 0,3418] | 0,6670 | 0,0919 |
| E1 (media NLI) | -0,0278 | [-0,2866; 0,1936] | 0,8640 | -0,0662 |

HHEM mostró mayor concordancia puntual que los NLI, pero ninguna κ fue distinta de cero
tras el ajuste BH y los intervalos son amplios. El orden de los candidatos es informativo
en esta muestra; no identifica una fidelidad verdadera de 0,55 frente a 0,30 ni confirma
superioridad entre instrumentos.

El contraste exp19b siguió el flujo `draft → extracción de claims → selección ms-marco →
regeneración → verificador`: sí se generó una nueva respuesta con la evidencia
seleccionada. HHEM no intervino en el selector, que usó el cross-encoder
`ms-marco-MiniLM-L-12-v2`; sin embargo, el aumento apareció solo con HHEM
(Δ=+0,0451, IC95 [0,0076; 0,0817], p=0,01798) y no con los dos NLI. Se interpreta, por
ello, como una mejora local de alineación con HHEM, no como una mejora de fidelidad
independiente del verificador. El TOST ubicó la diferencia dentro de la banda
preespecificada ±0,081 (p=0,03135; IC90 [0,0134; 0,0768]); esa banda admite efectos
mayores que el observado y no está validada externamente como umbral de irrelevancia.

## Sensibilidad al conjunto de evidencia

Al ampliar de uno a cinco chunks, 16 de 50 etiquetas cambiaron (32,0%; IC95 Wilson
[20,8%; 45,8%]). Esto demuestra sensibilidad al contexto presentado; no identifica por
sí solo sesgo causal ni establece que A sea una cota inferior formal.

| Juicio en A \ juicio en B | correcto | incorrecto | dudoso |
|---|---:|---:|---:|
| correcto | 28 | 1 | 2 |
| incorrecto | 9 | 3 | 2 |
| dudoso | 2 | 0 | 3 |

En la reducción `correcto`/resto hubo 11 transiciones hacia `correcto` y 3 en sentido
contrario. McNemar exacto dio p=0,057373. Entre todos los 16 flips, 11 fueron hacia
`correcto` y 5 tuvieron otra dirección (binomial exacta p=0,210114); este segundo resumen
no es McNemar. Ambos cálculos son descriptivos y están fuera de las familias BH.

## Taxonomía de `unsupported@0.5`

En la muestra de 40, los conteos no ponderados fueron 22 `externally correct`, 17
`externally incorrect` y 1 `dudoso`. La extrapolación Horvitz–Thompson a 759 fue 53,0%,
45,4% y 1,6%, respectivamente, con n efectivo de Kish 27,4. La tasa pequeña de `dudoso`
depende de un solo caso observado y no debe leerse como una estimación precisa.

Que un claim sea juzgado correcto pese a ser `unsupported@0.5` no demuestra conocimiento
paramétrico: también puede reflejar evidencia omitida por retrieval, información parcial
del prompt o patrones generales. Su procedencia no puede determinarse con este diseño.
Los estratos (`a_synthesis`, `b_parametric`, `c_unattributed` y
`d_threshold_artifact`) fueron candidatos de muestreo, no veredictos causales.

La taxonomía tuvo tres jueces ciegos. La concordancia pareada fue κ=0,571 entre Enzo y
Codex, κ=0,422 entre Enzo y Kimi y κ=0,712 entre Kimi y Codex; hubo mayoría de dos de tres
en 39/40. Es estabilidad local de mayoría, no validación externa. En A y B los jueces LLM
sin calibración específica mostraron baja concordancia con la referencia piloto
(κ=0,055 y κ=0,204, respectivamente), por lo que no ofrecieron un reemplazo fiable en
estas muestras.

## Confiabilidad intra-anotador y adjudicación

La tanda C repitió en ciego 20 ítems de A, con seed 42. El acuerdo crudo fue 11/20 = 55%
(IC95 Wilson [34,2%; 74,2%]) y κ de Cohen fue 0,268, por debajo de la meta
prerregistrada de 85%. Esta κ mide confiabilidad **intra-anotador**, no acuerdo entre
anotadores ni validez externa.

| Juicio original \ retest | correcto | incorrecto | dudoso |
|---|---:|---:|---:|
| correcto | 7 | 3 | 1 |
| incorrecto | 1 | 2 | 1 |
| dudoso | 2 | 1 | 2 |

El rótulo «fair» de Landis–Koch para κ=0,268 se menciona solo como convención histórica:
los puntos de corte fueron propuestos como arbitrarios y no constituyen un sello de
aceptabilidad ([Landis y Koch, 1977](https://jjcurtin.github.io/book_iaml/pdfs/landis_1977_kappa.pdf)).
La interpretación de κ depende de las etiquetas, sus prevalencias y el propósito del
corpus; no debe aislarse de la matriz y los marginales
([Artstein y Poesio, 2008](https://aclanthology.org/J08-4004/)).

Los nueve desacuerdos se adjudicaron con razones explícitas para construir una versión
reconciliada. Esa edición aporta una etiqueta operativa final, pero no aumenta ni corrige
retroactivamente la confiabilidad observada. La tabla siguiente conserva como principal
la versión preadjudicación y muestra dos análisis secundarios.

| Verificador | Pre: κ pond. / anchor | Reconciliada: κ pond. / anchor | Sin 9: κ pond. / anchor |
|---|---:|---:|---:|
| HHEM | 0,3033 / 0,3491 | 0,3150 / 0,3966 | 0,3090 / 0,3966 |
| E5 (`base AND HHEM`) | 0,1583 / 0,1748 | 0,1601 / 0,1702 | 0,1571 / 0,1702 |
| NLI small | 0,0860 / 0,0397 | 0,0916 / 0,0156 | 0,0844 / 0,0156 |
| NLI base | 0,0829 / 0,0919 | 0,0839 / 0,0780 | 0,0810 / 0,0780 |
| E1 (media NLI) | -0,0278 / -0,0662 | -0,0288 / -0,0800 | -0,0307 / -0,0800 |

El ordenamiento puntual fue estable, pero la baja confiabilidad y los intervalos amplios
limitan la interpretación de las magnitudes absolutas. La reconciliación no convierte la
referencia piloto en una verdad independiente.

## Limitaciones

La referencia depende de un solo anotador y sobremuestrea casos difíciles. Los claims
están anidados en respuestas y queries; una inferencia que los trate como independientes
puede subestimar la incertidumbre. La condición B cambia simultáneamente cantidad, orden
y longitud de evidencia. La taxonomía usa una muestra pequeña con pesos desiguales y la
corrección externa no identifica el origen del contenido. Finalmente, la concordancia
con una referencia inestable no valida por sí sola ningún verificador.

## Trazabilidad

- `paper/summer_ablation_log.md`, entradas 27, 28, 28b y 28c.
- `output/audit/gold_v4_analysis.json` y `output/audit/gold_v4_analysis.md`.
- `output/audit/descriptive_cis.json` y `output/audit/descriptive_cis.md`.
- `output/audit/taxonomy_calibration_report.md`.
- `output/audit/gold_v4_tandaC_resultado.json`.
- `output/audit/gold_v4_sensitivity.json` y `output/audit/gold_v4_sensitivity.md`.
- `scripts/analyze_gold_v4.py`, `scripts/analyze_taxonomy_calibration.py` y
  `scripts/run_gold_sensitivity.py`.
