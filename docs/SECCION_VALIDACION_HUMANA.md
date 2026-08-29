# Validación con gold humano

> Borrador para adaptar al informe. Los resultados de esta sección provienen de las
> entradas 28 y 28b de `paper/summer_ablation_log.md`; la entrada 27 documenta la
> validación metodológica previa de exp19b. La sensibilidad post-adjudicación permanece
> abierta mientras no exista el JSON final de adjudicación.

## Diseño del gold

La evaluación humana reunió 240 juicios en tres muestras complementarias. La etapa A
contuvo 150 claims estratificados para arbitrar los verificadores automáticos a partir del
mejor fragmento recuperado. La etapa B volvió a presentar 50 de esos claims con los cinco
fragmentos disponibles, lo que permitió medir cuánto cambia el juicio al ampliar la
evidencia. Una tercera muestra incluyó 40 claims no soportados de exp18 para calibrar su
taxonomía y extrapolarla al universo de 759 claims.

El muestreo de la etapa A sobrerrepresentó deliberadamente casos difíciles; por ello, los
resultados poblacionales se estimaron con pesos de diseño de Horvitz–Thompson. También se
reportó el estrato `random_anchor` sin ponderación como lectura menos dependiente del
diseño. La muestra taxonómica se estratificó de forma análoga y conservó por fila el
tamaño del estrato y su probabilidad de inclusión. En ambos análisis se informó el tamaño
efectivo de Kish para hacer visible la pérdida de precisión causada por pesos desiguales.

Los materiales se anotaron en interfaces HTML offline, con vocabulario controlado y
comentarios obligatorios en los casos dudosos. El proceso mantuvo separados los juicios
humanos y los de los jueces LLM durante la anotación. Los datos y procedimientos están
documentados en `docs/GUIA_ANOTACION_GOLD_V4.md`, los CSV de
`output/audit/claim_audit_sample_v4*.csv`, el analizador
`scripts/analyze_gold_v4.py` y el reporte
`output/audit/taxonomy_calibration_report.md`. El gold completo fue fusionado desde las
exportaciones del anotador con `scripts/merge_gold_v4.py`, dejando respaldo previo en
`output/audit/backups_pre_merge_2026-08-29/`.

## Concordancia entre verificadores y gold humano

En la etapa A hubo 132 juicios binarios utilizables; 18 casos `dudoso` se excluyeron de
κ conforme a la regla declarada. La tabla presenta κ de Cohen ponderada por el diseño y,
en la última columna, κ sin pesos dentro de `random_anchor`. El tamaño efectivo de Kish
fue 38,8 sobre un universo ponderado de 14 409 claims.

| Verificador | κ ponderada | IC95 | p ajustada BH | κ `random_anchor` |
|---|---:|---:|---:|---:|
| HHEM | 0,3033 | [-0,0206; 0,5622] | 0,3220 | 0,3491 |
| E5 (`base AND HHEM`) | 0,1583 | [-0,0849; 0,3851] | 0,4955 | 0,1748 |
| NLI small | 0,0860 | [-0,1964; 0,3518] | 0,6670 | 0,0397 |
| NLI base | 0,0829 | [-0,1961; 0,3418] | 0,6670 | 0,0919 |
| E1 (media NLI) | -0,0278 | [-0,2866; 0,1936] | 0,8640 | -0,0662 |

HHEM fue el candidato más alineado con el gold, pero ninguna κ alcanzó significación
después del ajuste BH. Por tanto, estos datos respaldan una ordenación —HHEM por encima
de los NLI—, no una diferencia confirmatoria entre instrumentos. La lectura sustantiva
es que la fidelidad real está más cerca de 0,55 que de 0,30: los verificadores NLI
subestimaron el soporte. La incertidumbre es amplia y el tamaño efectivo bajo, de modo
que no debe atribuirse a estas κ una precisión que el diseño no ofrece.

La etapa B mostró además un sesgo atribuible a la cantidad de evidencia visible. Al
presentar los cinco chunks, cambiaron 16 de 50 juicios (32 %), y 11 de esos cambios fueron
hacia `correcto`. En consecuencia, la concordancia de etapa A funciona como cota inferior
para verificadores —como HHEM máximo sobre evidencia— que evalúan el conjunto completo de
fragmentos.

## Taxonomía de los claims no soportados de exp18

La calibración humana de los 40 casos produjo un tamaño efectivo de Kish de 27,4. La
extrapolación Horvitz–Thompson a los 759 claims no soportados estimó 53,0 % de claims
correctos, 45,4 % incorrectos y 1,6 % dudosos. Así, “no soportado por el corpus” no equivale
automáticamente a alucinación: aproximadamente la mitad corresponde a conocimiento
paramétrico verdadero, aunque una fracción casi igual sí contiene errores.

Los estratos también difirieron de manera informativa. `a_synthesis` y `b_parametric`
alcanzaron aproximadamente 70 % de correctos, `c_unattributed` quedó en 50 %, y
`d_threshold_artifact` solo en 30 % de correctos, con 60 % de incorrectos. El último
estrato concentra, por tanto, claims genuinamente problemáticos además de posibles falsos
negativos del umbral.

La taxonomía tuvo tres jueces ciegos. La concordancia fue κ=0,571 entre Enzo y Codex,
κ=0,422 entre Enzo y Kimi y κ=0,712 entre Kimi y Codex; hubo mayoría de dos de tres en
39 de 40 casos. Esta triangulación es útil como control, pero no convierte al LLM en
sustituto del gold humano: en la etapa A, Enzo–Codex obtuvo κ=0,055, y en la etapa B,
κ=0,204.

## Confiabilidad del anotador

Una tanda C de test–retest volvió a presentar 20 ítems de la etapa A en ciego, con seed 42.
El acuerdo crudo fue 11/20 (55 %) y κ de Cohen fue 0,268, por debajo de la meta
prerregistrada de 85 %. Los nueve desacuerdos incluyeron cinco inversiones directas entre
`correcto` e `incorrecto` y cuatro casos con `dudoso` en uno de los dos juicios. El retest
se realizó el mismo día, después de unas cinco horas de anotación continua, aunque la guía
había previsto hacerlo al día siguiente. Este contexto puede haber aumentado el ruido,
pero no justifica descartarlo: incluso `random_anchor` mostró solo 1/3 acuerdos, con un n
muy pequeño.

Los nueve desacuerdos se resolverán mediante adjudicación razonada en
`output/audit/adjudicacion_tandaC.html`. El procedimiento exige un comentario para cada
veredicto final y deja trazabilidad en el comentario fusionado y en un backup del CSV. La
κ test–retest cruda se conserva como resultado de confiabilidad; la adjudicación no la
reemplaza ni la corrige retroactivamente.

### Sensibilidad a la adjudicación

Mientras no existe el JSON final, `scripts/run_gold_sensitivity.py` solo puede ejecutar la
variante que excluye los nueve ítems discordantes, identificados por
`output/audit/gold_v4_tandaC_resultado.json`. Quedan 126 juicios binarios utilizables. Los
resultados registrados en `output/audit/gold_v4_sensitivity.json` y
`output/audit/gold_v4_sensitivity.md` son:

| Verificador | κ ponderada sin discordantes | κ anchor sin discordantes | Δ frente a post-adjudicación |
|---|---:|---:|---:|
| HHEM | 0,3090 | 0,3966 | [PENDIENTE-ADJUDICACION] |
| E5 (`base AND HHEM`) | 0,1571 | 0,1702 | [PENDIENTE-ADJUDICACION] |
| NLI small | 0,0844 | 0,0156 | [PENDIENTE-ADJUDICACION] |
| NLI base | 0,0810 | 0,0780 | [PENDIENTE-ADJUDICACION] |
| E1 (media NLI) | -0,0307 | -0,0800 | [PENDIENTE-ADJUDICACION] |

La exclusión no altera el orden de los candidatos observado en el análisis original. Esta
comparación es descriptiva y parcial: faltan las variantes con etiquetas post-adjudicación
y pre-adjudicación reconstruida, además de sus deltas respecto de la primera.

- κ post-adjudicación: **[PENDIENTE-ADJUDICACION]**.
- κ pre-adjudicación reconstruida: **[PENDIENTE-ADJUDICACION]**.
- Deltas de ambas variantes: **[PENDIENTE-ADJUDICACION]**.

## Limitaciones

El gold depende de un único anotador humano y su test–retest quedó lejos de la meta
prerregistrada. El muestreo estratificado priorizó casos difíciles y redujo el tamaño
efectivo, por lo que los intervalos de κ son amplios. La adjudicación post-hoc mejora la
consistencia operativa del gold, pero no elimina esa limitación. Por ello, la evidencia
permite sostener el ordenamiento general HHEM > NLI y describir la composición de los
claims no soportados; no permite defender diferencias pequeñas ni una precisión fina de
las tasas.

## Trazabilidad

- `paper/summer_ablation_log.md`, entradas 27, 28 y 28b.
- `output/audit/gold_v4_analysis.json` y `output/audit/gold_v4_analysis.md`.
- `output/audit/taxonomy_calibration_report.md`.
- `output/audit/gold_v4_tandaC_resultado.json`.
- `output/audit/gold_v4_sensitivity.json` y `output/audit/gold_v4_sensitivity.md`.
- `scripts/analyze_gold_v4.py`, `scripts/analyze_taxonomy_calibration.py`,
  `scripts/merge_adjudicacion_tandaC.py` y `scripts/run_gold_sensitivity.py`.
