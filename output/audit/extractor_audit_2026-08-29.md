# Auditoría descriptiva del extractor de claims — 2026-08-29

## Alcance y declaración

Auditoría realizada por Codex (LLM), de forma descriptiva y acotada. No es anotación
humana, gold ni validación externa. No se ejecutaron el extractor, retrieval, NLI, HHEM,
Ollama ni otro modelo: se compararon exclusivamente respuestas y claims ya persistidos.
Los resultados se presentan como amenaza a la validez de constructo, no como una tasa
poblacional.

Fuentes: brazo exacto `baseline_repro | granite4.1-8b` de
`experiments/results/exp18_evidence_ceiling/results.json`; claims brutos y banderas
`artifact` de `claims_extraction.json`; claims retenidos de
`selection_scores_v2_index.json`.

## Selección y rúbrica

El universo fueron los 188 `query_id` con al menos un claim genuino en el índice v2. La
muestra se fijó con `random.Random(42).sample(sorted(qids), 15)` y se presenta ordenada:

`q009`, `q010`, `q011`, `q025`, `q030`, `q032`, `q039`, `q063`, `q068`, `q076`,
`q114`, `q147`, `q159`, `q171`, `q181`.

La revisión aplicó estas reglas operativas:

- **Cobertura:** una unidad omitida es una proposición explícita o un bloque de comando
  con contenido verificable que no aparece en ningún claim; no se cuentan repeticiones
  retóricas de una conclusión ya representada.
- **Merge:** un claim contiene al menos dos predicados que podrían puntuarse por separado.
- **Fragmento/degenerado:** encabezado, lista nominal, fuente, instrucción o comentario
  sobre el contexto sin una proposición factual autónoma.
- **Espurio:** el claim no está afirmado por la respuesta.
- **Conservación:** se comprobó en forma separada la polaridad de negaciones/modalidad y
  los valores numéricos explícitos.

Las categorías no son excluyentes: un claim puede ser a la vez un merge y meta-texto.

## Censo de la muestra

| qid | claims brutos | `artifact` | retenidos | unidades omitidas | merges claros | degenerados retenidos |
|---|---:|---:|---:|---:|---:|---:|
| q009 | 5 | 1 | 4 | 0 | 2 | 1 |
| q010 | 2 | 0 | 2 | 1 | 1 | 1 |
| q011 | 8 | 0 | 8 | 0 | 4 | 1 |
| q025 | 1 | 0 | 1 | 1 | 0 | 1 |
| q030 | 13 | 0 | 13 | 0 | 8 | 1 |
| q032 | 17 | 0 | 17 | 0 | 12 | 1 |
| q039 | 5 | 1 | 4 | 0 | 1 | 2 |
| q063 | 12 | 0 | 12 | 9 | 4 | 4 |
| q068 | 22 | 0 | 22 | 0 | 4 | 5 |
| q076 | 11 | 2 | 9 | 3 | 2 | 2 |
| q114 | 3 | 1 | 2 | 1 | 0 | 2 |
| q147 | 2 | 0 | 2 | 1 | 1 | 2 |
| q159 | 14 | 0 | 14 | 0 | 10 | 0 |
| q171 | 8 | 1 | 7 | 0 | 2 | 5 |
| q181 | 4 | 0 | 4 | 0 | 3 | 1 |
| **Total** | **127** | **6** | **121** | **16** | **54** | **29** |

La retención de 121/127 (95,3 %) mide **precisión aparente** del extractor —lo retenido
sí proviene de la respuesta—, no su exhaustividad: la cobertura se evalúa aparte en la
sección siguiente (16 unidades omitidas en 6/15 respuestas). Los seis descartes fueron
claims marcados con la bandera `artifact` (encabezados, fuentes o meta-texto), según el
censo por fila de la tabla.

## Cobertura

Se identificaron 16 unidades literales omitidas en 6/15 respuestas. Doce fueron ejemplos
de comandos: nueve comandos AWS CLI de q063 y tres bloques CLI de q076. Sus acciones
generales sí aparecen en claims narrativos, pero se perdieron el ejecutable, argumentos y
valores. Las otras cuatro fueron descripciones del alcance documental en q010, q025, q114
y q147. En las nueve respuestas restantes no se observó una omisión material bajo la
rúbrica.

Ejemplos literales de contenido omitido:

- q063 contiene `aws iam attach-role-policy --role-name ecsInstanceRole ...`; el claim
  conserva «Attach the AWS managed policy», pero no el comando.
- q076 contiene `--period 300` y `--evaluation-periods 1`; el claim agrupa los títulos de
  los pasos y omite todo el bloque de alarma.
- q010 afirma que el contexto cubre «EC2 Fleet quotas, ECS service quotas, Fargate
  throttling quotas, EC2 On-Demand Instance quotas»; esa delimitación no se extrajo.

## Atomicidad

Hubo 54/121 claims retenidos con dos o más predicados puntuables. Trece de las 15
respuestas presentaron al menos un merge claro. La concentración mayor apareció en listas
de capacidades: el extractor conservó cada viñeta como una sola unidad aunque la viñeta
incluyera varias acciones o propiedades. También retuvo 12 fragmentos de encabezado o
introducción como claims independientes; estos se contabilizan dentro de los 29
degenerados y representan sobre-segmentación estructural, no afirmaciones atómicas.

Ejemplos literales de merge:

- q032: «Multitenant Organizations: Collaborate across tenants within your organization,
  manage domain services, and support business-to-consumer (B2C) identity...» reúne tres
  capacidades comprobables.
- q030: «This upgrade is one-way and impacts service integrations and costs
  significantly.» mezcla irreversibilidad, integraciones y coste.
- q159: «Both services cater to different scaling needs: ACI excels in quick, flexible
  scaling..., while VMs provide robust, configurable compute power...» combina la
  caracterización de ambos servicios.

## Claims espurios y degenerados

No se encontró ningún claim espurio (0/121): todos los textos retenidos podían localizarse
en la respuesta. Por tanto, no hay ejemplos espurios que citar en esta muestra.

Sí se encontraron 29/121 claims degenerados o meta bajo la definición operativa. Ejemplos:

- q032: «The main capabilities of Azure Entra ID include:.» es un introductor sin
  proposición autónoma.
- q063: «[AWS > ECS > Tutorial: Using FSx for Windows File Server ...].» es una fuente
  aislada convertida en claim.
- q171: «AWS Lambda's features, pricing model, scaling behavior, supported languages, and
  integration capabilities..» es un fragmento nominal de una lista sobre documentación
  faltante, no una afirmación sobre Lambda.

La bandera `artifact` eliminó 6/127 claims brutos, pero dejó numerosos encabezados,
instrucciones, fuentes y comentarios sobre «el contexto». A la inversa, algunas frases
negativas informativas fueron marcadas como artefacto, aunque en estos 15 casos su sentido
quedó repetido por otro claim retenido.

## Negación, modalidad y números

Ocho respuestas contenían negativas explícitas sobre ausencia o insuficiencia de
documentación (q009, q010, q025, q039, q114, q147, q171 y q181). En las ocho se conservó
al menos una formulación equivalente; no se observó ninguna inversión de polaridad. En
q039, por ejemplo, se filtró «does not explicitly list all VM SKUs», pero quedó «does not
directly address the full range of VM SKUs». Las modalidades `cannot`, `would need` y
`may be required` se conservaron cuando la oración fue extraída.

Se revisaron cinco valores numéricos o direcciones con significado de dominio. «two SKUs»
de q039 se preservó. En q068, `0.0.0.0/0` y `203.0.113.25/32` fueron reemplazados por
`[CODE]`; en q076, `300` y `1` quedaron dentro del bloque de comando omitido. La numeración
de pasos/listas no se contó como dato factual.

Ejemplos positivos de conservación:

- q039 preservó literalmente «Public IP addresses have two SKUs: standard and basic».
- q159 separó correctamente «The maximum IOPS limits are independent...» de
  «Performance is constrained by the lower limit...», manteniendo la relación cuantitativa
  sin inventar valores.
- q010 mantuvo la negación: «none of these sections mention AWS S3» y la consecuencia de
  que no pueden determinarse cuotas S3 desde ese contexto.

## Amenaza a la validez de constructo

El denominador claim-level no es una transcripción neutral de las respuestas. En esta
muestra, el extractor cubrió la mayoría de conclusiones sustantivas y no inventó texto,
pero perdió detalle procedimental, mantuvo 29 unidades no factuales y fusionó con frecuencia
predicados distintos. Los reemplazos `[CODE]` también pueden borrar números o condiciones
que cambian la verificabilidad. En consecuencia, conteos como 2053 claims genuinos y tasas
`unsupported@τ` dependen de decisiones de segmentación y filtrado; deben interpretarse como
medidas del pipeline de extracción observado, no como el número exacto de afirmaciones
semánticas de las respuestas.
