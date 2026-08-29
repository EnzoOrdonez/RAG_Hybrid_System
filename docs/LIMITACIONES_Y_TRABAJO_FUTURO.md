# Limitaciones y trabajo futuro

> Borrador en español para adaptar al informe. Distingue resultados medidos de propuestas
> aún no evaluadas. Las cifras del gold proceden de las entradas 28 y 28b de
> `paper/summer_ablation_log.md`; el probe de proveedores es descriptivo y no pertenece a
> ninguna familia de contrastes ni corrección BH.

## Limitaciones medidas

### La evidencia visible cambia el juicio humano

La etapa A mostró al anotador un único chunk por claim, mientras que la etapa B presentó
los cinco chunks recuperados para los mismos 50 casos. Cambiaron 16/50 veredictos (32 %) y
11 cambios fueron hacia `correcto`. Por ello, la evaluación de etapa A puede subestimar el
soporte disponible para verificadores que recorren varios fragmentos. Este efecto está
registrado en `output/audit/gold_v4_analysis.md` y en la entrada 28 de
`paper/summer_ablation_log.md`; no debe confundirse con error aleatorio del verificador.

### La confiabilidad intra-anotador fue baja

La reanotación ciega de tanda C obtuvo acuerdo crudo de 11/20 (55 %) y κ=0,268, por debajo
de la meta de 85 %. El retest ocurrió el mismo día después de una sesión larga y la muestra
sobrerrepresentó casos difíciles, pero esos atenuantes no eliminan la limitación. Los nueve
casos discordantes requieren adjudicación razonada y una prueba de sensibilidad con y sin
ellos. El resultado crudo y el procedimiento se conservan en la entrada 28b de
`paper/summer_ablation_log.md` y en `output/audit/gold_v4_tandaC_resultado.json`.

### El diseño estratificado reduce la precisión efectiva

Los pesos de Horvitz–Thompson permiten extrapolar desde estratos sobremuestreados, pero
incrementan la varianza. En la etapa A, 150 anotaciones corresponden a un tamaño efectivo
de Kish de 38,8; en la taxonomía de 40 claims, el tamaño efectivo fue 27,4. Los intervalos
de κ son, en consecuencia, amplios y ninguna κ de la etapa A resultó significativa tras BH.
Las cifras ponderadas deben leerse como estimaciones poblacionales con incertidumbre, junto
con la lectura no ponderada de `random_anchor`. Véanse `output/audit/gold_v4_analysis.md`,
`output/audit/taxonomy_calibration_report.md` y la entrada 28 del ledger.

### El corpus conserva restos de conversión Markdown

El corpus descargado contiene encabezados incrustados, escapes como `\.` y tablas o bloques
Markdown parcialmente rotos. Esos restos pueden degradar el chunking, producir claims sin
proposición factual y dificultar que una persona identifique la evidencia decisiva. La guía
de anotación trata explícitamente estos casos en `docs/GUIA_ANOTACION_GOLD_V4.md` (§§1–2),
pero el tratamiento durante la anotación no repara el corpus subyacente. Una limpieza más
agresiva tendría que validarse contra la preservación de tablas, código y jerarquías antes
de reconstruir índices.

### Las consultas multi-nube siguen perdiendo proveedores

`scripts/probe_provider_coverage.py` analizó las 25 queries multi-nube de
`data/evaluation/test_queries.json` usando únicamente los top-5 persistidos en
`experiments/results/exp17_crosscloud_balanced/retrieval_ids.json`. La cobertura estricta
—al menos un chunk por cada proveedor mencionado— fue 2/25 (8 %) en el brazo baseline y
20/25 (80 %) en el brazo balanced. En ambos brazos, 0/125 chunks pertenecieron a un
proveedor ajeno a la query (0 %). El problema observado no es contaminación por un cuarto
proveedor, sino ausencia de al menos uno de los proveedores solicitados. El detalle queda
en `output/audit/provider_coverage_probe.json` y
`output/audit/provider_coverage_probe.md`.

El 20/25 también matiza el antiguo 25/25 de exp17: aquel indicador exigía solamente los
proveedores deseados que ya estaban disponibles en el pool; el probe actual exige todos
los mencionados en la query. Es un análisis descriptivo sobre evidencia congelada, sin
repetir retrieval y fuera de cualquier inferencia confirmatoria.

## Trabajo futuro

### Routing por metadatos de proveedor

El primer paso sería hacer explícito el conjunto de proveedores de la consulta y aplicar
un routing por `cloud_provider` antes de la selección final. Cada proveedor mencionado
debería aportar candidatos al pool y una cuota mínima al top-k, con una salida de auditoría
que distinga “sin candidato disponible” de “candidato descartado por ranking”. Esta defensa
ataca directamente los cinco casos que permanecen incompletos después del balanceo. Debe
evaluarse el intercambio entre cobertura y relevancia, manteniendo como referencia el
artefacto congelado de exp17 y sin reutilizar su métrica condicional como cobertura estricta.

### Descomposición de consultas multi-nube

Una consulta comparativa puede descomponerse en una subconsulta por proveedor y por eje de
comparación —por ejemplo, escalado, precio o disponibilidad—. Cada subconsulta recuperaría
evidencia dentro de su proveedor; después se normalizarían los atributos comparables y solo
entonces se generaría la síntesis transversal. Este diseño reduce la competencia entre
proveedores dentro de un único ranking top-k y permite abstenerse de manera localizada
cuando falta evidencia de un lado de la comparación. Su evaluación debería medir cobertura
por proveedor, relevancia, fidelidad por claim y costo de contexto.

### Grafo de tripletas como tercer canal

Un tercer canal de retrieval podría representar la documentación como tripletas
`servicio–relación–atributo` —por ejemplo, equivalencias, dependencias, límites y regiones—
en una base de grafos embebida, como LadybugDB. Las rutas recuperadas se fusionarían con
BM25 y búsqueda densa, conservando la procedencia del fragmento original. Además de aportar
evidencia estructurada a comparaciones multi-nube, el grafo permitiría verificar relaciones
explícitas antes de redactar un claim. Esta propuesta requiere un esquema, extracción y
resolución de entidades evaluados por separado; todavía no constituye evidencia del sistema
actual.

## Trazabilidad

- `paper/summer_ablation_log.md`, entradas 28 y 28b.
- `docs/GUIA_ANOTACION_GOLD_V4.md`.
- `output/audit/gold_v4_analysis.md`.
- `output/audit/taxonomy_calibration_report.md`.
- `output/audit/gold_v4_tandaC_resultado.json`.
- `output/audit/provider_coverage_probe.json` y
  `output/audit/provider_coverage_probe.md`.
- `experiments/results/exp17_crosscloud_balanced/retrieval_ids.json` y
  `experiments/results/exp17_crosscloud_balanced/retrieval_report.md`.
