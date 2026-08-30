# Ficha experimental

Esta ficha explicita las unidades, familias inferenciales y datos usados en el cierre de
exp18, exp19b y la referencia humana piloto. No ejecuta modelos ni experimentos: organiza
artefactos persistidos y cálculos descriptivos ligeros.

## Unidades y estimandos

| Componente | Unidad observada | Estimando o resumen | n y anidamiento | Fuente |
|---|---|---|---|---|
| Generación exp18/exp19b | Una respuesta por query y brazo | Respuesta condicionada al prompt y a los chunks del brazo | 194 queries por brazo; respuestas emparejadas por `query_id` | `scripts/run_exp19b_generation.py`; `experiments/results/exp19b_anchored_selector/results.json` |
| Extracción exp18 | Claim extraído de una respuesta | Descomposición de la respuesta en afirmaciones verificables | 2218 claims brutos en 194 respuestas; 165 marcados como artefacto y 2053 retenidos | `experiments/results/exp18_evidence_ceiling/claims_extraction.json` |
| Verificación exp18 | Claim frente al conjunto de chunks recuperados | Mejor score de soporte del claim en el pool; `unsupported@τ` si `best_over_pool <= τ` | 2053 claims anidados en 188 queries con al menos un claim genuino | `experiments/results/exp18_evidence_ceiling/selection_scores_v2_index.json` |
| Contraste exp19b | Diferencia pareada por query entre `claim_selected` y `baseline_repro` | Media de diferencias de fidelidad por verificador | **n=186 queries**: 8 declinaciones excluidas; 7 fallbacks incluidos con diferencia exactamente cero | `experiments/results/exp19b_anchored_selector/equivalence__{hhem,small,base}.md`; ledger, entrada 27 |
| Referencia piloto A | Claim muestreado | κ verificador–humano ponderada por diseño | 150 claims; 132 binarios usados; 139 clusters respuesta/configuración y 93 `query_id` únicos; Kish 38,8 | `output/audit/claim_audit_sample_v4_meta.json`; `output/audit/gold_v4_analysis.json` |
| Referencia piloto B | El mismo claim juzgado con más evidencia | Tasa y dirección de cambio A→B | 50 pares claim-level de A | `output/audit/claim_audit_sample_v4_stageB.csv`; `output/audit/descriptive_cis.json` |
| Test–retest | Claim de A reanotado en ciego | Acuerdo intra-anotador y κ de Cohen | 20 pares; 11 acuerdos | `output/audit/gold_v4_tandaC_resultado.json`; `output/audit/descriptive_cis.json` |
| Taxonomía exp18 | Claim `unsupported@0.5` muestreado | Tasas HT de corrección externa sobre 759 claims | 40 claims estratificados; Kish 27,409 | `output/audit/taxonomy_calibration_report.md`; `output/audit/descriptive_cis.json` |

Los claims no son observaciones independientes: están anidados en respuestas, queries y,
para A, configuraciones. La inferencia claim-level puede subestimar la incertidumbre si
ignora esa dependencia. En cambio, el **n=186 de exp19b es explícitamente query-level** y el
bootstrap es pareado por query, según los reportes `equivalence__*.md`.

## Familias de hipótesis

### Exp19b: familias preregistradas por verificador

Cada verificador constituye una familia de un solo contraste; por eso BH es la identidad y
`p_BH = p_crudo`. Los tres verificadores forman una triangulación, no una familia conjunta.

| Familia / hipótesis | Métrica y dirección | p crudo | p BH | IC95 | Efecto | Estatus |
|---|---|---:|---:|---|---|---|
| HHEM: `claim_selected − baseline_repro` | Δ pareada, bilateral | 0,01798 | 0,01798 | [0,0076; 0,0817] | Δ=0,0451; d_z=0,1727 | Preregistrado |
| NLI small: mismo contraste | Δ pareada, bilateral | 0,94206 | 0,94206 | [-0,0440; 0,0393] | Δ=-0,0021; d_z=-0,0073 | Preregistrado |
| NLI base: mismo contraste | Δ pareada, bilateral | 0,09958 | 0,09958 | [-0,0035; 0,0562] | Δ=0,0258; d_z=0,1232 | Preregistrado |

El TOST se declaró por separado para la banda preespecificada ±0,081. No se interpreta
como una familia BH ni como validación externa del margen:

| Verificador | Dirección | p TOST | IC90 | Resultado dentro de la banda |
|---|---|---:|---|---|
| HHEM | `-0,081 < Δ < 0,081` | 0,03135 | [0,0134; 0,0768] | Sí |
| NLI small | `-0,081 < Δ < 0,081` | 0,00015 | [-0,0375; 0,0332] | Sí |
| NLI base | `-0,081 < Δ < 0,081` | 0,00021 | [0,0004; 0,0513] | Sí |

### Referencia humana piloto: familia exploratoria de cinco candidatos

La hipótesis bilateral fue κ=0 frente a κ≠0. Los p crudos se reconstruyeron con la misma
función, bootstrap estratificado de 10 000 réplicas y seed 42 de
`scripts/analyze_gold_v4.py`; los p BH e intervalos son los persistidos en
`output/audit/gold_v4_analysis.json`. Esta familia arbitra concordancia local y no decide
la promoción del verificador primario.

| Candidato | Dirección | p crudo | p BH | IC95 de κ | κ ponderada | Estatus |
|---|---|---:|---:|---|---:|---|
| HHEM | Bilateral, κ≠0 | 0,0644 | 0,3220 | [-0,0206; 0,5622] | 0,3033 | Exploratorio |
| E5 (`base AND HHEM`) | Bilateral, κ≠0 | 0,1982 | 0,4955 | [-0,0849; 0,3851] | 0,1583 | Exploratorio |
| NLI small | Bilateral, κ≠0 | 0,5222 | 0,6670 | [-0,1964; 0,3518] | 0,0860 | Exploratorio |
| NLI base | Bilateral, κ≠0 | 0,5336 | 0,6670 | [-0,1961; 0,3418] | 0,0829 | Exploratorio |
| E1 (media NLI) | Bilateral, κ≠0 | 0,8640 | 0,8640 | [-0,2866; 0,1936] | -0,0278 | Exploratorio |

## Ficha del conjunto de evaluación

El conjunto versionado parte de 200 preguntas. La curación eliminó seis (`q126`, `q186`,
`q188`, `q194`, `q199`, `q200`) porque su sujeto principal era Kubernetes upstream o CNCF,
dominios retirados del corpus; quedaron 194. La bitácora conserva preguntas, campos y razón
de cada exclusión en `data/evaluation/test_queries_removed_log.json`. Los
`relevant_chunk_ids` de las 194 están vacíos por diseño: la relevancia de retrieval no es
una anotación humana persistida, sino un oráculo dinámico documentado en
`data/evaluation/README.md`.

| Dimensión en las 194 queries | Distribución contada desde `test_queries.json` |
|---|---|
| Tipo | 63 factual; 65 procedural; 66 comparative |
| Dificultad | 70 easy; 85 medium; 39 hard |
| Categoría | 1 ai_ml; 86 compute; 27 general; 19 networking; 15 security; 46 storage |
| Número de proveedores | 169 de uno; 8 de dos; 17 de tres |
| Combinación | 75 AWS; 57 Azure; 37 GCP; 8 AWS+Azure; 17 AWS+Azure+GCP |

Hay 25 consultas multi-nube, cifra confirmada tanto por `cloud_providers` como por
`output/audit/provider_coverage_probe.json`. En el retrieval persistido de exp17, la
cobertura de todos los proveedores mencionados fue 2/25 para el brazo baseline y 20/25
para el balanceado; no aparecieron chunks de proveedores ajenos a la consulta. Es un
resultado descriptivo fuera de las familias BH.

Exp18 generó respuestas para las 194 queries. El extractor produjo 2218 claims brutos y
marcó 165 artefactos, dejando 2053 genuinos. Seis respuestas (`q003`, `q008`, `q028`,
`q042`, `q045`, `q134`) quedaron sin claims genuinos; por eso el índice de scores contiene
188 queries, no 194. Estos conteos se obtienen directamente de
`claims_extraction.json` y `selection_scores_v2_index.json`.

El corpus declarado en `data/corpus_stats.json` contiene 2697 documentos procesados y
24 481 chunks (chunking adaptativo 500/50), con BGE large de 1024 dimensiones. La
descripción de construcción disponible es la bitácora del rebuild y del submuestreo
estratificado de Azure; no se afirma una procedencia más detallada de la que ese archivo
documenta.

## Repetibilidad interna y reproducibilidad externa

El repositorio aporta `requirements-lock.txt` generado con Python 3.14, CI sin GPU ni
servicios, configuración y seeds, manifiestos con revisión y SHA-256 de los modelos
Transformers (`output/audit/summer_models_manifest_2026-07-22.json` y
`output/audit/verifier_models_manifest_2026-07-23.json`) y una compuerta de replay que
exige 5/5 respuestas bit-idénticas antes de regenerar exp19b. La compuerta cubre
únicamente la regeneración y evaluación sobre artefactos archivados dentro del mismo
estado del runtime (replay de cinco respuestas archivadas contra cinco regeneraciones
directas); no cubre ejecuciones nuevas entre estados distintos del runtime, para las
cuales la sonda de ruido observó variabilidad no despreciable (|Δ| media 0,0616 entre
réplicas). Esto respalda repetibilidad condicional en el entorno congelado, no
determinismo general de nuevas generaciones.

No demuestra por sí solo reproducibilidad externa. Git no contiene archivos bajo
`data/raw`, `data/processed`, `data/chunks` ni `data/indices`; solo persiste la estadística
y configuración del corpus. Los manifiestos citados cubren modelos Transformers, pero no
registran un digest exacto del modelo Granite servido por Ollama. Un tercero necesita el
artefacto del corpus (o una lista completa con hashes), los índices o su reconstrucción,
acceso al modelo generativo con su digest Ollama exacto y el hardware/configuración
descritos en `REPRODUCE.md`.

## Fuentes de cierre

- `paper/summer_ablation_log.md`, entradas 25, 27, 28 y 28c.
- `data/evaluation/README.md`, `test_queries.json` y `test_queries_removed_log.json`.
- `output/audit/claim_audit_sample_v4_meta.json`, `gold_v4_analysis.json`,
  `provider_coverage_probe.json` y `descriptive_cis.json`.
- `experiments/results/exp19b_anchored_selector/equivalence__{hhem,small,base}.md`.
- `REPRODUCE.md` y `.github/workflows/ci.yml`.
