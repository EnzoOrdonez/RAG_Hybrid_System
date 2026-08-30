# PROMPT MAESTRO PARA CODEX — cierre documental post-revisión externa (2026-08-29)

```
CONTEXTO. Repo C:\Users\enziz\projects\hybrid-rag-system, rama summer/taxonomia-759.
Una revision externa (output/audit/revision_chatgpt_2026-08-29.txt, LEELA COMPLETA PRIMERO)
marco 4 puntos bloqueantes y varios importantes. La decision del proyecto: aceptarlos en
version DOCUMENTAL y de computo ligero; NO se reabre ningun experimento ni pipeline.
Todo lo de abajo usa solo datos ya persistidos. Español en los docs.

T1. Reescritura de docs/SECCION_VALIDACION_HUMANA.md (edita el borrador existente):
  - Sustituir "gold humano" por "referencia humana piloto" en todo el documento; la
    version PRE-adjudicacion queda como principal; la adjudicacion se presenta como
    construccion de una version reconciliada, NO como aumento de confiabilidad.
  - Reemplazar "240 juicios" por una tabla de 3 subconjuntos (proposito, poblacion,
    evidencia visible, muestreo, n).
  - Renombrar "sesgo de evidencia" a "sensibilidad al conjunto de evidencia" y aplicar
    TODAS las reformulaciones de la tabla D del dictamen que toquen a este documento
    (unsupported@τ / human-judged / externally incorrect; κ no estima nivel real;
    selector = alineacion HHEM, no mejora de fidelidad; TOST con margen preespecificado).
  - Añadir nota: κ test-retest mide confiabilidad INTRA-anotador; Landis-Koch "fair"
    citado como rotulo arbitrario, no como sello; citar Artstein & Poesio 2008 como
    advertencia de interpretacion (ambas referencias ya estan en el dictamen).

T2. scripts/compute_descriptive_cis.py (nuevo) + tests sinteticos. Calcula y escribe
  output/audit/descriptive_cis.{json,md}:
  - IC95 Wilson para la tasa de flips (16/50) y para el auto-acuerdo del retest (11/20);
    matriz de confusion 3x3 del retest (lee output/audit/gold_v4_tandaC_enzo_*.json +
    gold_v4_tandaC_resultado.json; NO leas gold_v4_juicios_enzo_*.json completo, solo
    via el resultado ya derivado).
  - McNemar exacto (binomial) para flips hacia 'correcto' vs otros: declara que es
    descriptivo, no confirmatorio.
  - Tasas de la taxonomia: reportar conteos NO ponderados (22/17/1) junto a las tasas HT
    ya publicadas, con nota de que con Kish 27.4 el 2% de 'dudoso' depende de ~1 caso.
  - Sensibilidad de tau: a partir de las probabilidades YA PERSISTIDAS que uso
    compute_exp18_unsupported_taxonomy.py, conteo de claims bajo tau 0.4/0.5/0.6
    (solo conteos; descriptivo). Si no son accesibles sin recomputar, declara bloqueo.

T3. docs/FICHA_EXPERIMENTAL.md (nuevo, media pagina cada tabla):
  - Tabla de unidades/estimandos: unidad de generacion, extraccion, verificacion,
    muestreo humano, unidad de cada contraste (lee paper/summer_ablation_log.md entradas
    27/28 y output/audit/claim_audit_sample_v4_meta.json para los n reales; NO inventes
    numeros). Incluir la salvedad: la inferencia a nivel claim puede subestimar
    incertidumbre por anidamiento en queries; si n=186 del contraste exp19b es a nivel
    query, dilo explicitamente con la fuente.
  - Tabla de familia BH: hipotesis, metrica, direccion, p crudo, p ajustado, IC, efecto,
    preregistrado/exploratorio (de las entradas del ledger).
  - Ficha de dataset: construccion de las 188/194 queries, exclusiones, cuantas son
    multi-nube (usa output/audit/provider_coverage_probe.json y test_queries.json;
    cuenta con codigo, no a ojo).
  - Parrafo de reproducibilidad externa: lo que YA existe (lockfile, manifest de modelos,
    compuerta de replay, CI) y lo que un tercero necesitaria (corpus, digest del modelo);
    sin afirmar mas de lo que los archivos prueban.

T4. Auditoria descriptiva del extractor de claims (amenaza a validez de constructo):
  - Toma 15 respuestas YA GENERADAS de experiments/results/ (declina cuales; seed 42),
    compara respuesta vs claims extraidos persistidos, y reporta en
    output/audit/extractor_audit_2026-08-29.md: cobertura (afirmaciones omitidas),
    atomicidad (splits/merges), claims espurios o degenerados (conteo con 2-3 ejemplos
    literales de cada tipo), conservacion de negacion/numeros. DECLARADO: auditoria por
    LLM, descriptiva, no gold; queda como amenaza explicita + evidencia acotada.
  - Si los claims extraidos no estan persistidos por respuesta, declara bloqueo.

PROHIBIDO: push; cambiar de rama; editar analyze_gold_v4.py ni
compute_exp15_ensemble_sweep.py; modificar CSV de output/audit; re-ejecutar experimentos;
GPU/Ollama/NLI/HHEM; inventar cifras (todo numero con fuente en archivo real).

CIERRE: corre la suite segura
  & "C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pytest -q -m "not slow and not gpu"
Commits atomicos por tarea, SIN push. REPORTE FINAL: Hecho / Auditoria (git status +
git log --oneline -10) / Bloqueos / Falta.
```
