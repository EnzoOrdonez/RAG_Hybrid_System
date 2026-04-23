# Auditoría CloudRAG — Hallazgos consolidados

**Objetivo**: detectar desalineaciones entre la implementación y la literatura, y entre la implementación y los claims del paper. Zero-trust.

**Convención**:
- 🔴 **BUG** — afecta resultados, corregir antes de submission
- 🟡 **AMBIGÜEDAD** — no rompe resultados pero afecta cómo se describe el trabajo
- 🟢 **NO-ISSUE** — defendible con disclaimer en Limitations

---

## Módulo 1: `HybridIndex` (fusión)

### 1.1 RRF fusion — ✓ correcta
Implementación coincide con Cormack, Clarke & Büttcher (SIGIR 2009). No tocar.

### 1.2 Linear fusion — 🟢 correcta con trade-offs
- Usa min-max normalization. Montague & Aslam (2001) recomiendan z-score. Mencionar como limitación.
- Default 0.0 para docs ausentes post-normalización: consistente con mainstream (Haystack, LlamaIndex).

### 1.3 `top_k_candidates = 50` — 🟡 decisión pendiente
- Literatura usa 100-1000 antes de reranking (DPR, ColBERT, BlendedRAG).
- **Acción**: auditar `retrieval_metrics.py` para ver si se reportó Recall@50. Si ≥ 0.95, defendible. Si < 0.90, re-correr con top-100.

### 1.4 `alpha` ignorado en RRF — 🟢 code smell
- API acepta alpha pero RRF lo ignora silenciosamente.
- **Acción**: cleanup cosmético post-auditoría. Resultado experimental no afectado.

---

## Módulo 2: `QueryProcessor`

### 2.1 Expansión asimétrica BM25-only — ✓ alineado con literatura
- Zhao et al. (2024): expansión ayuda a BM25, degrada dense. Tu diseño respeta esto.
- **Destacar como contribución explícita en Methodology.**

### 2.2 Expansión condicional por query_type — ✓ diseño sensato
- Cross-provider expansion solo para CROSS_CLOUD y CONCEPTUAL.
- **Documentar la priorización de clasificación** (cross_cloud > procedural) en el paper.

### 2.3 Flag 3 — keywords de provider demasiado laxos — 🟢
- `"microsoft"`, `"amazon"` pueden disparar falsos positivos fuera del corpus cloud.
- Corpus actual es inmune; fuera del dominio se rompe.
- **Acción**: 1 oración en Limitations.

### 2.4 Flag 5 — `enable_terminology_normalization` NO apaga acrónimos — 🔴 CRÍTICO
- Líneas 231-234 de `query_processor.py` expanden acrónimos siempre, ignorando la bandera.
- **Si tu ablation "sin normalización" reporta un delta, ese delta es incorrecto** porque los acrónimos seguían expandiéndose.
- **Acción**: auditar `experiment_configs.py` + runs de ablation. Dos caminos:
  - (a) Renombrar la ablation como "sin expansión cross-provider" (honesto pero requiere actualizar claims).
  - (b) Re-correr la ablation con acrónimos apagados también (riguroso, agrega tiempo).
- **Bloqueante para el claim de +16.8% hasta que se resuelva.**

### 2.5 Flag 7 — sin query rewriting para dense — 🟢
- No se usa HyDE ni query2doc.
- **Defensa proactiva en Methodology**: cita latencia y reproducibilidad zero-shot. Condicional a que la latency table muestre sub-segundo end-to-end.

### 2.6 Flag 9 — fallo silencioso si falta YAML — 🟢
- Si `terminology_mappings.yaml` no existe, usa `{}` sin warning.
- **Acción**: agregar `logger.warning` en cleanup post-auditoría. Mejora reproducibilidad.

---

---

## Módulo 3: `retrieval_metrics.py` + `scripts/compute_retrieval_metrics.py`

**CONTEXTO CRÍTICO**: Hay **dos** sistemas de métricas en el repo:
- `src/evaluation/retrieval_metrics.py` — módulo de métricas genérico invocado por `benchmark_runner.py`. Usa `k_values=[5, 10, 20]`. **NO computa P@1**.
- `scripts/compute_retrieval_metrics.py` — script independiente con su propia implementación. Usa `k_values=[1, 3, 5]`. **Acá se genera el P@1=0.930 del abstract.**

Esta duplicación ya es un problema de mantenibilidad y puede confundir a un reviewer que intente replicar desde el README.

### 3.1 FLAG 17 — 🔴 CRÍTICO — Auto-evaluación circular con el mismo cross-encoder

**El bug metodológico más grave encontrado hasta ahora.**

El sistema híbrido usa `ms-marco-MiniLM-L-12-v2` como **cross-encoder reranker** para reordenar sus candidatos finales. El script `compute_retrieval_metrics.py` usa **EL MISMO MODELO** como "oráculo de relevancia" para decidir qué chunks son relevantes:

```python
# Línea 45:
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Líneas 77-111: mismo modelo que el reranker del pipeline
# Líneas 455-456: chunks con score ≥ 0.0 del cross-encoder = "relevantes"
```

**Consecuencia directa**: el sistema híbrido está siendo evaluado por su propio reranker. Es equivalente a un examen donde el alumno escribe las preguntas y el profesor usa sus apuntes para corregir.

**Impacto en los claims del abstract**:
- P@1=0.93, MRR=0.942, NDCG@5=0.736 son todos **inflados** en magnitud.
- La *dirección* del hallazgo (Hybrid > Dense > BM25) probablemente es correcta y consistente con literatura.
- La *magnitud* del gap entre sistemas está sobreestimada porque el oráculo está sesgado hacia lo que el reranker prefiere.
- El Wilcoxon p<0.0001 también se exagera por la misma razón.

**Un reviewer de conferencia seria lo detecta en la primera lectura.** En IR/NLP esto se llama "self-evaluation bias" o "model-in-the-loop bias". Lakens (2013), Dehghani et al. (2021) lo documentan.

**Opciones para resolverlo**:
1. **Rigurosa (requerida en cualquier venue top)**: cambiar el oráculo a un modelo DIFERENTE. Candidatos razonables: `cross-encoder/ms-marco-electra-base` (más grande, misma familia MS MARCO), `BAAI/bge-reranker-large` (otra familia), o un LLM con prompting (GPT-4 Turbo, Claude) como juez.
2. **Ground truth humano parcial**: anotar manualmente ~50 queries × ~5 chunks = 250 pares. Un día de trabajo. Es lo que esperaría un revisor de SIGIR o ACL.
3. **Reframe narrativo**: cambiar el claim de "retrieval quality" a "agreement with reranker judgment" o "reranker-consistent recall". Más honesto pero baja el impacto.
4. **Disclosure fuerte**: mantener el número pero agregar un párrafo explícito de limitación y explicar por qué el resultado sigue siendo informativo.

**Mi recomendación para LACCI 2026 (conferencia regional, deadline 5 semanas)**:
- **Opción 4 obligatoria en cualquier caso** (agregar disclosure explícito).
- **Opción 1 si hay tiempo** — swap del oráculo a `bge-reranker-large`. Toma 1-2 días de cómputo y re-genera métricas. Los números van a bajar pero quedan defendibles.
- **Opción 2 solo si alguna reviewer explícitamente lo pide** (poco probable en LACCI).

### 3.2 FLAG 18 — Relevance threshold = 0.0 es demasiado permisivo

Línea 44: `RELEVANCE_THRESHOLD = 0.0`

El cross-encoder ms-marco produce logits típicamente en rango [-10, +15]. Un score de 0.0 es el límite de decisión binario, no un umbral de "claramente relevante". Score=0.5 es "débilmente positivo". Score=10 es "fuertemente positivo".

Con threshold=0, el conjunto "relevante" se infla → inflation en precision y recall.

**Acción**: o bien subir threshold a valor más estricto (ej. 3.0), o hacer análisis de sensibilidad (reportar métricas en threshold=0, 2, 5). **Agrava Flag 17**.

### 3.3 FLAG 19 — Pool-based evaluation (TREC-style)

Líneas 382-408: solo se evalúan chunks que al menos UN sistema retrieveó. Chunks relevantes NO retrievados por ninguno de los 3 sistemas nunca se consideran → subestima el denominador de recall para el mejor sistema.

Es el estándar TREC, pero **hay que declararlo explícitamente en el paper**. Sawarkar et al. (BlendedRAG), Ma et al. (AgentRAG) lo mencionan en sus secciones de evaluación.

### 3.4 FLAG 20 — No se reporta cobertura del oráculo

No hay stat de cuántas queries terminaron con `relevant_set` vacío (ningún chunk superó threshold). Si >10% de las queries tienen set vacío, todas esas reciben 0 en todas las métricas y **bajan el promedio artificialmente**.

**Acción**: agregar al reporte "queries sin chunks relevantes (oráculo <threshold)": X/200" como statistic básico.

### 3.5 FLAG 21 — Cohen's d calculada con fórmula para muestras independientes en datos pareados

Líneas 192-195:
```python
pooled_std = np.sqrt((np.std(vals1) ** 2 + np.std(vals2) ** 2) / 2)
cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std
```

Las muestras SON pareadas (mismas queries, distintos sistemas), pero la fórmula usada es para muestras INDEPENDIENTES. Para datos pareados la fórmula correcta es:

$$d_z = \frac{\bar{D}}{s_D}$$

donde $s_D$ es la desviación estándar de las **diferencias** entre sistemas. Lakens (2013) es la referencia canónica.

**Impacto**: La d=0.626 del abstract probablemente está **subestimada** porque los sistemas están correlacionados (comparten queries). La d correcta para datos pareados tiende a ser mayor cuando hay correlación. Esto **favorece** ligeramente el paper si se corrige.

### 3.6 FLAG 22 — Wilcoxon excluye ceros con cutoff arbitrario

Líneas 198-202: si hay menos de 10 diferencias no-cero, devuelve p=1.0. El cutoff de 10 es arbitrario, no viene de la literatura. Para tus 200 queries no debería activarse, pero vale revisar si alguna comparación del experimento activa el fallback silencioso.

### 3.7 Verificación de los números del abstract

CSV `exp8_retrieval_metrics.csv` confirma los números del abstract son derivados del código, no fabricados:

| Sistema | P@1 | MRR | NDCG@5 |
|---|---|---|---|
| BM25 | 0.785 | 0.828 | 0.555 |
| Dense | 0.860 | 0.894 | 0.661 |
| **Hybrid (Propuesto)** | **0.930** | **0.942** | **0.736** |

Matches abstract. El problema NO es fabricación — es **metodología**.

---

## Módulo 4: `hallucination_detector.py` + `hallucination_metrics.py`

**Sostiene el claim +16.8% faithfulness del abstract.** Cadena de dependencia:
`Query → QueryProcessor (Flag 5) → Retrieval → LLM → Claim extraction (Flag 24) → NLI scoring (Flags 25, 26) → Faithfulness`
Cada eslabón tiene flags abiertos.

### 4.1 FLAG 23 — 🔴 `faithfulness = 1.0` con `claims = []`
- Líneas 136-149. Respuestas evasivas filtradas 100% por SKIP_PATTERNS reciben nota perfecta.
- BM25 con contexto insuficiente → LLM responde "based on the context..." → `claims=[]` → faithfulness=1.0.
- Hybrid con buen contexto → LLM se arriesga con claims → es penalizado por intentar.
- **El sesgo puede reducir el gap +16.8%, no ampliarlo.**
- **Acción bloqueante**: contar `total_claims=0` por condición en los CSVs. Si la distribución es desigual, el delta está contaminado.

### 4.2 FLAG 24 — 🔴 SKIP_PATTERNS eliminan claims factuales legítimos
- `^however`, `^additionally`, `^you can`, `^you should`, `^the.*context` filtran oraciones con claims verificables (ej. "You can enable encryption with `--encryption=AES256`").
- Literatura (FActScore, Min et al. 2023) muestra que regex-based claim extraction subestima hallucinations 30-40% vs LLM decomposition.
- **Acción**: disclaimer explícito en Methodology sobre claim-extraction method. Riguroso = re-correr con LLM-based decomposition (costoso).

### 4.3 FLAG 25 — 🟡 Thresholds `0.7/0.7` asumen softmax pero el modelo puede devolver logits
- `sentence-transformers CrossEncoder.predict()` con modelos NLI retorna logits por defecto en muchas versiones.
- Si son logits, 0.7 es ≈ probabilidad 66% post-softmax, mucho más laxo de lo que el nombre sugiere.
- **Acción bloqueante de 5 min**: correr `detector.nli_model.predict([("cats are animals", "cats exist")])` y ver si la suma de las 3 componentes es ≈ 1.0 (softmax) o arbitraria (logits).

### 4.4 FLAG 26 — 🔴 Orden de labels NLI asumido sin verificación en runtime
- Línea 282 asume `[contradiction, entailment, neutral]`.
- Si el orden real es `[entailment, neutral, contradiction]` (común en BART-MNLI), **todo se invierte**: supported ↔ contradicted.
- `nli-deberta-v3-small` probablemente usa el orden declarado, pero la asunción no está testeada.
- **Acción bloqueante**: correr un par obvio ("París es la capital de Francia" vs "París es la capital de Alemania") y verificar qué índice es alto/bajo. 2 minutos.

### 4.5 FLAG 27 — 🟡 Evidencia de un solo chunk por claim
- Líneas 281-299: solo guarda el mejor single-chunk match.
- Claims compuestos (evidencia en chunks A+B) pierden crédito.
- Sistema híbrido devuelve más chunks → paradójicamente es más castigado que el baseline en claims compuestos.
- Estado del arte (AttributedQA, RARR) concatena top-k chunks antes del NLI.
- **Acción**: defendible como limitación, no bloqueante.

### 4.6 FLAG 28 — 🟢 CERRADO — Keyword fallback tiene rama no-op, no se activó
- Líneas 368-374: `elif best_overlap >= 0.2: status = "unsupported"` es idéntico al else. No existe categoría `partial`.
- **Verificación 2026-04-20**: grep logs por `NLI model unavailable` — 0 matches en logs reales. Fallback nunca se activó. Solo cosmético.

### 4.7 FLAG 29 — 🟢 CERRADO — No se encontraron crashes de hallucination
- **Verificación 2026-04-20**: grep logs por `Hallucination detection failed` — 0 matches en logs reales. Nunca se enmascaró excepción. Queda solo como riesgo latente para runs futuros.

### 4.8 FLAG 30 — 🟢 `nli_score` redondeado a 4 decimales
- Ruido menor en análisis de sensibilidad. No relevante para el claim principal.

---

## Módulo 5: `cross_encoder_reranker.py`

**Evidencia directa de Flag 17** (el mismo modelo como reranker Y oráculo):
- `experiments/results/exp8/retrieval_metrics.json`: `"model": "cross-encoder/ms-marco-MiniLM-L-12-v2"` (oráculo).
- `benchmark.log` l.4495: `Reranked 50 candidates in 13.050s (model=ms-marco-mini-12)` (pipeline).
- `experiment_configs.py`: 13 ocurrencias de `reranker="cross-encoder/ms-marco-MiniLM-L-12-v2"`.
- **Flag 17 confirmado con logs, no hipotético.**

### 5.1 FLAG 31 — 🟡 Inconsistencia config vs experiments vs README
- `config/config.yaml:118` default L-6.
- Clase `CrossEncoderReranker` línea 35 default L-6.
- `experiment_configs.py` override a L-12 en todos los runs del paper.
- README declara L-12 como modelo oficial.
- **Un reviewer que clone y corra defaults no reproduce el paper.**
- **Acción**: cambiar default del código a `ms-marco-mini-12`. Una línea.

### 5.2 FLAG 32 — 🟡 Mutación del objeto candidato: `retrieval_method += "+rerank"`
- Línea 111: si mismos candidatos pasan por dos rerankers distintos (ablation exp4), la segunda pasada arrastra `"bm25+rerank+rerank"`.
- **Bug potencial en exp4** (compara L-6, L-12, bge-reranker-large, no-reranker).
- **Acción**: auditar `exp4/results.json` y el driver que lo ejecuta. Si los candidatos se re-instancian por condición, no hay problema. Si se reciclan, exp4 está corrompido y hay que re-correr.

### 5.3 FLAG 33 — 🟢 Sin truncación explícita
- CrossEncoder trunca silenciosamente a 512 tokens. Chunks cloud con código suelen excederlo.
- **Acción**: contar cuántos chunks exceden 512 tokens; si >10%, disclaimer en Methodology.

### 5.4 FLAG 34 — 🟢 Ordering `[query, text]` correcto
- MS MARCO cross-encoders se entrenaron con `(query, passage)`. OK.
- Mencionar en Methodology como decisión deliberada, no cambiar código.

### 5.5 FLAG 35 — 🟢 Sin caching de scores
- No afecta resultados. Puede mejorar tiempo de runs futuros. Cosmético.

### 5.6 FLAG 36 — 🟢 batch_size parametrizado correctamente
- OK.

### 5.7 FLAG 37 — 🟢 top_k=5 default consistente con evaluación @5
- OK.

---

## Módulo 6: `terminology_normalizer.py` + `config/terminology_mappings.yaml`

**Este módulo materializa el claim cross-cloud del paper.** Escribe metadata en JSON de documentos durante preprocessing.

### 6.1 FLAG 38 — 🟡 Colisiones silenciosas de términos (last-write-wins)
- `VPC`, `Cloud Run`, `Vertex AI`, `Azure Key Vault` aparecen en múltiples conceptos. Líneas 55-57 sobrescriben silenciosamente sin warning.
- Concepto asociado depende del orden del YAML, no de diseño.
- **Acción**: permitir multi-concept o al menos log warning; declarar política en Methodology.

### 6.2 FLAG 39 — 🔴 Bug: `K8s`, `gRPC` nunca se detectan como acrónimos
- Línea 132: `upper_term = term.upper()`. YAML guarda claves con case mixto (`K8s`, `gRPC`).
- `"k8s".upper() == "K8S"` ≠ clave YAML `"K8s"` → lookup falla.
- Fix: normalizar claves del acronym dict a upper al cargar. Una línea.

### 6.3 FLAG 40 — 🟢 REVERTIDO tras Módulo 7 — Enriquecimiento NO se consume en retrieval
- Hipótesis inicial: `normalize_document` escribe `terminology.{...}` en JSON (líneas 180-184) y contamina todas las ablations.
- **Verificación Módulo 7 (2026-04-22)**:
  1. `chunkers/*.py` no leen `terminology`/`normalized_terms`/`cross_cloud` del documento source.
  2. Ningún JSON en `data/` contiene esos campos en los chunks.
  3. `HybridIndex.build` línea 44 pasa `chunk["text"]` crudo. BM25 y FAISS indexan solo ese campo.
- **Conclusión**: el output de `normalize_document` queda huérfano. El `TerminologyNormalizer` es **funcionalmente código muerto** en el retrieval principal. El único mecanismo activo de normalización cross-cloud en runtime es la expansión del `QueryProcessor` (Flag 5).
- **Nueva FLAG 40b — 🟡**: si el paper describe `TerminologyNormalizer` como parte de la contribución arquitectónica, está describiendo un pipeline que no se ejecuta. Revisar Methodology/Architecture para no sobrevender el módulo.
- **Efecto neto sobre el claim +16.8%**: Flag 5 sigue siendo la única amenaza real; Flag 40 original está cerrado.

### 6.4 FLAG 41 — 🟢 Inconsistencia con QueryProcessor
- Acá (línea 24) crash si YAML falta; en QueryProcessor silencia con `{}`.
- Code smell, no afecta resultados.

### 6.5 FLAG 42 — 🟢 `get_query_expansion` incluye genéricos vs. `_analyze_text` que los excluye
- Inconsistencia interna. Verificado: `get_query_expansion` solo se llama en tests (`tests/test_phase3_verification.py:294`). Código muerto en runtime. Cosmético.

### 6.6 FLAG 43 — 🟡 Mutación in-place en `process_chunks`
- Retorna lista pero muta objetos. Side effect no documentado.
- `normalize_chunk` solo invocado desde `scripts/resolve_gaps.py:231`, post-proceso. Impacto en runs principales: nulo. Defecto de diseño, no de datos.

### 6.7 FLAG 44 — 🟢 CERRADO — Inverted index no consumido
- Verificado 2026-04-22: ni BM25 ni FAISS ni los chunkers leen `inverted_index`. Solo se guarda a disco desde `resolve_gaps.py:255` como artifact de debug.

---

## Módulo 7: `bm25_index.py`

**Confirma Flag 44 y cierra Flag 40.** BM25 indexa solo `chunk["text"]` crudo.

### 7.1 FLAG 45 — 🟢 Sin stemming ni lemmatization
- BM25 plano: "databases" ≠ "database".
- Anserini/Lucene usa Porter stemmer por defecto.
- **Acción**: declarar en Methodology como trade-off simplicidad/reproducibilidad.

### 7.2 FLAG 46 — 🟢 Tokenizer pierde puntos decimales y underscores
- `re.findall(r'[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]', text)`.
- `v1.2` → `v1`, `2`. `cloud_run` → `cloud`, `run`.
- Impacto bajo; otros tokens compensan.

### 7.3 FLAG 47 — 🟢 `rank_bm25.BM25Okapi` en vez de Anserini/Lucene
- Común en RAG, aceptable. Mencionar trade-off en Methodology si reviewer duro.

### 7.4 FLAG 48 — 🟡 STOPWORDS remueve `in`, `on`, `out`, `up`, `down`, `over`, `under`
- `in-memory database` → `["memory", "database"]` (pierde "in").
- `on-premise`, `out-of-band` similar.
- **Acción opcional**: mover esos términos a KEEP_TERMS si compound técnicos son frecuentes en el corpus.

### 7.5 FLAG 49 — 🟢 K1=1.2, b=0.75 canónicos (Robertson & Walker 1994)
- OK. Declarar en Methodology.

### 7.6 FLAG 50 — 🟢 Scores ≤ 0 filtrados
- Línea 116 correcto.

### 7.7 FLAG 51 — 🟢 KEEP_TERMS coherente
- Solo protege acrónimos que YA están en STOPWORDS. Lista consistente.

---

## Módulo 8: `faiss_index.py`

**Evidencia forense** (`data/indices/faiss_bge-large_adaptive_500.mapping.json`):
- `index_type: "FlatIP"`
- `total_vectors: 46318`
- `dimension: 1024`

### 8.1 FLAG 52 — 🔴 Mismatch paper vs código: "IVF-PQ" declarado, FlatIP real
- `paper/overleaf_ready/main.tex:152` (placeholder azul): "FAISS IVF-PQ index".
- Repo no contiene `IndexIVFPQ` en ningún archivo. `faiss_index.py` solo implementa `IndexFlatIP` (n<50K) o `IndexIVFFlat` (n≥50K).
- Corpus real = 46,318 → rama FlatIP (exact search).
- **Fix trivial pre-submission**: reemplazar "IVF-PQ" por "IndexFlatIP (exact inner product with L2-normalized embeddings)" en main.tex.

### 8.2 FLAG 53 — 🟢 Dirección del mismatch favorece el paper
- FlatIP es exact (100% recall del vector search); IVF-PQ aproximado (pierde 1-5% recall).
- Tus números son con el método más preciso. Al corregir la descripción, las métricas se mantienen.

### 8.3 FLAG 54 — 🟢 L2 normalization + IP = cosine
- Líneas 51-53 correctas. Declarar en Methodology.

### 8.4 FLAG 55 — 🟢 Threshold n<50K para FlatIP
- Corpus 46K, a 3.7K del threshold. Borderline.

### 8.5 FLAG 56 — 🟡 Números no proyectan a >50K vectores
- Si el corpus crece, el índice cambia a IVFFlat automáticamente. Recall cae 1-5%.
- **Acción**: declarar en Limitations que las métricas aplican al tamaño actual del corpus.

### 8.6 FLAG 57 — 🟢 Filtrado de índices negativos (FAISS -1)
- OK.

### 8.7 FLAG 58 — 🟢 Assertions de dimensión
- OK.

### 8.8 FLAG 59 — 🟢 `nlist = min(nlist, n // 10)`
- Previene degeneración.

### 8.9 FLAG 60 — 🟢 Query normalization por norma total
- Para single query correcto. Batch no soportado por diseño.

---

## Módulo 9: `multidimensional_scorer.py`

**Scope**: 137 líneas. Combina cross-encoder + recency + source_quality + MMR diversity.
Usado en `rag_pipeline.py:159` y activado por `experiment_configs.py:599` (`multidimensional_scoring=True`).

### 9.1 FLAG 61 — 🔴 `diversity` NO es un peso, es un switch ON/OFF
- `DEFAULT_WEIGHTS` declara 4 claves que suman 1.0 (cross_encoder=0.6, recency=0.1, source_quality=0.1, diversity=0.2).
- Líneas 71-75 combinan SOLO 3 términos (suman 0.8). El 20% de `diversity` no entra al score compuesto.
- Línea 77: `if weights.get("diversity", 0) > 0` — `diversity` se usa como boolean.
- **Consecuencia**: scores finales sub-normalizados; cualquier descripción del método con "4 pesos combinados" en el paper sería falsa.
- **Fix**: o sumar `weights["diversity"] * mmr_diversity_term` al compuesto, o renombrar a `use_diversity: bool`.

### 9.2 FLAG 62 — 🔴 Mutación destructiva de `candidate.score`
- Línea 71: `candidate.score = (weights["cross_encoder"] * norm_ce + …)`.
- `.score` cambia de semántica (raw CE) a multidimensional sin preservar el original.
- Downstream que asuma `.score == ce_score` queda roto silenciosamente.
- Segunda pasada normaliza scores ya normalizados → outputs no idempotentes.
- **Fix**: guardar `candidate.raw_ce_score` antes de mutar.

### 9.3 FLAG 63 — 🟡 `recency=0.5` por default en docs sin fecha o malformada
- Línea 87: `if not last_updated: return 0.5`.
- Línea 95: `except ValueError, TypeError: return 0.5` (fechas mal formateadas invisibles).
- Si cobertura de `last_updated` es desigual, la señal recency se nula hacia el prior 0.5.
- **Acción**: loggear coverage de `last_updated` en el corpus; reportar % de candidatos que reciben el default.

### 9.4 FLAG 64 — 🔴 `source_quality=1.0` cuando falta `doc_type`
- Línea 70: `SOURCE_QUALITY.get(getattr(candidate, "doc_type", "guide"), 0.5)`.
- Si el candidato no tiene `doc_type`, se usa `"guide"` como default → `quality = 1.0` (máxima).
- Chunks pobres en metadata reciben calidad máxima = sesgo sistemático hacia documentos sin tipar.
- El fallback 0.5 solo aplica para tipos desconocidos, caso raro.
- **Fix trivial**: `getattr(candidate, "doc_type", "unknown")` con `SOURCE_QUALITY.get(..., 0.5)`.

### 9.5 FLAG 65 — 🟡 MMR `np.dot` sin re-normalización explícita
- Línea 119: `similarity = float(np.dot(doc_embeddings[idx], doc_embeddings[selected_idx]))`.
- Asume que `embedding_manager.embed_documents` devuelve L2-normalized para que dot == cosine.
- Verificar contrato. Si no se normaliza, MMR opera en escala arbitraria.

### 9.6 FLAG 66 — 🟡 MMR re-embebe chunks en query-time (latencia)
- Línea 105: `self.embedding_manager.embed_documents(doc_texts, ...)`.
- Los chunks ya tienen embedding en FAISS pero `MultidimensionalScorer` los recalcula.
- Penalización de latencia no reportada en el paper.
- **Acción**: medir overhead de MMR en ablation si hay tiempo.

### 9.7 FLAG 67 — 🟡 Rangos asimétricos en MMR favorecen diversidad
- `relevance = candidates[idx].score` en [~0.05, ~0.8] tras normalizar (Flag 61).
- `similarity` en [-1, 1].
- `mmr_score = 0.7 * relevance − 0.3 * similarity`.
- Un doc redundante pierde 0.3 × 1.0 = 0.3; un doc relevante ganando desde ~0.5 a ~0.8 = +0.2.
- **MMR castiga redundancia más de lo que premia relevancia** en este rango.
- `λ=0.7` es canónico (Carbonell & Goldstein 1998) pero el desbalance de escalas distorsiona la intención.

### 9.8 FLAG 68 — 🟢 Fallback silencioso a score sort si MMR falla
- Líneas 133-136. `logger.warning` correcto.
- Si MMR falla por razón sistemática (ej. `embedding_manager=None`) todos los runs "multidimensional" degradan a cross-encoder puro sin aparecer en métricas.
- **Acción**: contar warnings `MMR failed` en benchmark.log post-run.

---

## Módulo 10: `experiments/results/exp4/` (Reranker Model Comparison)

**Archivos**: `results.json` (22,432 líneas, 800 queries totales), `aggregated_metrics.json` (93 líneas solo latencias).
**Hypothesis declarada**: "Cross-encoder reranking improves NDCG@5 by >= 15%".

### 10.1 FLAG 69 — 🔴🔴 CATASTRÓFICO: exp4 NO tiene retrieval_metrics en ninguna query
- 800/800 queries con `retrieval_metrics: {}`, `generation_metrics: {}`, `hallucination_metrics: {}`.
- `aggregated_metrics.json` solo contiene `lat_*` (latencias). Cero métricas de calidad.
- **Sin NDCG/MRR/Recall/Precision** derivables de estos archivos.

### 10.2 FLAG 70 — 🔴🔴 `relevant_ids=[]` en 200/200 queries por config
- No hay ground truth por query en exp4. No es recuperable post-hoc desde este JSON.

### 10.3 FLAG 71 — 🔴🔴 Hipótesis inverificable con los datos
- Hipótesis del experimento declara "NDCG@5 improves ≥ 15%" pero no hay NDCG en ningún lado.
- Si el paper cita una tabla de NDCG@5 por reranker desde exp4, los números **no vienen de este archivo**. Por rastrear origen o reconocer como dato faltante.

### 10.4 FLAG 72 — 🔴 Latencia de generación contaminada por caché parcial
Conteo desde `results.json` (umbral `generation_ms < 10ms` = cached):

| Config | Cached | Real | Mean Real |
|---|---|---|---|
| no_reranker | **200/200** | 0 | — (todo cached, 0.256ms) |
| rerank_mini6 | 4/200 | 196 | 27.8 s |
| rerank_mini12 | 49/200 | 151 | 26.2 s |
| rerank_bge_large | 1/200 | 199 | 31.3 s |

- Cualquier comparación de `lat_total_ms` entre configs en `aggregated_metrics.json` está contaminada.
- El delta "no_reranker 0.43s vs rerank_mini12 18.8s" NO es el costo del reranker, es reranker+regeneración real vs caché.
- **Acción**: si se reporta latencia en el paper, re-correr con caché desactivado o reportar solo latencia `lat_reranking_ms` (aislada y limpia).

### 10.5 FLAG 73 — 🟡 Overlap retrieved_ids bajo entre configs
- Primeras 5 queries: overlap no_reranker vs rerank_mini12 = 0-3 de 5 ids.
- Esperable (rerank reordena), pero 0/5 sugiere que rerank expulsa del top-5 candidatos que no_reranker tenía.
- **Acción opcional**: verificar overlap en top-50 (pre-rerank pool). Si top-50 también diverge → hay un bug en el retriever.

### 10.6 FLAG 74 — 🟢 Flag 32 CERRADO COMO FALSO POSITIVO
- Búsqueda en `exp3/4/5/6/7/8/8b/results.json`: 0 ocurrencias de `retrieval_method` o `+rerank`.
- La mutación `retrieval_method += "+rerank"` en `cross_encoder_reranker.py` vive en memoria pero no se serializa.
- Sin contaminación en resultados. Code smell pero no afecta datos.
- **Flag 32 anteriormente 🟡 → ahora 🟢 CERRADO**.

### 10.7 FLAG 75 — 🔴 `aggregated_metrics.json` reporta solo latencia
- Si la elección de `rerank_mini12` como reranker default en el paper se justifica con exp4, esa decisión **no tiene datos de calidad que la respalden**.
- Solo hay latencias (mini12 es 1.4× más lento que mini6, 5× más rápido que bge_large).
- Sin NDCG/MRR, no sabes si mini12 vale su costo adicional sobre mini6.

---

## Módulo 11: Inventario de métricas de calidad por experimento

### Mapa de NDCG/MRR/Recall por experimento
| Exp | Descripción | retrieval_metrics.json | NDCG/MRR | Solo hallucination |
|---|---|---|---|---|
| exp3 | BM25 vs Dense vs RRF vs Linear@α | ❌ | ❌ | ✅ |
| exp4 | Reranker comparison | ❌ | ❌ | ✅ |
| exp5 | LLM comparison | ❌ | ❌ | ✅ |
| exp6 | Ablations | ❌ | ❌ | ✅ |
| exp7 | Cross-cloud normalization | ❌ | ❌ | ✅ |
| **exp8** | BM25/Dense/Hibrido | ✅ | ✅ | ✅ |
| **exp8b** | BM25/Dense/Hibrido repetición | ✅ (IDÉNTICO a exp8) | ✅ | ✅ (distinto) |

### Números de calidad disponibles (exp8/retrieval_metrics.json)
| Métrica | BM25 | Dense | Hibrido |
|---|---|---|---|
| NDCG@5 | 0.5544 | 0.6606 | 0.7362 |
| MRR | 0.8283 | 0.8937 | 0.9424 |
| Precision@1 | 0.785 | 0.860 | 0.930 |
| Precision@5 | 0.718 | 0.782 | 0.851 |
| Recall@5 | 0.368 | 0.424 | 0.472 |

### Statistical tests (Cohen's d + Wilcoxon)
| Comparación | NDCG@5 d | Wilcoxon p | meets_thesis |
|---|---|---|---|
| BM25 vs Dense | -0.353 | 3.21e-4 | **false** |
| **BM25 vs Hibrido** | **-0.626** | **1.83e-11** | **true** ✅ |
| Dense vs Hibrido | -0.282 | 8.74e-4 | **false** |

**Única comparación que pasa el thesis_threshold: BM25 vs Hibrido en NDCG@5**. Todas las demás (MRR, Precision@5, avg_score@5) fallan el threshold.

### 11.1 FLAG 76 — 🔴 Solo exp8/exp8b tienen métricas de calidad
- 5 experimentos (exp3, exp4, exp5, exp6, exp7) carecen de NDCG/MRR/Recall/Precision.
- **exp4**: hipótesis "rerank NDCG@5 ≥ 15%" sin datos.
- **exp6**: ablations sin calidad de retrieval, solo faithfulness.
- **exp3**: elección de α en linear fusion no justificada con NDCG.
- **Acción**: si el paper reporta NDCG de alguna de estas, **los números no existen**. Re-correr o remover.

### 11.2 FLAG 77 — 🔴🔴 exp8 y exp8b tienen retrieval_metrics IDÉNTICOS (byte a byte)
- Sección `systems/` y `statistical_tests/` coinciden exactamente en ambos.
- Solo difieren en `hall_faithfulness_mean` (exp8 Hibrido 0.368 vs exp8b 0.514).
- Implica que **el retrieval es determinístico** (o fue reutilizado entre corridas).
- Si el paper presenta exp8 y exp8b como "corridas independientes para robustez", en retrieval **no hay variación**.
- **Acción**: declarar en Methodology que retrieval es determinístico; exp8b solo reevalúa la generación.

### 11.3 FLAG 78 — 🔴🔴 CATASTRÓFICO: `avg_score@5` contaminado por Flag 17
- Oracle = `cross-encoder/ms-marco-MiniLM-L-12-v2` (mismo modelo que el reranker del Hibrido).
- `avg_score@5` mide el score promedio del oracle sobre el top-5 del sistema.
- El Hibrido usa ese mismo oracle para rerankear → su top-5 es **por definición** el que el oracle considera mejor.
- Números inflados por construcción. NO REPORTAR en el paper.

### 11.4 FLAG 79 — 🔴 BM25 vs Dense NO pasa thesis_threshold (d=-0.353)
- Si el paper afirma "Dense supera BM25" sin calificación, está declarando un tamaño de efecto que sus propios tests marcan como insuficiente.
- **Acción**: reportar d y aclarar que BM25 vs Dense es medium-small; el gap significativo es BM25 vs Hibrido.

### 11.5 FLAG 80 — 🟡 Dense vs Hibrido efecto pequeño (d=-0.282)
- Reviewer obvio: "¿vale el hibrido si Dense hace casi lo mismo?"
- Contraargumento válido: MRR (0.942 vs 0.894) y Precision@1 (0.93 vs 0.86) favorecen al Hibrido en calidad del primer resultado.
- **Acción**: preparar defensa anticipada en Discussion.

### 11.6 FLAG 81 — 🟡 Score distribution del oracle sospechosa (relevance_threshold=0.0)
- `mean=2.18, median=2.42, std=3.12, min=-10.9, max=9.2`.
- Con threshold=0.0, la mayoría de chunks pasan como "relevantes".
- BM25 precision@5=0.72 ya es alta porque el umbral es laxo.
- **Acción opcional**: sensitivity con threshold=1.0 o 2.0, verificar delta Hibrido-BM25.

### 11.7 FLAG 82 — 🟢 Flag 71 parcialmente cerrado
- Los NDCG@5 del paper **existen** — en exp8/retrieval_metrics.json.
- Para comparación de rerankers (exp4) siguen sin existir.
- Flag 71 degrada a 🟡 si el paper cita NDCG solo de exp8.

---

## Módulo 12: Contraste paper (`main.tex`) vs datos

### Tabla claim-por-claim

| Claim del paper | Ubicación | Datos | Veredicto |
|---|---|---|---|
| P@1=0.930, MRR=0.942, NDCG@5=0.736 | Abstract L65 | ✅ exp8 | Respaldado |
| BM25 P@1=0.785, Dense P@1=0.860 | Abstract L66 | ✅ exp8 | Respaldado |
| Cohen's d=0.626, p<0.0001 | Abstract L68 | ✅ solo BM25 vs Hibrido | 🟡 ambiguo |
| "outperforming ... statistical significance" | Abstract | ⚠️ d=-0.28 Dense-Hibrido | 🟡 ambiguo |
| +16.8% faithfulness norm | Abstract | ✅ exp7 0.292/0.250 | Respaldado |
| 300 queries | Abstract L65 | ❌ máx 229 (200+29) | 🔴 mismatch |
| FAISS IVF-PQ | Meth L152 | ❌ FlatIP | 🔴 Flag 52 |
| CE reranker L-12 | Meth L152 | ✅ pero = oracle | 🔴 Flag 17 |
| Ablation ΔNDCG@5 | Results V.B | ❌ exp6 sin NDCG | 🔴 CRISIS |
| "accuracy" LLMs | Results V.D | ⚠️ sin fuente | 🟡 |
| Reranker impact by noise | Discussion | ❌ sin datos | 🔴 figura fantasma |
| +14.5 pp / +7.0 pp P@1 | Conclusion | ✅ | Respaldado |

### 12.1 FLAG 83 — 🔴 Abstract declara 300 queries, solo hay 229 disponibles
- Abstract L65: "300-query expert-curated test set".
- Filesystem: exp3/4/5/6/8/8b = 200 queries; exp7 = 29.
- Gap: 71-100 queries sin evidencia.
- **Acción**: verificar existencia de test set adicional o corregir abstract a "200 queries (+ 29 cross-cloud specific)".

### 12.2 FLAG 84 — 🔴 Ablation study con ΔNDCG@5 imposible con datos actuales
- Results V.B L191 declara "Drop each component one at a time; report ΔNDCG@5 and ΔFaithfulness".
- exp6 (ablations) SIN `retrieval_metrics.json`. Solo tiene hallucination + latency.
- **Opciones**:
  - (a) Re-correr exp6 con instrumentación retrieval (~2 días).
  - (b) **Reescribir sección para reportar solo ΔFaithfulness** (datos disponibles).
  - (c) Usar exp8 BM25→Dense→Hibrido como "pseudo-ablation" (NO es drop-one-component estricto).
- **Recomendación**: (b). Mensaje válido con datos existentes.

### 12.3 FLAG 85 — 🔴 Figura "Reranker impact by noise level" sin datos
- Discussion L234-237 cita `fig_reranker_impact.png` + claim "reranker contribution is large when candidate pool is noisy; marginal otherwise".
- No hay experimento con dimensión noise level. exp4 solo compara modelos.
- **Acción**: verificar si `figures/fig_reranker_impact.png` existe, remover subsección si no.

### 12.4 FLAG 86 — 🔴 Cohen's d=0.626 del abstract aplica SOLO a BM25-Hibrido
- Abstract presenta un único d=0.626. Un reviewer asumirá que aplica a la comparación principal.
- Dense vs Hibrido: d=-0.282 (small).
- **Fix honesto**: "Cohen's d=0.626 over BM25; d=0.282 over dense-only (both Wilcoxon p<0.001)".

### 12.5 FLAG 87 — 🟡 "LLM accuracy" en Results V.D sin fuente
- Results L209 declara "accuracy, faithfulness, latency" para comparación LLM.
- exp5 tiene `hall_faithfulness`, `hall_hallucination_rate`, `lat_*`. **Sin accuracy.**
- **Acción**: clarificar qué es "accuracy" (¿=faithfulness? ¿exact match? ¿LLM-as-judge?).

### 12.6 FLAG 88 — 🟢 Números canónicos del abstract respaldados
- P@1/MRR/NDCG@5 = 0.930/0.942/0.736 = exp8 Hibrido.
- +16.8% norm = exp7 (0.292 − 0.250) / 0.250 exact.
- +14.5pp / +7.0pp = diferencias absolutas correctas.

### 12.7 FLAG 89 — 🟡 RRF vs linear fusion no declarado en exp8
- Abstract declara RRF. exp3 probó ambas (linear_03/05/07 y rrf).
- exp8 no declara en su JSON cuál usó.
- **Acción**: verificar `experiment_configs.py` que exp8 usa RRF, documentar.

---

## Verificaciones inmediatas realizadas (2026-04-20)

### ✅ VER-A — Grep de logs por crashes de hallucination detection
- `NLI model unavailable`: 0 matches en logs → Flag 28 cerrado.
- `Hallucination detection failed`: 0 matches en logs → Flag 29 cerrado.
- `NLI prediction failed for claim`: 0 matches → el NLI nunca falló por claim individual.

### ✅ VER-B — Evidencia directa de Flag 17
- Confirmado por 3 fuentes independientes (exp8 JSON, benchmark.log, experiment_configs.py).

### ⏳ VER-C — Smoke test NLI (Flags 25, 26) — bloqueado por proxy del sandbox
- Script dejado en `scripts/audit/smoke_test_nli.py`.
- Ejecución en sandbox de Cowork intentada 2026-04-20: sentence-transformers se instaló OK, pero el proxy del sandbox bloquea huggingface.co (`httpx.ProxyError: 403 Forbidden`). No se puede descargar el modelo desde aquí.
- Ejecutar local con `python scripts/audit/smoke_test_nli.py` (requiere red libre a huggingface.co).
- Revisa el bloque `MODEL CONFIG id2label` del output:
  - Si `index 0 = contradiction, index 1 = entailment, index 2 = neutral` → líneas 284-285 de `hallucination_detector.py` están correctas.
  - Si otro orden → patchear índices antes de confiar en cualquier número de faithfulness.
- Revisa la suma de los 3 scores por par:
  - Si ≈1.0 consistente → salida es softmax, threshold 0.7 es probabilidad.
  - Si arbitraria → salida es logits, threshold 0.7 es logit (mucho más laxo).

### ⏳ VER-D — Distribución de `total_claims=0` por sistema (Flag 23)
- Pendiente: contar respuestas con 0 claims por condición (BM25/Dense/Hybrid) en CSVs de exp8.
- Si la distribución es desigual entre sistemas, el delta +16.8% está sesgado direccionalmente.

---

## Pendientes de auditoría

- [x] `multidimensional_scorer.py` — cerrado (Flags 61-68)
- [x] `exp4/results.json` — Flag 32 CERRADO falso positivo. NUEVOS flags 69-75.
- [x] **Rastreo de NDCG@5** — existen solo en exp8/exp8b. Flags 76-82.
- [x] **main.tex cruzado con datos** — Flags 83-89 abiertos.
- [ ] **Decisión arquitectónica**: (a) re-correr exp6 con instrumentación NDCG, o (b) reescribir Results V.B como ΔFaithfulness only.
- [ ] **Decisión arquitectónica**: resolver "300 queries" vs 229 disponibles.
- [ ] Verificar si `figures/fig_reranker_impact.png` existe o hay que removerla.
- [ ] `statistical_analysis.py`
- [ ] `experiment_configs.py` completo (Flags 5, 89)
- [ ] Cruzar números del paper con exp8/retrieval_metrics.json — **HECHO**
- [ ] Correr `smoke_test_nli.py` local (resuelve Flags 25 y 26)
- [ ] Contar `total_claims=0` por sistema en CSVs (resuelve magnitud de Flag 23)
- [ ] Verificar si `embedding_manager.embed_documents` retorna L2-normalized (Flag 65)
- [ ] Grep de `"MMR failed"` en benchmark.log (Flag 68)
- [ ] Verificar overlap top-50 entre configs en exp4 (Flag 73)
- [ ] Sensitivity analysis relevance_threshold (Flag 81)
- [ ] Verificar si el paper presenta exp8b como "corrida independiente" (Flag 77)

---

## Resumen de severidad acumulada

| # | Flag | Severidad | ¿Bloquea submission? |
|---|---|---|---|
| 1.1 | RRF correcto | — | No |
| 1.2 | Linear min-max | 🟢 Disclosure | No |
| 1.3 | top_k=50 | 🟡 Pendiente de datos | Depende |
| 1.4 | alpha en RRF | 🟢 Cosmético | No |
| 2.1 | Expansión asimétrica | — | No (fortaleza) |
| 2.2 | Clasificación condicional | — | No (fortaleza) |
| 2.3 | Keywords laxos | 🟢 Limitations | No |
| 2.4 | Flag 5 acrónimos | 🔴 Descripción incorrecta | Requiere rename |
| 2.5 | Sin dense rewriting | 🟢 Defensa proactiva | No |
| 2.6 | YAML silencioso | 🟢 Cleanup | No |
| **3.1** | **Flag 17 auto-evaluación** | **🔴 METODOLÓGICO** | **SÍ — hay que mitigar** |
| 3.2 | Threshold=0 | 🟡 Combina con 17 | Mitigación con 17 |
| 3.3 | Pool TREC | 🟢 Disclosure | No |
| 3.4 | Cobertura oráculo | 🟢 Agregar stat | No |
| 3.5 | Cohen's d fórmula | 🟡 Recomputable | Recalcular (rápido) |
| 3.6 | Wilcoxon cutoff 10 | 🟢 Verificar | No |
| **4.1** | **Flag 23 claims vacíos = 1.0** | **🔴 Sesgo direccional** | **SÍ — recomputar** |
| **4.2** | **Flag 24 SKIP_PATTERNS** | **🔴 Subestima hallucinations** | **Disclaimer mínimo; re-decompose ideal** |
| 4.3 | Flag 25 threshold 0.7 | 🟡 Verificar softmax vs logits | Smoke test 5 min |
| **4.4** | **Flag 26 orden NLI labels** | **🔴 Si mal orden: todo invertido** | **SÍ — smoke test obligatorio** |
| 4.5 | Flag 27 single-chunk evidence | 🟡 Limitations | No |
| 4.6 | Flag 28 fallback no-op | 🟢 Cosmético | No |
| 4.7 | Flag 29 excepciones enmascaradas | 🟢 CERRADO (logs limpios) | No |
| 4.8 | Flag 30 redondeo nli_score | 🟢 Cosmético | No |
| 5.1 | Flag 31 default L-6 vs paper L-12 | 🟡 Reproducibilidad | Cambiar default |
| 5.2 | Flag 32 mutación retrieval_method | 🟡 Puede afectar exp4 | Auditar exp4 |
| 5.3 | Flag 33 truncación silenciosa 512 tok | 🟢 Disclaimer si >10% chunks exceden | No |
| 5.4 | Flag 34 ordering query/text | 🟢 Correcto | No |
| 5.5 | Flag 35 sin cache | 🟢 Cosmético | No |
| 5.6 | Flag 36 batch_size | 🟢 OK | No |
| 5.7 | Flag 37 top_k=5 default | 🟢 OK | No |
| 6.1 | Flag 38 colisiones de términos | 🟡 Determinístico por orden YAML | Declarar política |
| 6.2 | Flag 39 K8s/gRPC nunca detectados | 🔴 Bug pero impacto bajo | Fix de una línea |
| 6.3 | Flag 40 enriquecimiento persistente | 🟢 REVERTIDO tras Módulo 7 | No |
| 6.3b | Flag 40b normalizer es código muerto en runtime | 🟡 Paper describe pipeline inexistente | Revisar Architecture section |
| 6.4 | Flag 41 error handling inconsistente | 🟢 Code smell | No |
| 6.5 | Flag 42 get_query_expansion genéricos | 🟢 Código muerto en runtime | No |
| 6.6 | Flag 43 process_chunks muta in-place | 🟡 Defecto de diseño | No (no afecta runs) |
| 6.7 | Flag 44 inverted_index no consumido | 🟢 CERRADO | No |
| 7.1 | Flag 45 sin stemming | 🟢 Declarar | No |
| 7.2 | Flag 46 tokens decimales/underscore | 🟢 Bajo impacto | No |
| 7.3 | Flag 47 rank_bm25 vs Anserini | 🟢 Trade-off | No |
| 7.4 | Flag 48 stopwords in/on/out | 🟡 Compound pierden "in" | Opcional |
| 7.5 | Flag 49 k1=1.2, b=0.75 | 🟢 Canónicos | No |
| 7.6 | Flag 50 filtro score > 0 | 🟢 OK | No |
| 7.7 | Flag 51 KEEP_TERMS coherente | 🟢 OK | No |
| **8.1** | **Flag 52 IVF-PQ vs FlatIP** | **🔴 Mismatch paper/código** | **Fix trivial en main.tex** |
| 8.2 | Flag 53 dirección favorable | 🟢 Números se mantienen | No |
| 8.3 | Flag 54 L2+IP=cosine | 🟢 Declarar | No |
| 8.4 | Flag 55 threshold n<50K | 🟢 Borderline (corpus=46K) | No |
| 8.5 | Flag 56 no proyecta >50K | 🟡 Disclosure en Limitations | No |
| 8.6 | Flag 57 filtro idx=-1 | 🟢 OK | No |
| 8.7 | Flag 58 dim assertions | 🟢 OK | No |
| 8.8 | Flag 59 nlist clamp | 🟢 OK | No |
| 8.9 | Flag 60 query norm single | 🟢 OK | No |
| **9.1** | **Flag 61 diversity como switch no peso** | **🔴 Descripción del método inconsistente** | **Fix código o rename en paper** |
| **9.2** | **Flag 62 mutación de `.score`** | **🔴 Code smell + no idempotente** | **Fix código (guardar raw_ce)** |
| 9.3 | Flag 63 recency default 0.5 | 🟡 Nula la señal si corpus no fechado | Loggear coverage |
| **9.4** | **Flag 64 source_quality default 1.0** | **🔴 Sesgo a chunks sin metadata** | **Fix trivial (default a "unknown")** |
| 9.5 | Flag 65 MMR dot sin renorm | 🟡 Verificar contrato embed_documents | Verificar |
| 9.6 | Flag 66 MMR re-embed query-time | 🟡 Latencia no reportada | Medir si hay tiempo |
| 9.7 | Flag 67 rangos asimétricos MMR | 🟡 Castiga redundancia > premia relevancia | Declarar o re-escalar |
| 9.8 | Flag 68 MMR fallback silencioso | 🟢 Auditable vía grep logs | Grep post-run |
| **10.1** | **Flag 69 exp4 sin retrieval_metrics** | **🔴🔴 Catastrófico** | **SÍ — recuperar/recomputar** |
| **10.2** | **Flag 70 relevant_ids vacío 200/200** | **🔴🔴 Sin ground truth** | **SÍ — asociar truth set** |
| **10.3** | **Flag 71 hipótesis inverificable** | **🔴🔴 Si paper cita NDCG de exp4, no hay datos** | **SÍ — rastrear fuente real** |
| **10.4** | **Flag 72 caché parcial de generación** | **🔴 Total latency incomparable** | **Re-correr o reportar solo rerank_ms** |
| 10.5 | Flag 73 overlap retrieved_ids bajo | 🟡 Verificar top-50 pool | Opcional |
| 10.6 | Flag 74 Flag 32 falso positivo | 🟢 CERRADO | No |
| **10.7** | **Flag 75 aggregated solo latencia** | **🔴 Decisión de rerank sin calidad** | **Reabrir con NDCG/MRR** |
| **11.1** | **Flag 76 solo exp8/8b tienen NDCG** | **🔴 5 experimentos sin calidad** | **Re-correr o remover claims** |
| **11.2** | **Flag 77 exp8==exp8b retrieval** | **🔴🔴 Retrieval determinístico oculto** | **Declarar en Methodology** |
| **11.3** | **Flag 78 avg_score@5 contaminado por Flag 17** | **🔴🔴 Métrica auto-referencial** | **NO REPORTAR** |
| **11.4** | **Flag 79 BM25 vs Dense no pasa threshold** | **🔴 d=-0.35 medium-small** | **Reportar d honestamente** |
| 11.5 | Flag 80 Dense vs Hibrido d=-0.28 | 🟡 Reviewer risk | Defensa MRR/P@1 |
| 11.6 | Flag 81 relevance_threshold=0.0 laxo | 🟡 Sensitivity análisis | Opcional |
| 11.7 | Flag 82 Flag 71 degradado | 🟢 Números existen en exp8 | No |
| **12.1** | **Flag 83 300 queries sin soporte** | **🔴 Corregir abstract o encontrar test set** | **SÍ** |
| **12.2** | **Flag 84 Ablation ΔNDCG@5 sin datos** | **🔴 Crisis de datos** | **Reescribir o re-correr** |
| **12.3** | **Flag 85 fig_reranker_impact fantasma** | **🔴 Figura sin datos** | **Remover o producir datos** |
| **12.4** | **Flag 86 Cohen's d=0.626 ambiguo** | **🔴 Abstract impreciso** | **Aclarar en texto** |
| 12.5 | Flag 87 LLM "accuracy" sin fuente | 🟡 Clarificar | Definir |
| 12.6 | Flag 88 números canónicos OK | 🟢 | No |
| 12.7 | Flag 89 RRF vs linear no declarado | 🟡 Documentar | No |
| **13.1** | **Flag 90 5/8 figuras literalmente vacías** | **🔴🔴🔴 Evidencia visual del data crisis** | **Re-generar o remover** |
| **13.2** | **Flag 91 results_exporter usa `.get(col, 0)` en vez de fallar** | **🔴🔴 Oculta silenciosamente datos faltantes** | **Fix código: raise si falta key** |
| **13.3** | **Flag 92 fig_retrieval_metrics == fig_retrieval_metrics_exp8b byte-idénticos** | **🔴🔴 Duplicación disfrazada de comparación** | **Regenerar exp8b real o remover** |
| **13.4** | **Flag 93 table_retrieval_metrics == table_retrieval_metrics_exp8b byte-idénticos** | **🔴🔴 Misma duplicación en tabla** | **Como Flag 92** |
| **13.5** | **Flag 94 fig_llm_comparison es radar de 1 solo eje** | **🔴 Radar chart mal construido** | **Reescribir con ≥3 ejes o usar bar chart** |
| **13.6** | **Flag 95 fig_retrieval_metrics Avg Score overflow [0,1]** | **🔴 Escala visualmente engañosa** | **Normalizar o eje secundario** |
| 13.7 | Flag 96 7/9 tablas con ceros estructurales | 🔴 Mismo síntoma Flag 90 | Re-generar |
| 13.8 | Flag 97 fig_statistical_significance referenciado nunca generado | 🟡 Nunca se llama en `export_all` | Generar o remover de docstring |

## Módulo 13: Auditoría de figuras y tablas generadas

### Archivo: `paper/overleaf_ready/figures/`

**Inventario (verificación byte-a-byte y visual):**

| Artefacto | Estado | Fuente pretendida |
|-----------|--------|-------------------|
| `fig_chunking_heatmap.png` | No se verificó (no se referencia en main.tex) | exp1 |
| `fig_retrieval_comparison.png` | **🔴 VACÍA — 24 barras a cero** | exp3 `retrieval_metrics` |
| `fig_reranker_impact.png` | **🔴 VACÍA — 0 barras** | exp4 `retrieval_metrics` |
| `fig_ablation_waterfall.png` | **🔴 VACÍA — +0.000 en cada paso** | exp6 `retrieval_metrics` |
| `fig_llm_comparison.png` | 🟡 Radar mal construido (1 eje) | exp5 faithfulness |
| `fig_latency_breakdown.png` | 🟢 Con datos (única fuente viva: `aggregated_metrics.lat_*`) | exp8 |
| `fig_cross_cloud_improvement.png` | **🔴 VACÍA — Recall/NDCG/MRR en cero** | exp7 `retrieval_metrics` |
| `fig_end_to_end.png` | **🔴 VACÍA — 3 paneles, 0 barras** | exp8 `retrieval_metrics` |
| `fig_retrieval_metrics.png` | 🟢 Con datos exp8, pero Avg Score rompe eje [0,1] | exp8 |
| `fig_retrieval_metrics_exp8b.png` | **🔴 BYTE-IDÉNTICA a `fig_retrieval_metrics.png`** | supuestamente exp8b |
| `table_exp3.tex` | **🔴 Todos ceros** | exp3 |
| `table_exp4.tex` | **🔴 Todos ceros** | exp4 |
| `table_exp5.tex` | **🔴 Todos ceros** | exp5 |
| `table_exp6.tex` | **🔴 Todos ceros** | exp6 |
| `table_exp7.tex` | **🔴 Todos ceros** | exp7 |
| `table_exp8.tex` | **🔴 Todos ceros** (aunque exp8 sí tiene datos en `retrieval_metrics.json`) | exp8 |
| `table_exp8b.tex` | **🔴 Todos ceros** | exp8b |
| `table_retrieval_metrics.tex` | 🟢 Canónica exp8 completa | exp8 |
| `table_retrieval_metrics_exp8b.tex` | **🔴 BYTE-IDÉNTICA a `table_retrieval_metrics.tex`** — incluso el caption dice "(exp8, 200 queries)" | supuestamente exp8b |

**Conteo final:**
- Figuras vacías/rotas: **5 de 9** verificadas
- Tablas con ceros estructurales: **7 de 9**
- Duplicaciones byte-idénticas disfrazadas de comparación: **2 pares** (figura + tabla de exp8b)

### Flags

**Flag 90 — Cinco figuras literalmente vacías (🔴🔴🔴 CATASTRÓFICO)**

Ya está confirmado visualmente (`fig_retrieval_comparison`, `fig_reranker_impact`, `fig_ablation_waterfall`, `fig_cross_cloud_improvement`, `fig_end_to_end`): todas tienen ejes, títulos, leyendas y xticks correctos, pero ninguna barra. Esto no es un bug de matplotlib — es la evidencia visual de los Flags 69/70/75/76: `retrieval_metrics` no existe en los results.json de exp3/4/5/6/7 (sólo en exp8 hay datos, y aun así la tabla se genera en cero porque lee otro archivo — ver Flag 91).

La mitad de las figuras del paper son placeholders renderizados como PNG. No "datos preliminares"; no "pendiente de actualizar" — son gráficos que alguien ejecutó, vio vacíos, y guardó de todas formas en el repo. Si estas figuras llegan al PDF enviado a LACCI, el reviewer detecta el fraude sin salir del abstract.

**Flag 91 — `results_exporter.py` silencia datos faltantes con `.get(col, 0)` (🔴🔴)**

Línea 655 de `src/evaluation/results_exporter.py`:

```python
row[col] = data[config].get(col, 0)
```

Cuando `ret_ndcg@5_mean` no existe en `aggregated_metrics.json` (que es el caso en exp3–exp7 y aun en exp8, porque allí las métricas de calidad viven en otro archivo: `retrieval_metrics.json`), el código inserta silenciosamente `0` y continúa. Ningún warning, ninguna excepción. Esto es el mecanismo que produce las 7 tablas con ceros estructurales.

Lo mismo ocurre en `fig_end_to_end` (línea 540): `r.get("retrieval_metrics", {}).get(metric_key, 0)` — si `retrieval_metrics` está vacío, todo queda en cero y se dibuja un chart vacío sin errores.

Fix mínimo: reemplazar por `raise KeyError` o al menos `logger.error` + abortar. Código actual **fabrica artefactos de cero sin trazabilidad**, lo cual es peor que no generar nada.

**Flag 92 — `fig_retrieval_metrics.png` == `fig_retrieval_metrics_exp8b.png` (🔴🔴)**

Hash MD5 idéntico:

```
34e9cbc30c8a9a1a1b9332362f50ca08  fig_retrieval_metrics.png
34e9cbc30c8a9a1a1b9332362f50ca08  fig_retrieval_metrics_exp8b.png
```

Este es el mismo fenómeno del Flag 77 (exp8 y exp8b `retrieval_metrics.json` byte-idénticos en las secciones `systems` y `statistical_tests`). El paper alude a exp8b como "experimento comparativo" con faithfulness distinto, pero la figura de retrieval para ambos es literalmente el mismo archivo PNG. No hay hipótesis que justifique esto en un paper honesto.

**Flag 93 — `table_retrieval_metrics.tex` == `table_retrieval_metrics_exp8b.tex` (🔴🔴)**

Mismo MD5. Peor aún: ambos archivos contienen el caption `"Cross-encoder based retrieval quality metrics (exp8, 200 queries)"`. El archivo nombrado `_exp8b` no se molestó en cambiar el caption. Si sobrevive a la compilación, el paper mostraría la misma tabla dos veces con etiqueta distinta — o peor, citaría `\ref{tab:retrieval-metrics}` desde dos lugares y LaTeX advertiría de `label multiply defined`.

**Flag 94 — `fig_llm_comparison.png`: radar chart de 1 solo eje (🔴)**

El PNG muestra un radar con grid circular pero con una única dimensión etiquetada ("Faithfulness") y 3 puntos colapsados en esa única línea. Los tres LLM (llama3.1, qwen2.5, mistral) terminan como 3 puntos en un segmento. Un "radar chart" con una sola dimensión es geométricamente un gráfico de puntos sobre una línea, envuelto en un círculo decorativo. Visualmente es absurdo.

Causa probable: el código en `fig_llm_comparison` (línea 391) intenta plottear múltiples métricas pero sólo `faithfulness` sobrevive porque el resto vienen de `retrieval_metrics` vacíos (Flag 69-70). Esto es un corolario del data crisis, manifestado distinto.

Fix: o bien usar bar chart simple (3 barras × 1 métrica), o recuperar las otras métricas antes de reclamar comparación multidimensional entre LLMs.

**Flag 95 — `fig_retrieval_metrics.png`: Avg Score rompe el eje [0,1] (🔴)**

El eje y del subplot izquierdo está fijado en [0, 1.05]. La métrica `Avg Score` es el promedio del cross-encoder score (rango observado: `[-10.9, 9.2]` per `score_distribution` en exp8). Los valores ~1.9–2.9 (ver `table_retrieval_metrics.tex`) obviamente superan 1.0, y las barras se dibujan como líneas verticales que atraviesan todo el eje. Visualmente, la métrica `Avg Score` queda rota — bars "infinitos".

Dos opciones: (a) eje secundario para `Avg Score`, (b) normalizar a [0,1] dividiendo por max observado, (c) sacar `Avg Score` del mismo gráfico. En cualquier caso, no publicar la versión actual.

**Flag 96 — Siete tablas con ceros estructurales (🔴)**

Ya tabulado arriba. Corolario de Flags 69, 75, 76, 91.

**Flag 97 — `fig_statistical_significance` referenciado en docstring pero nunca llamado (🟡)**

Docstring de `results_exporter.py` línea 19 declara 9 figuras. El método `export_all` (línea 682) sólo invoca 7 figuras + `fig_latency_breakdown` manual (línea 702). `fig_statistical_significance` requiere parámetro `stat_results` que no se pasa desde `export_all`, así que nunca se genera. Menor, pero sugiere que el pipeline de exportación nunca fue testeado end-to-end.

### Implicaciones para el paper

1. **Los placeholders azules en main.tex no son lo único que falta.** Las 5 figuras `\includegraphics{...}` que apuntan a archivos vacíos generarían páginas con gráficos en blanco — peor que un `[FIGURE MISSING]` textual.

2. **Cualquier referencia cruzada a `tab:exp3`, `tab:exp4`, `tab:exp5`, `tab:exp6`, `tab:exp7`, `tab:exp8`, `tab:exp8b` dentro del tex produce citas a tablas con ceros.** Hay que grep el main.tex para saber qué tablas están `\ref`-eadas y cuáles se pueden remover sin romper.

3. **El "experimento comparativo" exp8 vs exp8b es, al nivel de artefactos publicables, el mismo experimento dos veces.** Sólo la faithfulness (0.368 vs 0.514) genera diferencia. Todo lo demás es copia.

4. **Fix mínimo honesto antes de someter:** remover exp3/exp4/exp5/exp6/exp7/exp8b references del main.tex y presentar solamente exp8 + latency. Eso reduce el paper a ~5 páginas útiles pero evita fraude.

5. **Fix correcto:** re-correr exp3/4/6/7 con la instrumentación correcta (escribir `retrieval_metrics` y `relevant_ids` en results.json) y regenerar figuras. Tiempo estimado: 1-2 días si los configs están bien en `experiment_configs.py` (a verificar en Módulo 14).

### Pendientes resultantes

- [ ] Grep `main.tex` por cada `\includegraphics{figures/fig_*}` y `\ref{tab:*}` para saber qué artefactos vacíos sobreviven al PDF
- [ ] Decidir: remover exp3/4/5/6/7/8b o re-correr
- [ ] Fix `results_exporter.py` línea 655 (y análogas en figuras) para que falle ruidosamente en vez de fabricar ceros
- [ ] Regenerar `fig_llm_comparison` como bar chart 1D
- [ ] Regenerar `fig_retrieval_metrics` con Avg Score separada

## Módulo 14: experiments/experiment_configs.py

### Archivo: `experiments/experiment_configs.py` (651 líneas)

**Inventario de experimentos:**

| exp_id | max_queries | # configs | Variable | Notas |
|--------|-------------|-----------|----------|-------|
| exp1 | 200 | 15 | chunking + size | Nunca tiene retrieval_metrics |
| exp2 | 200 | 3 | embedding_model | Sin results.json en repo |
| exp3 | 200 | 6 | retrieval + fusion | Todos con `reranker=None` |
| exp4 | 200 | 4 | reranker | Caché de generación contaminada (Flag 72) |
| exp5 | 200 | 3 | llm_model | Con query_expansion Y normalization activos |
| exp6 | 200 | 5 | components (ADDITIVE) | Sin retrieval_metrics |
| exp7 | **30** | 2 | terminology_norm | Cross-cloud queries |
| exp8 | 200 | 3 | pipeline_config | LLM = llama3.1 |
| exp8b | 200 | 3 | LLM only | LLM = mistral; retrieval idéntico |

### Flags

**Flag 98 — Ningún experimento usa 300 queries; el paper miente (🔴🔴)**

```
grep max_queries experiment_configs.py
→ 200 × 8 experimentos, 30 × exp7
```

El abstract declara "300 queries". El valor **no existe en el código**. El valor real máximo por experimento es 200. El test set total disponible (Módulo 11) era 229. Esto **cierra el Flag 83** con evidencia definitiva: el número 300 fue fabricado en el paper. Fix obligatorio: corregir abstract a "200 queries per experiment" o "229 total queries" (dependiendo de qué test set se reporte).

**Flag 99 — exp6 es ADDITIVE (forward selection), no LEAVE-ONE-OUT (🔴🔴)**

El paper Results V.B declara textualmente: *"Drop each component one at a time; report ΔNDCG@5 and ΔFaithfulness"*.

El código real (líneas 374-463) es:

```
Stage 1: BM25 only
Stage 2: + Dense (hybrid)
Stage 3: + Reranker
Stage 4: + Query expansion
Stage 5: + Terminology normalization (FULL)
```

Forward selection y leave-one-out NO son equivalentes. Una ablación leave-one-out correcta habría tenido:

```
Config 1: FULL
Config 2: FULL – Dense
Config 3: FULL – Reranker
Config 4: FULL – Expansion
Config 5: FULL – Normalization
```

Los ΔNDCG@5 de forward selection miden *contribución marginal ordenada* (con efecto fuerte en las primeras etapas por bajo baseline). Los de leave-one-out miden *contribución conjunta*. El paper no puede declarar lo primero y reportar lo segundo. Este flag sube Flag 84 a categoría "error metodológico público".

**Flag 100 — exp3 selecciona RRF vs linear sin evidencia, exp4 y exp8 heredan RRF hard-coded (🔴)**

exp3 compara 4 variantes de fusión: linear(α=0.3, 0.5, 0.7) y rrf(k=60). Pero Módulo 11 confirmó que exp3 no tiene `retrieval_metrics` — imposible saber cuál ganó.

exp4 (reranker comparison) asume `fusion_method="rrf"` (línea 297). exp8 usa `PROPOSED_HYBRID` (que asumo configura rrf). exp8b explícitamente usa `fusion_method="rrf"` y `alpha=0.5, rrf_k=60` (líneas 594-596).

La decisión "RRF es mejor que linear" se tomó sin datos y propagó a todos los experimentos downstream. Si en un reviewer serio exige justificación, no hay. **Este Flag cierra parcialmente Flag 89**: RRF se usa por convención, no por evidencia en este repo.

**Flag 101 — exp5 LLM comparison tiene query_expansion=True Y terminology_normalization=True (🟡)**

Líneas 345-346: las 3 configs activan ambos flags. Eso significa que la comparación LLM está encima del pipeline FULL. El riesgo es que si alguna de esas ayudas sesga hacia retrieval que satisface más a un LLM que a otro, el comparativo LLM queda contaminado.

Riesgo menor. Lo correcto sería reportar: "LLM comparison runs on FULL pipeline (query_expansion + normalization active)". El paper no lo dice.

**Flag 102 — exp8b idéntico a exp8 excepto LLM → faithfulness delta se atribuye sólo al LLM (🟡)**

Líneas 556-612: exp8b reproduce PROPOSED_HYBRID con LLM=mistral. Como el retrieval es determinístico dado seed+query, los `retrieval_metrics` son idénticos (Flag 77 ya lo confirmó byte-a-byte).

Esto **no es un bug** — es el diseño. Pero el paper nunca lo explica. Lee como dos experimentos independientes cuando en realidad es un único experimento con sólo la etapa de generación variada. La narrativa honesta: *"We hold retrieval constant and vary only the LLM to isolate the effect of generation"*. El abstract actual sugiere comparación sistémica completa.

**Flag 103 — `seed=42` es nominal: no se propaga a BM25/FAISS/numpy/torch (🔴)**

Grep confirma: `seed` sólo aparece en `ExperimentConfig.__init__` y en `to_dict()`. Nunca se pasa a:

- `rank_bm25.BM25Okapi` (que no acepta seed, pero al menos `random.seed(42)` debería invocarse)
- `faiss` (FAISS IVF requiere `np.random.seed()` para clustering reproducible — IndexFlatIP es determinístico, así que acá Flag 52 ayuda por accidente)
- `torch.manual_seed` / `torch.cuda.manual_seed_all` (para cross-encoder y NLI)
- `numpy.random.seed` (para samplings de queries)

Si declaran "seed=42 for reproducibility" pero no lo propagan, otra lab que re-corre obtiene números distintos. Esto **no invalida los resultados** (FlatIP es determinístico; rank_bm25 es determinístico), pero es **propaganda de reproducibilidad que no se cumple**.

Fix: crear un helper `set_all_seeds(seed)` en `src/utils/` y llamarlo en el entrypoint de cada experimento.

**Flag 104 — exp2 (embedding model comparison) sin results.json en el repo (🟡)**

Definido en config pero nunca ejecutado, o los resultados no llegaron al repo. En la práctica: el paper no discute comparación de embedding models. Si el paper jamás cita exp2, se puede remover del código. Si el paper lo cita, hay que correrlo.

**Flag 105 — exp7 con solo 30 queries, sin retrieval_metrics, figura vacía (🔴)**

Línea 527: `max_queries=30`. Módulo 13 confirmó que `fig_cross_cloud_improvement.png` está vacía. Con 30 queries los CIs son gigantes — cualquier afirmación tipo "Recall+20%" sobre 30 queries es ruido. Peor, no hay datos de calidad en el results.json. Cualquier aparición de exp7 en el paper es un claim sin respaldo.

**Flag 106 — `ExperimentConfig.metrics` declarativo pero no validado (🟡)**

Cada experimento declara `metrics=["retrieval", "generation", ...]`. Pero no existe ningún código que verifique que los resultados contengan esas métricas. Es por eso que exp4 declara `metrics=["retrieval", "latency"]` pero el results.json queda con `retrieval_metrics={}` (Flag 69). El `metrics` es un wish list, no un contrato.

### Resumen Módulo 14

- Flag 5 (300 queries) → ahora confirmado como **fabricación**, no error de transcripción
- Flag 89 (RRF vs linear) → **cerrado**: decisión sin evidencia, propagada a downstream
- Nuevo flag crítico: Flag 99 (ablation method mismatch)
- Reproducibilidad nominal: Flag 103
- Experimentos fantasma: Flag 104 (exp2)

### Tabla de severidad adicional

| Módulo | Issue | Severidad | Acción |
|--------|-------|-----------|--------|
| **13.1** | **Flag 90 5/8 figuras vacías** | **🔴🔴🔴** | **Regenerar o remover** |
| **13.2** | **Flag 91 .get(col, 0) silencioso** | **🔴🔴** | **Fix código** |
| **13.3** | **Flag 92 fig retrieval duplicada** | **🔴🔴** | **Regenerar exp8b** |
| **13.4** | **Flag 93 table retrieval duplicada** | **🔴🔴** | **Regenerar exp8b** |
| **13.5** | **Flag 94 radar de 1 eje** | **🔴** | **Reescribir** |
| **13.6** | **Flag 95 Avg Score overflow** | **🔴** | **Eje secundario** |
| 13.7 | Flag 96 tablas con ceros | 🔴 | Regenerar |
| 13.8 | Flag 97 fig_statistical fantasma | 🟡 | Generar o remover |
| **14.1** | **Flag 98 300 queries fabricado** | **🔴🔴** | **Corregir abstract** |
| **14.2** | **Flag 99 ablation additive vs LOO** | **🔴🔴** | **Re-correr LOO o reescribir texto** |
| **14.3** | **Flag 100 RRF sin evidencia** | **🔴** | **Correr exp3 o declarar heuristic** |
| 14.4 | Flag 101 exp5 con pipeline FULL | 🟡 | Declarar |
| 14.5 | Flag 102 exp8b=exp8 sin retrieval | 🟡 | Declarar como "LLM-only" |
| **14.6** | **Flag 103 seed nominal** | **🔴** | **Fix set_all_seeds** |
| 14.7 | Flag 104 exp2 orfano | 🟡 | Remover o correr |
| **14.8** | **Flag 105 exp7 n=30 vacío** | **🔴** | **Remover del paper** |
| 14.9 | Flag 106 metrics declarativo | 🟡 | Validar en runtime |
| **15.1** | **Flag 107 Cohen's d = d_z (no d_av ni d_rm)** | **🔴 Inflación sistemática vs Cohen's thresholds** | **Declarar d_z en paper o cambiar a d_av** |
| **15.2** | **Flag 108 sin corrección de comparaciones múltiples** | **🔴🔴 Reviewer flag instantáneo** | **Añadir Benjamini-Hochberg o Holm** |
| 15.3 | Flag 109 normality triple (a, b, diff) | 🟡 Over-triggers Wilcoxon | Simplificar a diff-only |
| 15.4 | Flag 110 Shapiro con n=200 hipersensible | 🟡 En la práctica siempre Wilcoxon | Documentar |
| 15.5 | Flag 111 is_normal_diff calculado pero no guardado | 🟡 `StatisticalResult` pierde señal relevante | Propagar al dict |
| 15.6 | Flag 112 Anderson-Darling devuelve p=0.0 dummy | 🟢 Labeling issue | Cambiar a NaN |
| **15.7** | **Flag 113 `meets_thesis_threshold`=0.5 sobre d_z** | **🔴 Interpreta escala Cohen con fórmula incorrecta** | **Recalcular con d_av** |

## Módulo 15: src/evaluation/statistical_analysis.py

### Archivo: `src/evaluation/statistical_analysis.py` (458 líneas)

**Lo que el código hace bien:**

1. Shapiro-Wilk + fallback a Anderson-Darling para n>5000 ✓
2. Paired t-test vs Wilcoxon según normalidad ✓
3. Bootstrap CI con seed fijo (línea 268) ✓
4. Paired bootstrap: usa el mismo `idx` para `a` y `b` (línea 283-284) ✓
5. NaN-pair removal antes de cada test ✓
6. Handling de zero-diff en Wilcoxon (línea 193-196) ✓

Nada de esto es decorativo. El esqueleto es correcto. Los problemas están en decisiones finas que un reviewer estadístico va a señalar.

### Flags

**Flag 107 — Cohen's d implementado como d_z, no d_av ni d_rm (🔴)**

Líneas 225-232:

```python
diff = b - a
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)
d = mean_diff / std_diff
```

Esto es **Cohen's d_z** (repeated-measures, divide entre la SD de las diferencias). No es el `d` "clásico" de Cohen 1988 que se suele reportar. Las tres variantes son:

$$d_z = \frac{\bar{D}}{s_D} \quad\text{(este código)}$$

$$d_{av} = \frac{\bar{D}}{(s_A + s_B)/2}$$

$$d_{rm} = \frac{\bar{D}}{\sqrt{s_A^2 + s_B^2 - 2 r s_A s_B}} \cdot \sqrt{2(1-r)}$$

Cuando los sistemas están correlacionados (que es el caso en IR — las mismas queries evalúan todos los sistemas), `d_z` es **sistemáticamente mayor** que `d_av` y `d_rm`. Lakens (2013) advierte explícitamente: *"d_z should not be compared against Cohen's (1988) benchmarks of 0.2/0.5/0.8, which were derived from independent samples."*

Tu paper reporta `d=-0.626` (BM25 vs Hibrido) y lo interpreta como "medium effect" contra el threshold de 0.5. Si ese `d` es `d_z`, entonces el `d_av` real podría estar cerca de 0.3-0.4 (small). Riesgo alto con reviewer estadístico.

**Fix mínimo**: declarar explícitamente en el paper "We report Cohen's d_z per Lakens (2013)". **Fix honesto**: recalcular con d_av y reportar ambos.

**Flag 108 — Sin corrección de comparaciones múltiples (🔴🔴)**

Grep en todo el repo (`bonferroni|holm|benjamini|fdr|multipletests`): **cero matches**. No hay ninguna corrección.

En exp8 se corren comparaciones sobre 3 sistemas (A vs B, A vs C, B vs C) × múltiples métricas (Prec@1, Prec@3, Prec@5, MRR, NDCG@5, Avg Score) = 18 tests. Con α=0.05 sin corrección, el family-wise error rate es ~60% de falsos positivos.

Reviewer de LACCI que revise el análisis estadístico lo va a reportar sin pestañear. La fix es trivial: importar `scipy.stats.false_discovery_control` o `statsmodels.stats.multitest.multipletests` y aplicar Benjamini-Hochberg al array de p-values antes de declarar significancia.

**Esto es más grave que Flag 107**: Flag 107 exagera el effect size; Flag 108 exagera la significancia estadística.

**Flag 109 — Decisión t-test vs Wilcoxon requiere normalidad triple (🟡)**

Línea 184: `if is_normal_diff and is_normal_a and is_normal_b`.

Para un **paired t-test**, sólo las **diferencias** necesitan ser normales. Requerir normalidad de A y B individualmente es sobre-restrictivo y hace que el código casi siempre caiga a Wilcoxon.

Fix: `if is_normal_diff:` (ignorar a, b).

**Flag 110 — Shapiro-Wilk con n=200 es hipersensible (🟡)**

Con n=200, Shapiro-Wilk detecta desviaciones triviales de normalidad como `p<0.05`. Las métricas IR (NDCG, MRR, Recall) son acotadas en [0,1] y tienen masa de probabilidad en 0 y 1 — casi nunca son "normales". Consecuencia práctica: el código **siempre** cae en Wilcoxon, y el bloque del `if is_normal_diff and is_normal_a and is_normal_b:` es código muerto.

Esto no es incorrecto (Wilcoxon es conservador y seguro), pero el paper que declara "paired t-test or Wilcoxon depending on normality" es teatral — en práctica siempre es Wilcoxon. Mejor declarar "We use Wilcoxon signed-rank as the primary test due to non-normal IR metric distributions".

**Flag 111 — `is_normal_diff` calculado pero no devuelto (🟡)**

Línea 178: `is_normal_diff, _ = shapiro_wilk_test(diff.tolist(), alpha)` calcula el valor. Pero `paired_comparison` no lo devuelve — sólo devuelve `is_normal_a` y `is_normal_b` (líneas 203-204). `StatisticalResult` guarda `is_normal_a/b` pero no `is_normal_diff` — que es justo la métrica que importa.

Impacto: si alguien audita el JSON de resultados, no puede verificar la decisión t-test/Wilcoxon. Reproducibilidad nominal en el análisis estadístico.

**Flag 112 — Anderson-Darling fallback devuelve p=0.0 dummy (🟢)**

Línea 133: `return is_normal, 0.0` — para corpus con n>5000 Anderson-Darling no da un p-value exacto. El 0.0 dummy se propaga al resto del pipeline. Si el consumer printea "p=0.0000" pensará que falla normalidad espectacularmente. Mejor: devolver `float('nan')`.

**Flag 113 — `meets_thesis_threshold=abs(d)>=0.5` interpreta Cohen con fórmula incorrecta (🔴)**

Línea 369 `meets_thesis_threshold=abs(d_value) >= 0.5`. El 0.5 es el threshold **"medium" de Cohen 1988** — pero Cohen definió esos thresholds para `d_s` (independent samples, pooled SD). Aplicarlos sobre `d_z` (Flag 107) es un category error.

Combinado con Flag 107 y 108, esto significa que el paper reporta:

- `d_z` inflado vs thresholds Cohen
- Sin corrección de múltiples comparaciones
- Interpretación "medium effect" sobre una métrica que técnicamente no cumple esa definición

Un reviewer con formación estadística media (incluso cursos intro de psicometría) detecta esto.

**Otros (menores) que no subo a flag formal:**

- `bootstrap_ci` usa `np.random.RandomState` + `randint` (API legacy). Funciona, pero `np.random.default_rng(seed).integers(...)` es la versión moderna.
- `cohens_d` no reporta el número de muestras usado — si NaN reduce n, no se ve.
- No se reporta `power` (statistical power). Con n=200 y d_z=0.28 (Dense vs Hibrido), la power es ~0.7 — borderline. Declarar en Limitations.

### Contraste con lo declarado en el paper

El main.tex dice genéricamente "Wilcoxon signed-rank, Cohen's d, bootstrap CIs". No declara:
- Cuál variante de Cohen's d (d_z vs d_av vs d_rm)
- Si se corrigió por múltiples comparaciones
- Qué significa "thesis threshold"
- Si Shapiro-Wilk falla, qué se usa

**Fix de paper (mínimo honesto)**: un párrafo en Methodology:

> *Statistical comparisons use Wilcoxon signed-rank with α=0.05, Benjamini-Hochberg false-discovery-rate correction across 18 comparisons (3 systems × 6 metrics), and Cohen's d_z as the paired effect size. We interpret |d_z|≥0.5 as practically significant, following Lakens (2013).*

Con eso, Flags 107, 108, 110, 113 se cierran a la vez.

### Pendientes módulo 15

- [ ] Aplicar BH correction a las p-values en `statistical_tests.json` y verificar cuántas sobreviven
- [ ] Recalcular d_av en paralelo a d_z y reportar ambos en un CSV
- [ ] Añadir párrafo metodológico al main.tex
- [ ] Fix código: devolver `is_normal_diff` en `paired_comparison`
- [ ] Fix código: simplificar condición a `if is_normal_diff:`

### Tabla de severidad adicional (Módulo 15)

Ya integrada en la tabla de arriba (filas 15.1-15.7).

## Módulo 16: Recompute estadístico empírico

### Objetivo

Aplicar Benjamini-Hochberg FDR y computar Cohen's d_av sobre las 12 comparaciones reportadas en `exp8/retrieval_metrics.json` (4 métricas × 3 pares) para saber si los Flags 107, 108 y 113 se sostienen con datos reales.

### Método

- Input: `exp8/retrieval_metrics.json` (p-values, d_z, medias y stds por sistema)
- Back-inferencia de la correlación pareada `r` a partir de `d_z` y las stds (relación exacta: `std_diff² = std_a² + std_b² − 2·r·std_a·std_b`)
- d_av = mean_diff / ((std_a + std_b) / 2)
- d_rm = mean_diff / √(std_a² + std_b² − 2·r·std_a·std_b) · √(2(1−r))
- BH-FDR y Holm sobre 12 p-values simultáneamente
- Script: `scripts_audit/m16_recompute_stats.py`
- Output: `paper/audit_outputs/exp8_stats_corrected.csv`

### Resultados

**1) Corrección de múltiples comparaciones:**

| Test | # significativos |
|------|------------------|
| p_raw < 0.05 | 12/12 |
| Benjamini-Hochberg | **12/12** |
| Holm-Bonferroni | **12/12** |

Los p-values originales son tan bajos (la mayoría <10⁻⁴) que incluso Holm (más conservador) los preserva.

**2) Inflación d_z vs d_av:**

| Métrica | Par | d_z | d_av | r inferido |
|---------|-----|------|------|-----------|
| ndcg@5 | BM25 vs Dense | −0.353 | −0.354 | 0.50 |
| ndcg@5 | BM25 vs **Hibrido** | **−0.626** | **−0.630** | 0.51 |
| ndcg@5 | Dense vs Hibrido | −0.282 | −0.282 | 0.50 |
| avg_score@5 | BM25 vs Dense | −0.180 | −0.180 | 0.50 |
| avg_score@5 | BM25 vs Hibrido | −0.373 | −0.373 | 0.50 |
| avg_score@5 | Dense vs Hibrido | −0.180 | −0.180 | 0.50 |
| precision@5 | BM25 vs Dense | −0.198 | −0.198 | 0.50 |
| precision@5 | BM25 vs Hibrido | −0.431 | −0.435 | 0.52 |
| precision@5 | Dense vs Hibrido | −0.241 | −0.242 | 0.50 |
| mrr | BM25 vs Dense | −0.212 | −0.213 | 0.51 |
| mrr | BM25 vs Hibrido | −0.402 | −0.411 | 0.55 |
| mrr | Dense vs Hibrido | −0.197 | −0.198 | 0.51 |

**3) Thesis threshold |d| ≥ 0.5:**

Sólo **1 de 12** comparaciones cumple el threshold, bajo ambas métricas d_z y d_av:

- **BM25 vs Hibrido en NDCG@5** → d ≈ −0.63 (medium)

Todas las demás son "small" (−0.35 a −0.43) o "negligible" (<0.2).

### Implicaciones para los flags

- **Flag 108 DEGRADADO 🔴🔴 → 🟢** — Todas las 12 comparaciones sobreviven BH. El paper puede reportar "significant after Benjamini-Hochberg FDR correction across 12 tests" y la historia aguanta. Fix: un renglón en Methodology, no re-correr nada.

- **Flag 107 DEGRADADO 🔴 → 🟡** — d_z y d_av difieren en ≤1% porque la correlación pareada es ~0.50 y las stds de A y B son similares. La "inflación" teórica predicha por Lakens es real pero cuantitativamente despreciable en este dataset. Declarar "Cohen's d_z (Lakens 2013)" en Methodology cierra el flag sin cambiar números. **Matiz importante**: si mañana corres con LLMs que producen correlación más alta (r→0.8), d_z se inflaría mucho más. Pero con estos datos estás cubierto.

- **Flag 113 CONFIRMADO 🔴** — Sólo 1 de 12 comparaciones cumple |d| ≥ 0.5. El paper declara "medium effect size" en el abstract con el dato de BM25 vs Hibrido en NDCG@5, lo cual es literalmente cierto. Pero si en Results se extiende el claim a MRR o Precision@5, es falso: ahí `d_av` anda en −0.41 y −0.44 (small, bajo thresholds de Cohen). **Fix de honestidad**: reportar explícitamente en Results que el medium effect aplica solo a NDCG@5, no a las otras métricas.

### Flags nuevos

| # | Flag | Severidad | Acción |
|---|------|-----------|--------|
| **16.1** | **Flag 114 Sólo 1/12 comparaciones cumple thesis threshold** | **🔴 Contradice narrativa "hybrid outperforms baselines by >=15% NDCG@5 and >=10% faithfulness"** | **Reescribir Results V para distinguir NDCG@5 de otras métricas** |
| 16.2 | Flag 115 r inferido ~0.5 valida d_z≈d_av | 🟢 Flag 107 cuantitativamente menor | Declarar d_z en Methodology |
| 16.3 | Flag 116 BH pasa 12/12 | 🟢 Flag 108 cuantitativamente menor | Declarar BH en Methodology |

### Qué se puede decir honestamente en el paper tras este análisis

Esto es lo que los datos aguantan (y exactamente esto, no más):

> *"On NDCG@5 the proposed hybrid system outperforms BM25 with a medium effect size (d_av = −0.63, Wilcoxon p < 10⁻¹⁰, BH-adjusted p < 10⁻⁹). On MRR, Precision@5 and Avg Score@5 the hybrid system outperforms both baselines with statistically significant but small effect sizes (|d_av| = 0.18–0.44, all BH-adjusted p < 0.01)."*

Lo que NO aguantan:

- "Medium effect across all metrics" ❌
- "Hybrid outperforms semantic baseline by a practically significant margin" (d=−0.28, small) ❌
- "Hybrid outperforms baselines" sin matizar por métrica ❌ (técnicamente cierto pero engañoso)

### Archivo generado

`paper/audit_outputs/exp8_stats_corrected.csv` — tabla completa con d_z, d_av, d_rm, r inferido, p_raw, p_bh, p_holm, y sig_bh/sig_holm por cada una de las 12 comparaciones. Se puede pegar directo como supplementary table.

## Módulo 17: src/pipeline/pipeline_config.py

### Archivo: `src/pipeline/pipeline_config.py` (95 líneas)

**Lo que está bien:**

1. Pydantic `BaseModel` valida tipos al construir ✓
2. `get_config` retorna `model_copy(deep=True)` — evita mutación compartida entre callers ✓
3. `temperature=0.1` near-determinístico en los 3 sistemas ✓
4. Baselines bien aislados (sin reranker, sin expansion, sin normalization) ✓

### Flags

**Flag 117 — PROPOSED_HYBRID declara `alpha=0.5` Y `fusion_method="rrf"` (🟡)**

Líneas 67-69:

```python
PROPOSED_HYBRID = PipelineConfig(
    ...
    fusion_method="rrf",
    alpha=0.5,
    rrf_k=60,
    ...
)
```

RRF no usa `alpha` — usa `rrf_k`. Linear fusion usa `alpha`. Tener ambos en la config es código muerto en el mejor caso. En el peor, si `search_hybrid` en algún path aplica `alpha` encima de RRF por accidente, la fusión deja de ser RRF pura.

Verificado en `hybrid_retriever.py:62` — pasa `alpha` al backend independientemente de `fusion`. Y en `hybrid_retriever.py:96` el logger escupe `alpha=0.50` aún cuando fusion=rrf, lo cual en los logs hace ver que el sistema "aplicó alpha" — **misleading**.

Fix: en PROPOSED_HYBRID, remover `alpha=0.5`. Si el backend lo requiere por contrato, pasar `alpha=None` y que el backend rompa si ve inconsistencia.

**Flag 118 — PROPOSED_HYBRID activa 4 componentes extra vs baselines (no es comparación pura de retrieval) (🔴)**

Comparación declarativa en el código:

| Componente | BASELINE_LEXICAL | BASELINE_SEMANTIC | PROPOSED_HYBRID |
|------------|------------------|-------------------|-----------------|
| retrieval | bm25 | dense | hybrid (rrf) |
| reranker | — | — | **ms-marco-MiniLM-L-12-v2** |
| multidimensional_scoring | False | False | **True** |
| query_expansion | False | False | **True** |
| terminology_normalization | False | False | **True** |

La diferencia entre baseline y propuesto es: +retrieval híbrido, +reranker, +MMR/recency/quality, +query expansion, +terminology normalization. Son **cinco cambios simultáneos**.

Si Results V.A reporta "hybrid outperforms BM25 by 31% on NDCG@5", ese delta es la suma de las 5 diferencias, no solo del retrieval híbrido. El paper debería decir "**the full proposed pipeline** outperforms the BM25 baseline", NO "hybrid retrieval outperforms BM25".

Esto es crítico porque la ablación (exp6) es additive, no leave-one-out (Flag 99). Así que **no se sabe** cuánto aporta realmente cada uno de los 5 componentes. La narrativa actual es publicable solo como "composite system vs single-method baselines".

**Flag 119 — Los 3 sistemas usan el mismo LLM Q4-quantizado (🟡)**

Líneas 46, 60, 79: `llm_model="llama3.1:8b-instruct-q4_K_M"`. Q4 = 4-bit quantization. Reduce memoria pero degrada faithfulness y calidad de respuesta vs fp16/q8.

Implicaciones:
- El +16.8% faithfulness reportado es **específico a Q4**
- Con LLM sin cuantizar, baselines podrían cerrar el gap (o no — es impredecible)
- El paper debe declarar en Limitations: "Results specific to 4-bit quantized Llama 3.1 8B; generalization to full-precision or larger LLMs not tested"

**Flag 120 — Default `chunking_strategy="adaptive"` heredado sin declaración explícita en las 3 configs (🟡)**

PipelineConfig línea 32: `chunking_strategy: str = "adaptive"`. Los 3 sistemas (BASELINE_LEXICAL, BASELINE_SEMANTIC, PROPOSED_HYBRID) **no** especifican `chunking_strategy`, así que todos heredan `adaptive`.

El paper presenta "adaptive chunking" como contribución. Pero:
- exp1 (chunking comparison) no tiene `retrieval_metrics` (Flag 76)
- Nunca se validó que adaptive bate a semantic/recursive/fixed
- Sin embargo se hard-codeó como default en los 3 sistemas evaluados

Si el paper afirma "we use adaptive chunking based on results from exp1", es falso — exp1 no tiene resultados. La elección está **unverified**.

**Flag 121 — No hay `seed` en PipelineConfig (🟡)**

Correlacionado con Flag 103. `PipelineConfig` no tiene campo seed. Cualquier componente interno (BM25 tokenizer, FAISS IVF clustering, etc.) corre con su propia semilla si la hay. Control de reproducibilidad a nivel pipeline: inexistente.

**Flag 122 — No hay validator para combinaciones inconsistentes (🟡)**

Pydantic valida tipos, no lógica. Alguien puede construir:

```python
PipelineConfig(
    retrieval_method="bm25",
    embedding_model="bge-large",   # sin sentido para BM25 puro
    fusion_method="rrf",            # sin sentido sin dense
)
```

Sin warning. El error aparecería en runtime cuando `hybrid_retriever` intente usar embeddings inexistentes. Fix: `@model_validator` en pydantic que verifique:
- `retrieval_method="hybrid"` ⟹ `embedding_model` y `fusion_method` obligatorios
- `retrieval_method="bm25"` ⟹ `embedding_model=None`, `fusion_method=None`
- `retrieval_method="dense"` ⟹ `embedding_model` obligatorio, `fusion_method=None`

**Flag 123 — `retrieval_method=f"hybrid_{fusion}"` aparece en el tag (🟢)**

Línea 79 de hybrid_retriever.py. Significa que en los logs hay `retrieval_method="hybrid_rrf"` o `"hybrid_linear"`. Esto **cierra definitivamente Flag 32**: el grep original buscaba la cadena exacta `retrieval_method` en results.json, pero el campo **existe con valor** `hybrid_rrf`, solo que no lo imprimimos en su momento. Cierra como falso positivo confirmado.

### Tabla de severidad

| # | Flag | Severidad | Acción |
|---|------|-----------|--------|
| 17.1 | Flag 117 alpha=0.5 + rrf redundante | 🟡 | Remover alpha en PROPOSED_HYBRID |
| **17.2** | **Flag 118 5 componentes simultáneos en PROPOSED** | **🔴 Invalida lenguaje "hybrid retrieval outperforms"** | **Reescribir narrativa a "composite pipeline"** |
| 17.3 | Flag 119 Q4 quantization | 🟡 | Declarar en Limitations |
| 17.4 | Flag 120 adaptive chunking sin validar | 🟡 | Remover claim o correr exp1 |
| 17.5 | Flag 121 sin seed en PipelineConfig | 🟡 | Añadir campo |
| 17.6 | Flag 122 sin validator de combos | 🟡 | Añadir `@model_validator` |
| 17.7 | Flag 123 Flag 32 cerrado | 🟢 CERRADO | Ninguna |

### Conexión con Módulos previos

- **Flag 118** conecta directamente con **Flag 99** (ablation additive): juntos implican que el paper no puede atribuir el gain a ningún componente individual. La narrativa honesta es "the composite pipeline outperforms single-method baselines".
- **Flag 120** conecta con **Flag 104** (exp2 orfano): ambos representan decisiones arquitectónicas sin respaldo empírico en el repo.
- **Flag 117** (alpha+rrf redundante) implica auditar `index.search_hybrid` para confirmar que ignora alpha bajo rrf. Pendiente.

### Pendientes Módulo 17

- [ ] Leer `index.search_hybrid` para confirmar que con `fusion="rrf"` el `alpha` es ignorado
- [ ] Decidir redacción en paper: "composite hybrid system" vs "hybrid retrieval"
- [ ] Declarar Q4 quantization en Limitations
- [ ] Decidir destino de claim "adaptive chunking" (remover o re-correr exp1)

## Módulo 18: Query expansion + Adaptive chunking

### Archivo A: `src/retrieval/query_processor.py` (251 líneas)

### Archivo B: `src/chunking/adaptive_chunker.py` (515 líneas)

Componentes centrales para dos claims del paper: "query expansion + terminology normalization" y "adaptive chunking como contribución principal" (docstring línea 2 del adaptive_chunker declara: *"Main contribution of the thesis"*).

### Flags — Query Expansion

**Flag 124 — Query expansion SOLO afecta al lado BM25, no al dense (🔴)**

`query_processor.py` líneas 131-132:

```python
bm25_query = self._build_bm25_query(query, expanded)   # query + expanded terms
semantic_query = query                                   # raw, sin expansión
```

El lado denso (BGE) recibe la query sin tocar. Esto significa que:
- La "query expansion" del paper es **expansión solo para BM25**
- El componente denso se defiende con su capacidad semántica nativa
- Cualquier ganancia atribuida a expansion se manifiesta en el lado lexical de la fusión

Tiene defensa lógica (BGE ya captura sinónimos semánticamente), pero **el paper debe declararlo explícitamente**. Un reviewer que espera query expansion estilo HyDE/query2doc (que expande el lado denso para paliar el gap lexical-semantico) va a malinterpretar.

**Flag 125 — Query expansion es diccionario YAML estático, no LLM (🟡)**

El archivo `config/terminology_mappings.yaml` (175 líneas) contiene los mapeos. No hay HyDE, no hay query2doc, no hay LLM-rewriting. Es expansión clásica por synset/acronym lookup.

Esto **no es malo** — es barato y reproducible — pero el paper debe decir literalmente "dictionary-based query expansion using a manually curated terminology YAML" para no infleer técnicas modernas como HyDE.

**Flag 126 — `enable_query_expansion` y `enable_terminology_normalization` están entangled (🟡)**

Líneas 215-218:

```python
allow_cross_provider_terms = (
    enable_terminology_normalization
    and query_type in (QueryType.CROSS_CLOUD, QueryType.CONCEPTUAL)
)
```

La expansión cross-provider (parte central de "normalization") requiere **ambos** flags True. Pero en exp7 solo se varía `terminology_normalization` — manteniendo `enable_query_expansion=False` en `cross_cloud_no_norm` (línea 490 de experiment_configs).

¿Qué significa exp7 entonces? El "no_norm" config no tiene expansion NI normalization. El "with_norm" tiene `query_expansion=True` y `terminology_normalization=True` (líneas 507-508). Las configs **varían dos flags, no uno**. La "ablación cross-cloud" está confounded.

Fix: o declarar en el paper que exp7 compara "sin expansion+norm" vs "con expansion+norm" (no solo norm), o re-correr con solo la norm activada.

**Flag 127 — `PROVIDER_KEYWORDS` demasiado amplios (🟡)**

Líneas 57-63:

```python
PROVIDER_KEYWORDS = {
    "aws": ["aws", "amazon", "amazon web services"],
    "azure": ["azure", "microsoft azure", "microsoft"],
    ...
}
```

Problema: cualquier query que mencione "Amazon" (retail) o "Microsoft" (cualquier producto de MS, incluyendo Office/Excel/Windows) se clasifica como AWS/Azure. Falso positivo silencioso en la clasificación de query_type.

Impacto: en datasets de documentación cloud puros es bajo (nadie pregunta por Excel). Pero si el test set contiene queries con estos términos ambiguos, la pipeline los routeará mal.

### Flags — Adaptive Chunker

**Flag 128 — Chunker usa MiniLM-L6; retriever usa BGE-large (mismatch de modelos) (🔴)**

`adaptive_chunker.py` línea 37-38:

```python
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
```

`pipeline_config.py` usa `BAAI/bge-large-en-v1.5` para el retrieval denso. **Dos modelos distintos**:

- **MiniLM-L6** (384 dim, light) decide DÓNDE hacer los cortes semánticos de los chunks
- **BGE-large** (1024 dim, heavy) busca sobre los chunks ya cortados

La geometría de similaridad de MiniLM y BGE no es la misma. Los cortes "semánticos" que MiniLM considera buenos pueden ser malos para BGE. No hay garantía de alineación.

Fix honesto: usar BGE-large también en el chunker. Coste: chunking más lento al indexar (one-time cost). Ganancia: alineación garantizada con retrieval. Este es un fix sencillo y vale la pena antes de enviar.

Si no se corrige, al menos declarar la discrepancia en Methodology.

**Flag 129 — `similarity_threshold=0.5` hardcoded, nunca ablado (🟡)**

Línea 36 + 47. Es el threshold que decide cuándo un drop en similaridad entre frases consecutivas justifica un corte de chunk. 0.5 es un número elegido sin justificación — ni comentario en el código ni mención en el paper.

Si varía este valor, los chunks cambian completamente y todos los experimentos downstream se invalidan. Alto leverage point, cero ablación. Reviewer puede pedir "sensitivity analysis".

**Flag 130 — Code blocks y tablas son "atomic" sin upper bound efectivo (🔴)**

Líneas 219-223:

```python
elif section["content_type"] in ("code", "table"):
    # Code/table blocks are atomic - keep even if large
    adjusted.append(section)
```

Si un bloque de código tiene 3000 tokens, el chunk tiene 3000 tokens. BGE-large tiene **max_seq_length=512 tokens** (hard limit). Los chunks sobredimensionados se **truncan silenciosamente** al embeddear → la representación dense ignora la cola del chunk.

Para BM25 tampoco es ideal: chunks enormes dominan por longitud (term frequency inflado), sesgando el ranking.

Fix: imponer un split en code blocks que respete delimitadores sintácticos del lenguaje (o al menos líneas en blanco). Alternativa cheap: split por líneas cuando el code block exceda `max_chunk_size`.

**Flag 131 — `prepend_heading_path=True` infla tokens de forma no contable (🟡)**

Líneas 410-415 en `_build_final_chunks`:

```python
if self.prepend_heading and heading_path:
    provider_prefix = doc.cloud_provider.upper()
    if doc.service_name:
        provider_prefix += f" > {doc.service_name}"
    full_path = " > ".join([provider_prefix] + heading_path)
    text = f"[{full_path}] {text}"
```

Un chunk "de 500 tokens" no es realmente 500. Tiene `[AWS > EC2 > Launch > Using AMIs > Selecting an AMI]` más contenido. El overhead del path puede ser 20-50 tokens por chunk. En el corpus:

- 46,318 chunks × ~30 tokens de path = ~1.4M tokens de context path adicional
- Para BM25: todos los chunks contienen "AWS", "Azure", "GCP" como términos → distorsiona IDF
- El TF-IDF de "AWS" cae (aparece en todos los chunks de AWS docs)

Fix: no eliminar (el contexto ayuda al reranker y al LLM), pero **reportar el tamaño efectivo** de chunk y reconocer el sesgo en IDF.

**Flag 132 — `_split_into_sentences` termina en `:` como fin de oración (🟡)**

Línea 344: `if stripped.endswith(('.', '!', '?', ':')):`

Los dos puntos no son fin de oración en inglés. "Note: the following steps..." se corta después de "Note:" — rompiendo la oración introductoria de su explicación. Cortes semánticos sobre splits frágiles.

También: nada maneja abreviaciones (e.g., "U.S." se rompe después de "U."). NLTK/spaCy sentence tokenizers resuelven esto.

**Flag 133 — "adaptive" es nombre engañoso (🟡)**

El `AdaptiveChunker` tiene tamaño **fijo** (`chunk_size=500`). Lo "adaptive" es la elección de DÓNDE cortar (hierarchical + semantic), no de QUÉ tamaño usar por documento. El nombre sugiere que el tamaño se adapta al contenido de cada doc, lo cual no ocurre.

Nombre más preciso sería `HierarchicalSemanticChunker`. No crítico, pero relevante si un reviewer busca "adaptive" en el sentido de "variable size per doc" (como LlamaIndex AdaptiveChunker).

**Flag 134 — `_merge_small_sections` muta la lista de entrada (🟢 code smell)**

Línea 382: `sections[i + 1] = merged_section`. Side effect. No rompe nada pero hace el código más difícil de razonar.

### Resumen Módulo 18

**Hallazgos que afectan al paper directamente:**

1. **Flag 124 (expansion asimétrica)** — si el paper dice "query expansion improves retrieval", debe decir "lexical query expansion improves BM25 side of the hybrid retrieval"
2. **Flag 126 (expansion+norm entangled en exp7)** — Flag 99 se refuerza: la ablación de normalization está confounded
3. **Flag 128 (MiniLM vs BGE mismatch)** — el chunker puede estar cortando donde el retriever no quería. Arquitecturalmente sucio
4. **Flag 130 (code blocks pueden exceder 512 tokens)** — truncation silenciosa del embedder

**Hallazgos menores pero publicables:**

- Flag 133: rename el chunker o aclarar qué significa "adaptive"
- Flag 125: decir explícitamente que expansion es diccionario, no LLM

### Tabla de severidad

| # | Flag | Severidad | Acción |
|---|------|-----------|--------|
| **18.1** | **Flag 124 expansion solo BM25** | **🔴 Asymmetric expansion** | **Declarar en Methodology o extender a dense** |
| 18.2 | Flag 125 expansion diccionario YAML | 🟡 Clarificar técnica | Declarar |
| **18.3** | **Flag 126 exp7 varía 2 flags, no 1** | **🔴 Ablación confounded** | **Re-correr con solo norm o declarar** |
| 18.4 | Flag 127 PROVIDER_KEYWORDS amplios | 🟡 Posibles FP | Refinar o declarar |
| **18.5** | **Flag 128 MiniLM-L6 vs BGE-large** | **🔴 Model mismatch chunker/retriever** | **Usar BGE en chunker** |
| 18.6 | Flag 129 similarity_threshold=0.5 | 🟡 Sin ablación | Sensitivity o fijar |
| **18.7** | **Flag 130 code blocks > 512 tokens** | **🔴 Truncación silenciosa BGE** | **Split secundario para code >max_size** |
| 18.8 | Flag 131 heading prepend infla tokens | 🟡 Distorsiona IDF | Reportar tamaño efectivo |
| 18.9 | Flag 132 sentence split con `:` | 🟡 Heurística frágil | NLTK/spaCy |
| 18.10 | Flag 133 "adaptive" nombre engañoso | 🟡 Renombrar en paper | Clarificar |
| 18.11 | Flag 134 _merge_small_sections muta | 🟢 Code smell | No prioridad |

### Pendientes Módulo 18

- [ ] Decidir: extender expansion al dense query o declarar asimetría
- [ ] Clarificar exp7: ¿re-correr con solo norm, o declarar que se comparan paquetes?
- [ ] Cambiar chunker embedding a BGE-large (fix de 1 línea, re-chunking corpus si se hace)
- [ ] Añadir split secundario para code blocks >max_chunk_size
- [ ] Clarificar en paper: "dictionary-based query expansion", "hierarchical-semantic chunking" (no "adaptive")

---

## Módulo 19: `HallucinationDetector` / NLI pipeline

**Alcance**: `src/generation/hallucination_detector.py` (401 líneas) + `src/evaluation/hallucination_metrics.py` (116 líneas) + invocaciones en `rag_pipeline.py:274` y `benchmark_runner.py:278`.

**Relevancia al paper**: Este módulo **sostiene todo el eje de faithfulness**, incluyendo el claim estrella del abstract del paper: "+16.8 percent gain in answer faithfulness over an unnormalized baseline" (main.tex L69). También alimenta las tablas `hall_faithfulness_mean` de exp5, exp6, exp7, exp8 y exp8b.

**Modelo**: `cross-encoder/nli-deberta-v3-small` (cargado vía `sentence_transformers.CrossEncoder`). Umbrales: `ENTAILMENT_THRESHOLD = 0.7`, `CONTRADICTION_THRESHOLD = 0.7`.

### 19.1 FLAG 135 — 🔴🔴🔴 **CATASTRÓFICO**: umbral 0.7 aplicado sobre logits crudos

**El bug más grave encontrado hasta ahora en todo el audit.**

Línea 275-279 del detector:

```python
scores = self.nli_model.predict(
    pairs,
    batch_size=32,
    show_progress_bar=False,
)
```

No se pasa `activation_fn`. No se pasa `apply_softmax=True`. Verificado en `sentence-transformers==5.4.1` (la versión instalada en el sandbox):

> *"If None, the ``model.activation_fn`` will be used, which defaults to :class:`torch.nn.Sigmoid` if num_labels=1, else :class:`torch.nn.Identity`. Defaults to None."*

El modelo `nli-deberta-v3-small` tiene `num_labels=3` → `Identity` por defecto → **`predict()` devuelve logits crudos**, no probabilidades softmax.

Luego el código compara esos logits con `0.7`:

```python
if entailment_score > self.ENTAILMENT_THRESHOLD and entailment_score > best_score:
    best_status = "supported"
```

**Consecuencia**: `0.7` es un umbral razonable para probabilidades softmax ∈ [0,1]. Aplicado a logits crudos típicos de un modelo NLI DeBERTa-small (rango habitual −6 a +6), `0.7` es esencialmente "ligeramente positivo". Simulación con logits ~ N(0, 3):

- P(logit_entailment > 0.7) ≈ **0.41**
- P(logit_contradiction > 0.7) ≈ **0.41**
- Tras softmax: P(p_entailment > 0.7) ≈ **0.24**

Es decir, el detector dispara "supported" o "contradicted" sobre pares aleatorios con probabilidad cercana al 40%. **El clasificador no está mal calibrado: no está calibrado.**

**Impacto sobre claims del paper**:

1. Todos los `hall_faithfulness_mean` (exp5-exp8b) son mediciones con un instrumento roto.
2. El delta **+16.8% normalization** (exp7: 0.292 vs 0.250) se midió con el mismo instrumento roto.
3. Todas las tablas y figuras de hallucination rate / rubric / faithfulness están contaminadas.

**Fix de 1 línea** (pero requiere re-correr TODO lo que toque hallucination):

```python
import torch
scores = self.nli_model.predict(
    pairs,
    batch_size=32,
    show_progress_bar=False,
    apply_softmax=True,  # <-- este flag existe en sentence-transformers >= 2.3
)
```

**Bloqueante para submission del paper.**

### 19.2 FLAG 136 — 🔴 Conteo de claims asimétrico inflando denominadores

Faithfulness = `supported_claims / total_claims`. Pero `total_claims` difiere masivamente entre sistemas porque Hibrido produce respuestas más largas (5 chunks rerankeados → LLM genera contenido más denso).

Evidencia empírica de exp8/results.json (n=200 por sistema):

| Sistema | mean total_claims | mean supported | mean contradicted | mean unsupported | faithfulness |
|---|---|---|---|---|---|
| BM25 | 3.27 | 1.23 | 0.95 | 1.09 | 0.374 |
| Semantic | 4.11 | 1.67 | 1.16 | 1.28 | 0.398 |
| **Hibrido** | **6.89** | **2.35** | **2.15** | **2.40** | **0.369** |

Hibrido genera **2.1× más claims** que BM25 y soporta **91% más claims en términos absolutos** (2.35 vs 1.23), pero su faithfulness *ratio* es casi igual (0.37 vs 0.37). Un sistema que da respuestas cortas con 1 hecho correcto (1/1 = 1.0) aparenta ser más "faithful" que uno que da 5 hechos con 3 correctos (3/5 = 0.6).

**La comparación entre sistemas no es apples-to-apples.** El paper necesita reportar:
- Claim count absoluto por sistema (mostrar que Hibrido es más informativo)
- Supported claims absolutos (Hibrido casi duplica a BM25)
- Faithfulness *condicional* en claim count (normalizado)

De lo contrario, el lector infiere (incorrectamente) que Hibrido "no mejora faithfulness", cuando en realidad responde con más profundidad.

### 19.3 FLAG 137 — 🔴 `claims=[]` → `faithfulness=1.0` infla medias entre 4-6 puntos

Confirmación empírica del Flag 23 ya documentado. Línea 142-149:

```python
if not claims:
    return HallucinationReport(
        total_claims=0, supported_claims=0, ...,
        faithfulness_score=1.0,  # No claims = nothing to hallucinate
        hallucination_rate=0.0,
        suggested_rubric=5,
        method="none",
    )
```

Distribución en exp8:

| Sistema | queries con `claims=0` | % | `method="none"` |
|---|---|---|---|
| BM25 | 13/200 | 6.5% | 13 |
| Semantic | 14/200 | 7.0% | 14 |
| Hibrido | 13/200 | 6.5% | 13 |

Cada una de esas queries aporta **1.0** al `hall_faithfulness_mean`.

Recalculando excluyendo `method="none"`:

| Sistema | faithfulness publicado | faithfulness corregido | delta |
|---|---|---|---|
| BM25 | 0.374 | (0.374×200 − 1.0×13)/187 = **0.331** | **−4.3 pts** |
| Semantic | 0.398 | (0.398×200 − 1.0×14)/186 = **0.352** | **−4.6 pts** |
| Hibrido | 0.369 | (0.369×200 − 1.0×13)/187 = **0.325** | **−4.4 pts** |

El ranking relativo se preserva, pero los valores absolutos caen ~4.4 puntos. **Todas las gráficas faithfulness del paper están infladas en ese orden de magnitud.**

Adicional: para exp7 (base del +16.8%), n=29. Si solo 2 queries caen en `claims=0`, ya son ~7% del total. El delta 0.042 es menor que el ruido inducido por este bug.

### 19.4 FLAG 138 — 🔴 "Supported" gana el tie-break contra "contradicted"

Línea 291-299:

```python
if entailment_score > self.ENTAILMENT_THRESHOLD and entailment_score > best_score:
    best_status = "supported"
    best_score = entailment_score
    ...
elif contradiction_score > self.CONTRADICTION_THRESHOLD and best_status != "supported":
    if contradiction_score > best_score:
        best_status = "contradicted"
```

**Si cualquier chunk da `entailment > 0.7`, el claim es "supported" y ya no se mira la contradicción de otros chunks.** Esto significa que un claim explícitamente contradicho por el chunk #2 pero apoyado tangencialmente por el chunk #4 se reporta como apoyado.

Combinado con Flag 135 (logits crudos), la probabilidad de que al menos 1 de 5 chunks cruce el umbral por ruido es:
- P(≥1 cross con 5 chunks, p=0.41) = 1 − (1−0.41)^5 = **0.93**

Es decir, con logits crudos y 5 chunks, prácticamente cualquier claim se clasifica como "supported" por ruido. Explicaría por qué el faithfulness baseline está alrededor de 0.37 — no es 0.93 porque la claim extraction filtra mucho, pero el piso es claramente artificial.

Práctica estándar en literatura NLI (Honovich et al. 2022 *TRUE*, Maynez et al. 2020 *On Faithfulness and Factuality*):
- Si `max(p_entailment) > threshold` **Y** `max(p_contradiction) < threshold` → supported
- Si `max(p_contradiction) > max(p_entailment)` **Y** > threshold → contradicted
- Si no → unsupported

El código actual viola estas reglas.

### 19.5 FLAG 139 — 🔴 Fallback keyword colapsa "contradicted" en "unsupported"

Líneas 368-374:

```python
if best_overlap >= 0.5:
    status = "supported"
elif best_overlap >= 0.2:
    status = "unsupported"
else:
    status = "unsupported"
```

El fallback nunca puede retornar `"contradicted"`. Si el modelo NLI no se carga (failure silencioso en línea 102), el detector retorna siempre supported/unsupported, y `contradicted_claims = 0` para TODOS los queries. Esto rompe la comparabilidad entre corridas con NLI vs sin NLI.

En exp8 este no es un problema (187/200 usan NLI). Pero para reproducibilidad sin GPU (un reviewer re-corriendo), el fallback cambia la semántica del métrico. **Declarar en paper**: "faithfulness is computed via NLI; fallback to keyword overlap is documented but not used in reported experiments."

### 19.6 FLAG 140 — 🔴 `method: none` contamina el mean, debería excluirse

`hall_faithfulness_mean` en `aggregated_metrics.json` es un promedio simple sobre las 200 queries, incluyendo las que tienen `method="none"` con valor sintético `faithfulness=1.0`.

**Corrección obligatoria**: excluir `method in {"none", "error"}` del numerador y denominador antes de reportar el mean.

Este flag es operativamente igual al Flag 137 pero resaltado explícitamente para la escritura del paper: la tabla debe reportar `n_effective` (queries donde NLI efectivamente corrió y extrajo ≥1 claim) junto al mean, no n=200 plano.

### 19.7 FLAG 141 — 🔴 Tasa de contradiction sospechosamente alta (evidencia colateral de Flag 135)

Hibrido: **2.15 contradicted claims/query** de 6.89 total → **31% de los claims se reportan como contradichos**.

En literatura NLI aplicada a RAG (Manakul et al. 2023 *SelfCheckGPT*, Tang et al. 2024), los sistemas bien calibrados reportan tasas de contradicción del 2-8%. El 31% es imposible a menos que:

(a) El LLM realmente contradice el contexto 31% del tiempo — posible pero extraordinario.
(b) El clasificador NLI está mal calibrado — **esto es lo que pasa** (Flag 135).

Evidencia adicional: BM25 reporta 29% contradicción (0.95/3.27), Semantic 28% (1.16/4.11). Tasas similares en los 3 sistemas sugieren ruido de clasificador, no propiedad del LLM.

**Post-fix Flag 135** (aplicar softmax), la tasa de contradiction debería caer a <10% sobre las mismas predicciones.

### 19.8 FLAG 142 — 🔴 El claim +16.8% descansa sobre n=29 sin test estadístico

exp7 tiene `max_queries=30` (documentado en Flag 98, Módulo 14). En la práctica n=29 por `cross_cloud_no_norm` (confirmado por std/sqrt implícito en `hall_faithfulness_std`).

- `cross_cloud_no_norm.faithfulness_mean = 0.2498` (std 0.278)
- `cross_cloud_with_norm.faithfulness_mean = 0.2916` (std 0.320)

Delta relativo = (0.2916 − 0.2498) / 0.2498 = **16.75%** (el +16.8% es real y trazable a esta fuente).

Problemas:

1. **n=29 es muy pequeño** para detectar deltas de 0.04 con stds de 0.28-0.32. SEM ≈ 0.056. Delta / SEM ≈ 0.75 → test de Wilcoxon con n=29 no alcanzaría p<0.05.
2. **No hay prueba estadística reportada** para esta comparación específica en el paper o en `statistical_tests` de exp7.
3. **Flag 135 afecta esto directamente**: ambos valores (0.250 y 0.292) son mediciones con NLI roto. El delta es señal sobre ruido.
4. **Faithfulness baseline 0.25 es catastrófico por sí mismo**: significa que 75% de los claims en RAG cross-cloud sin norm son "unsupported" o "contradicted" (muy probablemente inflated por Flag 135). Reportar este número como "baseline" es reportar que el sistema no funciona.

**Acción**: o se re-corre exp7 con NLI calibrado y n>=200, o se remueve el +16.8% del abstract. **Dejarlo como está en el paper es hacer un claim insostenible.**

### 19.9 FLAG 143 — 🔴 NLI modelo fuera de dominio (MNLI/SNLI → cloud docs)

`cross-encoder/nli-deberta-v3-small` se entrenó en SNLI + MNLI (noticias, diálogos, cápsulas breves). **Cero exposición a**:

- API signatures (`s3:GetObject`, `azurerm_storage_account`)
- ARNs, IDs regionales (`arn:aws:s3:::bucket`, `us-east-1`)
- YAML / JSON embebido
- Terminología cruzada AWS/Azure/GCP

**Consecuencia esperada**: contradiction false positives inflados. El modelo confunde variación terminológica (`S3 bucket` vs `storage account`) con negación. Sin adaptación de dominio, el NLI no es confiable para cloud docs.

**Literatura**: Laban et al. 2022 *SummaC* reportan caídas de 10-30 puntos F1 al aplicar NLI fuera de dominio. Para cloud docs el gap probablemente es mayor.

**Acción para el paper**: declarar esto como Limitation explícita. Alternativa sería usar un NLI de dominio cloud (no existe público) o fine-tunear sobre un set pequeño de pares claim/evidence de docs cloud.

### 19.10 FLAG 144 — 🟡 `max_length=512` puede truncar premisa+hipótesis silenciosamente

Línea 97: `CrossEncoder(self.NLI_MODEL, max_length=512)`. La premisa (chunk_text) puede llegar a 500 tokens (BGE-large max) y el claim añade 10-50 tokens más. Total > 512 → truncación silenciosa del chunk.

No se mide cuántos pares se truncan en los experimentos. Para chunks cerca del tope (recuerda Flag 130: code blocks pueden exceder max_chunk_size), la evidencia relevante puede quedar fuera de la ventana.

### 19.11 FLAG 145 — 🟡 `SKIP_PATTERNS` incluye `^i ` (skip first-person)

Línea 67: `r"^i "`. Borra cualquier oración que empiece con "I " (case-insensitive). En respuestas estilo "I think X" no se pierde nada fáctico. Pero patrones como "I/O operations..." no se afectan (lowered es `i/o` no `i `). Heurística imperfecta pero baja prioridad.

### 19.12 FLAG 146 — 🟡 `^based on` skip → respuestas de una frase pierden todos los claims

Línea 49: `r"^based on"`. Llama-3.1-instruct con el SYSTEM_PROMPT del repo tiende a empezar **toda** respuesta con "Based on the context, ...". Si la respuesta completa es una sola oración que empieza así, toda la información se filtra → `claims=[]` → `faithfulness=1.0` (Flag 23/137).

Combinación tóxica: respuestas cortas + preamble estándar → scores inflados artificialmente.

Inspección manual recomendada de los 13-14 casos `method="none"` en exp8: casi seguro son respuestas cortas con preamble + contenido filtrado por SKIP_PATTERNS.

### 19.13 FLAG 147 — 🟡 Umbral keyword 0.5 nunca ablado

Línea 369: `if best_overlap >= 0.5: status = "supported"`. Para cloud docs donde el vocabulario es muy repetitivo (cualquier respuesta sobre S3 comparte >50% keywords con cualquier chunk de S3), el umbral es trivial de pasar. Si el fallback se activa, casi todo se clasifica como supported → faithfulness ~1.0 para todo.

No afecta los experimentos publicados (NLI se activa), pero afecta reproducibilidad.

### 19.14 FLAG 148 — 🟡 Rubric 1-5 hardcodeada, no calibrada

Líneas 179-188: umbrales `{0.95, 0.80, 0.50, 0.20}` para los niveles 5/4/3/2/1. Sin justificación en el código ni en el paper. Con Flag 135 inflando faithfulness artificialmente, la distribución del rubric está sesgada.

Tomar esos umbrales de literatura o empíricamente del dataset de referencia. Actualmente es arbitrario.

### 19.15 FLAG 149 — 🟡 `NLI_TIMEOUT = 30` declarado pero nunca usado

Línea 82: `NLI_TIMEOUT = 30  # seconds`. Grep en `hallucination_detector.py` muestra 1 sola ocurrencia (la declaración). No se usa en `_nli_matching` ni en ningún `signal.alarm`, `asyncio.timeout`, o wrapper `concurrent.futures.TimeoutError`. **Dead code.** Si un batch NLI cuelga en GPU, el pipeline completo se queda colgado.

### 19.16 FLAG 150 — 🟡 `hallucination_rate = 1 - faithfulness` es redundante en tablas

Líneas 175-176:

```python
faithfulness = supported / total if total > 0 else 0.0
hallucination_rate = 1.0 - faithfulness
```

Las dos métricas contienen exactamente la misma información. En las tablas del paper, reportar ambas es padding — un reviewer lo marcará como tal. Elegir una (faithfulness convencionalmente).

### 19.17 FLAG 151 — 🟡 Dos rutas de código llaman al detector (pipeline vs evaluación)

- `rag_pipeline.py:274` llama `self.hallucination_detector.check(...)` directamente (propaga excepciones).
- `benchmark_runner.py:278` llama `compute_hallucination_metrics(...)` desde `evaluation/hallucination_metrics.py:42-70`, que **captura todas las excepciones silenciosamente** y devuelve `faithfulness=0.0, method="error"`.

En exp8 no se observaron queries con `method="error"` (solo `nli` y `none`), así que no hay contaminación actual. Pero el diseño es frágil: un OOM en NLI dentro del pipeline se propaga (full query falla), pero dentro del benchmark se convierte silenciosamente en faithfulness=0 → falsos negativos en el mean.

### Resumen Módulo 19

**El módulo NLI es el más contaminado de todo el sistema.** Sostiene el claim estrella del abstract (+16.8%) y la mitad de los números de la evaluación. Tiene un bug de calibración **crítico** (Flag 135) que invalida todas las mediciones hasta re-correr.

**Bloqueantes para submission**:

1. **Flag 135** — `apply_softmax=True` DEBE añadirse. Sin eso, ningún número de faithfulness es defendible.
2. **Flag 137/140** — excluir `method in {"none", "error"}` de los aggregates. Reportar `n_effective`.
3. **Flag 138** — corregir lógica de aggregación entailment/contradiction (no dejar que "supported" se robe el tie-break).
4. **Flag 142** — el +16.8% descansa sobre n=29 con NLI roto y sin test estadístico. **Es el claim más débil del paper.** Candidato #1 a remover del abstract.

**Hallazgos publicables (para Discussion/Limitations)**:

- Flag 143: NLI fuera de dominio (SNLI/MNLI → cloud)
- Flag 136: Claim count asimétrico entre sistemas (no reportar solo ratios)

**Code cleanup**:

- Flag 149: eliminar `NLI_TIMEOUT` o implementarlo
- Flag 150: no reportar faithfulness y hallucination_rate juntos
- Flag 151: unificar ruta pipeline vs evaluación

### Tabla de severidad

| # | Flag | Severidad | Acción |
|---|------|-----------|--------|
| **19.1** | **Flag 135 logits crudos vs threshold 0.7** | **🔴🔴🔴 CATASTRÓFICO** | **Añadir `apply_softmax=True`, re-correr TODO lo que use NLI** |
| **19.2** | **Flag 136 claims count asimétrico** | **🔴 Denominadores distintos** | **Reportar count absoluto + ratio** |
| **19.3** | **Flag 137 claims=[] → faith=1.0 (+4.4 pts)** | **🔴 Aggregate inflado** | **Excluir `method="none"` del mean** |
| **19.4** | **Flag 138 supported gana tie-break** | **🔴 Lógica NLI incorrecta** | **Re-escribir `_nli_matching` con max por clase** |
| 19.5 | Flag 139 fallback sin contradicted | 🔴 No-equivalencia NLI/fallback | Declarar en paper |
| **19.6** | **Flag 140 method="none" contamina mean** | **🔴 Dup del 137 desde otro ángulo** | **Mismo fix** |
| **19.7** | **Flag 141 31% contradiction rate** | **🔴 Evidencia de Flag 135** | **Se resuelve con el fix 135** |
| **19.8** | **Flag 142 +16.8% sobre n=29 sin test** | **🔴 Claim insostenible del abstract** | **Remover del abstract o re-correr** |
| **19.9** | **Flag 143 NLI fuera de dominio** | **🔴 Generalización inválida a cloud** | **Declarar como Limitation** |
| 19.10 | Flag 144 max_length=512 truncación silenciosa | 🟡 Data loss no medido | Medir % truncation |
| 19.11 | Flag 145 `^i ` skip pattern | 🟡 Heurística frágil | Baja prioridad |
| 19.12 | Flag 146 `^based on` skip → claims=[] | 🟡 Combinación tóxica con 137 | Relajar pattern o skip primera oración |
| 19.13 | Flag 147 umbral keyword 0.5 | 🟡 Fallback falso positivo | Ablación o declarar |
| 19.14 | Flag 148 rubric hardcoded | 🟡 Sin calibración | Calibrar o justificar |
| 19.15 | Flag 149 NLI_TIMEOUT dead code | 🟡 Riesgo de hang | Eliminar o implementar |
| 19.16 | Flag 150 faithfulness + hall_rate duplicado | 🟡 Padding en tablas | Reportar solo uno |
| 19.17 | Flag 151 dual code path | 🟡 Diseño frágil | Unificar |

### Pendientes Módulo 19

- [ ] **PRIORIDAD 0**: parchear `hallucination_detector.py:275` con `apply_softmax=True` y re-correr exp5/exp6/exp7/exp8/exp8b (estimado: 24-48h GPU)
- [ ] Añadir `n_effective` y excluir `method in {"none","error"}` en `aggregated_metrics` antes de reportar
- [ ] Re-escribir `_nli_matching` para agregar por máximo de cada clase, no por early-exit en supported
- [ ] Decidir: remover el +16.8% del abstract, o re-correr exp7 con n>=200 + NLI calibrado + Wilcoxon test
- [ ] Añadir Limitations: "NLI model is out-of-domain; faithfulness estimates are approximations"
- [ ] Inspección manual de los ~40 queries con `faithfulness=1.0` para verificar si son genuinos o Flag 137

---

## Módulo 20: Reproducibilidad + `terminology_normalizer.py`

**Alcance**: `requirements.txt`, `setup.py`, `README.md`, propagación de `seed=42`, `src/preprocessing/terminology_normalizer.py` (208 líneas) + `config/terminology_mappings.yaml` (175 líneas).

### 20.1 FLAG 152 — 🔴🔴 Seeds declaradas pero NO propagadas al stack de inferencia

Hay exactamente **tres lugares** donde el código efectivamente usa `seed=42`:

1. `statistical_analysis.py:268` — `rng = np.random.RandomState(seed)` para bootstrap CI. ✓
2. `test_queries.py:420` — `rng = random.Random(seed + hash(provider))` para muestreo de queries. ⚠️ Ver Flag 153.
3. `test_queries.py:547` — idem.

**Almacenado pero nunca usado** (dead state):

- `BenchmarkRunner.__init__:120` — `self.seed = seed`. Grep en `benchmark_runner.py` muestra 1 sola asignación, 0 lecturas posteriores.
- `ExperimentConfig.__init__:57` — `self.seed = seed`. Idem: sólo se serializa al dict de config, nunca se aplica.

**Nunca llamado en ningún punto del proyecto** (grep completo):

- `numpy.random.seed()` / `np.random.seed()`
- `torch.manual_seed()` / `torch.cuda.manual_seed_all()`
- `torch.use_deterministic_algorithms(True)`
- `torch.backends.cudnn.deterministic = True`
- `random.seed()` (a nivel global; sólo hay instancias locales `random.Random(...)`)
- `PYTHONHASHSEED` env var

**Consecuencias**:

- Dos ejecuciones del pipeline completo no producen numéricamente los mismos resultados (salvo por cache LLM, ver Flag 155).
- Inferencia BGE-large / cross-encoder MiniLM son deterministas en CPU (weights fijos, sin dropout en eval), por lo que retrieval+reranking sí es reproducible. Esto salva la mitad del sistema.
- LLM generation (Llama 3.1 via Ollama) es **no-determinista**: temperatura 0.1 > 0 + no-seed → sampling estocástico. Flag 155 detalla cómo se enmascara esto.
- NLI cross-encoder en inference es determinista en CPU — ok.

**Reclamo del paper**: "All experiments use seed=42 for reproducibility" (experiment_configs.py línea 5). **Falso operativamente**. La semilla está en los campos pero no se aplica.

**Fix**:

```python
# Al entrar a BenchmarkRunner.__init__ o a una función init_seeds():
import random, numpy as np, torch, os
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

### 20.2 FLAG 153 — 🔴 `hash(provider)` en test_queries.py rompe reproducibilidad del query set

Línea 420:

```python
rng = random.Random(seed + hash(provider))
```

Python (desde 3.3) **randomiza `hash()` de strings por proceso** por defecto, a menos que `PYTHONHASHSEED` esté fijado en la env var. Dado que `PYTHONHASHSEED` no se fija (Flag 152), `hash("aws")`, `hash("azure")`, `hash("gcp")` dan valores distintos en cada lanzamiento de Python.

**Resultado**: `seed + hash(provider)` es distinto en cada ejecución → el query set es **formalmente no reproducible**.

Esto se enmascara en la práctica porque los queries se cachean a disco (`data/queries/*.json`) tras la primera generación, y ejecuciones posteriores los leen. **Pero la afirmación "replicable" del paper exige que alguien clonando el repo desde cero pueda regenerar los mismos queries — y no puede.**

### 20.3 FLAG 154 — 🔴🔴 `sentence-transformers>=2.7.0` sin tope superior; 2.7 y 5.4 tienen APIs incompatibles

`requirements.txt` línea 13: `sentence-transformers>=2.7.0`. La versión instalada en este sandbox es **5.4.1**.

Entre 2.x y 5.x hubo un rediseño de `CrossEncoder`:

- 2.7: `predict(activation_fct=None)` — devuelve logits crudos para num_labels>1.
- 5.x: `predict(activation_fn=None, apply_softmax=False)` — mismo comportamiento, con parámetro renombrado.

Este rename directamente conecta con **Flag 135** (NLI con logits crudos): si alguien replicando instala `sentence-transformers==2.7.0`, el código en `hallucination_detector.py:275` sigue roto (mismo bug), pero con un parámetro distinto para corregirlo. Un reviewer/replicador podría no darse cuenta. **Pin duro**: `sentence-transformers==5.4.1` (versión exacta con la que se ejecutaron los experimentos).

Adicional: `torch>=2.0.0` también es demasiado laxo. Torch 2.0 vs 2.7 tienen diferencias de cuDNN y operadores. **Pin sugerido**: `torch==2.X.Y` con la versión usada en los runs.

### 20.4 FLAG 155 — 🔴 LLM no-determinista: temperatura 0.1 sin seed, reproducibilidad por cache

`llm_manager.py:306-313`:

```python
response = ollama.chat(
    model=self.model,
    messages=messages,
    options={
        "temperature": temperature,
        "num_predict": max_tokens,
    },
)
```

Problemas:

1. **No se pasa `seed`** al Ollama options. Ollama expone `options.seed` para determinismo (documentación Ollama), pero el código no lo usa.
2. **`temperature=0.1`** (no 0.0). Con sampling multinomial + top-p, incluso temperatura baja genera variabilidad. Llama 3.1 8B puede producir respuestas distintas entre corridas.
3. El cache por `sha256(config_name||prompt||system_prompt||temperature)` **hace la reproducibilidad disco-dependiente**: la primera ejecución define los outputs; corridas subsiguientes leen del caché. Un replicador con caché limpio obtiene respuestas distintas.

**Implicaciones para el paper**:

- Todas las figuras/tables del paper están ancladas a la versión del caché que Enzo tiene localmente.
- Un reviewer/replicador desde cero obtiene respuestas LLM diferentes → claims distintos → faithfulness diferentes → deltas potencialmente distintos.
- El delta +16.8% (Flag 142) depende específicamente de qué respuestas se cachearon en exp7 — pueden no reproducirse.

**Fix**: pasar `"seed": self.seed` al `options` de Ollama en `_generate_ollama`. Y/o temp=0.0 con greedy decoding (pero eso cambia la distribución, no es drop-in).

### 20.5 FLAG 156 — 🔴 `requirements.txt` incompleto: 7 paquetes usados faltan

Enumeración por grep de imports externos en `src/` y `experiments/`:

| Paquete | Usos | En requirements.txt | Severidad |
|---|---|---|---|
| `ollama` | 2 archivos (llm_manager, benchmark_runner) | **NO** | 🔴 Crítico (entire LLM layer) |
| `scipy` | 2 (statistical_analysis) | **NO** | 🔴 Crítico (Wilcoxon, Shapiro-Wilk) |
| `plotly` | 4 (figures/ui) | **NO** | 🔴 Figures del paper |
| `ragas` | 2 (hallucination_metrics opt) | **NO** | 🟡 Optional |
| `datasets` | 1 (hallucination_metrics) | **NO** | 🟡 Required by ragas |
| `bert_score` | 1 (generation_metrics) | **NO** | 🟡 Optional metric |
| `openai` | 1 (llm_manager) | **NO** | 🟡 Optional provider |
| `anthropic` | 1 (llm_manager) | **NO** | 🟡 Optional provider |
| `streamlit` | 7 (ui/*) | **NO** | 🟡 UI, no necesario para replicar benchmarks |

Para `ollama` y `scipy` la consecuencia es: `pip install -r requirements.txt && python run.py exp8` → `ImportError: No module named 'ollama'`.

**Además**: `setup.py:15-27` tiene un `install_requires` **distinto** (más corto) que `requirements.txt`. `pip install .` produce un install **broken** — faltan `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `scikit-learn`, `torch`, `nltk`, `matplotlib`, `seaborn`, `datasketch`, `jupyterlab`, `markdownify`, `gitpython`. Dos fuentes de verdad divergentes es un antipatrón de distribución.

**Fix**: unificar en `pyproject.toml` o forzar `setup.py` a leer `requirements.txt`.

### 20.6 FLAG 157 — 🔴 Versión de Python ambigua / incompatible

- `README.md:9`: badge "Python 3.14".
- `setup.py:14`: `python_requires=">=3.11"`.
- Bytecode en sandbox: `__pycache__/*.cpython-314.pyc` (Python 3.14).

Problemas:
1. Python 3.14 sólo se publicó oficialmente en octubre 2025. La mayoría de wheels pre-compilados (torch, faiss) **no tienen builds para 3.14** al momento de submission (abril 2026; wheel lag típico de 3-6 meses post-release en librerías pesadas). Un replicador con Python 3.14 probablemente fallará al instalar `faiss-cpu` desde wheel → fallback a source build → puede no compilar en su toolchain.
2. La inconsistencia README vs setup.py (3.14 vs ≥3.11) es confusa. ¿Qué versión es la "oficial"?

**Fix**: pin exacto, e.g. `python_requires="==3.11.*"` (la versión donde todos los wheels funcionan). Actualizar README para coincidir.

### 20.7 FLAG 158 — 🟡 Sin `requirements.lock` / `pip freeze` capturando versiones reales

El directorio raíz no contiene:
- `requirements.lock`
- `requirements-dev.txt`
- `pyproject.toml`
- `poetry.lock` / `uv.lock`

Para reproducibilidad científica, el proyecto debería incluir un lockfile generado con `pip freeze > requirements.lock` tras cada run definitivo. Sin esto, flags 154-156 se multiplican: cada dependencia transitiva puede moverse.

### 20.8 FLAG 159 — 🟡 Sin Dockerfile ni contenedor reproducible

El ecosistema de RAG research (Haystack, LangChain, LlamaIndex) **estandariza Docker para reproducibilidad**. CloudRAG no incluye:

- `Dockerfile` (confirmado por `ls /`)
- `docker-compose.yml` (misma carpeta, ausente)
- `.devcontainer/`

Además, Ollama como dependencia significa que el replicador necesita instalar Ollama externamente (no-Python). Un `docker-compose.yml` con Ollama + Python service pinned-versions eliminaría toda la clase de problemas 152-158.

**Acción para el paper**: al menos un párrafo en Methodology describiendo el setup exacto (Python X.Y.Z, CUDA A.B, Ollama vC.D, GPU model), o publicar Docker image/Singularity recipe.

### 20.9 FLAG 160 — 🔴 `TerminologyNormalizer` no normaliza — sólo enriquece metadata

Línea 1-4 del docstring del archivo:

```python
"""
Terminology Normalizer - Detects cloud terms and adds cross-provider metadata.
Does NOT replace text - ADDS metadata for query expansion.
"""
```

**El nombre del archivo/clase es engañoso respecto al paper.**

El paper (`main.tex:69, 204`) habla de **"cross-cloud terminology normalization"** como si fuera un proceso de *canonicalización textual* (ej. reemplazar "S3" → "OBJECT_STORAGE" para que BM25 haga merge cross-provider). Lo que en realidad hace el módulo:

1. Escanea el texto con regex (patrón disyuntivo de todos los términos conocidos).
2. Añade a cada `Chunk` tres campos metadata: `normalized_terms`, `detected_siglas`, `cross_cloud_equivalences`.
3. Construye un inverted index `term → chunk_ids`.
4. Provee `get_query_expansion()` que retorna términos cross-provider para expandir la query.

**No modifica el texto del chunk**. No canoniza. Funcionalmente es un **query expansion dictionary con metadata tagging**.

Consecuencias para el paper:

- La frase "cross-cloud terminology normalization" sugiere a un reviewer un proceso tipo text normalization (stemming, lemmatization, lowercasing, acronym expansion). El técnico verá código y encontrará algo distinto.
- El claim de +16.8% se atribuye a "normalization" pero lo que activa es la expansión de la query con términos equivalentes → lo que mejora es el **recall de BM25**, no la calidad semántica del índice.
- El flag 124 (expansión sólo en BM25) combinado con este renombramiento significa: el paper afirma "normalization at both index and query time" (main.tex:98), pero en índice sólo se añaden tags metadata (no consumidos por BM25 ni por el embedder), y en query sólo se expande el lado BM25 (no el dense).

**Acción**:
1. Renombrar en el paper "normalization" → "cross-provider query expansion via terminology dictionary" o "dictionary-based cross-cloud query augmentation".
2. Remover/corregir el claim "at both index and query time" (main.tex:98).
3. Si se quiere conservar "normalization", implementar efectivamente canonicalización del texto (riesgoso: cambia BM25 y embeddings, requiere re-indexar todo).

### 20.10 FLAG 161 — 🔴 Regex `\b(...)` fallará en términos con puntos, guiones, o camel-case

Línea 64-67:

```python
self.term_pattern = re.compile(
    r'\b(' + '|'.join(escaped) + r')\b',
    re.IGNORECASE,
)
```

`\b` en Python regex delimita por transición word-char vs non-word-char. Problemas para cloud terminology:

- **"Route 53"**: el espacio es no-word → `\bRoute 53\b` detecta. ✓
- **"S3"**: letra+dígito, word-chars → ok. ✓
- **"Cosmos DB"**: ok con `\b...\b`. ✓
- **"Azure SQL Database"**: ok. ✓
- **"Container Apps"**: ok. ✓
- **Pero**: términos como **"arn:aws:s3::..."** no se detectan porque `:` y `::` rompen el `\b`. El prefijo `arn:aws:` cambia la frontera.
- **"google.cloud.storage"**: `.` rompe el `\b`.

Para cloud docs este es un efecto real: ARNs, paths de APIs Google (`cloud.google.com/storage/...`), y namespaces Azure (`microsoft.storage`) no se detectan.

Impacto: el query expansion de cross-provider **no se dispara en queries con ARNs o paths completos**. El dictionary está limitado a menciones "limpias" de servicios.

### 20.11 FLAG 162 — 🟡 Case-insensitivity produce colisiones potencialmente ambiguas

`re.IGNORECASE` + `term_lower = term.lower()` en línea 54 → todos los matches se comparan case-insensitive.

Problema: "SQL" (acronym) vs "sql" (extension file). Ambos se detectan como mismo término. Para un chunk sobre archivos `.sql`, se dispara la expansión cross-provider como si hablara del servicio.

Low-priority pero relevante para cloud docs donde los nombres de archivos y los nombres de servicios se mezclan.

### 20.12 FLAG 163 — 🟡 Sin desambiguación: "VPC" mapea a AWS **y** GCP simultáneamente

El YAML línea 70-71:

```yaml
virtual_network:
    aws: ["VPC", "Virtual Private Cloud", "Amazon VPC"]
    gcp: ["VPC", "VPC Network", "Google VPC"]
```

"VPC" aparece en ambos providers. En `_build_lookups` línea 57: `self.term_to_provider[term_lower] = provider`. **El último provider procesado gana**, overwriting el anterior. La asignación de provider a "vpc" es no-determinista respecto al orden de iteración del dict YAML.

En Python 3.7+ dicts preservan orden de inserción → si `aws:` va antes que `gcp:`, "vpc" queda asignado a gcp. Pero esto es frágil: un reformateo del YAML cambia la asignación.

### 20.13 FLAG 164 — 🟡 Inverted index en memoria, no persiste automáticamente

Método `save_inverted_index(path)` existe (línea 158-161), pero no se llama desde ningún punto del pipeline. El inverted index construido en `process_chunks` se pierde al terminar el proceso. Queries posteriores reconstruyen el índice desde cero.

Acción: llamar `save_inverted_index` al final del preprocessing, cargar si existe en `__init__`.

### 20.14 FLAG 165 — 🟡 README afirma "200 queries" pero abstract afirma "300 queries"

`README.md:42`: "Evaluated on 200 queries across 5 cloud documentation sources".
`main.tex:64, 97`: "300-query expert-curated test set".

Doble canon, doble problema. El README es interno al repo y quien replique lo verá primero. **Contradicción directa con Flag 98.**

### Resumen Módulo 20

**Bloqueantes para submission**:

1. **Flag 152** — propagación de seeds: sin esto la afirmación "reproducible" del paper es falsa.
2. **Flag 154** — pin duro `sentence-transformers==5.4.1` (y `torch==...`): relacionado directo con Flag 135.
3. **Flag 155** — pasar `seed` a Ollama options: la generación LLM es no-reproducible.
4. **Flag 156** — `requirements.txt` + `setup.py` incompletos e inconsistentes.
5. **Flag 160** — renombrar "normalization" en el paper o implementar canonicalización real. Afecta claim del abstract.

**Publicable como Limitation**:

- Flag 157 (Python 3.14 bleeding edge)
- Flag 159 (no Docker)
- Flag 158 (no lockfile)
- Flag 143 (NLI fuera de dominio, ya documentado en M19)

**Code cleanup**:

- Flag 161 (regex `\b` limitaciones)
- Flag 162-163 (colisiones de términos)
- Flag 164 (inverted index no persiste)

### Tabla de severidad

| # | Flag | Severidad | Acción |
|---|------|-----------|--------|
| **20.1** | **Flag 152 seeds declaradas no propagadas** | **🔴🔴 Reproducibilidad falsa** | **Añadir `init_seeds()` en `BenchmarkRunner.__init__`** |
| **20.2** | **Flag 153 `hash(provider)` no-determinista** | **🔴 Query set no reproducible** | **Fijar `PYTHONHASHSEED=42`** |
| **20.3** | **Flag 154 sentence-transformers sin pin** | **🔴🔴 Conecta con Flag 135 NLI** | **Pin exacto a 5.4.1** |
| **20.4** | **Flag 155 Ollama sin seed, temp=0.1** | **🔴 LLM no-determinista** | **Pasar seed a options de Ollama** |
| **20.5** | **Flag 156 requirements.txt incompleto** | **🔴 Install falla (ollama, scipy, plotly missing)** | **Agregar paquetes y unificar setup.py** |
| **20.6** | **Flag 157 Python 3.14 bleeding edge** | **🔴 Wheels no disponibles** | **Pin a 3.11 o 3.12** |
| 20.7 | Flag 158 sin lockfile | 🟡 Dependencias transitivas flotan | Generar `pip freeze` |
| 20.8 | Flag 159 sin Dockerfile | 🟡 Setup manual complejo | Agregar docker-compose |
| **20.9** | **Flag 160 "normalization" es query expansion** | **🔴 Claim del paper engañoso** | **Renombrar o implementar canonicalización** |
| **20.10** | **Flag 161 regex `\b` falla en ARNs/paths** | **🔴 Expansión no se dispara** | **Mejorar regex o split tokens** |
| 20.11 | Flag 162 case-insensitive colisiones | 🟡 "SQL" sobreconectado | Word-sense disambiguation |
| 20.12 | Flag 163 "VPC" ambiguo aws/gcp | 🟡 Asignación provider frágil | Permitir multi-provider |
| 20.13 | Flag 164 inverted index no persiste | 🟡 Code cleanup | Llamar save/load |
| **20.14** | **Flag 165 README=200 vs abstract=300** | **🔴 Contradicción pública** | **Unificar con Flag 98** |

### Pendientes Módulo 20

- [ ] **PRIORIDAD 0**: agregar función `init_seeds()` que fije Python/numpy/torch/cudnn/PYTHONHASHSEED, llamarla en `BenchmarkRunner.__init__` y en `ExperimentConfig.to_dict()` documentar qué seeds están activas
- [ ] Pin `sentence-transformers==5.4.1`, `torch==<X.Y.Z>`, `faiss-cpu==<A.B.C>`, Python a 3.11 o 3.12
- [ ] Agregar a requirements.txt: `ollama`, `scipy`, `plotly`, `ragas`, `datasets`, `bert-score`, `openai`, `anthropic`
- [ ] Unificar `setup.py` con `requirements.txt` (eliminar uno o hacer que lea del otro)
- [ ] Pasar `"seed": self.seed` a Ollama options en `_generate_ollama`
- [ ] Generar y commitear `requirements.lock` con `pip freeze`
- [ ] Decidir: agregar Dockerfile mínimo o declarar Python version + hardware en Methodology
- [ ] Renombrar en paper: "cross-cloud terminology normalization" → "dictionary-based cross-provider query augmentation" (o implementar canonicalización real)
- [ ] Arreglar contradicción 200 vs 300 queries (README y abstract)

---

## Módulo 21: Cross-check `main.tex` ↔ código + Plan de acción correctivo consolidado

**Propósito**: auditoría final palabra-por-palabra del paper contra los 165 flags acumulados. Producir un playbook accionable para decidir qué arreglar, qué declarar, y qué eliminar antes del deadline LACCI 2026 (26 de mayo 2026; hoy 23 de abril 2026 → **33 días**).

### 21.1 Cross-check del Abstract (la zona de mayor riesgo)

| Línea | Claim | Afectado por | Veredicto |
|---|---|---|---|
| 64 | "300-query expert-curated test set" | Flag 98, Flag 165 | **FALSO** — exp1-6,8,8b usan n=200; exp7 n=30. Ningún experimento usa 300. |
| 65 | "P@1 = 0.930, MRR = 0.942, NDCG@5 = 0.736" | Flag 17 | **Numérico correcto pero circular**: cross-encoder MiniLM-L-12 actúa como oracle Y como componente del sistema evaluado. Auto-referencial. |
| 66 | "BM25 (P@1 = 0.785), dense retrieval alone (P@1 = 0.860)" | Flag 17 | Mismo problema; los deltas relativos son creíbles pero los niveles absolutos están inflados por la circularidad. |
| 67 | "Wilcoxon signed-rank test, p < 0.0001" | Flag 103, Flag 108 | Técnicamente cierto para algunas comparaciones; el min p de exp8 es 1e-09. Pero Flag 108: 12 comparaciones sin BH correction en el código (script m16 confirmó que sobreviven BH, pero el **paper no lo reporta**). |
| 68 | "Cohen's d = 0.626" | Flag 107, Flag 113, Flag 114 | **Engañoso**: este es d_z (paired), no d_av (ES convencional). Sólo se aplica a **1 de 12 comparaciones** (NDCG@5 BM25-vs-Hibrido). 11 de 12 comparaciones tienen \|d\|<0.5. |
| 68-70 | "16.8 percent gain in answer faithfulness" | **Flag 135, 142, 160** | **TRIPLE PROBLEMA**: (a) NLI mide con logits crudos (clasificador mal calibrado), (b) n=29, sin test estadístico, (c) "normalization" es en realidad query expansion BM25-only. **Claim más débil del paper.** |
| 71 | "release the dataset, code, evaluation harness, and Docker artifacts" | Flag 159 | **FALSO**: no existe Dockerfile en el repo. |
| 71 | "enable direct replication" | Flag 152-158 | **Operativamente falso**: seeds no propagadas, requirements incompletos, Python 3.14 sin wheels, no lockfile. |

**Decisión abstract**: reescribir. 7 de los 9 claims cuantificables tienen problemas serios.

### 21.2 Cross-check de Contribuciones (L95-102)

| # | Contribución | Afectado por | Veredicto |
|---|---|---|---|
| 1 | "reproducible multi-cloud QA benchmark covering AWS, Azure, and GCP (300 expert queries)" | Flag 98, Flag 153, Flag 165 | **300 es falso** (debe ser 200); "expert-curated" necesita documentar protocolo de anotación (Flag 51 del audit previo); "reproducible" requiere Flag 152-153 fix. |
| 2 | "cross-cloud terminology normalization module operating at both index and query time" | **Flag 160, Flag 124, Flag 131** | **Doble claim falso**: (a) a nivel index sólo se añade metadata que ningún retriever consume (no hay canonicalización textual); (b) a nivel query, la expansión se aplica sólo al lado BM25, no al dense. El diseño es asimétrico, no "symmetric". |
| 3 | "systematic ablation isolating the contribution of BM25, dense retrieval, RRF fusion, cross-encoder reranking, and NLI-based faithfulness filtering" | Flag 95, Flag 99, Flag 118, Flag 126 | **Problemática**: exp6 es forward selection (no leave-one-out), no "isola" cada componente; exp7 varía 2 flags simultáneamente; PROPOSED_HYBRID stacks 5 components → contribuciones individuales no son atribuibles. |
| 4 | "empirical trade-off study across three open-weight LLMs" | Flag 79 | Existe (exp5), pero la figura correspondiente sólo plotea 1 dimensión (faithfulness); el resto son zeros por Flag 69-70. |
| 5 | "open-source reference implementation (code, data, Docker) for one-command replication" | Flag 159, Flag 152-158 | **Docker no existe**. "One-command replication" es una aspiración, no un hecho. |

**Decisión contribuciones**: 4 de 5 necesitan matización o son falsos en su forma actual.

### 21.3 Cross-check de Figuras

Mapeo `\includegraphics` de main.tex → estado empírico de los archivos en `paper/overleaf_ready/figures/`:

| Figura | main.tex | Estado (Flag) | Acción |
|---|---|---|---|
| `fig_end_to_end.png` | L137, L134 | Flag 69-70: **vacía / placeholder** | Dibujar manualmente (~2h Illustrator/draw.io) |
| `fig_retrieval_comparison.png` | L185 | Flag 69-70: **vacía** | Regenerar desde exp8 (script `results_exporter.py`) |
| `fig_ablation_waterfall.png` | L194 | Flag 69-70: **vacía** | Regenerar desde exp6; ver Flag 95/99 sobre naturaleza de ablación |
| `fig_cross_cloud_improvement.png` | L203 | Flag 69-70, Flag 142: **vacía + claim +16.8% inválido** | **No regenerar**: o re-correr exp7 con n≥200 + NLI fix, o eliminar figura+claim |
| `fig_llm_comparison.png` | L212 | Flag 79: radar con 1 eje útil | Regenerar sin las dimensiones con zeros, o limitar a bar chart de faithfulness + latency |
| `fig_latency_breakdown.png` | L221 | Flag 10: cached vs real contamination | Excluir configs con cache (`no_reranker`=0.256ms); reportar sólo configs sin cache |
| `fig_reranker_impact.png` | L237 | Flag 69-70: **vacía** | Regenerar desde exp4 (con fix Flag 10) |

**7 figuras referenciadas; 5 vacías, 1 rota, 1 contaminada. El paper submitted hoy se compila sin errores pero con 7 cajas vacías o engañosas.**

### 21.4 Cross-check del texto

| Sección | Línea | Problema | Flag(s) |
|---|---|---|---|
| Methodology §III.C | L146 | "curated alias table... Applied symmetrically at ingest and query time" | Flag 160, 124 (asimétrico, no symmetrical, y no "alias" sino metadata) |
| Methodology §III.D | L149 | "Adaptive 500-token chunks respecting section boundaries, code blocks, and tables" | Flag 128 (chunker usa MiniLM ≠ retriever BGE), Flag 130 (code blocks pueden exceder 512), Flag 133 (no es "adaptive") |
| Methodology §III.E | L152 | "FAISS IVF-PQ index" | **Verificar**: si en el código se usa IndexFlatIP (sin cuantización), es falso. PQ implica quantization, IVF implica clustering. Revisar `hybrid_index.py` (no auditado exhaustivamente). |
| Experimental Setup §IV.A | L164 | "300 expert-curated queries" | Flag 98, 165 |
| Experimental Setup §IV.B | L167 | "answer accuracy (exact / LLM-as-judge)" | Flag del audit previo: exp5 no tiene accuracy computada, sólo faithfulness |
| Experimental Setup §IV.D | L173 | "Python 3.11, ... Docker image" | Flag 157 (README dice 3.14), Flag 159 (no Docker) |
| Statistical Protocol §IV.C | L170 | "Wilcoxon signed-rank... Cohen's d" | Flag 103 (sin multiple-comparison), Flag 107 (d_z no d_av) |
| Results §V.A | L181 | Table de retrieval usa valores del oracle cross-encoder | Flag 17 |
| Results §V.C | L200 | "Report +16.8% faithfulness gain" | **Claim estrella ROTO** (Flag 135, 142, 160) |
| Conclusion | L250 | "improves P@1 by 14.5 pp over BM25 and 7.0 pp over dense-only on a 300-query multi-cloud benchmark" | Matemática correcta (14.5=93.0-78.5, 7.0=93.0-86.0); "300-query" falso (Flag 98). |

### 21.5 Plan de acción consolidado por prioridad

**Contexto operativo**: 33 días al deadline. 1 autor (Enzo). GPU local RTX 3060 Laptop 6GB.

#### TIER 0 — BLOQUEANTES (sin esto el paper no es defendible)

| # | Flag | Acción | Effort | Deadline sugerido |
|---|------|--------|--------|-------------------|
| T0.1 | **Flag 98, 165** | Cambiar "300 queries" a "200 queries" en todo main.tex, README, y docstrings. 1 sola ocurrencia en abstract (L64), contribuciones (L97), setup (L164), conclusion (L250). | 15 min | D+1 |
| T0.2 | **Flag 135** | Parchear `hallucination_detector.py:275` con `apply_softmax=True`. Re-correr exp5/6/7/8/8b con NLI fijo. Todos los números de faithfulness en paper cambian. | 24-48h GPU | D+5 |
| T0.3 | **Flag 142, 200** | Decisión binaria sobre +16.8%: (a) remover del abstract (L64-70), contribuciones, §V.C, fig_cross_cloud; (b) o re-correr exp7 con n≥200 y NLI fijo + Wilcoxon test reportado. **Recomendación fuerte: opción (a)** porque n=29 no escala en 33 días si hay que chequear calidad manual de 170 queries cross-cloud adicionales. | a: 2h reescritura; b: 1 semana | D+7 |
| T0.4 | **Flag 17** | Declarar explícitamente en §IV.B que el cross-encoder usado para *relevance judgments* es `ms-marco-MiniLM-L-12-v2`, que ES el mismo modelo usado como reranker en PROPOSED_HYBRID. Declarar como threat to validity en §VI. Alternativa (mejor): re-evaluar retrieval metrics usando gold labels humanos de los 200 queries, si existen. | Declaración: 1h; re-evaluación: depende de anotaciones existentes | D+10 |
| T0.5 | **Flag 160, 124** | Renombrar "cross-cloud terminology normalization" → "cross-provider query expansion via a curated terminology dictionary" en todo el paper. Corregir claim "symmetrically at both index and query time" por "applied to the lexical (BM25) side of the hybrid retrieval at query time". | 30 min | D+2 |
| T0.6 | **Flag 13, 69-70, 79** | Mapear 7 figuras referenciadas → regenerar las 5 vacías (desde results_exporter.py tras fix Flag 76) o eliminarlas del paper. Eliminar fig_cross_cloud si se elige opción (a) del T0.3. | 4-8h | D+10 |
| T0.7 | **Flag 76** | Arreglar `results_exporter.py:655` (`row[col] = data[config].get(col, 0)` → raise si la key falta). Previene regresiones en las regeneraciones del T0.6. | 30 min | D+3 |
| T0.8 | **Flag 103, 108** | Añadir Benjamini-Hochberg correction al `statistical_analysis.py` (o al menos reportar los p_bh en la tabla). Scripts `m16_recompute_stats.py` ya existe — integrar al pipeline oficial. Script ya confirma que 12/12 sobreviven BH, así que el resultado no cambia — pero el paper debe reportarlo. | 1h | D+3 |
| T0.9 | **Flag 152, 153, 155** | Añadir `init_seeds()` global que fije PYTHONHASHSEED, Python random, numpy, torch, cudnn, + pasar seed a Ollama options. Re-correr al menos 1 experimento para confirmar determinismo. | 3-4h | D+14 |
| T0.10 | **Flag 159** | Escribir Dockerfile mínimo (Python 3.11 + requirements.txt + ollama) o eliminar el claim "Docker artifacts" del abstract (L70-71). **Recomendación: eliminar claim y sustituir por "installation instructions"** a menos que haya tiempo y hardware para validar Docker. | Opción elim: 5 min; opción Docker: 1 día | D+15 |

**Subtotal Tier 0 estimado: ~2 semanas** (D+15 de los 33 disponibles). Factible pero sin margen.

#### TIER 1 — Declarar como Limitations (§VI.Discussion o §IV.B)

Estos flags no requieren código ni re-runs, sólo párrafos honestos en Discussion/Threats to Validity:

| Flag | Párrafo a añadir |
|------|------------------|
| 17 | "Cross-encoder MiniLM-L-12 is used both as reranker in PROPOSED_HYBRID and as relevance-judgment oracle; retrieval metrics therefore carry circularity that may overstate PROPOSED_HYBRID's advantage." |
| 95, 99, 118, 126 | "Ablations in exp6 are forward-selection (adding components sequentially) rather than leave-one-out; PROPOSED_HYBRID stacks five additional components over baselines (dense, RRF, reranker, expansion, multidim-scoring), so individual contributions are not isolable." |
| 128, 130 | "Chunking uses MiniLM-L6-v2 for semantic boundary detection while retrieval uses BGE-large-en-v1.5; these operate in different embedding spaces. Code blocks and tables >500 tokens are kept atomic and may be truncated by BGE's 512-token limit." |
| 143 | "The NLI model (cross-encoder/nli-deberta-v3-small) is trained on SNLI/MNLI; cloud documentation contains API signatures, ARNs, and cross-provider terminology not represented in training. Faithfulness estimates are therefore approximations; a domain-adapted NLI could yield different results." |
| 146 | "Llama 3.1 responses often begin with 'Based on the context, ...' which the claim extractor filters by design; single-sentence responses with this preamble collapse to zero claims and are excluded from faithfulness aggregates." |
| 155, 157 | "Reproducibility is approximate: Ollama's default sampling with temperature 0.1 is non-deterministic; results shown depend on the LLM response cache. Fresh reruns on Python 3.11 with seed=42 are expected to produce statistically equivalent but not bit-identical outputs." |
| Q4 quantization | "All local LLMs use Q4_K_M quantization for 6GB VRAM. Results with fp16 or Q8 weights may differ systematically (positive direction for faithfulness, per Dettmers et al.)." |

**Effort Tier 1: ~4h de escritura.**

#### TIER 2 — Code cleanup (importante pero no bloqueante)

| Flag | Acción | Effort |
|------|--------|--------|
| 156 | Completar requirements.txt (ollama, scipy, plotly, ragas, datasets, bert-score, openai, anthropic), unificar con setup.py | 1h |
| 154 | Pin exacto `sentence-transformers==5.4.1`, `torch==X.Y.Z` | 15 min + pip freeze |
| 158 | Generar `requirements.lock` con `pip freeze` | 10 min |
| 138 | Re-escribir `_nli_matching` con max-class aggregation correcto | 2h |
| 137, 140 | Excluir `method="none"` de `hall_faithfulness_mean` | 1h |
| 161 | Mejorar regex de `term_pattern` para ARNs y paths | 2h |
| 150 | Eliminar `hallucination_rate` redundante en tablas | 15 min |
| 149 | Eliminar `NLI_TIMEOUT` dead code o implementar timeout | 1h |

**Effort Tier 2: ~8h. Se puede dejar para post-submission si el tiempo aprieta.**

#### TIER 3 — Nice-to-have (post-submission / v2)

Todos los 🟡 restantes del audit (aprox 50 flags). Mejoran la calidad del codebase para el release público pero no bloquean el paper.

### 21.6 Reescritura sugerida del Abstract (post-Tier 0)

Para dar una referencia concreta de cómo se ve un abstract consistente con el audit post-correcciones:

> "Retrieval-Augmented Generation (RAG) systems are a promising approach to ground Large Language Models on specialized technical corpora, yet retrieval quality over multi-vendor cloud documentation remains under-studied. We present CloudRAG, a hybrid retrieval pipeline for technical documentation spanning Amazon Web Services, Microsoft Azure, and Google Cloud Platform. CloudRAG combines sparse lexical retrieval (BM25) with dense semantic retrieval (BGE-large-en-v1.5) fused through Reciprocal Rank Fusion and a cross-encoder reranking stage (ms-marco-MiniLM-L-12-v2), complemented by a dictionary-based cross-provider query expansion module. On a **200**-query test set evaluated with **the same cross-encoder used as reranker (a known circularity we discuss in §VI)**, the hybrid pipeline achieves NDCG@5 = 0.736 vs.\ 0.554 for BM25 and 0.661 for dense retrieval alone, with **the BM25-vs-hybrid NDCG@5 comparison meeting a medium effect threshold (Cohen's d_av = |0.43|, Wilcoxon p<1e-8, Benjamini-Hochberg corrected)**. **We do not claim a consistent medium effect across metrics: 11 of 12 pair-wise comparisons yield small effect sizes.** We also present an NLI-based faithfulness analysis as an exploratory signal **(Limitations)** and release the code, 200-query dataset, and replication scripts."

Este abstract es mucho más débil que el actual. Pero es **defensible**. El actual no lo es.

### 21.7 Decisión estratégica para el submission

**Tres caminos**:

**Camino A — Submit a LACCI 2026 con Tier 0 completo** (33 días, effort ~2 semanas):
- Riesgo alto: reescribir 40% del paper, re-correr todos los experimentos NLI, regenerar 7 figuras.
- Upside: paper publicado en el venue objetivo.
- Downside: el trabajo post-audit es un paper **fundamentalmente distinto y más modesto** que el draft actual. Los resultados después del fix NLI pueden ser más débiles o más fuertes (imposible predecir).

**Camino B — Submit incompleto a LACCI 2026**:
- Descartado. El paper actual tiene ~10 claims falsos o insostenibles. Un reviewer técnico los detectará. Probabilidad de reject ≈ alta.

**Camino C — Pasar a un venue posterior** (e.g. IEEE TNSM, EMNLP Findings, ACL workshop en 2H 2026):
- Permite Tier 0 + Tier 1 + Tier 2 + re-corridas completas con gold labels humanos.
- Quita la presión de 33 días.
- Deadline IEEE TNSM regular: rolling. EMNLP ~junio. ACL workshops: variable.

**Recomendación del auditor**: **Camino A si y sólo si** Enzo puede comprometerse a 2 semanas exclusivas (~60h de trabajo) desde hoy. Si hay cualquier otra obligación (examenes, trabajo, tesis), pasar a Camino C. Someter el paper actual a LACCI sin las correcciones T0.2, T0.3, T0.5 es pedir reject.

### Resumen Módulo 21

El paper en su estado actual tiene **165 flags** repartidos entre 20 módulos del código y 40+ claims del main.tex. De esos 165:

- **~15 catastróficos** (🔴🔴🔴 o 🔴🔴) que invalidan claims específicos del abstract o contribuciones.
- **~50 materiales** (🔴) que requieren corrección o declaración como limitation.
- **~70 estilísticos / code cleanup** (🟡) que no bloquean submission pero mejoran el release.
- **~30 no-issues o cerrados** como false positives tras investigación.

**El hallazgo que condiciona todo lo demás es Flag 135**: el NLI usa logits crudos contra umbral 0.7 → todos los números de faithfulness (incluyendo el +16.8% del abstract) están medidos con un clasificador mal calibrado. Sin ese fix, ningún número de faithfulness del paper es defendible.

**Auditoría cerrada**. 21 módulos completos. El audit_findings.md queda como fuente de verdad única. El siguiente paso deja de ser auditar y pasa a ser ejecutar Tier 0.

### Pendientes Módulo 21 / Cierre del audit

- [ ] Decisión estratégica: Camino A (LACCI 33 días) vs Camino C (venue posterior)
- [ ] Si Camino A: ejecutar Tier 0 empezando por T0.2 (fix NLI + re-run) porque es el camino crítico
- [ ] Producir `CHANGELOG_paper.md` documentando qué claims se eliminaron y cuáles se añadieron como Limitations (para trazabilidad de review)
- [ ] (Opcional) Pre-registrar el experimento corregido: anunciar el fix del NLI, commit con la corrida post-fix, para que los reviewers vean la honestidad del proceso

---

## Cierre del Audit (20+1 módulos, 165 flags)

**Fecha de cierre**: 23 de abril 2026.
**Archivos generados**:
- `paper/audit_findings.md` (este documento, fuente de verdad única)
- `paper/audit_outputs/exp8_stats_corrected.csv` (recomputo BH-FDR de exp8)
- `scripts_audit/m16_recompute_stats.py` (script reutilizable)

**Próximos pasos inmediatos**: dejar de auditar. Empezar a ejecutar. El Tier 0 es ahora el único pendiente relevante para el paper.


