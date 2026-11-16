[Fable 5.2] | 2026-11-16 11:07 | rama `fable/rag-auditoria` @ `0ea4d04` | tarea F-propuesta (sin número, ver §9 del informe de auditoría) | modo: sondeo
Capas: 1 secretos=exit 0 | 2 tests=no repetida (este cambio solo añade documentación; última corrida completa: 2406 pasan, 0 fallan, 6 omitidas, 8 xfail estrictos) | 3 docs=exit 0
Cambios: este documento y dos notas de corrección en el informe de conceptos | Sin verificar: no tengo el A.3 (vive fuera del repo); no comprobé si el paper ya está en IEEE Xplore; del PDF certificado solo cotejé la página 3

# Afirmación principal sobre el híbrido: propuesta frase por frase

Enzo pidió preparar el cambio de la afirmación principal del paper, frase por frase, a la luz de
`exp20_rerank_arms`. Este documento **no cambia ninguna prosa**. Lista cada frase que habla de la
ventaja del híbrido, dice si se sostiene y, donde hace falta, propone un texto. Cada fila tiene un
identificador para que el OK sea frase por frase («OK S1, S7c; no S13»).

Continúa el [informe de conceptos](fable-rag-conceptos-2026-11.md) (§5 y §6).

## 1. Tres hechos que cambian el encargo

**1. El paper publicado ya atribuye la ganancia al reranker.** El texto citable es el camera-ready
`docs/Paper_IEEE_RAG_Hibrido_LACCI_v9.tex` (PDF certificado `docs/2026305869.pdf`). En el resumen,
en los resultados, en la discusión y en la conclusión dice que la ventaja sobre el denso procede
del reranking y no de la fusión RRF. exp20 **no contradice ninguna frase publicada sobre la
atribución**. Lo que hace es sustituir una inferencia (comparar el híbrido antes y después del
reranker) por un control directo (el mismo reranker sobre el top-50 denso), y eso vuelve
imprecisas por omisión algunas frases.

**2. Corrección mía.** En el informe de conceptos (§6, decisión 2) escribí que cambiar la
afirmación principal era «la consecuencia más seria de esta ronda». Lo escribí a partir de los
ledgers y del resumen de resultados, sin cotejar el camera-ready. Cotejado, es falso: la
afirmación principal del paper se sostiene. Lo serio de exp20 no es que el paper esté mal, sino
que la frase informal «el híbrido supera al denso» no puede usarse sin el matiz. Queda anotado en
ese informe.

**3. El texto publicado ya no se puede editar.** El camera-ready está certificado y, según el
README, el congreso fue del 3 al 6 de noviembre de 2026. Los textos propuestos sirven para cuatro
destinos reales, y tú decides cuáles:

| Destino | Quién lo aplica |
|---|---|
| A.3 y tesis (fuera del repo) | Tú; aquí van los textos en español (§5) |
| README del repo público | Yo en esta rama, si das el OK (§4) |
| Defensa y diapositivas | Tú |
| Versión extendida o fe de erratas | Tú; aquí van los textos en inglés (§3) |

Aparte: `paper/overleaf_ready/main.tex` **no es el paper**. Es un esqueleto antiguo con cifras ya
retiradas (P@1 = 0,930; «+16,8 %»; 200 consultas; Llama 3.1, Mistral y Qwen 2.5). No lo toqué
(`paper/**` es de solo lectura para esta rama). Conviene que no se tome como fuente; si quieres
archivarlo con un aviso, es una decisión tuya (ADR-0011: se mueve y se declara).

## 2. Cifras de apoyo

Todas salen de `experiments/results/exp20_rerank_arms/analysis.json` (evidencia firmada el
2026-11-16). Juez: `bge-reranker-large`. 194 consultas.

| Brazo | NDCG@5, pool de 7 sistemas | NDCG@5, pool de 4 (tabla del paper) |
|---|---|---|
| Léxico (BM25) | 0,428 | 0,442 |
| Híbrido solo RRF | 0,582 | 0,603 |
| Denso (BGE) | 0,603 | 0,624 |
| Léxico + reranker | 0,646 | no existe |
| Denso + reranker | 0,699 | no existe |
| Reranker sobre la unión de pools | 0,706 | no existe |
| Híbrido + reranker (pipeline completo) | 0,714 | 0,740 |

| Contraste (pool de 7) | Diferencia | d_z | p (Wilcoxon) | p agrupando por plantilla |
|---|---|---|---|---|
| Híbrido + reranker − denso + reranker | +0,016 | 0,11 | 0,075 | 0,16 |
| Denso + reranker − denso | +0,095 | 0,37 | 2 × 10⁻⁶ | 4 × 10⁻⁶ |
| Híbrido + reranker − híbrido solo RRF | +0,133 | 0,53 | 9 × 10⁻¹¹ | 8 × 10⁻⁸ |
| Híbrido solo RRF − denso | −0,022 | −0,08 | 0,19 | 0,37 |
| Híbrido + reranker − léxico + reranker | +0,068 | 0,33 | 2 × 10⁻⁵ | 0,003 |

El intervalo del primer contraste es de −0,004 a +0,035 (95 %, t pareada; se obtiene de la media
y del d_z firmados: desviación 0,140, error estándar 0,010, 193 grados de libertad).

Tres reglas para no escribir algo falso con estas cifras:

1. **No mezclar pools.** El NDCG de este proyecto es relativo al pool. El 0,740 del paper y el
   0,699 de exp20 no se pueden poner en la misma frase: en el pool de exp20 el pipeline completo
   vale 0,714. Se cita la diferencia, o los dos niveles del mismo pool.
2. **«No significativo» no es «igual».** El intervalo deja abierto que la fusión aporte hasta
   +0,035. Lo demostrado es que no se detecta ventaja y que, si existe, es como mucho de 0,035
   frente a los 0,095 que aporta el reranker.
3. **El híbrido sí ayuda frente al léxico.** Con el mismo reranker, el híbrido supera a BM25 en
   +0,068, significativo también agrupando por plantilla. Lo que no se demuestra es la ventaja
   frente al denso. «La fusión no sirve» sería tan incorrecto como «la fusión es la que gana».

Dato exploratorio, sin corregir por comparaciones múltiples: en las 115 consultas sin filtro de
proveedor la diferencia híbrido − denso (ambos con reranker) es +0,026 (p = 0,007); en las 79 con
filtro es +0,000 (p = 0,66). No recomiendo citarlo como hallazgo.

## 3. Camera-ready v9, frase por frase

Las líneas son las de `docs/Paper_IEEE_RAG_Hibrido_LACCI_v9.tex`. La página 3 del PDF certificado
se cotejó a la vista y coincide con el `.tex` en S6, S7, S8 y S11.

### 3.1 Frases que se sostienen y no necesitan cambio

| ID | Línea | Frase (extracto) | Por qué se sostiene |
|---|---|---|---|
| S0 | 15 | Título: «…Comparative Evaluation of Lexical, Semantic, and Hybrid Pipelines…» | Descriptivo; no afirma superioridad |
| S2 | 41 | «…how much of the apparent hybrid advantage is real rather than an artifact of judging relevance with the pipeline's own reranker?» | Es una pregunta; exp20 la responde mejor |
| S3 | 42 | «…the hybrid advantage over dense is real but inflated under the circular judge and stems from reranking, not fusion» | exp20 lo confirma con control directo |
| S4 | 44 | «Hybrid methods… consistently match or exceed the better single paradigm» | Afirmación sobre la literatura; aquí la fusión iguala al denso (−0,022; n.s.) |
| S7a | 87 | «…the reranked hybrid outperforms both baselines on every metric (NDCG@5 0.442→0.740; P@5 0.443→0.637)» | Cierto en la tabla firmada; los dos brazos base no llevan reranker (lo cubre S5) |
| S7b | 87 | «…the hybrid before reranking (RRF only) does not beat dense (NDCG@5 0.603 vs. 0.624), so the gain is attributable to the cross-encoder, not the fusion» | exp20 lo confirma: −0,022 (n.s.) y +0,095 del reranker sobre el denso |
| S8 | 88–103 | Tabla II (cuatro sistemas, pool de 4) | Cifras firmadas de exp11; no se toca. Añadir un brazo cambiaría todas las cifras de la tabla (regla 1) |
| S10 | 143 | Título de la subsección: «…and the Gain Is from Reranking» | Confirmado |
| S11b | 144 | «Decomposing the pipeline shows RRF fusion alone does not beat dense, whereas the cross-encoder does… cautioning against attributing gains to “hybridization” in the aggregate» | Confirmado; es la frase más prudente del paper |

### 3.2 Frases que se sostienen y conviene afinar

**S1. Resumen (línea 33).**

- Actual: «…the reranked hybrid reaches NDCG@5 0.740 (versus 0.995 under the circular judge), and
  its advantage over the dense baseline, though real, originates in the reranking stage rather
  than in RRF fusion.»
- Problema: «the dense baseline» no lleva reranker. La frase es cierta, pero un lector entiende que
  el pipeline híbrido es mejor que una alternativa densa comparable.
- Propuesta: «…the reranked hybrid reaches NDCG@5 0.740 (versus 0.995 under the circular judge),
  and its advantage over the unreranked dense baseline originates in the reranking stage rather
  than in RRF fusion: applying the same reranker to dense retrieval alone yields no significant
  difference (ΔNDCG@5 = +0.016, 95% CI −0.004 to +0.035).»

**S5. Escenarios (línea 83).**

- Actual: «…**lexical** (BM25 top-5), **dense** (BGE top-5), and **hybrid** (RRF + cross-encoder,
  top-5).»
- Problema: aquí vive el confundido. Solo un escenario lleva reranker y la frase no lo advierte.
- Propuesta (frase añadida a continuación): «Only the hybrid scenario includes the cross-encoder;
  a control that applies the same reranker to the lexical and dense top-50 is reported
  separately.»

**S7c. Resultados (línea 87, final del párrafo).**

- Actual: «…the RRF-only hybrid does not differ from dense (d_z = −0.08, p = 0.240), and reranking
  alone accounts for the gain (d_z = +0.52, p < 0.001).»
- Estado: se sostiene. En exp20 las mismas comparaciones dan d_z = −0,08 y d_z = +0,53. «Alone» es
  algo fuerte: el reranker explica el 86 % de la distancia entre el pipeline y el denso (0,095 de
  0,111) y la fusión el 14 % restante, no significativo.
- Propuesta (frases añadidas a continuación): «A direct control confirms this. Scored in a common
  seven-system pool, the same reranker applied to the dense top-50 reaches NDCG@5 0.699 against
  0.714 for the full pipeline (Δ = +0.016, d_z = 0.11, p = 0.075), while it lifts dense retrieval
  by +0.095 (p < 0.001). Fusion does matter relative to lexical retrieval: with the same reranker,
  the hybrid exceeds BM25 by +0.068 (p < 0.001). NDCG levels depend on the pool, so 0.714 here
  corresponds to 0.740 in Table II.»

**S11a. Discusión (línea 144, primera frase).**

- Actual: «Hybrid retrieval delivers large relative gains over BM25 (NDCG@5 +67% under the
  independent oracle)…»
- Estado: la aritmética es correcta (0,740 / 0,442 = 1,675). El sujeto es impreciso: el +67 %
  incluye al reranker. Con el mismo reranker, el híbrido supera a BM25 en +0,068 (un 11 %, en el
  pool de 7).
- Propuesta: «The reranked hybrid pipeline delivers large relative gains over unreranked BM25
  (NDCG@5 +67% under the independent oracle)…»

**S11c. Discusión (línea 144, última frase).**

- Actual: «…doing so reduces the hybrid's score to a still-leading 0.740.»
- Problema: «still-leading» vale entre los cuatro sistemas comparados. Con el control de exp20, un
  sistema denso con reranker no difiere de forma significativa.
- Propuesta: «…doing so reduces the hybrid's score to 0.740, still the highest of the four systems
  compared, although a dense-plus-reranker control does not differ significantly from it.»

**S13. Conclusión (línea 153).**

- Actual: «Hybrid retrieval reaches NDCG@5 0.740 under an independent oracle; the advantage over
  dense is real but moderate and originates in reranking, not fusion.»
- Propuesta: «The reranked hybrid pipeline reaches NDCG@5 0.740 under an independent oracle; its
  advantage over unreranked dense retrieval is real but moderate and originates in reranking, not
  fusion: with the same reranker, dense retrieval alone does not differ significantly from it.»

### 3.3 Error de hecho, ajeno a exp20

**S6. Descripción del conjunto de consultas (línea 84; página 3 del PDF certificado).**

- Actual: «…provider counts overlapping for the 42 multi-cloud queries…»
- Dato: `data/evaluation/test_queries.json` tiene 194 consultas; 169 nombran un proveedor, 8
  nombran dos y 17 nombran tres. Son **25 consultas multi-proveedor**, las mismas de
  `data/evaluation/cross_cloud_subset.json`. El 42 es el número de etiquetas de más
  (100 + 82 + 54 = 236; 236 − 194 = 42 = 8 × 1 + 17 × 2), no el de consultas.
- Origen: la frase entró el 2026-08-30 entre los arreglos del camera-ready
  (`paper/summer_ablation_log.md`, sección de ese día, «Recomendables adoptados»), para explicar
  por qué los conteos suman 236.
- Alcance: es un error en la descripción del conjunto. No toca ninguna métrica ni contraste.
- Propuesta: «…provider counts overlap because 25 queries name more than one provider (8 name two
  and 17 name all three)…»
- Como la cifra ya está entregada, la decisión es tuya: anotarlo en `paper/correction_log.md`,
  corregirlo en el A.3 y decidir si merece una nota de erratas en el README.

### 3.4 Límites que faltan

**S12. Limitaciones (línea 151).**

- Actual: «The oracle is a cross-encoder rather than human-annotated ground truth; we mitigate the
  self-consistency bias by reporting an independent oracle…»
- Problema: «independent» es exacto respecto del reranker del pipeline, pero el juez
  (`bge-reranker-large`) es de la misma familia que el recuperador denso (`bge-large-en-v1.5`).
  Además faltan dos límites medidos: el NDCG es relativo al pool y las consultas no son
  independientes.
- Propuesta (frases añadidas): «The independent oracle differs from the pipeline's reranker but
  belongs to the same model family as the dense retriever (BGE); whether this favors the dense leg
  cannot be resolved without human relevance labels. NDCG is computed against the pool of chunks
  retrieved by the compared systems and is therefore relative to that pool. The 194 queries fall
  into 74 groups that share a question stem, so they are not fully independent; the retrieval
  conclusions hold when the tests are clustered by group.»
- Nota: «74 groups that share a question stem» describe lo que mide la sonda (el arranque de la
  pregunta con el proveedor enmascarado). Las consultas por proveedor sí se generan con plantillas
  (`PROVIDER_TEMPLATES`, `src/evaluation/test_queries.py:269`), pero el 74 es el número de grupos
  que mide la sonda, no un conteo de las plantillas del código. Por eso la frase no dice
  «generated from 74 templates».

## 4. README del repo

El README no es prosa del paper, pero es público y repite la afirmación. No lo edité.

| ID | Línea | Actual (extracto) | Propuesta |
|---|---|---|---|
| R1 | 35 | «**Why hybrid?** … CloudRAG fuses both to get the best of each approach — then re-ranks with a cross-encoder for precision.» | «**Why hybrid?** Pure keyword search misses semantically related content; pure embedding search can miss exact technical terms. CloudRAG fuses both and then re-ranks with a cross-encoder. In our own evaluation the measurable gain comes from the re-ranker: fusion clearly helps over lexical-only retrieval, but with the same re-ranker it is not significantly better than dense-only retrieval (exp20).» |
| R2 | 67–70 | «Hybrid(post) > Dense is significant… the advantage comes from the **reranking stage**, not the RRF fusion (pre-rerank ≈ Dense, n.s.).» | Se sostiene. Frase añadida: «A same-reranker control (`exp20_rerank_arms`, seven-system pool) puts dense + reranker at 0.699 against 0.714 for the full pipeline (Δ = +0.016, p = 0.075, n.s.).» |
| R3 | 326 | Tabla de experimentos, fila `exp10-11` | Fila nueva: «exp20 — Same-reranker control arms (retrieval only) — Dense + reranker does not differ significantly from hybrid + reranker (Δ +0.016); the reranker adds +0.095 over dense» |
| R4 | 48 | «19 logical experiments (exp3-exp19b + exp8b), stored in 20 result directories» | Con exp20 firmada son 20 experimentos lógicos y 21 directorios (contados en `experiments/results/`) |

R4 no es de redacción: el conteo queda desfasado en cuanto esta rama se integre.

## 5. Textos en español para el A.3 y la defensa

No tengo el A.3: vive fuera del repo. Estos son los equivalentes en español de las propuestas de
§3. Si pegas los párrafos del A.3 que hablan del híbrido, hago la correspondencia frase por frase.

- **A-S1 (resumen):** «El híbrido con reranker alcanza un NDCG@5 de 0,740 con el oráculo
  independiente (0,995 con el juez circular). Su ventaja sobre la recuperación densa sin reranker
  procede de la etapa de reranking y no de la fusión RRF: al aplicar el mismo reranker a la
  recuperación densa, la diferencia no es significativa (+0,016; IC 95 %: −0,004 a +0,035).»
- **A-S5 (diseño):** «Solo el escenario híbrido incluye el reranker. Un control que aplica el mismo
  reranker al top-50 léxico y al top-50 denso se reporta aparte (`exp20_rerank_arms`).»
- **A-S7 (resultados):** «Un control directo lo confirma. Calificados en un pool común de siete
  sistemas, el reranker sobre el top-50 denso alcanza 0,699 frente a 0,714 del pipeline completo
  (+0,016; d_z = 0,11; p = 0,075), mientras que eleva la recuperación densa en +0,095
  (p < 0,001). La fusión sí importa frente a la recuperación léxica: con el mismo reranker, el
  híbrido supera a BM25 en +0,068 (p < 0,001). El NDCG depende del pool: el 0,714 de este control
  corresponde al 0,740 de la tabla principal.»
- **A-S11 (discusión):** «Con un juez independiente, el híbrido baja a 0,740. Sigue siendo el más
  alto de los cuatro sistemas comparados, aunque un control denso con reranker no difiere de él de
  forma significativa.»
- **A-S6 (conjunto de consultas):** «Los conteos por proveedor se solapan porque 25 consultas
  nombran más de un proveedor (8 nombran dos y 17 nombran los tres).»
- **A-S12 (límites):** «El oráculo independiente es distinto del reranker del pipeline, pero
  pertenece a la misma familia de modelos que el recuperador denso (BGE); sin juicios humanos de
  relevancia no se puede decidir si eso favorece a la pierna densa. El NDCG se calcula respecto
  del pool de fragmentos recuperados por los sistemas comparados y es, por tanto, relativo a ese
  pool. Las 194 consultas forman 74 grupos que comparten el arranque de la pregunta, así que no son
  del todo independientes; las conclusiones de recuperación se mantienen al agrupar las pruebas.»
- **A-S13 (conclusión):** «El pipeline híbrido con reranker alcanza un NDCG@5 de 0,740 con un
  oráculo independiente. Su ventaja sobre la recuperación densa sin reranker es real pero moderada
  y procede del reranking, no de la fusión: con el mismo reranker, la recuperación densa sola no
  difiere de él de forma significativa.»

Una frase para la defensa, si te preguntan «¿entonces para qué el híbrido?»: «El pipeline completo
es el mejor brazo medido y supera con claridad al léxico. Frente al denso, el componente que
decide es el reranker; con estos datos no puedo afirmar que la fusión añada algo, ni descartar que
añada hasta 0,035 de NDCG@5.»

## 6. Qué necesito de ti

1. El OK o el rechazo de cada fila: S1, S5, S7c, S11a, S11c, S13, S6, S12 y R1 a R4.
2. Para S6: si se anota en `paper/correction_log.md` y si quieres una nota de erratas pública.
3. Para el A.3: los párrafos actuales, si quieres la correspondencia exacta.

Con tu OK aplico en esta rama solo lo del README (R1 a R4), en un commit. No edito `paper/**` ni
el `.tex` del camera-ready.

## 7. Verificaciones de este documento

| Comprobación | Resultado |
|---|---|
| Cifras de §2 frente a `analysis.json` firmado | Leídas del archivo; el intervalo se recalculó desde la media y el d_z |
| Frases de §3 frente al `.tex` del camera-ready | Copiadas de las líneas indicadas |
| Página 3 del PDF certificado | Coincide con el `.tex` en S6, S7, S8 y S11 |
| Conteo de S6 | 169, 8 y 17 consultas con uno, dos y tres proveedores; etiquetas 100, 82 y 54 |
| Conteo de R4 | 21 directorios `exp*` en `experiments/results/` |
| Capas 1 y 3 | exit 0 |
