[Fable 5.2] | 2026-11-16 01:12 | rama `fable/rag-auditoria` @ `0e21e32` | tarea F-propuesta (sin número, ver §9 del informe de auditoría) | modo: ejecución
Capas: 1 secretos=exit 0 | 2 tests=exit 0 (2403 pasan, 0 fallan, 6 omitidas, 8 xfail estrictos) | 3 docs=exit 0
Cambios: `scripts/audit/exp20_rerank_arms.py`, `scripts/audit/fable_concept_probes.py`, sus tests, las salidas en `output/audit/` y este informe | Sin verificar: no se ejecutó ningún LLM, NLI ni HHEM; exp20 no está firmada; `init.sh` completo corrió sobre `0e21e32`, antes del commit que solo añade documentación (capas 1 y 3 repetidas después)

# Lagunas y errores de concepto en la evaluación de CloudRAG

Enzo pidió, además de explicar las diferencias de métricas, buscar en todo el código y en los
planteamientos lagunas, errores de concepto o malentendidos del mismo tipo que la circularidad del
oráculo: cosas que se dieron por buenas y que una revisión lógica y de código desmiente o matiza.

Método: cada sospecha se convirtió en una sonda de código que da un número refutable. Nada se
afirma por opinión. Cada resultado lleva una etiqueta:

- **Demostrado**: se sigue de la aritmética o del código, con test.
- **Medido**: número obtenido de la evidencia firmada o de la corrida exp20.
- **No resuelto**: la sonda no puede decidirlo; se dice qué haría falta.
- **Refutado**: la sospecha se contrastó y los datos no la sostienen.

Ninguna cifra firmada cambió. La corrida exp20 **no es evidencia firmada**: vive en
`output/audit/exp20_rerank_arms_2026-11-15/` y nunca se escribió en `experiments/results/`.

Código: [`scripts/audit/exp20_rerank_arms.py`](../../scripts/audit/exp20_rerank_arms.py) y
[`scripts/audit/fable_concept_probes.py`](../../scripts/audit/fable_concept_probes.py). Salidas:
[`analysis.md` de exp20](../../output/audit/exp20_rerank_arms_2026-11-15/analysis.md) y
[`fable_concept_probes_2026-11-15.md`](../../output/audit/fable_concept_probes_2026-11-15.md).
Este informe continúa [el del porqué](fable-rag-porque-2026-11.md) y
[el de auditoría](fable-rag-auditoria-2026-11.md).

## 1. Respuesta corta

1. **Con el mismo reranker, el híbrido no supera de forma demostrable al denso.** NDCG@5 con el
   oráculo independiente: 0,714 el híbrido y 0,699 el denso con reranker. Diferencia +0,016
   (IC 95 %: −0,004 a +0,035; p = 0,075; p = 0,16 agrupando por plantilla). La ventaja que hoy se
   atribuye al «híbrido» frente al denso (0,111) la pone el reranker: según el orden en que se
   descomponga, la fusión aporta entre −0,022 y +0,016, y ninguna de las dos cifras es
   significativa.
2. **La fusión sola no mejora al denso** (−0,022; no significativo). Y BM25 con reranker (0,646)
   supera al denso sin reranker (0,604). El componente que decide es el reranker.
3. **El juez «independiente» es de la misma familia BGE que el recuperador denso**, y los dos
   jueces disponibles coinciden solo a medias (Spearman 0,47 a 0,55, cota inferior). Si eso sesga
   el resultado no se puede decidir con estos datos: una primera regresión mía que parecía
   mostrarlo era un artefacto de selección y queda retirada (C2).
4. **El NDCG de este proyecto es relativo, no absoluto.** Depende de qué sistemas entran al pool, y
   0,74 no significa «74 % relevante»: el propio juez da al top-5 del mejor sistema una
   probabilidad media de relevancia de 0,42.
5. **El truncado a 512 tokens, anotado como «no medido», es grande:** el 28,9 % de los fragmentos
   no cabe en el embedder y el verificador deja sin leer más contexto en el escenario híbrido que
   en el léxico. No se detecta efecto sobre la fidelidad, pero tampoco se puede descartar.
6. **Las 194 consultas equivalen a unas 69 independientes** (74 plantillas). Las conclusiones de
   recuperación sobreviven; los contrastes de generación quedan aún más lejos de la potencia
   necesaria.
7. **Tres afirmaciones mías eran erróneas y se retiran:** «Gemma supera a Granite» (no resiste otro
   estimador), «el oráculo no está en disco» (sí estaba) y «el juez favorece a la pierna densa»
   (artefacto de selección, detectado por la revisión independiente de los scripts).

## 2. exp20: el control que faltaba

### 2.1 Qué se corrió

La evidencia firmada compara «híbrido + reranker» con «denso sin reranker» y con «léxico sin
reranker». Con ese diseño no se puede separar cuánto aporta la fusión y cuánto el reranker. exp20
añade los brazos que faltaban, sobre las mismas 194 consultas y con los mismos objetos que
`scripts/run_retrieval_only.py`:

| Brazo | Pool de candidatos | Reranker |
|---|---|---|
| `lexico`, `denso`, `hibrido_pre`, `hibrido_rerank` | los cuatro sistemas de exp11 | como en exp11 |
| `denso_rerank` | top-50 denso | sí |
| `lexico_rerank` | top-50 léxico | sí |
| `union_rerank` | unión de los tres pools | sí (cota de referencia) |

Modelos cargados, todos locales y en CPU: `bge-large-en-v1.5` (encoder denso),
`ms-marco-MiniLM-L-12-v2` (reranker) y, para calificar, `BAAI/bge-reranker-large` (oráculo
independiente). No se cargó Ollama, NLI ni HHEM. No se descargó nada. La recuperación tardó 66,6
minutos y la calificación de 2 957 pares, 59,3.

**Corrección de un error mío.** En el turno anterior dije que el oráculo independiente no estaba en
disco y que haría falta descargarlo. Era falso. El enlace `data/models` de este worktree apunta a
una copia parcial con tres modelos; la copia completa, con `bge-reranker-large`, está en
`data/models` del checkout principal. La leí en modo lectura, sin escribir nada allí, y comprobé
que el SHA-256 de `model.safetensors` coincide con el del manifiesto de la descarga aprobada en
julio (`summer_models_manifest.json`). La pregunta sobre autorizar una descarga ya no hace falta.

**Alcance de tu autorización, para que lo revises.** Autorizaste «exp20 del brazo denso con
reranker (encoders locales, sin Ollama)». Entendí que calificar esa corrida con el oráculo local
entra en ese permiso: es un encoder local, es el mismo juez de la evidencia firmada y sin él la
corrida solo tenía una medida circular. Si no lo ves así, la calificación está aislada en un
archivo (`oracle_scores__bge-reranker-indep.json`) y se puede descartar sin tocar lo demás.

### 2.2 El instrumento reproduce la evidencia firmada

| Comprobación | Resultado |
|---|---|
| Top-5 de `lexico`, `denso` e `hibrido_rerank` frente a exp11 firmado (identificadores y orden) | 194 de 194 en los tres |
| Top-5 de `hibrido_pre` frente a exp11 | mismo conjunto en 194 de 194; mismo orden en 193 |
| NDCG@5 circular firmado de los cuatro sistemas, recalculado con los puntajes de exp20 | desviación máxima 8 × 10⁻⁸ |
| NDCG@5 independiente firmado de los cuatro sistemas, recalculado | desviación máxima 2 × 10⁻⁷ |
| Puntuar dos veces el mismo par (consulta, fragmento) | desviación máxima 2,9 × 10⁻⁶ |

Esto es lo que da valor a los brazos nuevos: se miden con un instrumento que devuelve los números
firmados cuando se le piden los brazos viejos.

### 2.3 Resultado con el oráculo independiente (medido)

NDCG@5 con un pool común de los siete brazos. Los valores son algo menores que los firmados porque
el pool es mayor (ver C3); el orden entre los cuatro sistemas firmados no cambia.

| Brazo | NDCG@5, pool de 4 (firmado) | NDCG@5, pool de 7 | Probabilidad media de relevancia del top-5 |
|---|---|---|---|
| `lexico` | 0,4421 | 0,4276 | 0,255 |
| `hibrido_pre` | 0,6026 | 0,5817 | 0,348 |
| `denso` | 0,6237 | 0,6035 | 0,359 |
| `lexico_rerank` | no existe | 0,6461 | 0,370 |
| `denso_rerank` | no existe | 0,6985 | 0,415 |
| `union_rerank` | no existe | 0,7062 | 0,424 |
| `hibrido_rerank` | 0,7405 | 0,7142 | 0,423 |

| Contraste | Diferencia | d_z | p (Wilcoxon) | p por plantilla | gana / empata / pierde |
|---|---|---|---|---|---|
| reranker sobre el pool léxico (`lexico_rerank − lexico`) | +0,219 | +0,82 | 4 × 10⁻²⁰ | 7 × 10⁻¹² | 159 / 0 / 35 |
| reranker sobre el pool híbrido (`hibrido_rerank − hibrido_pre`) | +0,133 | +0,53 | 9 × 10⁻¹¹ | 8 × 10⁻⁸ | 134 / 0 / 60 |
| reranker sobre el pool denso (`denso_rerank − denso`) | +0,095 | +0,37 | 2 × 10⁻⁶ | 4 × 10⁻⁶ | 128 / 0 / 66 |
| fusión sin reranker (`hibrido_pre − denso`) | −0,022 | −0,08 | 0,19 | 0,37 | 95 / 0 / 99 |
| **fusión con reranker (`hibrido_rerank − denso_rerank`)** | **+0,016** | **+0,11** | **0,075** | **0,16** | 79 / 53 / 62 |
| híbrido frente a BM25, ambos con reranker | +0,068 | +0,33 | 2 × 10⁻⁵ | 0,003 | 95 / 42 / 57 |
| reranker sobre la unión frente a sobre el pool híbrido | −0,008 | −0,07 | 0,25 | 0,49 | 49 / 83 / 62 |

Lectura:

- El reranker mejora cualquier pool, y el efecto es grande y robusto.
- La brecha entre el híbrido completo y el denso sin reranker es 0,111. La descomposición depende
  del orden. Por el camino «denso → añadir reranker → añadir fusión»: +0,095 del reranker y +0,016
  de la fusión. Por el camino «denso → añadir fusión → añadir reranker»: −0,022 de la fusión y
  +0,133 del reranker. En los dos caminos la parte de la fusión no es significativa y la del
  reranker sí.
- «No significativa» no es «nula». Con 194 consultas el efecto mínimo detectable es d_z = 0,20; el
  observado es 0,11 y harían falta unas 620 consultas para detectarlo. Lo que sí acota el dato: la
  fusión añade como mucho +0,035 de NDCG@5 con un 95 % de confianza, frente a los +0,095 del
  reranker.
- El híbrido y el denso con reranker devuelven listas parecidas: 3,81 fragmentos en común de 5, e
  idénticas en 55 de 194 consultas.

**Por subgrupo (exploratorio, no planificado de antemano).** El contraste de la fusión se concentra
en las 115 consultas sin filtro de proveedor: +0,026 (p = 0,007 con Wilcoxon; p = 0,08 agrupando
por plantilla). En las 79 con filtro es +0,000. Es un indicio, no un resultado: el subgrupo se
definió después de ver los datos y no resiste la agrupación por plantilla.

**¿Se debe a que el híbrido recibe un pool más pequeño cuando hay filtro?** No. El híbrido de
producción fusiona 50 + 50 y filtra después sin reponer (pool medio de 45,3 en esas 79 consultas),
mientras los brazos base piden 100 antes de filtrar. Para comprobarlo se derivó, sin ejecutar
modelos, el brazo `hibrido_justo_rerank`: fusiona los pools ya filtrados (50 + 50) con el mismo RRF
y reordena con los puntajes guardados del reranker. Coincide con el híbrido de producción en las
115 consultas sin filtro (115 de 115) y solo cambia el top-5 en 6 de las 79 con filtro. Su
diferencia con el híbrido de producción es −0,001 (p = 0,46) y con el denso con reranker, +0,015
(p = 0,13). La asimetría de sobre-recuperación existe en el código pero no explica nada.

### 2.4 ¿Le quita la fusión candidatos buenos al reranker? (refutado con el juez independiente)

Si al reranker se le diera la unión de los tres pools, en 111 de 194 consultas pondría en su top-5
algún fragmento que no llegó al pool híbrido (151 de 970 huecos). Con su propio puntaje, eso
parece una pérdida (+0,10 logits; es una medida circular). Con el juez independiente no lo es:
`union_rerank − hibrido_rerank` vale −0,008 (p = 0,25). Darle más candidatos al reranker no mejora
el resultado. Consecuencia práctica: un parche que amplíe el pool de la fusión no está justificado
por estos datos.

El coste de los primeros resultados densos que la fusión expulsa también queda medido: ocurre en
82 de 194 consultas. Ese fragmento tiene una probabilidad media de relevancia de 0,434 y el que la
fusión pone primero en su lugar, 0,413 (diferencia +0,021; p = 0,71). No hay pérdida demostrable.
Los 82 siguen en el pool de 50 y el reranker devuelve 29 al top-5 final.

## 3. Hallazgos de concepto

| # | Hallazgo | Etiqueta | ¿Estaba en el ledger? |
|---|---|---|---|
| C1 | «Híbrido supera a denso» compara un brazo con reranker contra brazos sin reranker | Medido (exp20) | El diseño sí; el control no existía |
| C2 | El oráculo «independiente» es de la misma familia (BAAI BGE) que el recuperador denso, y los dos jueces discrepan con frecuencia | El parentesco, demostrado; el sesgo, no resuelto | No |
| C3 | El NDCG es relativo al pool: añadir brazos baja la nota de todos sin cambiar ningún ranking | Demostrado y medido | «Sesgo de pooling» documentado; esta consecuencia no |
| C4 | P@k, R@k y MRR llaman «relevante» a lo que supera la mediana del pool | Demostrado | No |
| C5 | Truncado a 512 tokens en embedder, reranker y NLI | Medido | Flags 33, 130 y 144: conocidos, «no medido» |
| C6 | `adaptive_500` no significa fragmentos de 500 tokens o menos | Medido | Flag 130, sin cifra |
| C7 | El reranker prefiere fragmentos largos: los escenarios con reranker entregan más texto al LLM | Medido | No |
| C8 | Las 194 consultas salen de 74 plantillas: no son independientes | Medido | No |
| C9 | El 8 % del índice es texto duplicado y gasta huecos del top-5 | Medido | No |
| C10 | 26 de las 51 consultas tipadas `cross_cloud` son de un solo proveedor | Medido | No (F2 del ledger de verano cubre otro caso) |
| C11 | Mi contraste «Gemma supera a Granite» no resiste otro estimador | Refutado (corrección propia) | No aplica |

### C1. La ventaja del híbrido es, sobre todo, la del reranker (medido)

El addendum del ledger ya decía que la ventaja del híbrido «procede del reranking», y la tabla de
revisión daba por bueno que los brazos base estuvieran «bien aislados, sin reranker». Las dos cosas
juntas son el problema: sin un brazo base con reranker, la frase «el híbrido supera al denso» no se
puede atribuir. exp20 lo mide (§2.3): el reranker aporta entre +0,095 y +0,133 según el orden de
la descomposición, y la fusión entre −0,022 y +0,016, sin significación en ningún caso.

Esto no dice que el sistema desplegado sea peor. El pipeline completo sigue siendo el mejor brazo
medido. Dice que la tesis «la fusión híbrida mejora la recuperación» no está demostrada con estos
datos; lo demostrado es «añadir un reranker cross-encoder mejora la recuperación, sobre cualquier
pool».

Por qué la fusión aporta tan poco, con la aritmética del informe del porqué (§2): con `k = 60` y
50 candidatos por pierna, RRF coloca arriba lo que está en las dos piernas, y las dos piernas
coinciden poco (0,75 de 5 en el top-5). El pool resultante es 24 % consenso, 39 % solo denso y
38 % solo léxico. El reranker reordena ese pool y acaba eligiendo casi lo mismo que elegiría del
pool denso.

### C2. El oráculo independiente comparte familia con el recuperador denso

La circularidad original era: el reranker del pipeline (`ms-marco-MiniLM-L-12-v2`) también juzgaba
la relevancia. Se corrigió cambiando el juez a `BAAI/bge-reranker-large`. Pero el recuperador denso
es `BAAI/bge-large-en-v1.5`: mismo laboratorio y misma familia de modelos. El ledger
(`paper/audit_findings.md`, sección del oráculo) solo pide «un modelo diferente» respecto de
MS MARCO; no considera el parentesco con la pierna densa.

Lo medido:

1. Al cambiar de juez, el NDCG@5 firmado del brazo léxico cae de 0,552 a 0,442 (−0,110) y el del
   denso solo de 0,649 a 0,624 (−0,026). Es compatible con un sesgo y también con que el juez nuevo
   sea simplemente mejor.
2. Los dos jueces coinciden a medias. Spearman dentro de cada consulta: media 0,47 sobre los 2 957
   pares puntuados (por debajo de 0,5 en 86 de 194 consultas) y 0,55 sobre los 2 140 pares de los
   brazos sin reranker. Son cotas inferiores: solo se puntuaron fragmentos que algún brazo puso en
   su top-5, y ese rango restringido atenúa la correlación. «Relevante» depende de quién juzga.

**Un análisis mío que se retira.** Intenté medir la afinidad de cada juez con cada pierna con una
regresión: el puesto que un juez da a un fragmento, explicado por el puesto que le da el otro juez
y por si el fragmento venía del top-50 denso o del léxico. Con todos los pares, `bge-reranker`
parecía favorecer lo que trajo la pierna densa (+0,134; p = 4 × 10⁻²²) y `ms-marco` no favorecía a
ninguna. La revisión independiente de los scripts detectó que era un artefacto: cuatro de los siete
brazos eligen su top-5 con `ms-marco` y ninguno con `bge-reranker`, y esa selección anula por
construcción los coeficientes de `ms-marco`. En los 2 140 pares que entraron solo por brazos sin
reranker, la asimetría desaparece:

| Muestra | Juez | β «estaba en el top-50 denso» | β «estaba en el top-50 léxico» |
|---|---|---|---|
| todos los brazos (seleccionada) | `bge-reranker` | +0,134 | +0,060 |
| todos los brazos (seleccionada) | `ms-marco` | +0,007 | −0,023 |
| solo brazos sin reranker | `bge-reranker` | +0,096 | +0,019 (p = 0,22) |
| solo brazos sin reranker | `ms-marco` | +0,121 | +0,093 |

Los dos jueces puntúan más alto lo que trajo la pierna densa, y `ms-marco` no menos que
`bge-reranker`. No hay evidencia de afinidad especial del juez BGE. El script conserva las dos
filas con el aviso.

**Estado.** El parentesco entre juez y recuperador es un hecho y debe declararse como límite. Si
sesga las cifras no está demostrado ni descartado. Solo lo resuelven juicios humanos de relevancia
o un tercer juez de otra familia.

Nota: el contraste central de C1 (híbrido frente a denso, ambos con reranker) es el menos expuesto
a este problema, porque los dos brazos pasan por el mismo reranker y ninguno es de la familia BGE
en su último paso.

### C3. El NDCG es relativo al pool (demostrado y medido)

`scripts/compute_retrieval_metrics.py` calcula el DCG ideal con los mejores puntajes de **todos los
fragmentos que algún sistema recuperó para esa consulta**. El denominador depende, por tanto, de
qué sistemas participan. Test: `test_the_same_ranking_scores_lower_when_another_system_adds_a_better_chunk`.

Medido sin cambiar ningún ranking, solo añadiendo los tres brazos de exp20 al pool:

| Brazo | Juez circular: pool de 4 → pool de 7 | Juez independiente: pool de 4 → pool de 7 |
|---|---|---|
| `lexico` | 0,5516 → 0,5434 | 0,4421 → 0,4276 |
| `denso` | 0,6494 → 0,6356 | 0,6237 → 0,6035 |
| `hibrido_pre` | 0,6681 → 0,6576 | 0,6026 → 0,5817 |
| `hibrido_rerank` | 0,9948 → 0,9695 | 0,7405 → 0,7142 |

Consecuencias:

1. Un NDCG de este estudio no se puede comparar con el de otro artículo, ni con el de otro
   experimento del propio proyecto que use otro conjunto de sistemas.
2. No es una nota absoluta. Con el juez independiente, el híbrido tiene NDCG@5 = 0,74, pero el
   propio juez asigna a su top-5 una probabilidad media de relevancia de 0,42 (0,36 al denso y
   0,26 al léxico). «0,74» no significa «el 74 % es relevante».

### C4. «Relevante» significa «por encima de la mediana del pool» (demostrado)

Las métricas binarias (P@k, R@k, MRR) usan como umbral el percentil 50 de los puntajes del pool.
Con el juez independiente ese umbral es una probabilidad de 0,131. Por construcción, la mitad de
los fragmentos del pool son «relevantes», sea cual sea su calidad real. Estas métricas ordenan
sistemas; no dicen cuántos fragmentos son relevantes. La variante de umbral 0 que también se
guarda vale 1,00 para todos los sistemas con este juez, porque una probabilidad nunca es negativa:
no discrimina nada.

### C5. Truncado silencioso a 512 tokens (medido)

Los Flags 33, 130 y 144 del ledger lo anotan como riesgo «no medido». Esta es la medida, contando
tokens con el vocabulario de cada modelo:

| Modelo | Qué lee | Cuánto se pierde |
|---|---|---|
| Embedder `bge-large` | primeros 510 tokens del fragmento | 7 072 de 24 481 fragmentos no caben (28,9 %); el 13,5 % de los tokens del corpus no está representado en ningún vector |
| Reranker `ms-marco` | consulta más fragmento, 512 en total | 26,4 % de los pares del pool híbrido truncados; en el top-5 final, 40,3 % de los pares y 14,5 % de los tokens |
| Verificador NLI | fragmento más afirmación, 512 en total | con una afirmación de 40 tokens: 21,1 % de los fragmentos del contexto léxico, 29,4 % del denso y 39,9 % del híbrido |

Tres consecuencias de concepto:

1. **Las dos piernas no ven el mismo documento.** BM25 indexa el texto completo; el vector denso
   solo el principio. Un término que aparece al final de un fragmento largo existe para BM25 y no
   para la pierna densa.
2. **El Flag 33 fijó su propio criterio:** «disclaimer si más del 10 % de los fragmentos excede».
   Se excede con holgura en los tres modelos. El aviso corresponde.
3. **El truncado no es igual entre escenarios.** El verificador deja de leer el 7,4 % del contexto
   léxico y el 13,5 % del híbrido, y el LLM sí recibe el texto completo. Una afirmación apoyada en
   la cola de un fragmento largo cuenta como no apoyada, y eso ocurre más en el escenario híbrido.

¿Se nota en la fidelidad? Regresión dentro de cada par (modelo, consulta), 912 respuestas, errores
agrupados por consulta: el coeficiente de la fracción no leída es +0,52 (p = 0,17) con el
verificador small y +0,02 (p = 0,96) con base. **No se detecta efecto.** El diseño es
observacional y de poca potencia: no demuestra que el efecto sea nulo. La prueba limpia sería
volver a verificar con el fragmento partido en ventanas (corrida `exp20+`, solo NLI, sin Ollama).

### C6. `adaptive_500` no acota el tamaño (medido)

Longitud de los fragmentos en tokens del embedder: mediana 299, percentil 95 de 734, máximo
11 187. El nombre sugiere un tope de 500; el 28,9 % lo supera. El paper debe describir la
distribución real, no el parámetro nominal.

### C7. El reranker prefiere fragmentos largos (medido)

Dentro de cada consulta, la correlación de Spearman entre el puntaje del reranker y la longitud
del fragmento es +0,15 de media, positiva en 144 de 194 consultas (p = 2 × 10⁻¹⁵). Efecto sobre lo
que recibe el LLM:

| Brazo | Tokens medios por fragmento del top-5 |
|---|---|
| `lexico` | 317 |
| `hibrido_pre` | 340 |
| `denso` | 376 |
| `lexico_rerank` | 401 |
| `hibrido_rerank` | 408 |
| `denso_rerank` | 417 |

El escenario híbrido de exp12 entrega un 29 % más de texto que el léxico. La comparación de
generación «híbrido frente a base» mezcla, por tanto, tres cosas: qué fragmentos se eligen,
cuánto texto se entrega y cuánto de ese texto puede leer el verificador (C5). Con los datos
firmados no se pueden separar.

### C8. Las consultas no son independientes (medido)

Las 194 consultas salen de 74 plantillas (proveedor enmascarado y primeras cuatro palabras; la
mayor tiene 15 consultas y 41 son únicas). Las pruebas firmadas tratan cada consulta como una
observación independiente.

| Contraste (NDCG@5 circular, pool de 7) | ICC por plantilla | Efecto de diseño | n efectivo | p independiente | p por plantilla |
|---|---|---|---|---|---|
| híbrido+reranker − denso | 0,42 | 2,83 | 69 | 6 × 10⁻⁴⁷ | 7 × 10⁻²² |
| híbrido+reranker − léxico | 0,43 | 2,85 | 68 | 9 × 10⁻⁵² | 2 × 10⁻²⁰ |
| híbrido+reranker − léxico+reranker | 0,39 | 2,68 | 72 | 4 × 10⁻⁸ | 2 × 10⁻⁴ |

Lectura: para los contrastes grandes de recuperación, la información real equivale a unas 69
consultas independientes, no a 194. **Las conclusiones de recuperación sobreviven** (los p siguen
siendo minúsculos, también con el juez independiente: §2.3). El punto importa donde el margen es
estrecho: el contraste de la fusión pasa de p = 0,075 a p = 0,16, y los contrastes de generación,
que ya estaban por debajo de la potencia necesaria (informe del porqué, §3), lo están más. La
definición de plantilla es heurística; el ICC cambia con ella, el hecho de que hay agrupación no.

### C9. Texto duplicado (medido)

1 951 de los 24 481 fragmentos del índice (8,0 %) tienen el mismo texto que otro fragmento, en 806
grupos (el mayor, de 69). En el top-5: el brazo léxico gasta 61 de 970 huecos (6,3 %) en
repeticiones y el híbrido 43 (4,4 %); entre 23 y 34 consultas de cada brazo reciben algún texto
repetido. Dos efectos: el LLM recibe menos evidencia distinta de la que indica `k = 5`, y un mismo
texto cuenta varias veces en las métricas de recuperación. Deduplicar cambiaría rankings: es
clase B y no se toca.

### C10. `cross_cloud` no significa «varios proveedores» (medido)

El procesador de consultas tipa por reglas 51 consultas como `cross_cloud`. El conjunto de prueba
anota varios proveedores solo en 25. Las otras 26 comparan dos servicios del mismo proveedor
(«Compare AWS EKS and AWS EC2…»): reciben la plantilla de comparación entre nubes y se recuperan
sin filtro de proveedor. Además, 64 consultas procedimentales de un solo proveedor tampoco llevan
filtro.

Efecto medido en recuperación, pequeño: en las 64 procedimentales, el 10,0 % del top-5 léxico es
de un proveedor que la pregunta no nombra (4,4 % en el denso y 3,1 % en el híbrido); en las 26
mal tipadas, entre 0 y 2,3 %. El efecto de la plantilla sobre la generación no se puede medir sin
ejecutar el LLM. Para el texto: «51 consultas cross-cloud» es una propiedad del procesador, no del
conjunto; las consultas de varios proveedores son 25.

### C11. Corrección propia: «Gemma supera a Granite» no es robusto (refutado)

En el informe del porqué (§4) presenté como hallazgo exploratorio que, con la pregunta y el
contexto fijos, Gemma supera a Granite en 0,08 con los dos verificadores (Holm p = 0,016 y 0,022).
Ese número venía de una regresión conjunta de los cuatro modelos. Con el estimador más simple y
con menos supuestos (diferencias pareadas solo en las celdas donde ambos modelos contestaron):

| Verificador | Gemma − Granite | p Holm, unidad consulta | p Holm, unidad plantilla |
|---|---|---|---|
| small | +0,057 | 0,30 | 0,36 |
| base | +0,078 | 0,066 | 0,14 |

No es significativo con ninguna de las dos unidades. Con small el único par significativo es
Granite − Mistral (−0,061; Holm 0,001) y con base, Mistral − Qwen (−0,060; Holm por plantilla
0,020). **Ningún par es significativo con los dos verificadores.** Retiro el hallazgo: el «0 de 18
pares robustos» del ledger N9 se mantiene sin matices. El párrafo del informe del porqué queda
corregido.

## 4. Sospechas que los datos no sostienen

Se anotan porque una auditoría que solo reporta lo que confirma no es una auditoría.

| Sospecha | Resultado |
|---|---|
| La fusión le quita al reranker candidatos buenos | Refutado con el juez independiente (§2.4) |
| El híbrido sale perjudicado porque fusiona 50 candidatos y filtra por proveedor después, sin reponer, mientras los brazos base piden 100 antes de filtrar | Refutado (§2.3): con la misma sobre-recuperación el top-5 cambia en 6 de 79 consultas y el NDCG en −0,001 |
| El juez BGE favorece a la pierna densa más que el otro juez | Refutado: artefacto de selección de mi primera regresión (C2) |
| Los primeros resultados densos que la fusión expulsa eran mejores que lo que pone en su lugar | No demostrado: 0,434 frente a 0,413, p = 0,71 (§2.4) |
| El truncado del verificador baja la fidelidad del escenario híbrido | No detectado (C5); tampoco descartado |
| Tipar mal 26 consultas contamina la recuperación con otros proveedores | Efecto pequeño (C10) |
| Las consultas de varios proveedores puntúan peor porque el NLI verifica cada afirmación contra un solo fragmento | Sin patrón coherente entre verificadores (sonda puntual, no incorporada al script) |
| Gemma supera a Granite | No robusto (C11) |
| El oráculo independiente no estaba en disco | Falso (§2.1) |

## 5. Qué se puede afirmar y qué no

Textos propuestos. La prosa del paper y del A.3 solo cambia con tu OK frase por frase. exp20 no
está firmada: para citar sus cifras hay que repetirla bajo un ID propio (§6, decisión 1).

| Tema | Lo que los datos sostienen |
|---|---|
| Afirmación principal | «El pipeline completo (fusión RRF y reranker cross-encoder) supera a la recuperación densa y a la léxica sin reranker. Un control con el mismo reranker sobre el top-50 denso alcanza un NDCG@5 de 0,699 frente a 0,714 del pipeline completo; la diferencia atribuible a la fusión, +0,016 (IC 95 %: −0,004 a +0,035), no es significativa. La mejora procede sobre todo del reranker.» |
| Fusión | «La fusión RRF sin reranker no mejora a la recuperación densa (−0,022; no significativo).» |
| NDCG | «El NDCG se calcula respecto del pool de fragmentos recuperados por los sistemas comparados y con un juez automático. Es una medida relativa: no es comparable entre estudios ni equivale a una proporción de fragmentos relevantes.» |
| Juez | «El juez de relevancia (`bge-reranker-large`) es de la misma familia que el recuperador denso (`bge-large-en-v1.5`). No se dispone de juicios humanos de relevancia.» |
| Métricas binarias | «Un fragmento se cuenta como relevante si su puntaje supera la mediana del pool.» |
| Truncado | «Los tres modelos leen como máximo 512 tokens. El 28,9 % de los fragmentos excede ese límite en el embedder; el verificador de fidelidad no lee entre el 7 % y el 14 % del contexto, según el escenario.» |
| Fragmentación | «Los fragmentos tienen una mediana de 299 tokens y un percentil 95 de 734.» |
| Consultas | «Las 194 consultas se generaron a partir de 74 plantillas. Las pruebas por consulta las tratan como independientes; al agrupar por plantilla, las conclusiones de recuperación se mantienen.» |
| Cross-cloud | «25 consultas nombran más de un proveedor.» (No 51.) |

## 6. Decisiones que te tocan

1. **¿Se firma exp20?** Recomiendo que sí: cambia la redacción de la afirmación principal y hoy
   solo existe como corrida de auditoría. Exige repetirla bajo `experiments/results/exp20_…`
   (unas 2 horas de CPU, sin Ollama), declararla viva en `src/utils/signed_evidence.py` (la guarda
   B1 lo pide) y anotarla en el ledger.
2. **¿Se cambia la afirmación principal del paper y del A.3?** Texto propuesto en §5. Es la
   consecuencia más seria de esta ronda.
3. **¿Se valida el juez?** Una muestra de juicios humanos de relevancia (por ejemplo, 200 pares
   estratificados por acuerdo entre los dos jueces) resuelve C2 y da por primera vez una medida
   absoluta. Encaja con la fase humana ya prevista.
4. **¿Se vuelve a verificar la fidelidad con ventanas en lugar de truncar?** Corrida `exp20+`, solo
   NLI. Resuelve C5.
5. **¿Se aplica el aviso del Flag 33?** Su propio umbral (10 %) está superado.
6. Siguen abiertas B3, B4, B5, B7 y B8 (informe de auditoría, §5 y §8). No recomiendo un parche de
   fusión de clase B: §2.4 indica que ampliar el pool no mejora el resultado.

## 7. Verificaciones de esta ronda

| Comprobación | Resultado |
|---|---|
| Reproducción del top-5 firmado de exp11 con los modelos reales | 194 de 194 en tres brazos; 193 de 194 en orden y 194 en conjunto en `hibrido_pre` |
| NDCG@5 circular firmado recalculado desde exp20 | desviación máxima 8 × 10⁻⁸ |
| NDCG@5 independiente firmado recalculado desde exp20 | desviación máxima 2 × 10⁻⁷ |
| SHA-256 del oráculo frente al manifiesto de julio | coincide |
| Tests nuevos (`tests/test_fable_concept_probes.py`) | 14 rápidos y 1 `slow` que recorre todas las sondas; pasan. Incluyen la paridad de `ndcg_graded` con la función firmada sobre 300 pools aleatorios |
| Revisión independiente de los dos scripts (agente revisor, solo lectura) | 1 hallazgo alto (la regresión de afinidad, retirada), 4 medios (descomposición dependiente del orden, subgrupos, comparación top-1 frente a media, acuerdo atenuado) y 2 bajos (tests). Todos corregidos. Recalculó por su cuenta +0,0157 y p = 0,0745 |
| `bash init.sh`, tres capas | exit 0 (2403 pasan, 0 fallan, 6 omitidas, 8 xfail estrictos) |
| Verificadores offline frente a la línea base | SHA-256 normalizado igual en los dos |
| `git diff --stat 1574d22..HEAD -- experiments paper config src/ui scripts/study_operator .harness` | vacío |

## 8. Límites

- exp20 no es evidencia firmada.
- No hay juicios humanos de relevancia. Todo NDCG de este proyecto es el acuerdo con un modelo.
- El contraste de la fusión no tiene potencia para efectos pequeños (d_z < 0,20).
- La longitud de las afirmaciones no está guardada: el truncado del NLI se da para 20, 40 y 60
  tokens.
- Los contrastes agrupados por plantilla dependen de una definición heurística de plantilla.
- La regresión de C5 es observacional. El análisis por subgrupo de §2.3 es exploratorio.
- El juez independiente es de la misma familia que una de las piernas (C2); el contraste central
  (híbrido frente a denso, ambos con el mismo reranker) es el menos expuesto, porque el último paso
  de los dos brazos es el mismo modelo, de otra familia.
- No se ejecutó ningún LLM: nada de aquí mide el efecto de estos hallazgos sobre las respuestas.
- La calificación corrió en CPU y la firmada posiblemente en GPU; la reproducción a 10⁻⁷ indica que
  no importa.

## 9. Cómo reproducirlo

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUTF8=1 .venv-app/Scripts/python.exe scripts/audit/fable_concept_probes.py
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUTF8=1 .venv-app/Scripts/python.exe scripts/audit/exp20_rerank_arms.py --analyze
```

Ninguno de los dos carga pesos de modelos. Repetir la corrida de recuperación (unos 67 minutos de
CPU) o la calificación con el oráculo (`--grade --oracle-path <carpeta local del modelo>`, unos 60
minutos) sí los carga y requiere tu autorización.
