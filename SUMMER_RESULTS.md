# SUMMER_RESULTS — Informe de hallazgos de la fase de verano

**Estado: EN CURSO** (arranque 2026-07-22). Este documento acumula los resultados de la ablación
(Fase 1), el diagnóstico (Fase 1b) y las mejoras (Fase 2). Ledger de decisiones:
`paper/summer_ablation_log.md`. Línea base: tag `summer-baseline` (cifras v4 verificadas,
`output/audit/phase0_verification_summer_2026-07-22.md`).

## Pregunta central
¿Por qué una mejor recuperación (NDCG@5 0.740 híbrido vs 0.442 léxico, oráculo independiente)
NO mejora la fidelidad de la respuesta (0/12 pares RAG-vs-RAG significativos; Granite
0.235/0.247/0.299)? ¿Instrumento, generación, contexto, o techo real?

## Respuesta corta (actualizada 2026-07-24)
**¿Por qué mejor recuperación ≠ mejor fidelidad? Porque la fidelidad responde a QUÉ evidencia entra
(selección de contenido), no a la calidad topical del ranking, ni al arreglo del contexto, ni al prompt.**
Diagnóstico completo de la fase:
- **Instrumento (Tier 3):** el 0/12 NO es robusto — HHEM revela granite híbrido>léxico (1/12) que el NLI
  ruidoso (22% falso-contradicted) enmascara.
- **Arreglo del contexto (Tier A):** NULO robusto (0/4 en 3 instrumentos) — rerank/top-k/orden/lost-middle
  no mueven la fidelidad. Lost-in-the-middle descartado.
- **Prompt / decodificación anclada (exp16):** NULO/negativo (0/2 en 3 instrumentos) — citar o abstenerse
  no mejora; solo sube declinación y recorta contenido.
- **Selección de contenido / cobertura (exp17):** **POSITIVO** — balancear la cobertura de proveedores en
  cross-cloud (7/25→25/25) sube la fidelidad comparativa en los 3 instrumentos (HHEM +0.081; guardas
  confirman genuino: declina menos, dice más, no copia). Piloto n=25, no sig, direccional-consistente.

Converge: el único eje que mueve la fidelidad es la selección de evidencia (Tier 3 + exp17), no su
presentación (Tier A) ni la instrucción (exp16). Todo pendiente de gold humano.

## Respuesta Tier 3 (detalle, 2026-07-23 — sujeta a validación con gold)
**El nulo 0/12 NO es robusto al instrumento: un verificador de grounding limpio (HHEM) revela un efecto
retrieval→fidelidad para granite que el NLI ruidoso enmascara.** Test pareado between-scenario, familia
BH v4-consistente (24, incl sin_rag):
- NLI small: **0/12** (granite hib-vs-lex p_bh 0.085). NLI base: **0/12**.
- **HHEM: 1/12** — **granite hibrido-vs-lexico p_bh 0.020, d_z −0.35, SIGNIFICATIVO** (mistral 0.067,
  cerca). El efecto híbrido>léxico existe para el modelo determinista pero solo se detecta con un
  instrumento menos ruidoso.
- **Nivel instrument-relative:** HHEM (especificidad buena: falso-grounded 0.033) da +0.307 sobre NLI
  (granite 0.40-0.44 vs 0.23-0.30). El NLI marca **22% de texto ALEATORIO** como contradicted → baja el
  nivel Y enmascara el contraste. El "0.30" publicado es relativo al instrumento NLI.
**Correcciones de rigor (3 iteraciones):** entrada 6 ("todo artefacto NLI, HHEM 0.99") sobre-vendió
(bug de carga HHEM); entrada 8 ("0/12 robusto, sin efecto") sub-vendió (bug de familia BH excluyó
sin_rag). La verdad está en medio: NLI enmascara un efecto real pequeño granite-específico que HHEM
revela. Ver `hhem_vs_nli.md` + `tier3_negative_control_finding` (ledger 6,7,8,9). Report-before-prose:
el 0/12 depende del instrumento → material de discusión; NO cambia cifras firmadas.

## Línea base v4 (referencia congelada)
| Métrica | Valor | Fuente |
|---|---|---|
| NDCG@5 híbrido (oráculo indep.) | 0.7405 | exp11 |
| NDCG@5 híbrido (circular, no citar como real) | 0.9948 | exp11 |
| Fidelidad Granite lexico/denso/hibrido | 0.235(75)/0.247(85)/0.299(87) | exp12 v4_small |
| RAG-vs-RAG significativos | 0/12 | exp12 v4 ambos verificadores |
| Entre-modelos robusto | 0/18 | exp12 v4 small∩base |
| Expansión cross-cloud (25 q) | OFF≈ON (retirada) | exp13 |

## Tabla de ablación Tier A (2026-07-24, 5 brazos × 60q, exp15_ablation_tierA)
Contexto = ids firmados exp11 híbrido full-rerank, **transformados sin re-recuperar**. granite temp0 seed42.
Contraste pareado within-session vs `baseline_repro` (Wilcoxon+d_z+bootstrap seed42, familia BH de 4).
Fidelidad = rescore vb_agree τ0.7 (NLI) / max_chunk τ0.5 (HHEM). Ver `arm_stats__{small,base,hhem}.md`.

| Brazo | Transform | Fid. NLI small | NLI base | HHEM | Sig BH (los 3) | Veredicto |
|---|---|---|---|---|---|---|
| baseline_repro | identidad | 0.308 | 0.204 | 0.450 | ancla | reproduce exp12 (drift +0.03 n.s.) |
| reranker_off | RRF pre-rerank | 0.243 | 0.144 | 0.438 | 0/3 | reranking NO sube fidelidad |
| final_top_k_3 | top-3 | 0.282 | 0.222 | 0.438 | 0/3 | recorte de contexto NO mueve |
| context_reversed | orden invertido | 0.315 | 0.214 | 0.469 | 0/3 | orden NO importa |
| context_lost_middle | relevante al centro | 0.332 | 0.202 | 0.424 | 0/3 | **lost-in-the-middle NO** |

**0/4 brazos significativos bajo NLI-small, NLI-base Y HHEM → el nulo de ablación de contexto es ROBUSTO
al instrumento** (a diferencia del 0/12 entre-escenarios, que HHEM sí rompe a 1/12). Mecanística: la
fidelidad responde débilmente a QUÉ documentos elige el *método* (híbrido vs léxico), NO a cómo se arregla
un pool ya recuperado (rerank/top-k/orden/posición). Deriva H5 baseline_repro vs junio: +0.033 (n.s.
p=0.083), r=0.86, 39% queries idénticas → valida gate relajado y diseño pareado within-session.

## Diagnóstico (Fase 1b) — hipótesis y estado
| Hipótesis | Estado | Evidencia |
|---|---|---|
| Instrumento NLI ruidoso/descalibrado (¿0/12 artefacto del punto de operación?) | **Tier 0 + Tier 3-A COMPLETOS**: (Tier 0) κ 0.30–0.36; nulo robusto bajo base (0/64); bajo small granite hib-vs-lex sig con ent≤0.6, consistente 128/128. (Tier 3-A) small sobre-contradice 1.8× y es 2.5× más frágil al umbral → **el verificador runtime es el ruidoso**; 128 falso-contradicted; agregador `max` sub-acredita evidencia distribuida (noisy_or→granite hib>lex p_bh 0.009). Control negativo: NLI marca **22% de texto aleatorio como contradicted**. → parte del 0.30 es instrumental; **cuánto** pendiente de HHEM (bug de carga corregido, re-corriendo) + gold | `exp15_ablation_nli/{sweep,disagreement,negative_control}_*`; ledger 2,4,6,7 |
| Generación no ancla en la evidencia | **parcial (Tier A)**: degradar el contexto (rerank off, orden, posición) no baja la fidelidad → el anclaje no depende del arreglo del pool; el cuello está en generación/selección de contenido, no en presentación | `arm_stats__*` |
| Lost in the middle | **DESCARTADO (Tier A)**: context_lost_middle vs baseline 0/3 instrumentos (small +0.027, base −0.005, HHEM −0.025, todos n.s.) | `arm_stats__{small,base,hhem}.md` |
| Corte de contexto / nº fragmentos | **parcial (Tier A)**: final_top_k_3 (5→3) 0/3 n.s.; recorte de fragmentos no mueve la fidelidad a este tamaño | `arm_stats__*` |
| Reranking sube fidelidad | **DESCARTADO (Tier A)**: reranker_off 0/3, incluso tiende abajo (−0.05..−0.06 NLI, n.s.) | `arm_stats__*` |
| Declinación confunde la métrica | parcialmente tratado en v2/v4 | denominadores decline-aware |

## Matriz de factibilidad — Trabajos Futuros del A.3 (CERRADA 2026-07-24)
Viabilidad en esta laptop (RTX 3060 6 GB) antes de las encuestas. Cerrada tras Tier A + exp16 + exp17.

| Línea (A.3) | Viable aquí | Costo | Payoff esperado | Veredicto preliminar |
|---|---|---|---|---|
| **1a. Decodificación anclada** (citar/atribuir evidencia, temperatura, prompt) | **Sí** | Bajo (infra lista) | **Nula (probado)** | **IMPLEMENTADA Y PROBADA 2026-07-24 (exp16) — SIN ganancia local.** anchored_cite + strict_abstain: 0/2 bajo NLI-small/base/HHEM; anchored tiende ABAJO (cita≠grounding), ambos suben declinación y recortan contenido. Descarta la línea como victoria local |
| **1b. Modelo de mayor capacidad** | **No en 6 GB** | — | Alto pero incuantificable local | **DISEÑO/NUBE** — granite@4096 ya no cabe 100% GPU (hallazgo); ≥13B exige otra máquina/nube. Reportar trade-off |
| **2. Anotación humana (relevancia + gold)** | **Parcial** (diseño sí, ejecución no) | ~4-5 h humano | Alto (rompe circularidad, arbitra instrumento) | **ENTREGADO EL DISEÑO** — `claim_audit_sample_v4` N≈200 listo; ejecuta Enzo/anotadores |
| **3. Verificador de fidelidad estable (Tier 3)** | **Sí** | Bajo-medio (CPU + descargas hechas) | **Alto** (κ 0.32; NLI 22% falso-contradicted) | **HECHO (selección espera gold)** — NLI 22% falso-contradicted; HHEM especificidad buena (falso-grounded 0.033) revela efecto que el NLI enmascara; front-runner ensemble E5_base+hhem (0.003). Selección definitiva = gold humano |
| **4. Ablación de componentes (Tier A/B)** | **Sí** | Medio (GPU, gate determinismo relajado) | Alto (aísla qué mueve la fidelidad) | **Tier A HECHO 2026-07-24** — 0/4 robusto (rerank/top-k/orden/lost-middle no mueven fidelidad en 3 instrumentos); descarta lost-in-the-middle y reranking. Tier 0 hecho; Tier B oráculo listo |
| **5. Cross-cloud: reescritura/expansión densa** | **Sí** | Bajo (25 q) | **Positivo (piloto)** | **PILOTO CON SEÑAL POSITIVA 2026-07-24 (exp17).** La expansión léxica falló (exp13), pero **rebalancear la cobertura por proveedor** (7/25→25/25) sube la fidelidad comparativa en 3 instrumentos (HHEM +0.081, guardas confirman genuino). No sig a n=25 → confirmatorio con OK. Trabajo Futuro accionable |
| **6. Memoria semántica (tripletes/KG/versionada)** | **No (verano)** | Alto | Incierto | **SOLO DISEÑO** — excede el verano; entregar veredicto de factibilidad |

Nota clave (hallazgo que ata 1b + corte de contexto): granite@4096 es simultáneamente el techo que truncó
exp12 (input máx=4096 exacto) Y el máximo que casi-no-cabe en 6 GB → subir contexto O modelo exige salir
de esta laptop. Esto acota fuertemente qué "generación más fiel" es implementable localmente.

## Mejoras (Fase 2)
### exp16 — decodificación anclada (2026-07-24): RESULTADO NEGATIVO triangulado
3 brazos de prompt sobre el mismo pool híbrido (solo cambia system+sufijo), pareado within-session vs
baseline_repro fresco (`--no-cache`, co-temporal). Ver `exp16_anchored_finding_2026-07-24.md`, ledger 11.

| Brazo | Δ small | Δ base | Δ HHEM | Sig (3 instr.) |
|---|---|---|---|---|
| anchored_cite (cita [N] por claim) | −0.034 | −0.040 | −0.056 | 0/3 — tiende ABAJO |
| strict_abstain (omitir lo no explícito) | −0.002 | +0.031 | +0.038 | 0/3 — plano |

**La decodificación anclada NO mejora la fidelidad** (0/2 bajo NLI-small/base/HHEM). Guardas anti-gaming:
ambos brazos suben la declinación (51.7%→58/60%) y recortan contenido (palabras 352→223/191, claims
11.95→7.55/5.27); anchored_cite baja el solape (0.061, no copia) pero igual baja la fidelidad → **cita ≠
grounding**. Junto con Tier A (nulo de recuperación): ni contexto ni prompt mueven la fidelidad → techo de
capacidad del modelo (1b, fuera de 6GB) o instrumento (Tier 3, gold pendiente). Caveat: declinación
baseline 51.7% → n efectivo ≈29, underpowered; dirección + guardas argumentan contra un positivo oculto.

### exp17 — recuperación balanceada por proveedor (2026-07-24): PRIMER POSITIVO de la fase
Diagnóstico: solo 7/25 queries comparativas cross-cloud recuperan TODOS los proveedores en top-5 (sesgo a
un proveedor pese a NDCG 0.85) → comparación imposible de anclar. 2 brazos del MISMO pool híbrido (baseline
= exp13 exp_off, validado overlap 5.0/5; balanced = ⌈5/|P|⌉ por proveedor). Cobertura 7/25 → 25/25.
Ver `exp17_crosscloud_finding_2026-07-24.md`, ledger 12.

| Instrumento | baseline | balanced | Δ | d_z | p |
|---|---|---|---|---|---|
| NLI small | 0.199 | 0.236 | +0.037 | 0.13 | 0.40 |
| NLI base | 0.151 | 0.196 | +0.045 | 0.21 | 0.14 |
| HHEM | 0.477 | 0.558 | **+0.081** | 0.26 | 0.24 |

**Balancear la cobertura de proveedores sube la fidelidad comparativa en los 3 instrumentos** (efecto
pequeño, no sig a n=25 per-query, pero direccional-consistente). Guardas confirman mejora GENUINA (opuesto
a exp16): declinación 56%→32% (BAJA), palabras 349→427 (SUBE), claims 11.5→14.9 (SUBE), solape 0.11→0.13
(plano). La cobertura (no el arreglo ni el prompt) es la palanca. Converge con Tier 3: el eje que mueve la
fidelidad es QUÉ evidencia entra.

**Reanálisis de mayor potencia (mismas 25 q, claim-level, pool agotado → sin queries nuevas; ledger 13):**
GLMM `supported ~ arm + (1|query)` una-cola: HHEM OR 1.25 **p=0.021**, base 0.056 (marginal), small 0.152;
bootstrap conservador por query NO cruza (HHEM 0.23). Sugestivo, no concluyente: real en dirección, al
borde según el modelo. Caveats: una-cola, 3 verificadores (no sobrevive Bonferroni ×3), GLMM puede
sobre-estimar potencia → bootstrap = guarda. Confirmatorio real exigiría queries pre-registradas nuevas
(decisión de Enzo) o gold.

**Síntesis Fase 2:** Tier A (arreglo contexto) nulo · exp16 (prompt) nulo/negativo · **exp17 (cobertura de
contenido) POSITIVO**. La fidelidad responde a la selección de evidencia, no a su ordenamiento ni al prompt.
(Candidata restante: verificador estable [3], espera gold.)

## Configuración recomendada para SUS/Likert (Fase 3, cerrada 2026-07-24)
Config del sistema a poner frente a los participantes de las encuestas, derivada de los hallazgos de la
fase. Objetivo: la mejor variante DEFENDIBLE que corre en esta laptop, sin cambiar nada de la evidencia
firmada (esto es la config de despliegue para usuarios, no una cifra del paper).

**Pipeline base (sin cambios — es lo que ya funciona y está medido):**
- Recuperación: **híbrido** (BM25 + denso bge-large + RRF, k=60) + rerank cross-encoder ms-marco-L12,
  final top-5. NDCG@5 indep 0.74 (vs léxico 0.44). Es el mejor retrieval medido; Tier A confirmó que su
  arreglo (orden/recorte) no hay que tocarlo.
- Generación: granite4.1:8b, temp 0, seed 42, prompt canónico. exp16 mostró que anclar por prompt NO
  mejora (y sube declinación) → **no cambiar el prompt**. Es la única opción a 6 GB (1b exige nube).
- Métrica de fidelidad mostrada/registrada: reportar relativa al instrumento (el "0.30" es NLI-relativo;
  HHEM da ~+0.31 de nivel). Para la encuesta, la fidelidad es contexto interno, no se le pide juzgarla al
  usuario; se mantiene el verificador runtime actual pero se DOCUMENTA su ruido (22% falso-contradicted).

**Único cambio recomendado (bajo riesgo, con señal positiva): cobertura balanceada por proveedor en
queries comparativas cross-cloud.** exp17: el retrieval híbrido deja 18/25 comparaciones sin uno de los
proveedores pedidos → respuestas ancladas a un solo lado. El rebalanceo (⌈5/|P|⌉ por proveedor del mismo
pool) sube cobertura 7/25→25/25 y la fidelidad comparativa en los 3 instrumentos (mejora genuina: menos
declinación, más contenido, sin copiar). Aplicarlo SOLO al ramo comparativo cross-cloud (detección ya
existe: `QueryProcessor` clasifica `cross_cloud`); no toca las queries de un solo proveedor. Es la única
palanca de mejora que dio positivo en toda la fase.

**Lo que NO se recomienda tocar para las encuestas:** decodificación anclada (exp16 nula/negativa);
modelo mayor (no cabe en 6 GB); memoria semántica/KG (fuera de alcance). Todo eso queda como Trabajo
Futuro en A.3 con veredicto de factibilidad ya escrito.

**Pendientes que NO bloquean las encuestas (pero sí el cierre del paper):** gold humano
(`claim_audit_sample_v4`, N≈200) para (a) arbitrar el nivel de fidelidad NLI vs HHEM y (b) seleccionar el
verificador definitivo; confirmatorio pre-registrado de exp17 si se quiere cruzar significancia con n mayor.

## Estado de cierre de la fase de verano (2026-07-24)
Diagnóstico + mejoras + matriz: **COMPLETOS**. La fidelidad responde a QUÉ evidencia entra (selección de
contenido: Tier 3 + exp17), no a su presentación (Tier A) ni a la instrucción (exp16). Config de encuestas
definida. Ramas `summer/ablacion` (Tier A/3) y `summer/mejoras` (exp16/17) committeadas local, sin push
(GATE). Falta: gold humano (ejecución de Enzo) y, si se decide, confirmatorio pre-registrado de exp17.

---

# Fase post-verano (arranque 2026-07-30)

**Mandato:** métodos nuevos sobre la palanca correcta, diagnóstico de si el techo es de cómputo,
y pulido del código. Ledger detallado: `paper/summer_ablation_log.md` entradas 15-17.

## Correcciones de rigor (Bloque 0)

| # | Defecto | Estado | ¿Cambia alguna cifra? |
|---|---|---|---|
| D1 | familia BH hardcodeada: exp16 declaraba 4 contrastes (son 2), exp17 declaraba 4/n=60/`baseline_repro` (son 1/n=25/`baseline`) | corregido, 9 artefactos regenerados con `contrasts` **byte-idéntico** | **No** — la corrección BH sí se aplicó sobre la familia real |
| D2 | el gold mostraba 1 chunk al humano y 5 a HHEM → κ(humano,HHEM) sesgada a la baja por construcción | rediseñado en 2 etapas | No (aún sin anotar) |
| D4 | el gold no tenía analizador | `scripts/analyze_gold_v4.py` construido y verificado | — |
| F1 | `RAGPipeline` nunca enruta el prompt por `query_type` → plantilla distinta a la ruta medida en **115/194** queries | perilla `prompt_routing`, default apagado | No — legado intacto (exp8) |
| F2 | detección de proveedores pierde GCP; devuelve `k8s`/`cncf` sin corpus | resolvedor propio, **25/25** vs etiquetas exp17 | No |
| C2 | dos definiciones vivas de "declinación": guards usaban 1 substring exacto, la métrica 28 regex | unificado a `classify_response`; guards regenerados | **Sí, en el ledger** — los veredictos se refuerzan |
| C3 | el caveat «n efectivo ≈29» de la entrada 11 era falso; el `n_paired` real es 50-59 | corregido en el ledger | No |

### C2 — las tasas de declinación corregidas refuerzan ambos veredictos

Los guards probaban un substring exacto case-sensitive; la métrica usa `classify_response` con 28
patrones case-insensitive y **tres** clases. Unificado, y la partición en tres cambia la lectura:

| | `pure_decline` | `answered` |
|---|---|---|
| exp16 baseline → anchored_cite | 46,7 % → **65,0 %** (+18,3 pp; antes se reportó +6,6) | 40 % → 28,3 % |
| exp16 baseline → strict_abstain | 46,7 % → **68,3 %** (+21,6 pp; antes +8,3) | 40 % → 23,3 % |
| exp17 baseline → balanced | 56 % → **36 %** (−20 pp) | 20 % → **48 %** (se dobla) |

exp16 hace callar al modelo **el doble** de lo reportado. **exp17 más que dobla las queries
plenamente respondidas** — el positivo es más fuerte que el del piloto. Report-before-prose: material
para el paper, sin tocar prosa.

**Ojo con la palabra "declinación":** 37 de 60 respuestas del baseline de Tier A llevan una frase de
rechazo y **aun así afirman claims** y se puntúan normal (q002 declina, responde 128 palabras y saca
fidelidad 1,0 sobre 1 claim). El denominador decline-aware solo descarta las que no tienen **ningún**
claim genuino (3/60). `hedged_partial` no es abstención.

> **Corrección 2026-08-04 (defecto #7, ledger entrada 22): `pure_decline` TAMPOCO es abstención.**
> Mide un **prefijo** (marcador en los primeros 300 chars), no un rechazo. Sobre el baseline de exp18,
> **84 de 89** filas así etiquetadas afirman claims genuinos (media 5,6) y llevan la **peor** tasa de
> claims no soportables (0,471 vs 0,330 en `answered`). La forma dominante es *"I cannot find
> sufficient information … **However, I can outline general steps**"* — memoria paramétrica tras el
> prefijo. **La tasa real de respuestas que no afirman nada es 3,1 % (6/194), no 45,9 %.** Las tablas
> de arriba siguen siendo correctas como **tasas de prefijo**; para cualquier argumento de abstención
> o usabilidad usar `asserts_nothing_rate` de `guards.json`. Medido así, el negativo de exp16 se
> **refuerza**: 6,7 % (baseline) → 15,0 % (`strict_abstain`) → 16,7 % (`anchored_cite`).
>
> Consecuencia sobre el denominador: la familia **PRIMARIA** de v4 excluye justo esas filas, lo que
> sube el nivel reportado **+0,0914 (HHEM)** y **+0,0983 (small)** frente a `sens_c` — casi la banda
> TOST entera. **Ninguna cifra publicada es errónea** (la familia la eligió Enzo el 2026-06-11 y
> `sens_c` se publica al lado); el gap ahora se **declara** en la salida del script para que no se lea
> como ruido.

## Gold humano — diseño de dos etapas (decisión de Enzo 2026-07-30)

Los instrumentos no ven lo mismo que el anotador: NLI `vb_agree` lee los 5 chunks y HHEM puntúa
`max_chunk` sobre los 5 (premisa a 1500 chars), pero el CSV mostraba **uno**. Cerrar el confound para
los 150 costaba 3-4× el tiempo del anotador (medido: 120 k → 566-879 k chars; 139 contextos distintos,
agrupar no comprime). Diseño adoptado:

- **Etapa A** — los 150 claims, 1 chunk @800 **+ la pregunta**. Selección verificada **idéntica** a la
  anterior. ≈4-5 h.
- **Etapa B** — 50 de esos mismos claims (submuestreo proporcional por estrato), **5 chunks @1500**
  (paridad exacta con HHEM), barajada, después de la A. ≈3,5 h.

La etapa B convierte el confound de *caveat* en *corrección*: mide cuántos juicios cambian al ver la
evidencia completa. `analyze_gold_v4.py` pondera por **Horvitz-Thompson** re-ejecutando el muestreador
real (los estratos se solapan: 413 de 14 409 claims llevan >1 flag, así que no hay forma cerrada) y
reporta el **n efectivo de Kish = 42,7** sobre 150 — el gold se diseñó para discriminar verificadores,
no para estimar una κ poblacional, y el script no lo esconde.

## El techo de contexto está a k>5, no a k=5

| k | p50 tok | p90 tok | supera 4096 |
|---|---|---|---|
| **5 (config actual)** | 1973 | 2999 | **0/60** |
| 10 | 3685 | 5737 | 24/60 (40 %) |
| 20 | 7109 | 11212 | 55/60 (92 %) |

**La ventana de 4096 no ata a la configuración desplegada** (concuerda con exp12: 2/194 tocaron el
límite). Matiza la matriz de factibilidad: la nube compra **capacidad de modelo**, y compra contexto
solo si más fragmentos ayudan — testeable limpio únicamente por encima de k≈7.

## Config desplegable para las encuestas — empaquetada

`SURVEY_DEPLOY` en `src/pipeline/pipeline_config.py` = el sistema medido con **exactamente dos**
perillas: `prompt_routing=True` (sin ella el demo usa otra plantilla que la ruta medida en 115/194) y
`balance_cross_cloud_providers=True` (exp17). Fuera de `PIPELINE_CONFIGS` para que
`get_config("hybrid")` siga devolviendo el sistema medido. **Test de aceptación:** replicar la regla
sobre el pool guardado de exp17 devuelve los `balanced_ids` exactos en **25/25**.

## Reproducibilidad

`REPRODUCE.md` (5 niveles) + `scripts/verify_summer_offline.py`, que re-deriva **cada** celda de Tier A
/ exp16 / exp17 / **exp18** desde las probs persistidas y recomputa los contrastes pareados: **todo
cuadra exacto, sin GPU**. Suite **130 tests, 0 fallos, 0 omitidos** (2026-08-04).

## exp18 — la compuerta, CERRADA (2026-08-04)

Análisis pre-registrado (ledger 19/20), sin desviaciones. Familia BH = 3 contrastes brazo-vs-baseline
por verificador; TOST bilateral contra ±0,081; los 3 verificadores. Ancla HHEM 0,4638, dentro de la
guarda de carga 0,40-0,55. Determinismo del re-puntuado: **bit-idéntico**.

| Brazo | Δ fidelidad (HHEM) | d_z | p_BH | Veredicto |
|---|---|---|---|---|
| `evidence_swapped` | **−0,3185** | −0,857 (grande) | **0,000** | **sig. en los 3 verificadores** |
| `oracle_evidence` | +0,0217 | 0,060 | 0,288 | **TOST EQUIVALENTE** (HHEM p=0,012 · small 0,0009 · base 0,023) |
| `final_top_k_10` | +0,0161 | 0,050 | 0,477 | n.s. (no truncadas +0,0259; truncadas +0,0005) |

**1. El generador SÍ usa el contexto.** Con la evidencia de otra consulta: `answered` 38,7 % →
**1,7 % (1 de 60)**, claims 10,6 → 4,8, jaccard-5grama 0,036, **reaparición de claims del baseline
0,0009**. Diverge casi por completo y **se calla en vez de fabricar**. Descarta la fila 4 de la matriz.

**2. Seleccionar óptimo por relevancia tópica NO compra fidelidad.** Equivalencia positiva dentro de
±0,081 con oráculo independiente (`bge-reranker-large`). El claim-level da un positivo pequeño bajo
GLMM que el bootstrap de cluster no confirma (HHEM p=0,114) — mismo patrón que exp17.

**3. Más evidencia compra COBERTURA, no anclaje.** k=10: `answered` 38,7 % → **67,0 %**, claims 10,6 →
**15,0**, "no afirma nada" 3,1 % → **0,5 %**; fidelidad plana. Costo: latencia p50 **49,0 s → 79,0 s
(+61 %)**, 90,4 s en el estrato truncado, **74/194 prompts (38 %) en el límite de 4 096**.

**Cota de selección** (n=188): baseline 0,4552 · alcanzable k=5 **0,5834** · superior 0,5879. Validada
contra el scorer canónico con **0,0000 de discrepancia en 188 queries**, bracket sin violaciones.

> **La cota NO es un objetivo.** Mantiene la respuesta fija y solo reordena chunks. Un selector real
> cambia la respuesta —el propio brazo de oráculo da reaparición de claims 0,0132, ~99 % de lo afirmado
> cambia— así que **no acota a un selector real** y "% de los +0,128 recuperado" sería una cifra
> fabricada. Lo que sí establece: **el pool k=50 respalda el 58,3 % de los claims que el baseline
> escribió, y los 5 elegidos solo el 45,5 %**. El ranking tópico deja anclaje sobre la mesa.

**Lectura conjunta = fila 3 de la matriz pre-registrada:** hay margen, el ranking tópico no lo
encuentra, falta un **selector guiado por anclaje** — método **local y sin gasto**.

### Nube: **NO-GO por la vía de la selección**

exp18 cerró todas las vías por las que el gasto podía justificarse como diagnóstico del techo de
selección. **Caveat honesto: exp18 no testea capacidad de generación directamente.** Un modelo mayor
podría anclar mejor; esa pregunta queda **abierta**, no resuelta. Cero gasto ejecutado.

## En curso / pendiente

- **exp19** — selector guiado por anclaje. Primaria = Δ fidelidad + TOST ±0,081 en los 3 verificadores;
  la cota solo como motivación. Selector = rerank claim-level sobre borrador con ms-marco-L12, **sin
  ningún verificador en el bucle**, para que los tres queden evaluadores limpios y `bge-reranker-large`
  siga siendo el oráculo independiente. Precede una sonda **offline de coste cero** que puede matar el
  experimento antes de gastar GPU.
- **Config de encuestas (congela a mediados de agosto)** — k=5 vs k=10 sobre claims, cobertura y
  latencia, confirmado por la **ruta de despliegue** (`SURVEY_DEPLOY`, recuperación en vivo), no por
  los contextos congelados de exp18. **El hueco de transferencia está cuantificado:** exp18 generó
  con `rgm.build_prompt` (mismo routing de prompt) pero **sin** `balance_cross_cloud_providers`, y
  el balanceo solo actúa sobre queries `cross_cloud`, que son **51 de 194 = 26,3 %** del set. No es
  despreciable. Además exp18 reporta tiempo **total** mientras la UI hace streaming
  (`chat_page.py:189` → `query_stream`), así que lo que percibe un encuestado es el **TTFT**.
  Decisión de Enzo.
- **759 claims sin respaldo** — mejor score p50 0,226 (τ=0,5); 123 (16 %) a menos de 0,1 del umbral.
  Taxonomía en curso desde artefactos persistidos: síntesis legítima vs memoria paramétrica vs
  alucinación vs fallo del verificador.
- **`deberta-large` — RETIRADO** (2026-08-04). Quedó en 11/12 configs y **nunca se usó para ninguna
  cifra**: el runner solo promueve el artefacto al completar las 12. El estándar de la fase es
  **NLI-small + NLI-base + HHEM: tres verificadores, dos familias** (dos NLI de entailment + un modelo
  de grounding ortogonal). **Nunca hubo un "trío NLI"**; un tercer NLI habría sido un voto
  correlacionado. Ninguna cifra publicada depende de él — ver ledger entrada 22 para la revisión
  candidato por candidato.
- **Nube** — `docs/CLOUD_EXPERIMENT_DESIGN.md`: A100 80 GB, 5-9 h ≈ USD 10-18, techo sugerido USD 50.
  **NO-GO por selección** tras exp18; sobrevive solo como test de **capacidad de generación**, y
  **cero gasto sin OK explícito**.
- **Gold humano** — etapas A y B listas para anotar (Enzo).
