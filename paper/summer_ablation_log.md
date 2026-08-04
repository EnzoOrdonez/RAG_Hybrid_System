# Ledger — Fase de verano (ablación + mejoras)

Bitácora de decisiones y corridas de la fase de verano (post-Nota 3, pre-encuestas SUS/Likert).
Complementa, **no toca**, los ledgers de Nota 3 (`paper/audit_findings.md` inmutable,
`paper/audit_findings_cc_addenda.md` N1–N9, `paper/correction_log.md`).

Reglas de la fase (autorizadas por Enzo, 2026-07-22):
- Generación LLM nueva permitida SOLO bajo IDs `exp15+`; `exp3..exp13`+`exp8b` (tag
  `nota3-evidencia-2026-06-11`) y `exp14` intactos, solo lectura.
- Ramas `summer/ablacion` (esta), `summer/mejoras` (al iniciar Fase 2); merge a main solo con OK.
- Modelo de iteración: Granite 4.1 8B determinista, subconjunto estratificado; escala a 194 q /
  otros modelos solo con OK por brazo.
- GATE antes de todo `git push`; STOP ante `.env`/secretos.

---

## Entrada 0 — Fase 0: arranque y línea base (2026-07-22)

**Qué se hizo:**
1. Entorno verificado: py 3.14.3, torch 2.10.0+cu126 (CUDA, RTX 3060), pydantic 2.12.5,
   Ollama **0.22.1** (congelado durante la fase — contexto H5/exp14), granite4.1:8b presente,
   NLI locales, índice `bge-large_adaptive_500`, 77.5 GB libres.
2. **Cifras v4 reproducidas offline** desde JSONs firmados (`scripts/verify_v4_offline.py`,
   reporte `output/audit/phase0_verification_summer_2026-07-22.md`): 16/16 celdas exactas
   (tol 1e-9) ambos verificadores; 24/24+18/18 pares estadísticos exactos; 0/12 RAG-vs-RAG;
   1/18 small + 1/18 base disjuntos → 0/18 robusto; CSV tabla6==JSON; exp11 NDCG@5 8/8
   (híbrido 0.7405 indep / 0.9948 circular); exp13 v2 off 0.285 / on 0.324.
3. pytest: 7 passed / 1 skipped (esperado). Evidencia firmada intacta (diff vs tag: 0 deleciones).
4. Mapa de perillas: `docs/KNOB_MAP_summer.md`. Hallazgo estructural: `config/config.yaml`
   (retrieval) y `config/evaluation_config.yaml` **muertos en runtime**; perillas vivas en
   `src/pipeline/pipeline_config.py` + `scripts/run_generation_matrix.py`.
5. Git: segunda opinión N9 commiteada a main (decisión Enzo); tag **`summer-baseline`** (29cea4a);
   rama `summer/ablacion` creada desde el tag.

**Hallazgos que condicionan la fase:**
- **Caché HF purgada** (≈2026-06-30): faltan bge-large-en-v1.5 (~1.3 GB), ms-marco-MiniLM-L-12-v2
  (~134 MB) y bge-reranker-large (~2.2 GB). Sin ellos no hay retrieval denso/híbrido nuevo ni
  oráculo. exp11 firmado NO es re-ejecutable en el estado actual de la máquina (sus JSONs quedan
  como evidencia). **Decisión Enzo: descarga única aprobada a `data\models\`** (snapshot durable,
  revisiones+sha256 a este ledger al ejecutarla); después re-congelar `HF_HUB_OFFLINE=1`.
- exp11 guarda **top-5 ids** por query × 4 configs (incl. pre-rerank RRF) → brazos de generación
  gratis sin re-retrieval: `reranker_off`, `final_top_k_3`, permutaciones de orden de contexto.
- H5/exp14: entorno no determinista bajo presión de VRAM → harness exp15 con pases desacoplados
  (nunca embedder/NLI residentes durante generación), sonda de determinismo 3× por brazo, y brazo
  ancla `baseline_repro` (toda comparación brazo-vs-ancla es mismo-entorno).

**Decisiones de diseño (Enzo, 2026-07-22):**
- Subconjunto de iteración: **n=60** estratificado (query_type × difficulty proporcional,
  largest-remainder, seed=42) → `data/evaluation/summer_subset.json`; los **25 cross-cloud** como
  corrida separada para brazos sensibles (patrón exp13). Fidelidad en subset = triage (atrición
  deja n efectivo ≈27–35 → detecta solo d_z≈0.55–0.63); confirmatorio a 194 q con OK por brazo.
- Prioridad Fase 1: **Tier 0** instrumento NLI (cero LLM: rescore small+base con persistencia de
  probs crudas + sweep CPU variante×umbral + kappa) → **Tier A** generación desde ids exp11
  (baseline_repro, reranker_off, final_top_k_3, context_reversed, context_lost_middle; ≈6–9 h GPU)
  → **Tier B** retrieval nuevo post-descarga (rrf_k {10,30,100}, linear, top_k_candidates {20,100},
  final_top_k_8; generación gateada por |ΔNDCG@5 indep| ≥ 0.02 o sig_bh; ≈4–8 h GPU).

**Pendiente:** harness `scripts/run_exp15_ablation.py` + registro de brazos
`experiments/ablation_arms.json` + subset (`scripts/make_summer_subset.py`).

---

## Entrada 1 — Enmienda CLAUDE.md + descarga de modelos + arranque Tier 0 (2026-07-22)

- **CLAUDE.md enmendado** con OK de Enzo (commit `7560184`): generación LLM solo bajo `exp15+`;
  evidencia inmutable ampliada a `exp3..exp14`.
- **Descarga única aprobada EJECUTADA** (sesión online supervisada, `scripts/download_summer_models.py`;
  entorno re-congelado offline después). Proveniencia (manifiesto completo con sha256 por archivo:
  `output/audit/summer_models_manifest_2026-07-22.json`, copia en `data/models/`):
  - `BAAI/bge-large-en-v1.5` → `data/models/bge-large-en-v1.5`, revisión `d4aa6901d3a4`, 1.25 GB (safetensors).
  - `cross-encoder/ms-marco-MiniLM-L-12-v2` → `data/models/ms-marco-MiniLM-L-12-v2`, revisión `7b0235231ca2`, 0.13 GB.
  - `BAAI/bge-reranker-large` → `data/models/bge-reranker-large`, revisión `55611d7bca2a`, 2.11 GB.
  → **Tier B desbloqueado.** Nota: los loaders aún cargan por nombre de hub; el ajuste local-first
  se hará en el harness exp15 (los modelos de data/models se pasan por ruta).
- **Tier 0 GO** (decisión Enzo): `scripts/rescore_nli_exp15.py` = clon parametrizado de
  `rescore_nli_v3.py` que (a) escribe SOLO en `exp15_ablation_nli/` (exp12 read-only), (b) persiste
  probs crudas por (config, query, claim, chunk) en gzip — el sweep variante×umbral pasa a ser
  re-agregación CPU, y (c) emite la agregación vb_agree@0.7 en formato v3 como **validación cruzada
  fila-por-fila contra el rescore firmado** antes de confiar en cualquier punto de operación nuevo.
  Protocolo: smoke 2 queries → corrida completa small → base → sweep.

---

## Entrada 2 — Tier 0 COMPLETO: mapa de robustez del instrumento NLI (2026-07-23)

**Config:** `exp15_ablation_nli/`. Scoring GPU una vez por verificador (small ~8 min, base ~15 min,
fp16, pooling idéntico a v3); sweep = re-agregación CPU pura, 64 puntos = 4 variantes
(v0, vb_agree, va_margin d0.1/d0.2) × umbrales ent×contr {0.5,0.6,0.7,0.8}², metodología v4
completa por punto (primary_answered + exclude-vacuous + Wilcoxon/d_z/bootstrap/BH, seed 42).

**Validación previa (condición para confiar en el sweep):** las filas vb_agree@0.7/0.7
reconstruidas desde las probs crudas son **idénticas 1798/1798** al rescore firmado de exp12 en
AMBOS verificadores (ancla automática del sweep; también verificado archivo-vs-archivo).
El instrumento queda reproducido bit-perfect antes de mover cualquier perilla.

**Resultados (sweep_results.json / sweep_summary.md):**
1. **Verificador base: el nulo 0/12 es TOTALMENTE robusto — 0/64 puntos con algún par
   RAG-vs-RAG significativo.**
2. **Verificador small: 32/64 puntos muestran EXACTAMENTE un par significativo, siempre el mismo:
   granite hibrido-vs-lexico.** Patrón nítido: significativo ⟺ ent_t ≤ 0.6 (las 4 variantes,
   los 4 contr_t); con ent_t ≥ 0.7 → 0/12 en todos. Punto canónico (vb_agree 0.7/0.7):
   d_z −0.309, p_bh 0.085, n 53 (near-miss).
3. **Dirección consistente 128/128:** granite hibrido>lexico en TODOS los puntos × ambos
   verificadores (d_z, convención b−a, negativo = hibrido mejor: small −0.20..−0.39,
   base −0.20..−0.23). Efecto pequeño real plausible, sub-potenciado en el punto canónico.
4. **La variante de guarda y contr_t son casi irrelevantes** (sig y kappa apenas cambian):
   el eje sensible del instrumento es el umbral de ENTAILMENT, no la vía de contradicción.
5. **Kappa small-vs-base a nivel claim: 0.30–0.36 en toda la grilla** (canónico 0.323,
   n=14 469 claims). Acuerdo pobre → cualquier hallazgo mono-verificador es frágil; la regla
   framing B (doble verificador) queda justificada por diseño.
6. Nivel de fidelidad fuertemente dependiente de ent_t (granite hibrido 0.372@0.5 → 0.299@0.7 →
   0.248@0.8): la métrica es relativa al instrumento; comparar solo dentro del mismo punto.

**Veredicto Tier 0:** el "0/12" publicado (punto canónico, doble verificador) queda VERIFICADO y
es robusto bajo base en toda la grilla. PERO el contraste central granite hibrido-vs-lexico es
direccionalmente consistente en el 100% de la grilla y cruza significancia bajo small con ent
laxo → hipótesis actualizada: **efecto real pequeño (híbrido > léxico en granite) enmascarado por
un instrumento ruidoso (κ≈0.32) y potencia limitada (n≈53 tras exclusiones)**. No cambia ninguna
cifra publicada; contextualiza el hallazgo central. Reportado a Enzo antes de tocar prosa
(regla 3). Alcance: el sweep cubre la familia between-scenario; between-model no re-barrido
(no era la pregunta).

**Implicación para Tier A:** el eslabón retrieval→fidelidad merece el test directo
(`reranker_off` y permutaciones de contexto) con n máximo disponible; considerar confirmatorio
a 194 q del par granite hibrido-vs-lexico si Tier A lo respalda.

---

## Entrada 3 — CAUSA RAÍZ DE H5 IDENTIFICADA + protocolo de corrida limpia (2026-07-23)

**Cronología:** smoke Tier A pasó sonda de determinismo (3× idénticas); 1 h después las sondas de
`baseline_repro` y `reranker_off` FALLARON (corrida completa). Diagnóstico en caliente:

1. **Mecanismo:** base de escritorio ≈1.7 GB VRAM (Edge/WebView2/Brave/apps Electron/overlays) →
   granite4.1:8b @ num_ctx 4096 (Ollama 0.22.1 lo dimensiona en 6.2 GB; con flash-attention +
   KV q8_0 baja a 5.7 GB; los pesos solos ≈4.9 GB) **no cabe** en los ~4.4 GB libres → split
   33–43 % CPU / resto GPU → **generación no determinista**. Patrón medido: 1.ª generación ≠
   2.ª/3.ª (idénticas entre sí) — divergencia frío-vs-caliente del prompt-cache con KV mixto.
2. **No fue la versión:** logs de la app confirman **Ollama 0.22.1 ya el 2026-06-07** (exp12).
   La diferencia junio↔hoy es la VRAM libre al cargar: junio = boot limpio (~0.4 GB base,
   granite 5 351 MB **100 % GPU**, bit-determinista). Matiz: qwen3.5 corrió en junio con offload
   parcial Y fue determinista → el split no rompe siempre; aquí sí (frío/caliente).
   Esto cierra H5: "determinista a temp=0" = propiedad de **(modelo, versión, VRAM libre al
   cargar)**; precisión ya exigida por el ledger N9, ahora con mecanismo.
3. **Hallazgo colateral (exp12, para párrafo de limitación — NO cambia cifras):** el techo de
   contexto se tocó en exp12: `tokens.input` máx = **4096 exacto** en hibrido|granite (≥1 query
   con prompt truncado), p95 = 3 087; con salida ≤1 024, la cola larga (input+output > 4096)
   sufrió context-shifting. Reportar a Enzo antes de cualquier prosa.
4. **Higiene:** purgadas 37 entradas de caché LLM (claves recomputadas por sha256) y el
   checkpoint 30/60 de baseline_repro — todo generado bajo split, propio y sin commitear.
   Entradas de la era exp12 intactas (config_name distinto → claves distintas).
5. **Protocolo de corrida limpia:** `scripts/launch_tierA_clean.ps1` — tras reinicio limpio:
   server Ollama con defaults (condiciones canónicas de junio, sin KV q8), gate duro
   **"100 % GPU o aborta"**, Pass G 5 brazos (sonda 3× por brazo) + Pass N small y base,
   checkpointeado/reanudable. Decisión Enzo: reinicio inmediato y lanzamiento.

---

## Entrada 4 — Tier 3 arranque: descargas + Bloque A (anatomía del desacuerdo NLI) (2026-07-23)

Segundo prompt maestro (libertad total para diagnosticar/mejorar). Tier 3 = verificador de fidelidad
más estable (máximo impacto dado κ≈0.32). Decisiones Enzo: relajar gate Tier A; descargar HHEM+large;
diseñar gold N≈200. Corrida limpia Tier A del 12:18 **abortó correctamente en el gate 100% GPU**
(`offloaded 30/41 layers`) → confirma la causa raíz: **granite@4096 NO cabe 100% GPU en 6 GB con
Ollama 0.22.1** (pesos 5.1 GB + KV 640 + compute 533 ≈ 6.3 GB > 6.0 usable; auto num_ctx=4096).

**Descargas aprobadas EJECUTADAS** (`scripts/download_verifier_models.py`, manifiesto sha256 en
`output/audit/verifier_models_manifest_2026-07-23.json`):
- `cross-encoder/nli-deberta-v3-large` → `data/models/nli-deberta-v3-large`, rev `bab4bc717883`, 1.75 GB
  (3.er voto NLI, API idéntica, id2label [contr,ent,neut] confirmado igual a small/base).
- `vectara/hallucination_evaluation_model` (HHEM-2.1) → `data/models/hhem-2.1`, rev `8e4a2e6e96c7`,
  0.44 GB (grounding, familia ortogonal). Dependencia dura: foundation `google/flan-t5-base`
  (config+tokenizer, ~2 MB, SIN pesos — el safetensors de HHEM puebla el backbone) → `data/models/flan-t5-base`.

**Bloque A — anatomía del desacuerdo** (`scripts/analyze_exp15_disagreement.py`, CPU sobre probs Tier 0;
salidas en `exp15_ablation_nli/disagreement_{analysis.json,summary.md}` + `false_contradicted_candidates.csv`):
1. **Confusión small×base, 14 469 claims: κ=0.323, acuerdo 67.3%.** Desacuerdo dominado por la frontera
   supported↔unsupported (1968+1118=3086 claims) — el gate de ENTAILMENT, consistente con Tier 0.
2. **small sobre-etiqueta contradicted 1.8×** (1696 vs 956 de base) y es más decisivo (empuja claims fuera
   de unsupported hacia supported Y contradicted); base es conservador (estaciona en unsupported).
3. **small 2.5× más frágil al umbral**: 3.04% de TODOS los claims cambian etiqueta supported al mover
   ent_t ±0.05, vs 1.19% en base. El verificador **runtime (small) es el ruidoso** — argumento fuerte
   para migrar el punto de operación a base o a un ensemble.
4. **128 falso-contradicted** (small=contradicted conf≥0.9 ∧ base=supported): granite 52, qwen 43,
   mistral 27, gemma 6; lexico 55, denso 36, hibrido 37 → `false_contradicted_candidates.csv`, alimenta
   el estrato contradicted del gold (Bloque D). Familia del artefacto q085.
5. **Sensibilidad de agregación — GATE A-G1 DISPARADO:** el `max`-over-chunks NO es inocuo.
   `mean_top2` cambia 32% de etiquetas (shift sistemático); `noisy_or` cambia solo 5.5% PERO **hace el
   par granite hibrido-vs-lexico SIGNIFICATIVO bajo small** (d_z −0.42, **p_bh 0.009**, 1/12) — porque
   noisy-or acredita evidencia DISTRIBUIDA entre chunks, que la recuperación híbrida aporta más, mientras
   max solo mira el mejor chunk. **El agregador max sub-acredita evidencia distribuida** → decisión Enzo
   del agregador canónico antes de fijar el instrumento. (Nota: noisy-or también infla contradicción;
   evaluar en Bloque B con control negativo, no adoptar por mover el contraste — regla anti-p-hacking.)

**Lectura:** dos evidencias convergen en que "baja fidelidad" es en parte artefacto de medición:
(a) el verificador runtime (small) es el más ruidoso/frágil y sobre-contradice; (b) el agregador max
sub-acredita evidencia distribuida, ocultando una señal híbrido>léxico que noisy-or revela. Ninguna
cambia cifras publicadas aún; A-G1 requiere decisión de Enzo. Reporte antes de prosa.

**Siguiente:** scoring deberta-large (en curso, GPU) → HHEM (tras liberar GPU) → Bloque B ensembles
con los 4 verificadores + control negativo → gold N≈200.

---

## Entrada 5 — Bloque D (gold N≈200) + reorden GPU (2026-07-23)

**Reorden de scoring:** deberta-large resultó ~11× más lento por par que base (~7 pairs/s: large fp16
apenas cabe en 6 GB → thrashing), ETA ~3 h. Es el verificador de MENOR valor (misma familia → errores
correlacionados). HHEM (ortogonal, alto valor, responde "¿es artefacto de familia NLI?") estaba bloqueado
detrás. Decisión: matar large (resumible por `.partial`, 2/12 hecho), correr HHEM primero.

**HHEM smoke (2q/config):** carga de código custom offline OK (foundation pineado a `data/models/flan-t5-base`).
Grounding scores mean **0.993** (casi todo grounded a τ=0.5). Dos lecturas posibles: (a) HHEM lenient
inútil aquí, o (b) **hallazgo ortogonal clave** — un modelo de grounding RAG-específico ve estos claims
COMO grounded, contra la "baja fidelidad" NLI → "baja fidelidad" sería artefacto de familia NLI. **El
control negativo decide**: si HHEM también dice grounded a chunks aleatorios → lenient inútil; si NO →
discrimina y la señal NLI-baja es artefacto. Probs crudas persistidas (τ = perilla CPU).

**Bloque D — gold v4 GENERADO** (`scripts/build_gold_v4.py`; entregable para Enzo/anotadores):
- 150 claims NUEVOS (seed 42), disjuntos de los 50 de v3 → total objetivo 200.
- Estratos (sobre-muestrean la señal de desacuerdo): disagreement 50, near_threshold 40,
  false_contr 30, random_anchor 30. Todos alcanzados.
- **Juicio HUMANO CIEGO**: la plantilla (`output/audit/claim_audit_sample_v4.{csv,md}`) muestra claim +
  mejor-evidencia + juicio vacío (correcto/incorrecto/dudoso); NO muestra etiquetas de verificadores
  (evita anclaje). El scoring verificador-vs-humano es join post-hoc por (config,qid,claim). Estratos +
  etiquetas de verificadores en `_meta.json` (no visible al anotador).
- Potencia: binding = κ(verif,humano) CI half-width ≤0.1; con 200 y κ≈0.32 (muchos pares discordantes)
  cubre también McNemar 10 pp entre verificadores (power 0.8). Esfuerzo ~4-5 h.
- **Pendiente Enzo/anotadores:** llenar `juicio_humano`. Sin gold, la selección de verificador (Bloque B)
  usa solo el control negativo como criterio provisional (anti-p-hacking).

---

## Entrada 6 — HALLAZGO MAYOR: control negativo → "baja fidelidad" ≈ artefacto NLI (2026-07-23)

Reporte dedicado: `output/audit/tier3_negative_control_finding_2026-07-23.md`. **Report-before-prose:
NO cambia cifras firmadas; contextualiza el 0,30. No tocar A.3/LACCI sin OK.**

Control negativo (400 claims × 5 chunks ALEATORIOS no relacionados, seed 42;
`scripts/build_negative_control.py` + `score_negative_control.py` + `analyze_negative_control.py`;
reproduce el método de `h2_variant_eval.json`):

| Verificador | Falso-positivo en aleatorio | Datos reales |
|---|---|---|
| NLI small vb_agree | **falso-contradicted 0,237** | fidelidad ≈0,30 |
| NLI base vb_agree | **falso-contradicted 0,215** | fidelidad ≈0,30 |
| NLI v0 legacy | 0,54–0,60 | — |
| HHEM-2.1 τ=0,5 | **falso-grounded 0,010** (mean 0,002) | grounding ≈0,99 |

**Los NLI marcan ~22 % de texto ALEATORIO no relacionado como "contradicted"** — falso-positivo
sistemático de contradicción. HHEM (grounding ortogonal) ~1 % falso-grounded Y ≈0,99 en datos reales →
discrimina nítidamente y ve los claims COMO anclados. **El enigma central se re-enmarca: el ≈0,30 es en
gran parte artefacto del instrumento NLI** (sobre-dispara contradicción, sub-acredita entailment en docs
técnicos). Converge con Tier 3-A (small ruidoso/frágil, 128 falso-contradicted).

**Cautelas:** (1) HHEM ≈0,99 roza techo → puede no discriminar escenarios (varianza baja) — no invalida
el punto del NIVEL pero sí implica techo; cuantificar con corrida completa. (2) Arbitraje objetivo
requiere el gold humano (v4, pendiente): ¿HHEM o NLI? Selección de instrumento anclada en gold + control
negativo, NUNCA en downstream. (3) HHEM sin truncar chunks largos (T5 sin límite duro).

**En curso:** cadena overnight HHEM full + deberta-large + neg-control large (~5-6 h, checkpointeada) →
ensemble sweep (Bloque B) con los 4 verificadores.

---

## Entrada 7 — CORRECCIÓN: bug de carga de HHEM invalida los números HHEM de la entrada 6 (2026-07-23)

**Retracción parcial de la entrada 6.** La afirmación "HHEM ve las respuestas como grounded ≈0,99 → la
baja fidelidad es artefacto NLI" era ERRÓNEA. Cadena:
- HHEM full (datos reales, config lexico|granite, 186 resp) dio fidelidad **0,038**, NO 0,99. El smoke de
  2 queries (0,99) era no representativo Y —resultó— basura de un modelo mal cargado.
- Test controlado localizó el bug: contradicción "the sky is red" → 1,0; grounded "the sky is blue" →
  0,12 (invertido/aleatorio). **Causa:** los pesos del safetensors llevan prefijo `t5.`
  (`t5.classifier.weight`) y se cargaban en `model.t5` (espera sin prefijo) → `strict=False` descartó
  TODOS los pesos → T5 aleatorio. **Fix:** `model.load_state_dict(state)` + guarda que aborta si
  <100 tensores cargan. Tras el fix el test da correcto (sky-blue 0,856, sky-red 0,005, EKS 0,968,
  no-relacionado 0,003).
- **VOID:** todos los números HHEM previos (smoke 0,99; neg-control falso-grounded 0,010; fidelidad real
  0,038). La conclusión "baja fidelidad = artefacto NLI" NO está respaldada; pendiente de re-corrida.

**SIGUE VÁLIDO** (verificadores NLI deberta CrossEncoder, cargan bien): control negativo NLI
falso-contradicted small 0,237 / base 0,215 (v0 0,54–0,60) — los NLI marcan ~22 % de texto ALEATORIO
como contradicted. Tier 3-A completo (small ruidoso, 2,5× más frágil, 128 falso-contradicted). Esto
evidencia que parte del ruido viene de contradicciones inventadas por el NLI, pero **no cuantifica cuánto**
del 0,30 es artefacto — para eso HHEM bien cargado (re-corriendo) + gold.

**Lección de proceso:** nunca reportar un verificador nuevo sin (a) test controlado de cordura y (b)
coherencia smoke-vs-full. El smoke de 2 queries indujo una conclusión apresurada; corregido.

**En curso:** deberta-large 8/12 (terminando); al liberar GPU: re-correr HHEM (control negativo + datos)
con carga corregida + truncación/batch menor (era 1,9 h/config + OOM). Luego ensemble sweep real.

---

## Entrada 8 — RESULTADO DEFINITIVO Tier 3: el 0/12 es ROBUSTO AL INSTRUMENTO (2026-07-23)

HHEM corregido (fix de load + truncación 1500c + batch 16 → 122 s/config vs 6694 s roto) sobre los 12
configs. Análisis reproducible: `scripts/compute_exp15_hhem_analysis.py` →
`exp15_ablation_nli/hhem_vs_nli.{json,md}`. Reporte: `output/audit/tier3_negative_control_finding_2026-07-23.md`.

**1. NIVEL: NLI sub-acredita sistemáticamente.** HHEM > NLI-small en los 12 configs, gap medio **+0,307**
(+0,14..+0,43). granite HHEM 0,40-0,44 vs NLI 0,23-0,30; gemma 0,74-0,80 vs 0,32-0,41; qwen 0,63-0,69;
mistral 0,49-0,58. Coherente con el 22 % falso-contradicted del NLI → el instrumento NLI baja el NIVEL.

**2. CONTRASTE: 0/12 ROBUSTO AL INSTRUMENTO.** Test pareado between-scenario bajo HHEM (metodología v4,
BH) = **0/12 RAG-vs-RAG significativos**, igual que NLI-small y NLI-base. Par más fuerte granite
hibrido-vs-lexico: d_z −0,35 p_bh **0,11** (NLI-small daba p_bh 0,085) — direccionalmente consistente
hib>lex en los 3 instrumentos, nunca cruza BH.

**VEREDICTO:** "mejor recuperación no mejora significativamente la fidelidad (0/12)" **se sostiene bajo
tres familias de verificador** (NLI small, NLI base, HHEM grounding ortogonal). El ≈0,30 es
relativo-al-instrumento (HHEM da ≈0,55 medio) pero el **contraste entre escenarios es genuinamente
pequeño/nulo, NO artefacto**. Esto **corrige la afirmación errónea de la entrada 6** ("todo artefacto
NLI") y **refuerza** el hallazgo central de la tesis (robusto al instrumento). Report-before-prose:
material para reforzar el 0/12 y para una nota de Limitaciones (fidelidad absoluta instrument-relative);
NO tocar A.3 sin OK.

**Pendiente:** gold humano (arbitrar NIVEL 0,30 vs 0,55 + validar HHEM); deberta-large 8/12 (3.er voto).

---

## Entrada 9 — CORRECCIÓN de la entrada 8: familia BH inconsistente → HHEM es 1/12, NO 0/12 (2026-07-23)

**Retracción del titular de la entrada 8** ("0/12 ROBUSTO AL INSTRUMENTO, se sostiene bajo HHEM"). Error
metodológico: `compute_exp15_hhem_analysis.py` construía la familia BH between-scenario **excluyendo los
pares sin_rag** (12 pares) → corrección BH distinta de la del v4 publicado, que la construye
**incluyendo sin_rag** (24 pares = 4 modelos × C(4,2); confirmado por `verify_v4_offline.py` y
`compute_faithfulness_metrics.main()`). Recomputado con la familia v4-consistente (vía
`compute_exp15_nli_sweep.evaluate_point`):

| Instrumento | RAG-vs-RAG sig (familia v4, 24) | granite hib-vs-lex |
|---|---|---|
| NLI small | 0/12 | p_bh 0,085 (no) |
| NLI base | 0/12 | — |
| **HHEM (grounding)** | **1/12** | **p_bh 0,020 (SÍ), d_z −0,35** |

(mistral hib-vs-lex bajo HHEM: p_bh 0,067 — cerca, no sig.)

**VEREDICTO CORREGIDO:** el nulo 0/12 **NO es robusto al cambio a HHEM**. Bajo el instrumento de
grounding limpio (familia v4-consistente), **granite hibrido-vs-lexico CRUZA significancia (1/12,
p_bh 0,020)** donde los NLI ruidosos no (p_bh 0,085). Lectura: **el efecto retrieval→fidelidad SÍ existe
para el modelo determinista (granite: híbrido > léxico), pero solo es detectable con un instrumento
menos ruidoso** — el NLI lo enmascara (22 % falso-contradicted). Es 1/12 (solo granite), d_z pequeño
(−0,35), τ-dependiente → **matizar, no sobre-vender; pendiente gold humano para validar HHEM**.

Las tres iteraciones convergen: entrada 6 ("todo artefacto NLI") sobre-vendió; entrada 8 ("0/12 robusto,
sin efecto") sub-vendió por el bug de familia; **la verdad está en medio: NLI enmascara un efecto real
pequeño y granite-específico que HHEM revela.** Lección añadida a las reglas: verificar la construcción
de la familia estadística antes de cualquier titular.

**Report-before-prose:** esto toca la interpretación del hallazgo central (el 0/12 depende del
instrumento). NO cambia cifras firmadas (nuevo exp15). Reportar a Enzo antes de tocar A.3/LACCI.
Corregidos: `hhem_vs_nli.{json,md}`, reporte, SUMMER_RESULTS, memoria.

**Bloque B (ensemble sweep, ejecutado):** front-runner por control negativo (validez de constructo,
menor falso-positivo=mejor): **E5_base_and_hhem 0,003** > hhem 0,033 > **E1_mean NLI 0,09** (mitad del
NLI solo) > base 0,215 > small 0,237. noisy_or 0,55 RECHAZADO (su "significancia" viene de inflar
contradicción — resuelve A-G1). Selección definitiva espera el gold. `ensemble_{results.json,summary.md}`.

## Entrada 10 — Tier A COMPLETO: la ablación de contexto es NULA y robusta al instrumento (2026-07-24)

Re-corrida Tier A con gate relajado (advertir+warmup, decisión Enzo entrada previa). 5 brazos × 60q,
granite temp0 seed42, contexto = ids firmados exp11 híbrido full-rerank **transformados sin re-recuperar**.
Passes desacoplados (G generación 4,75 h; N scoring NLI small+base; + HHEM triangulación). Contraste
**pareado within-session por query_id vs baseline_repro** (Wilcoxon + d_z + bootstrap seed42, familia BH
de 4; decline-aware: pares None descartados, vacuous=1.0). Scripts nuevos: `run_exp15_ablation.py::pass_g`
(gate warn+warmup), `compute_tierA_arm_stats.py`, `rescore_grounding_tierA.py`.

**Determinismo 3×:** baseline_repro / reranker_off / context_reversed = idénticos; final_top_k_3 y
context_lost_middle = 1.ª gen difiere (cold-vs-warm cache, H5) → registrado en probe_report, contraste
pareado sigue válido (misma sesión/queries).

**RESULTADO — 0/4 brazos significativos bajo NLI-small, NLI-base Y HHEM (robusto al instrumento):**

| Brazo | Transform | NLI small Δ (p_BH) | NLI base Δ (p_BH) | HHEM Δ (p_BH) |
|---|---|---|---|---|
| reranker_off | RRF pre-rerank | −0,058 (0,60) | −0,050 (0,82) | +0,007 (0,96) |
| final_top_k_3 | top-3 | −0,019 (0,60) | +0,029 (0,53) | −0,006 (0,96) |
| context_reversed | orden invertido | +0,043 (0,60) | +0,047 (0,53) | +0,060 (0,96) |
| context_lost_middle | relevante al centro | +0,027 (0,60) | −0,005 (0,53) | −0,025 (0,96) |

Nivel baseline_repro: 0,308 (small) / 0,204 (base) / 0,450 (HHEM). HHEM baseline 0,450 = exp12 granite
hibrido HHEM 0,40–0,44 → **carga HHEM verificada** (no basura 0,04).

**CONTRASTE CLAVE con Tier 3 (publicable):**
| | Tier 3 entre-escenarios (léx/denso/híb) | Tier A transforms del MISMO pool híbrido |
|---|---|---|
| bajo NLI | 0/12 | 0/4 |
| bajo HHEM | **1/12** (granite híb>léx cruza sig) | **0/4** (sigue nulo) |
| robusto al instrumento | **NO** | **SÍ** |

**Lectura mecanística:** la fidelidad responde (débil, granite, solo-HHEM) a **QUÉ documentos** selecciona
el *método* de recuperación (híbrido vs léxico, NDCG 0,74 vs 0,44), pero **NO** a cómo se arregla un pool
ya recuperado — reranking, top-k, orden y lost-in-the-middle son nulos en los 3 instrumentos. El efecto
pequeño que existe es de **selección de contenido**, no de **ordenamiento/reranking/recorte**.
→ **DESCARTA lost-in-the-middle y el reranking como palancas de fidelidad**; el cuello está en la
generación/selección, no en la presentación del contexto.

**Deriva H5 (baseline_repro julio vs exp12 hibrido junio, NLI small, 59q):** media +0,033 (Wilcoxon
p=0,083, **n.s.**), |deriva por-query| 0,087, **corr r=0,858**, 39% (23/59) queries con fidelidad idéntica.
La deriva (+0,033) es del orden de los efectos de brazo → **valida el diseño pareado within-session** (los
brazos NO se comparan contra junio) y la decisión de relajar el gate (el ancla deriva poco, n.s.).

**Report-before-prose:** refuerza el hallazgo central desde la generación ("mejor recuperación ≠ mejor
fidelidad" se sostiene incluso degradando el contexto). Material de discusión/limitaciones, NO cambia
cifras firmadas (exp15 nuevo). Reporte: `output/audit/tierA_ablation_finding_2026-07-24.md`. NO tocar A.3
sin OK frase por frase.

## Entrada 11 — Fase 2 exp16: decodificación anclada NO mejora la fidelidad (negativo triangulado) (2026-07-24)

Rama `summer/mejoras` (desde 3b60ed3; infra commit 7f331e1). 3 brazos de prompt sobre el MISMO pool
híbrido (identidad de contexto), solo cambia system+sufijo. Contraste pareado within-session vs
`baseline_repro`. **baseline regenerado fresco `--no-cache`** — hallazgo operativo: el caché LLM se indexa
por `config_name‖prompt`, NO por exp-id, así que exp16 baseline_repro (prompt canónico) colisionó con el
caché de Tier A y se sirvió stale (~7h antes); fresh q016 2247 vs cached 2704 → deriva H5 cross-sesión pese
a det3x=True within-session. `--strip-inline-cites` quita los `[N]` antes de extraer claims, uniforme en
los 3 brazos (path firmado `_extract_claims` intacto).

**RESULTADO — 0/2 bajo NLI-small, NLI-base Y HHEM (ninguna mejora, robusto al instrumento):**

| Brazo | small Δ | base Δ | HHEM Δ | veredicto |
|---|---|---|---|---|
| anchored_cite (cita [N] por claim) | −0,034 | −0,040 | −0,056 | 0/3; tiende ABAJO en los 3 |
| strict_abstain (omitir lo no explícito) | −0,002 | +0,031 | +0,038 | 0/3; plano |

Nivel baseline 0,296/0,226/0,498 (small/base/HHEM); |d_z|≤0,19. HHEM baseline 0,498 → carga verificada.

**Guardas anti-gaming (mecanismo del fallo):** ambos brazos suben la declinación (baseline 51,7 % →
anchored 58,3 % / strict 60,0 %) y recortan contenido (palabras 352→223/191; claims 11,95→7,55/5,27).
anchored_cite baja el solape verbatim (0,122→0,061 → NO copia) y aun así baja la fidelidad → **cita ≠
grounding** (teatro de citación). strict_abstain sube el solape (0,233 → copia más) y fidelidad plana.
Ninguna compra fidelidad; solo hacen que granite diga menos y se abstenga más.

**Veredicto:** la decodificación anclada por prompt no mejora la fidelidad de granite (n=60). Junto con
Tier A (nulo de recuperación): ni el arreglo del contexto ni el prompt mueven la fidelidad → techo de
capacidad del modelo (1b, fuera de 6 GB) o instrumento (Tier 3, gold pendiente). Caveat de potencia:
declinación baseline 51,7 % → n efectivo ≈29, underpowered; la DIRECCIÓN (anchored abajo, strict plano) +
las guardas argumentan contra un positivo oculto. Confirmatorio 194q solo con OK.

**Report-before-prose:** toca la matriz de factibilidad (línea 1a A.3: IMPLEMENTAR → IMPLEMENTADA Y
PROBADA, sin ganancia local — resultado negativo honesto). NO cambia cifras firmadas (exp16 nuevo). NO
tocar prosa A.3 sin OK. Reporte: `output/audit/exp16_anchored_finding_2026-07-24.md`.

## Entrada 12 — exp17 piloto cross-cloud: cobertura balanceada por proveedor SUBE la fidelidad (PRIMER positivo) (2026-07-24)

Rama `summer/mejoras` (infra 46a7a53). Diagnóstico que motivó el piloto: de 25 queries comparativas
cross-cloud, **solo 7/25 recuperan TODOS los proveedores pedidos en el top-5** (18/25 pierden ≥1 proveedor
entero pese a NDCG 0.85) → la comparación es imposible de anclar. Falla de SELECCIÓN DE CONTENIDO, no de
ranking topical; la expansión léxica de exp13 nunca la tocó.

2 brazos del MISMO pool híbrido (aísla la cobertura), pareado within-session, granite temp0 seed42
`--no-cache`. baseline = rerank(pool)[:5] (**validado idéntico a exp13 exp_off: overlap 5.0/5 en 25/25**);
balanced = ⌈5/|P|⌉ por proveedor pedido del mismo pool reordenado. **Cobertura 7/25 → 25/25** (set cambió
en 22/25).

**RESULTADO — balanced > baseline en los 3 instrumentos (primer positivo direccional de la fase):**

| Instrumento | baseline | balanced | Δ | d_z | p |
|---|---|---|---|---|---|
| NLI small | 0,199 | 0,236 | +0,037 | 0,13 | 0,40 |
| NLI base | 0,151 | 0,196 | +0,045 | 0,21 | 0,14 |
| **HHEM** | 0,477 | 0,558 | **+0,081** | 0,26 | 0,24 |

Ninguno cruza significancia (n=25, familia BH de 1, underpowered), pero los TRES apuntan arriba y HHEM (el
más limpio, per Tier 3) da el mayor efecto. HHEM baseline 0,477 → carga verificada.

**Guardas — patrón OPUESTO a exp16 (mejora GENUINA, no gaming):** declinación 56 %→**32 %** (BAJA),
palabras 349→**427** (SUBE), claims 11,5→**14,9** (SUBE), solape verbatim 0,109→0,129 (plano, no copia).
exp16 subía declinación y recortaba contenido; exp17 hace lo contrario → la fidelidad sube porque, con
ambos proveedores presentes, granite deja de declinar y ancla más claims. **La cobertura es la palanca.**

**Síntesis de la fase:** Tier A (arreglo del contexto) NULO · exp16 (prompt) NULO/negativo · **exp17
(selección de contenido = cobertura) POSITIVO.** Converge con Tier 3: el único eje que mueve la fidelidad
es QUÉ evidencia entra, no cómo se ordena ni cómo se instruye.

**Caveat honesto:** piloto n=25, no significativo; señal direccional consistente + guardas, NO conclusión.
Confirmatorio a mayor n (ampliar el set comparativo cross-cloud) requiere OK.

**Report-before-prose:** matriz línea 5 A.3: PILOTO → PILOTO CON SEÑAL POSITIVA. NO cambia cifras firmadas
(exp17 nuevo). NO tocar prosa A.3 sin OK. Reporte: `output/audit/exp17_crosscloud_finding_2026-07-24.md`.

## Entrada 13 — exp17 reanálisis de mayor potencia (mismas 25 q, claim-level): sugestivo, no concluyente (2026-07-24)

Enzo eligió "más potencia sin queries nuevas" (el pool cross-cloud está AGOTADO en 25; las 4 removidas son
inválidas — corpus K8s/CNCF borrado en el rebuild; autorar queries = result-chasing, rechazado). Reanálisis
a resolución de claim conservando el pareo por query, SIN datos nuevos:
- **GLMM binomial** `supported ~ arm + (1|query)` (VB): contraste within-query a nivel claim.
- **Bootstrap de cluster por query** (unidad válida) del diff micro-promediado, seed 42.
Tests una-cola (H1 balanced>baseline, dirección pre-especificada). Condicional a claim genuino.

| Verificador | micro base→bal | GLMM OR | GLMM p(1c) | boot p(1c) |
|---|---|---|---|---|
| NLI small | 0,213→0,247 | 1,15 | 0,152 | 0,30 |
| NLI base | 0,167→0,210 | 1,24 | 0,056 | 0,14 |
| **HHEM** | 0,571→0,605 | **1,25** | **0,021** | 0,23 |

**Veredicto honesto:** el modelo con pareo a nivel claim sube **HHEM a p=0,021 una-cola** (base marginal
0,056), pero el bootstrap conservador NO cruza (HHEM 0,23) → efecto real en dirección, **al borde de la
significancia según el modelo, sugestivo no concluyente.** Caveats registrados: una-cola; 3 verificadores
(HHEM 0,021 no sobrevive Bonferroni ×3 = 0,063); el GLMM puede sobre-estimar potencia (correlación residual
intra-respuesta) → el bootstrap es la guarda. Balanced además tiene MÁS claims genuinos (372 vs 287, menos
declinación) — ganancia extra no capturada por el análisis condicional.

Se reforzó el rigor sin perseguir significancia con datos inventados. `powered_reanalysis.{json,md}`.
Report-before-prose: sigue siendo piloto; NO cambia cifras firmadas. Confirmatorio real exigiría autorar
queries pre-registradas (decisión de Enzo, no tomada) o gold humano.

## Entrada 14 — Fase 3: cierre de la fase + config de encuestas (2026-07-24)

Consolidación (docs, sin GPU, sin generación nueva). Matriz de factibilidad CERRADA (1a probada-nula, 1b
diseño/nube, 2 diseño entregado, 3 hecho-espera-gold, 4 Tier A hecho, 5 exp17 piloto positivo, 6 solo
diseño). Config recomendada para SUS/Likert escrita en SUMMER_RESULTS:
- Pipeline base sin cambios (híbrido+rerank+granite temp0 prompt canónico; exp16 mostró no tocar el prompt).
- ÚNICO cambio recomendado: cobertura balanceada por proveedor SOLO en el ramo comparativo cross-cloud
  (exp17, única palanca positiva de la fase; detección cross_cloud ya existe en QueryProcessor).
- No recomendado: decodificación anclada (nula), modelo mayor (no cabe 6 GB), memoria semántica (fuera).

**Cierre de la fase:** diagnóstico (Tier 0/A/3) + mejoras (exp16 negativo honesto, exp17 positivo) + matriz
= COMPLETOS. Respuesta central: la fidelidad responde a QUÉ evidencia entra (selección de contenido), no a
su presentación ni a la instrucción. Pendiente (no bloquea encuestas, sí el paper): gold humano N≈200 +
confirmatorio pre-registrado de exp17 si se busca significancia. Todo committeado local en `summer/ablacion`
+ `summer/mejoras`; NADA pusheado (GATE). Report-before-prose: nada de A.3/LACCI tocado sin OK.

## Entrada 15 — Fase post-verano · Bloque 0: correcciones de rigor y desbloqueo del gold (2026-07-30)

Bloque **sin GPU de generación y sin datos nuevos**: solo re-análisis offline, corrección de defectos
y construcción de las piezas que faltaban. Auditoría previa antes de tocar nada:
`git diff --name-status nota3-evidencia-2026-06-11 -- experiments/results` = **solo altas (`A`), cero
modificaciones/bajas** → evidencia firmada `exp3..exp14`+`exp8b` intacta.

### D1 — familia BH mal declarada en los artefactos de exp16/exp17 (CORREGIDO)

`scripts/compute_tierA_arm_stats.py` hardcodeaba tres campos del JSON de salida ignorando
`--exp-dir`/`--baseline-arm`, con lo que exp16 y exp17 heredaban la metadata de Tier A:

| Artefacto | Declaraba | Real |
|---|---|---|
| exp15_ablation_tierA | familia 4 · n=60 · `baseline_repro` | correcto |
| exp16_anchored_decoding | familia **4** · n=60 · `baseline_repro` | familia **2** |
| exp17_crosscloud_balanced | familia **4** · n=**60** · **`baseline_repro`** | familia **1** · n=**25** · **`baseline`** |

**Los p-valores nunca estuvieron mal** — la corrección BH sí se aplicó sobre la familia real
(exp17 p_bh = p; exp16 p_bh = p×2) y el Markdown ya imprimía «BH family = 1 contrasts». Mentía solo
la metadata del JSON. Es la misma clase de defecto que obligó a retractar la entrada 8, así que se
trata igual: campos derivados (`args.baseline_arm`, `len(rows_base)`, `len(contrasts)`), los 9
artefactos regenerados y **verificado que el bloque `contrasts` queda byte-idéntico** en los 9 →
ninguna cifra publicada cambia. Regresión permanente en `tests/test_arm_stats.py` (31 tests): el test
falla contra los artefactos antiguos y pasa contra los nuevos, comprobado.

### D2 — el gold v4 medía a HHEM con una mano atada (CORREGIDO, diseño de dos etapas)

`build_gold_v4.py` mostraba al anotador **un solo chunk** (argmax-entailment de NLI-small) y la
columna `question` **vacía**. Pero los instrumentos no ven eso: NLI `vb_agree` lee los 5 chunks
(contradicción exige ≥2 de acuerdo) y HHEM puntúa `max_chunk` sobre los 5 con premisa truncada a
1500 chars. → κ(humano, HHEM) salía sesgada **a la baja por construcción**, justo en la decisión que
el gold existe para arbitrar (nivel NLI 0,30 vs HHEM 0,55).

Cerrar el confound mostrando los 5 chunks a los 150 claims cuesta **3-4× el tiempo del anotador**
(medido: 120 k chars → 566 k @800 / 879 k @1500; los 150 claims abarcan **139 contextos distintos**,
así que agrupar no comprime). Decisión de Enzo: **dos etapas**.

- **Etapa A** — los 150 claims, 1 chunk @800 (+ ahora la **pregunta**, sin la cual un claim con
  pronombre no es juzgable). Selección **verificada idéntica** a la anterior: mismos 150 claims,
  mismos estratos, mismo orden; la única columna que cambia en el CSV es `question`.
- **Etapa B** — submuestreo **proporcional por estrato** de 50 de esos mismos claims, con los **5
  chunks @1500** (paridad exacta con HHEM). Se rellena **después** de la A y sin consultarla.

La etapa B convierte el confound de *caveat* en *corrección*: mide cuántos juicios **cambian** al ver
la evidencia completa. Ciego reforzado — la etapa B **no marca** qué chunk usó el instrumento (marcarlo
dirigiría la atención); el índice argmax vive solo en el `_meta.json`. Orden barajado (seed 42) para
que la posición no filtre estrato. Limitación registrada: arrastre de memoria entre etapas.

### D4 — el gold no tenía consumidor (CONSTRUIDO)

Existía el constructor, no el analizador: nada leía el CSV de vuelta. Nuevo
`scripts/analyze_gold_v4.py`, reutilizando `label_one()` de `compute_exp15_ensemble_sweep.py` para
etiquetar por claim a los 5 candidatos (small, base, hhem, E1_mean, E5_base_and_hhem). Entrega
κ, IC95 bootstrap, curva de confiabilidad + ECE, precisión/recall, barrido de umbral y el sesgo de la
etapa B. Familia BH declarada = 5 tests candidato-vs-humano.

**Hallazgo metodológico durante la construcción:** la primera versión ponderaba replicando los
estratos por orden de prioridad, y su propio guard de auto-validación la tumbó (16/150 discrepancias).
Causa real: `build_gold_v4.pick()` muestrea **secuencialmente sobre estratos SOLAPADOS** (413 de 14 409
claims llevan más de un flag), así que el estrato de un claim depende de **qué pase lo sacó**, no de una
prioridad fija — no hay forma cerrada para la probabilidad de inclusión. Sustituido por **Horvitz-Thompson
con π estimada re-ejecutando el muestreador real** (400 réplicas), agrupando por patrón de flags (los
claims con flags idénticos son intercambiables bajo el muestreador, lo que colapsa el error Monte Carlo).
Las π recuperadas coinciden con lo esperado analíticamente (patrón sin flags: 0,00209 vs 30/14 409 =
0,00208), lo que valida la réplica.

Consecuencia honesta que el script reporta en portada: **n efectivo de Kish = 42,7 sobre 150**. El gold
se diseñó para *discriminar verificadores*, no para estimar una κ poblacional; la κ ponderada es
insesgada pero de varianza alta, así que se reportan las tres lecturas (ponderada, `random_anchor`
sin supuestos, y por estrato). Verificado de punta a punta con `--simulate` (datos sintéticos, no
escribe nada). Segundo defecto propio detectado y corregido: la etapa B se escribe **barajada**, así
que el join debe ir por la columna `stage_a_idx`, nunca por posición.

### D5/D6 — config muerta y artefacto suelto

- `config/evaluation_config.yaml`: **cero consumidores** (grep .py/.yaml/.md/.sh/.ps1) → movido a
  `config/deprecated/` con cabecera que apunta a los valores vivos. `config/config.yaml` **sí se
  consume** (lado corpus: `ingestion_pipeline.py:21`, `deduplicator.py:313`, `text_cleaner.py:199`,
  `build_index.py:39`) → **no se retira**; se marcan sus secciones muertas. Éstas **contradecían al
  sistema real**: `query_expansion.enabled: true` (vivo: **False**, retirada en N4/exp13) y
  `reranking.default_model: ms-marco-mini-6` (vivo: **ms-marco-MiniLM-L-12-v2**). Los 4 YAML siguen
  parseando.
- `nli_probs__large.partial.json.gz`: 8/12 configs (granite **completo** en los 3 escenarios; faltan
  mistral-híbrido y qwen ×3 ≈ 24 k pares). Decisión de Enzo: **terminarla** — desbloquea el tercer voto
  NLI, hoy `NLI_TRIO` filtra por existencia de archivo y los ensembles E1/E2/E3 corren con 2 miembros
  (E2_vote, mayoría, queda mal definido con 2). Corriendo, reanuda desde el `.partial`.

### D-KNOB — corrección de `docs/KNOB_MAP_summer.md`

La afirmación «`RAGPipeline.query()` **NO** replica la ruta del prompt canónico» era **imprecisa**.
Verificado en `rag_pipeline.py:270-291`: la construcción del prompt es la misma (`build_context` →
`get_template` → rama `cross_cloud` con `context_by_provider` → `SYSTEM_PROMPT`), y `rgm.build_prompt`
se documenta a sí mismo como réplica de ella. La diferencia real es el **origen del contexto** (ids
firmados de exp11 vs recuperación en vivo). Importa porque habilita empaquetar `RAGPipeline` como
artefacto desplegable de las encuestas; la paridad de prompt queda pendiente de test antes de empaquetar.

**Estado:** suite `pytest` verde (38 pasan, 1 skip) antes y después. Ninguna cifra publicada cambia.
Report-before-prose: nada de A.3/LACCI tocado. Sin push (GATE).

## Entrada 16 — Bloque 4 (parcial): dos divergencias entre el pipeline MEDIDO y el DESPLEGABLE (2026-07-30)

Al empaquetar la config de encuestas aparecieron dos defectos que **no** estaban en la lista de tareas.
Ninguno cambia una cifra publicada; ambos cambian qué significa "desplegar lo que medimos".

### F1 — `RAGPipeline.query()` nunca enruta el prompt por `query_type`

`rag_pipeline.py:118-121` instancia el `QueryProcessor` **solo si `config.query_expansion`**, y ese flag
es **False** desde N4/exp13 en los tres sistemas de la tesis. Con `self.query_processor = None`, `query()`
cae en `query_type = "default"` → `RAG_PROMPT` para **todas** las preguntas. La ruta medida
(`run_generation_matrix.py:167`) construye un `QueryProcessor` incondicionalmente y **sí** enruta:
de las 194, `cross_cloud` 51 + `procedural` 64 = **115/194 usan una plantilla distinta de la default**
(y las 51 cross_cloud además usan `_build_cross_cloud_context`, agrupado por proveedor).

→ El demo/UI y el runner medido **no comparten prompt en el 59 % de las queries**. Esto matiza —y hace
más preciso— lo corregido en la entrada 15 sobre el KNOB_MAP: la *construcción* del prompt sí es idéntica,
pero el `query_type` que la alimenta no lo es, porque `RAGPipeline` no lo calcula.

**Alcance en evidencia firmada:** `_build_retriever()` hace `qp = self.query_processor or QueryProcessor()`,
así que **la recuperación no se ve afectada** — el fallo aísla solo el prompt en `query()`. Pero **exp8**
("End-to-End System Comparison") **sí tiene respuestas** y salió por esa ruta, y `exp8_stats_corrected.csv`
es **inmutable**. Cambiar el comportamiento por defecto haría que una re-corrida de exp8 discrepara de su
propio artefacto firmado.

**Resolución (aditiva, default apagado):** nueva perilla `PipelineConfig.prompt_routing` (False = legado
exacto). `RAGPipeline` gana un `_routing_qp` separado — el enrutado del prompt es una preocupación de
prompt, no de expansión; que viajara sobre `query_expansion` era el bug. Test:
`test_legacy_configs_keep_both_knobs_off`.

### F2 — `QueryProcessor._detect_providers` pierde GCP y detecta proveedores sin corpus

`PROVIDER_KEYWORDS['gcp'] = {gcp, google cloud, google cloud platform}` — sin `google` a secas y sin
acrónimos de servicio. Consecuencia medida sobre las 25 queries cross-cloud de exp17: **solo 20/25**
resuelven el mismo conjunto de proveedores que la etiqueta `cloud_providers` usada por el piloto.

| qid | etiqueta exp17 | detecta | causa |
|---|---|---|---|
| q173 | aws, azure, gcp | aws, azure, **k8s** | "GKE" invisible; "Kubernetes" sí matchea |
| q184 | aws, azure, gcp | aws, azure | "Google Artifact Registry" (no dice "google cloud") |
| q187 | aws, azure, gcp | aws, azure | ídem |
| q189 | aws, azure, gcp | **k8s** | "EKS vs AKS vs GKE": ningún keyword de proveedor |
| q197 | aws, azure, gcp | aws, azure | "Google Eventarc" |

Agravante: el corpus vivo tiene **solo aws/azure/gcp** (24 481 chunks; K8s/CNCF borrados en el rebuild,
ledger entrada 13), pero el detector sigue devolviendo `k8s`/`cncf` — q189 resolvería a un proveedor con
**cero chunks**. Afecta también a `_get_provider_filter` en queries `single_provider`, no solo al balanceo.

**Resolución (sin tocar `QueryProcessor`):** `resolve_wanted_providers()` en
`src/retrieval/coverage_balancer.py` une detector + alias (`google`) + nombres de servicio **derivados del
propio `chunk_map`** (EKS→aws, AKS→azure, GKE→gcp), y descarta proveedores sin chunks. Derivar del índice
en vez de hardcodear hace imposible esta clase de obsolescencia. **Verificado 25/25** contra las etiquetas
de exp17. NO se parchea `QueryProcessor` porque `run_generation_matrix.py` le pide el `query_type` que
enruta el prompt: cambiarlo alteraría re-corridas de exp11/exp12 firmados. **Decisión pendiente de Enzo:**
si se arregla el detector de raíz para exp18+.

### Empaquetado desplegable

`balance()` movido a `src/retrieval/coverage_balancer.py` (el script de exp17 lo importa → exp17 sigue
reproduciéndose byte a byte). Nueva `SURVEY_DEPLOY` = `PROPOSED_HYBRID` con **exactamente dos** perillas
cambiadas (`prompt_routing`, `balance_cross_cloud_providers`), fuera de `PIPELINE_CONFIGS` para que
`get_config("hybrid")` siga devolviendo el sistema medido. Balanceo cableado tras el rerank, solo si
`query_type == "cross_cloud"`; el rerank pasa a top_k=50 en esa rama (el cross-encoder ya puntúa todos los
candidatos y trunca después → sin coste extra) y la rama no-balanceada queda byte-idéntica.

**Tests:** `tests/test_coverage_balancer.py` (18) — contrato de `balance()`, resolución de proveedores,
y **aceptación**: replicar la regla sobre el pool guardado de exp17 devuelve los `balanced_ids` exactos en
las **25/25** queries; `SURVEY_DEPLOY` difiere del sistema medido en exactamente 3 campos (nombre + 2
perillas). `pytest.ini` con marcadores `slow`/`gpu`/`needs_artifacts`: suite completa 56 pasan + 1 skip
(25 s), suite rápida 54 en **1,5 s**.

## Entrada 17 — Bloque 3 (diseño de nube) + infra de exp18 + el techo de contexto es a k>5, no a k=5 (2026-07-30)

### Hallazgo: la ventana de 4096 NO ata a la configuración desplegada

La matriz de factibilidad daba por sentado que "contexto completo sin truncar" era una palanca de la
línea 1b. Medido, no lo es **a k=5**. Calibración `tokens ≈ 0,2228·chars + 261` ajustada sobre los
`tokens.input` reales de exp12 granite/híbrido (n=192 no truncados, R²=0,922), proyectada sobre el
subset de 60 q escalando el contexto de cada query por su propio tamaño medio de chunk:

| k | p50 tok | p90 tok | max | supera 4096 |
|---|---|---|---|---|
| **5 (actual)** | 1973 | 2999 | 3154 | **0/60 (0 %)** |
| 10 | 3685 | 5737 | 6047 | 24/60 (40 %) |
| 15 | 5397 | 8474 | 8940 | 50/60 (83 %) |
| 20 | 7109 | 11212 | 11833 | 55/60 (92 %) |

Concuerda con el dato directo de exp12: **2/194** prompts tocaron 4096. → La nube compra **capacidad
de modelo**; compra **contexto solo si más fragmentos ayudan**, y eso empieza a poder testearse recién
por encima de k≈7. (Estimación, no medición: supone que los chunks 6-20 miden como los 1-5. Los
`tokens.input` reales se registran al correr exp18.)

**Consecuencia de diseño para exp18:** el brazo `final_top_k_10` **no es un test limpio de cantidad** —
a k=10 el 40 % del subset ya viene truncado, así que confunde "más evidencia" con "evidencia cortada"
**por construcción**. Se reclasifica como **sonda del límite de truncamiento** y se analiza **partido**
por truncado/no-truncado, con los tokens observados registrados en `results.json::observed_truncation`.
Un test limpio de cantidad exige una ventana mayor, es decir, nube.

### Infra de exp18 (escrita, sin correr — la GPU está ocupada con deberta-large)

- `scripts/build_exp18_evidence_arms.py` — 4 listas de ids desde el MISMO pool híbrido k=50:
  `baseline_repro` (rerank[:5]), `oracle_evidence` (top-5 por **bge-reranker-large**),
  `evidence_swapped` (top-5 de OTRA query del mismo `query_type`), `final_top_k_10`.
  - **Anti-circularidad (Flag 17):** el oráculo de selección es bge-reranker-large, **independiente**
    de los verificadores que puntúan (NLI small/base, HHEM). Seleccionar evidencia con el mismo
    instrumento que mide anclaje fabricaría el resultado.
  - El emparejamiento del swap es un **desarreglo** dentro de `query_type` (seed 42): ninguna query
    conserva su propia evidencia. Los tipos con un solo miembro se registran aparte.
  - Auto-validación: `baseline_repro` debe solapar 5,0/5 con los ids firmados de exp11 híbrido.
- `scripts/run_exp18_ceiling.py` — generación por brazo con el prompt canónico (`rgm.build_prompt`),
  granite temp0 seed42, `--no-cache`, warmup + sonda de determinismo 3× por brazo, checkpoint cada 10.
  La pregunta y su `query_type` son siempre los **propios** de la query, también en `evidence_swapped`:
  el brazo pregunta si el generador sigue la evidencia que le dieron para la pregunta que le hicieron.
  Escribe el esquema estándar → los scorers existentes funcionan vía `--exp-dir`.
  **Familia BH declarada = 3 contrastes brazo-vs-`baseline_repro` por verificador.**

### Bloque 3 — `docs/CLOUD_EXPERIMENT_DESIGN.md` (propuesta, cero gasto)

Cuatro brazos, y el segundo es el que casi todo el mundo se salta:

| | Brazo | Motor | Modelo | Gen. |
|---|---|---|---|---|
| A | control de replicación | **Ollama** (idéntico a local) | granite4.1:8b | 60 |
| B | **puente de motor** | vLLM bf16 | granite4.1:8b | 60 |
| C | **capacidad** | vLLM bf16 | modelo mayor ~32B | 60 |
| D | confirmatorio exp17 | vLLM bf16 | ambos | 100 |
| E | contexto *(condicional a exp18)* | vLLM bf16 | mayor, top-20 | 60 |

Sin **B**, "modelo mayor en vLLM vs granite en Ollama" confunde capacidad con motor de inferencia.
Con B: capacidad = C−B a motor constante; efecto de motor = A−B; validez del puerto = A vs local.
**bf16 sin cuantizar** en C (cuantizar confunde capacidad con precisión numérica → es la razón de pedir
80 GB). Contextos congelados (ids de exp11): no se re-recupera nada en la nube. **La puntuación no se
paga en la nube** — vuelven solo los JSON y se puntúa en local con los scripts existentes, lo que además
mantiene el instrumento idéntico al del resto de la fase.

**Costo: A100 80 GB, 5-9 h de reloj ≈ USD 10-18; techo sugerido USD 50.** Aviso registrado: vLLM no es
bit-determinista al variar el tamaño de batch → sonda con batch=1 o registrar y tratar como pareado
dentro de sesión. **A es compuerta:** si el control no reproduce el nivel local dentro de la deriva H5
conocida (+0,033 n.s., r=0,86), nada de B/C/D es interpretable.

**Compuerta general: nada se lanza antes de exp18**, que cuesta ~3-4 h de GPU propia y cero dinero, y
decide qué vale la pena pagar. Trade-off de contribución explícito en §7 del documento: si C rompe el
techo es un **hallazgo**, no automáticamente la config recomendada — reencuadrar la contribución
"corre en 6 GB" es decisión de Enzo.

### Bloque 4 — pulido

- `pytest.ini` con marcadores `slow`/`gpu`/`needs_artifacts`. Suite **71 tests**: rápida **68 en 1,6 s**,
  completa ~25 s (antes: 39 tests, 66 s, sin forma de separar).
- `tests/test_decide_nli_status.py` (14) — la regla en la que descansa **toda** cifra de fidelidad no
  tenía cobertura directa: `test_nli_calibration.py` prueba el MODELO (¿son probabilidades?), nunca la
  DECISIÓN. Fija la asimetría de N8/Tier 3: `supported` lleva guarda (`max_ent > max_contr`) en todas
  las variantes; `contradicted` bajo v0 no lleva ninguna (de ahí el 22 % de falso-contradicted del
  control negativo); `vb_agree` exige ≥2 chunks. Incluye la propiedad `vb_agree ⊆ v0` y los bordes
  estrictos (`>` no `>=`) en el umbral.
- `scripts/verify_summer_offline.py` — re-deriva **cada** celda de fidelidad de Tier A/exp16/exp17
  desde las probs persistidas y recomputa los contrastes pareados. **Todo pasa**; niveles HHEM del
  ancla 0,4499 / 0,4983 / 0,4772, que cuadran con el ledger (0,450 / 0,498 / 0,477). Incluye la guarda
  de carga del instrumento (0,40-0,55; un HHEM mal cargado puntuaba ~0,04 y "corría") y la de familia
  BH declarada.
- `REPRODUCE.md` — reproducción desde limpio en 5 niveles, de segundos a horas, con los avisos
  operativos que muerden: H5 (nunca comparar contra una sesión anterior) y la clave del caché LLM
  (`config_name‖prompt`, no exp-id — lo que sirvió respuestas stale a exp16).

## Entrada 18 — Bloque B: una sola definicion de "declinacion" (C2) y correccion del caveat de potencia (C3) (2026-07-30)

Dos defectos hallados al criticar el plan de exp18. Ninguno cambia un veredicto; ambos cambian
cifras que el ledger citaba, y uno de ellos **refuerza notablemente el positivo de exp17**.

### C2 — habia dos definiciones vivas de "declinacion", discrepantes hasta 24 pp

`compute_exp16_guards.py:26` probaba **un substring exacto y sensible a mayusculas**;
`compute_faithfulness_metrics.py` usa `classify_response` sobre **28 regex case-insensitive**
(los 14 canonicos de `response_formatter.DECLINE_PATTERNS` + los 14 de
`EXTENDED_REFUSAL_PATTERNS`). Discrepaban en **todos** los brazos de Tier A, exp16 y exp17.

Unificado: los guards ahora **importan `classify_response`** del modulo que usa la metrica — la
misma funcion, no una copia — y reportan sus **tres** clases, porque esa distincion es la que
carga la interpretacion: `pure_decline` (marcador en los primeros 300 chars: el modelo abre
rechazando), `hedged_partial` (marcador mas tarde: hedgea y **aun asi responde**), `answered`.

**Cifras corregidas (los veredictos no cambian; se refuerzan):**

| | pure_decline | answered | (antes, regla estrecha) |
|---|---|---|---|
| exp16 baseline | 46,7 % | 40,0 % | 51,7 % |
| exp16 anchored_cite | **65,0 %** (+18,3 pp) | 28,3 % | +6,6 pp |
| exp16 strict_abstain | **68,3 %** (+21,6 pp) | 23,3 % | +8,3 pp |
| exp17 baseline | 56,0 % | 20,0 % | 56 % |
| exp17 balanced | **36,0 %** (−20 pp) | **48,0 %** (se dobla) | 32 % |

exp16 hace callar al modelo **el doble** de lo reportado. Y exp17 no solo declina menos: **mas que
dobla las queries plenamente respondidas** (20 % → 48 %) — el positivo es mas fuerte de lo que
decia el piloto. Regenerados `guards.{json,md}` de exp16 y exp17. Regresion permanente en
`tests/test_decline_rule.py` (12): los guards deben usar el clasificador canonico, no re-implementarlo;
el contrato de las tres clases incluido el borde puro-vs-hedged; y los artefactos committeados deben
coincidir con el clasificador.

### C3 — el caveat de potencia de la entrada 11 es FALSO

La entrada 11 dice «declinacion baseline 51,7 % → n efectivo ≈29, underpowered». **El `n_paired`
real es 50-59 de 60** en los tres verificadores (Tier A 55-59, exp16 50-55, exp17 25/25).

Causa: **37 de 60 respuestas que contienen una frase de rechazo NO son declinaciones** — declinan
y *ademas* responden. Caso q002: dice "I cannot find sufficient information…" y luego responde 128
palabras; puntua **fidelidad 1,0 sobre 1 claim genuino**. El denominador decline-aware solo descarta
las que no tienen **ningun** claim genuino (3/60 `vacuous`, 0 `None`), no las que hedgean.
→ El caveat de exp16 sobre-vendia la falta de potencia. La conclusion de exp16 (nulo/negativo) no
cambia, pero su razon declarada era incorrecta.

### C4 — la varianza no viene del n, viene del denominador

11,8 claims genuinos por query de media, pero **25 % de las queries tienen ≤2** → su fidelidad solo
puede valer 0 / 0,5 / 1. SD de la fidelidad por query = 0,311. Eso, y no el tamano muestral, es lo
que infla la varianza del contraste pareado. Motiva usar el analisis claim-level como secundario en
exp18 (708 claims por brazo a n=60 frente a 57 queries).

### C5 — no existe ancla humana de relevancia ni respuesta de referencia

`relevant_chunk_ids` esta **vacio en las 194** y `answer` **vacio en las 194**. Descarta dos disenos
que se consideraron para exp18 (un brazo de evidencia gold y un control positivo con respuesta de
referencia) y confirma que el "oraculo independiente" siempre sera otro modelo. No es defecto nuevo
—el A.3 ya lo declara asi— pero conviene tenerlo explicito antes de interpretar cualquier techo.

**Verificacion:** suite completa **82 pasan + 1 skip**; `verify_summer_offline.py` exit 0;
`git diff` vs `nota3-evidencia-2026-06-11` **solo altas**. Ninguna cifra del A.3/LACCI tocada.
**Report-before-prose:** las tasas de declinacion corregidas y el 20 %→48 % de exp17 son material
para el paper; NO se toca prosa sin OK frase por frase.

## Entrada 19 — exp18 PRE-REGISTRO (escrito y committeado ANTES de generar un solo dato) (2026-07-30)

Se registra aqui, antes de correr, para que el analisis no pueda elegirse despues de ver los
numeros. La matriz de decision de exp18 **acepta un nulo** para argumentar que el techo es de
capacidad y que el gasto en nube esta justificado; un nulo asi tiene que ser una **afirmacion
positiva de equivalencia**, no una ausencia de significancia.

### Pregunta
El techo de fidelidad (HHEM ~0,45-0,55 / NLI ~0,30) ¿es de **seleccion de evidencia**, de
**capacidad de generacion**, o del **instrumento**? La ablacion solo podia quitar componentes;
nunca midio el techo con evidencia (casi) ideal.

### Brazos y escala (los 4 salen del MISMO pool hibrido k=50; solo cambia la SELECCION)

| Brazo | Construccion | n | Por que ese n |
|---|---|---|---|
| `baseline_repro` | rerank(pool)[:5] ms-marco-L12 | 194 | ancla |
| `oracle_evidence` | top-5 por **bge-reranker-large** | **194** | unico brazo cuyo NULO decide; lleva el TOST |
| `final_top_k_10` | rerank(pool)[:10] | **194** | se analiza partido por truncamiento; a n=60 el split era 24/36 |
| `evidence_swapped` | top-5 de OTRA query del mismo tipo de routing | 60 | espera efecto grande |

**Justificacion de la escala (medida, no supuesta):** con la SD de la diferencia pareada
observada en Tier A/HHEM (**0,318**), el MDE(80 %, α .05 bilateral) es **0,118 a n=57** y **0,066
a n=185**. El mayor efecto positivo de toda la fase (exp17 HHEM **+0,081**) seria **indetectable
a n=60**; harian falta n=121.

### Analisis primario
Fidelidad **HHEM** por query, contraste pareado vs `baseline_repro` (Wilcoxon + d_z + bootstrap
seed 42). **Familia BH declarada = 3 contrastes brazo-vs-baseline, por verificador.** Bilateral.

### Equivalencia pre-registrada (brazo `oracle_evidence`)
**TOST con banda ±0,081**, fijada en `compute_exp18_diagnosis.py::TOST_BAND` antes de correr.
La banda es el efecto HHEM de exp17: *"el oraculo no compra ni lo que compro balancear la
cobertura"*. Es una cantidad **preexistente y ciega a este contraste**, no un umbral ajustado
hasta que algo pase. α=0,05; se declara equivalencia si ambos tests unilaterales rechazan
(equivalente a que el IC90 quepa entero en la banda).

**Potencia de equivalencia verificada por simulacion** (efecto real 0, SD 0,318, 400 replicas):
n=57 → **20 %**, n=121 → 74 %, n=185 → **94 %**. Coincide con el calculo analitico (22 % / 93 %) y
es la razon de escalar el brazo a 194. El TOST se valido ademas contra 4 casos de respuesta
conocida (nulo a n=185 → equivalente; nulo a n=57 → no concluyente; efecto +0,15 → no
equivalente; efecto justo en la banda → no equivalente).

### Secundario
GLMM binomial claim-level `supported ~ arm + (1|query)` + bootstrap de cluster por query,
reusando `compute_exp17_powered.py`. Motivo (C4): 25 % de las queries tienen ≤2 claims genuinos,
asi que su fidelidad solo puede valer 0/0,5/1 — la coarseness del denominador, no el n, es lo que
infla la varianza. **Secundario, no primario:** exp17 ya mostro que el GLMM puede sobre-estimar
potencia, y el bootstrap por cluster es la guarda.

### Triangulacion
Los 3 verificadores (NLI small, NLI base, HHEM). Guardas anti-gaming obligatorias con la regla de
declinacion **unificada** (entrada 18): pure_decline / hedged_partial / answered, palabras, claims,
solape verbatim.

### Lectura de `evidence_swapped` (no es fidelidad)
Divergencia de la respuesta vs baseline (jaccard 5-grama, jaccard de tokens, tasa de respuestas
identicas, reaparicion de claims). Solape **ALTO** ⇒ el generador no lee el contexto, y entonces
el nulo de recuperacion queda explicado de raiz. La fidelidad sola no puede distinguir eso de un
modelo que sigue correctamente evidencia equivocada: ambos puntuan bajo.

### Caveat declarado de antemano
`final_top_k_10` **no es un test limpio de cantidad**: a k=10 una fraccion grande supera los 4096
tokens, asi que confunde "mas evidencia" con "evidencia cortada" **por construccion**. Se analiza
partido por truncamiento **observado** (`tokens.input`, medido en la corrida, no estimado); solo el
estrato no-truncado es lectura limpia. Un test limpio de cantidad exige una ventana mayor (nube).

### Anti-circularidad
El oraculo de seleccion es **bge-reranker-large**, independiente de los verificadores que puntuan
(Flag 17). Prohibido seleccionar evidencia con el mismo instrumento que mide anclaje.

### Sanity-checks obligatorios antes de interpretar
`baseline_repro` debe solapar **5,0/5** con los ids firmados de exp11 hibrido · nivel HHEM del
ancla en **0,40-0,55** (guarda de carga: un HHEM mal cargado puntuaba ~0,04 y "corria") · sonda de
determinismo 3× por brazo · `--no-cache` (el cache se indexa por `config_name‖prompt`, no por
exp-id — sirvio respuestas stale a exp16).

## Entrada 20 — exp18: matriz de decision CORREGIDA, committeada ANTES de puntuar (2026-08-02)

La generacion esta hecha (entrada previa) pero **nadie ha visto una sola cifra de fidelidad**. Se fija
aqui como se lee, por el mismo motivo que se pre-registro el TOST: si la matriz decide un gasto, no
puede elegirse despues de ver los numeros.

### Por que se reescribe la matriz propuesta

La version propuesta decia: *"oraculo equivalente ⇒ nube justificada"*. **Es un non sequitur**, y es
justo la fila que sostiene el gasto. La equivalencia bajo un oraculo de RELEVANCIA descarta unicamente
el margen del **ranking topico**. Sobreviven cuatro explicaciones:

  (a) capacidad de generacion,
  (b) suelo del instrumento (22 % falso-contradicted sobre texto aleatorio),
  (c) recall del pool k=50 — la evidencia puede sencillamente no estar,
  (d) **relevancia ≠ anclabilidad**: `bge-reranker-large` ordena por relevancia topica a la CONSULTA,
      mientras la metrica pregunta si los claims que el modelo DECIDIO AFIRMAR estan respaldados. Un
      chunk puede ser muy relevante y no contener el hecho concreto afirmado.

→ El brazo de oraculo mide el techo de la **seleccion optima-por-relevancia**, que es una **cota
inferior** del techo de seleccion. Su nulo, por si solo, no prueba "no hay margen".

### Matriz de decision (lectura CONJUNTA, no del oraculo solo)

| `evidence_swapped` | `oracle_evidence` | Lectura | Nube |
|---|---|---|---|
| diverge mucho (usa el contexto) | **sube** sig. | margen de seleccion alcanzable en LOCAL | no urgente |
| diverge mucho | **equivalente** (TOST) **y cota ≈ baseline** | seleccion agotada de verdad; queda capacidad | **justificada** |
| diverge mucho | **equivalente** pero **cota >> baseline** | hay margen, el ranking topico no lo encuentra | **no** — falta un selector guiado por anclaje, y es local |
| apenas diverge (ignora el contexto) | cualquiera | el generador NO usa la evidencia → reencuadra el nulo de recuperacion entero | **justificada** (capacidad/atencion) |
| — | ni sig. ni equivalente | **INCONCLUSO** | **no justifica gasto**; se reporta asi, sin redondear |

**Regla dura: un nulo simple NO justifica gasto. Solo la equivalencia TOST junto con la cota.**

**Banda ±0,081: se mantiene.** Es un tamano de efecto, invariante al n; el n solo cambia la POTENCIA
(94 % de equivalencia a n=185, simulado). Supuesto declarado: se midio en 25 queries cross-cloud con
HHEM y se aplica a 194 mixtas, en calidad de "el menor efecto que nos importaria".

### Cota de seleccion condicionada a la respuesta (desambigua (c) y (d))

Para cada query: tomar los claims que el baseline **ya escribio** y elegir del pool k=50 los 5 chunks
que MAXIMIZAN su soporte HHEM. Es la fidelidad maxima alcanzable **por seleccion** para esa respuesta.

**Circular por construccion, y declarado como tal: vale SOLO como cota superior.** Nunca como
estimacion de efecto, nunca como brazo, nunca dentro de la familia BH ni del TOST. Su valor es logico,
no inferencial: **ningun metodo de seleccion no-circular puede superarla**, asi que si la cota no sube
sobre el baseline, la seleccion esta agotada de verdad — y si la cota sube pero el oraculo no, entonces
el techo NO es capacidad sino que falta un selector guiado por anclaje, que es un metodo LOCAL y no
requiere gastar un centavo.

### Lecturas de apoyo, tambien fijadas de antemano

- **`evidence_swapped` es un control de validez, no un brazo mas.** Si la respuesta apenas cambia al
  darle la evidencia de OTRA consulta, la generacion no esta usando el contexto y ninguna mejora de
  recuperacion podria haber movido nunca la fidelidad — eso reencuadraria el hallazgo central. Se lee
  por DIVERGENCIA de respuesta (jaccard 5-grama y de tokens, tasa de identicas, reaparicion de claims),
  no por fidelidad: un brazo con evidencia ajena puntua bajo tanto si el modelo la ignoro como si la
  siguio correctamente, y la fidelidad no distingue esos dos casos.
- **Denominadores desiguales (C9).** Claims genuinos por respuesta medidos sobre las 642 ANTES de
  puntuar: baseline **10,6** · oracle **10,7** · **swapped 4,8** · **top10 15,0**. La fidelidad es
  `soportados/genuinos`, asi que parte de cualquier Δ entre brazos es un cambio de denominador. Cada Δ
  se reporta junto al conteo de claims, y el **GLMM claim-level pasa a lectura de primera linea**.
  (De paso: que el swap afirme menos de la MITAD de claims ya sugiere que el generador si reacciona a
  la evidencia — pero es senal previa, no conclusion.)
- **`final_top_k_10` se lee PARTIDO** por truncacion observada (74 truncadas / 120 no). Solo el estrato
  no-truncado es lectura limpia de "mas evidencia ayuda".
- **Familia BH declarada = 3 contrastes brazo-vs-baseline_repro, por verificador.** Triangulacion en
  los 3 (NLI small, NLI base, HHEM).

### Guardas de instrumento antes de interpretar nada

Nivel HHEM del ancla en **0,40-0,55** (un HHEM mal cargado puntuaba ~0,04 y "corria"). La cota debe ser
**≥ la fidelidad del baseline en TODA query por construccion**; si alguna la incumple, hay un bug y se
para. Si el nivel del ancla cae fuera de rango, se para y se diagnostica.

### Otros hallazgos del barrido de supuestos fragiles

- `vacuous → faithfulness = 1.0` esta duplicado en **9 sitios**. Interaccion peligrosa: si el brazo
  swapped produjera respuestas cuyos claims son todos artefactos, puntuarian 1,0 y el brazo disenado
  para mostrar anclaje bajo saldria ALTO. **Medido: 2-3 % en todos los brazos (top10 0 %) → no es
  amenaza aqui.** Descartado con datos. Deuda tecnica.
- Umbrales 0,7 definidos en **6 sitios** (constantes de clase de `hallucination_detector` + 5 scripts
  con su propio ENT_T/CONTR_T). Todos valen 0,7 hoy → sin bug vivo, pero cambiar la constante de clase
  no propagaria. Deuda tecnica, severidad menor que la regla de declinacion (que si divergia).
- `_extract_claims` y `classify_artifact`: **una sola implementacion**, importada por todos. Limpio.

### Correccion de una afirmacion propia

Se dijo antes que «oraculo ∩ baseline = 2,046/5 ⇒ hay margen de seleccion real». Eso establece que los
brazos **DIFIEREN** (el brazo no es degenerado y su nulo seria informativo), **no** que el oraculo sea
mejor. Overclaim; lo que puede responderlo es la cota.

## Entrada 21 — exp18: fallo silencioso de pass_N, barrido de su familia, y la cota de seleccion (2026-08-03)

### El fallo

Los 4 pases de puntuacion de exp18 salieron con **exit 0**, pero NLI puntuo **1 de 4 brazos**.
HHEM tenia los 4 (194/194/60/194); `faithfulness_rows__{small,base}__vb_agree.json`,
`nli_probs__*.json.gz` y `claims_extraction.json` tenian solo `baseline_repro`.

**Causa:** `main()` resolvia `args.arms` por defecto desde el **registro de arms**, que por defecto
es el de **Tier A**. `pass_n` filtra con `if args.arms and arm not in args.arms: continue`.

| | |
|---|---|
| registro Tier A | `baseline_repro, reranker_off, final_top_k_3, context_reversed, context_lost_middle` |
| brazos de exp18 | `baseline_repro, oracle_evidence, evidence_swapped, final_top_k_10` |
| interseccion | **solo `baseline_repro`** |

El chequeo `unknown` validaba lo que el usuario pasa en `--arms` **contra el registro**, nunca el
registro **contra `results.json`**. El defecto de fondo: `pass_n` toma su lista de trabajo de
`results.json` pero la filtra por un registro de **otro experimento**. `rescore_grounding_tierA.py`
itera `results.json` directo y por eso HHEM no fallo — **esa asimetria entre los dos scorers era el
olor**.

**Alcance verificado — ninguna evidencia previa afectada:** exp15 5/5, exp16 3/3, exp17 2/2 completos
en los 3 verificadores. exp16/exp17 tienen registro propio y se corrieron con `--arms-file`; exp18 no
tiene registro (usa runner propio) y cayo al de Tier A.

**Fix:** `args.arms` se resuelve **por pase** — G desde el registro (autoritativo para *construir*), N
desde `results.json` (autoritativo para *puntuar*) — mas un **gate de completitud** que aborta con
exit != 0 nombrando los brazos que faltan, **antes** de escribir artefactos.

### Barrido de la misma familia: no era el ultimo, habia tres mas (dos mios)

**S1 — el "trio" NLI han sido 2 miembros, y nada lo registraba.** `NLI_TRIO` en
`compute_exp15_ensemble_sweep.py` se construye por **existencia de fichero**; como deberta-large quedo
11/12, `nli_probs__large.json.gz` nunca existio. `E1_mean`/`E2_vote`/`E3_conservative` se documentan
como ensembles del **trio** y se computaron con **dos**. `E2_vote` exige `n>=2` acuerdos, que con tres
es **mayoria** y con dos es **unanimidad**: otro estimador bajo el mismo nombre. Ahora se registran
`nli_members_expected/used/degraded` y los candidatos afectados se **renombran**
(`E2_vote[2m=unanimity]`). No se recomputa (large se queda 11/12 por decision). No afecta la seleccion
del front-runner (E5 = base+hhem, ajeno al trio), pero las cifras E1/E2/E3 committeadas son de 2
miembros.

**S2 — `verify_summer_offline.py` tenia `EXPERIMENTS` fija de 3** → exp18 **no se verificaba**. Ahora
auto-descubre por **forma del artefacto** y deriva el ancla de `results.json`. Distingue ademas `PEND`
(generado, sin analizar) de `FAIL` (analisis parcial), para que el gate siga siendo util justo en la
ventana en que mas hace falta.

**S3 — `analyze_gold_v4.py` recortaba `CANDIDATES` con `HAS_HHEM`**, sacando `hhem`/`E5` del analisis
**y de la familia BH** en silencio. Ahora registra `candidates_intended/available/missing` y avisa.

**S4 (latente, no disparado) — `compute_faithfulness_metrics.py` (default `exp12_matrix`) y
`compute_retrieval_metrics.py` (default `exp8`) escriben in-place y sus defaults apuntan a evidencia
FIRMADA.** Nuevo `src/utils/signed_evidence.py::guard_write` rehusa **sobrescribir un fichero
existente** dentro de `exp3..exp14`/`exp8b`; crear ficheros `_vN` **nuevos** sigue permitido, que es la
via sancionada. La lista firmada vive en **un solo sitio**, con test que lo fija — duplicarla seria el
mismo patron que causo todos estos defectos.

**Descartados tras revisar:** `rescore_nli_{v2,v3,exp15}`, `rescore_grounding_exp15`,
`compute_exp15_hhem_analysis`, `build_claim_audit_sample` llevan `MODELS`x`SCENARIOS` de exp12
hardcodeados, pero su `OUT_DIR` tambien es fijo (son exp12-especificos por diseno) y un config ausente
da `KeyError` **ruidoso**, no un salto silencioso.

**Patron comun de los cinco:** una lista de trabajo derivada por conveniencia (registro, existencia de
fichero, constante) en vez de derivada de la fuente de verdad, **y sin registrar que se uso**. Por eso
los fixes son todos "deriva del artefacto real y **registra el conjunto usado**".

### Cota de seleccion (n=188 de 194; 6 queries sin claims genuinos)

| | media HHEM |
|---|---|
| baseline (su propio top-5) | **0,4552** |
| **alcanzable con k=5** (greedy, suelo = baseline) | **0,5834** |
| **cota superior** (cualquier chunk del pool k=50) | **0,5879** |

**Validacion cruzada:** el baseline calculado por el script de la cota coincide con el scorer canonico
(`faithfulness_rows__hhem.json`) en **0,0000, cero discrepancias en las 188 queries**. Dos rutas de
codigo independientes dan lo mismo. Bracket sin violaciones. Nivel del ancla 0,4552, dentro de la
guarda de carga 0,40-0,55.

**Margen alcanzable sobre el baseline: +0,1282**, frente a la banda de equivalencia pre-registrada de
**±0,081**. El margen es ~1,6x la banda, y el `alcanzable` (0,5834) captura casi toda la `cota`
(0,5879): no es un optimo teorico inalcanzable, es una seleccion de 5 chunks que existe.

**Claims que ningun chunk del pool soporta: 759**, mejor score p50 0,226 / p90 0,439 (tau 0,5);
**123 (16 %) a menos de 0,1 del umbral**. O sea: la mayor parte de lo no soportable **no esta en la
evidencia recuperada** (no es artefacto de umbral), pero un 16 % si es sensible a donde se puso tau y
debe reportarse como sensibilidad.

**Lectura bajo la matriz pre-registrada (entrada 20), con el caveat de que el brazo de oraculo aun no
esta puntuado en NLI:** la fila 2 —"cota ≈ baseline ⇒ seleccion agotada ⇒ **nube justificada**"—
**queda descartada**: la cota NO es ≈ baseline. Quedan vivas la fila 1 (el oraculo sube ⇒ margen local)
y la fila 3 (el oraculo plano pero cota alta ⇒ **falta un selector guiado por anclaje, que es LOCAL**).
En ambas, **la nube deja de estar justificada por la via de "la seleccion esta agotada"**.

Recordatorio de estatus: la cota es **circular por construccion** (se elige la seleccion usando los
claims que la propia respuesta escribio). Es una **cota**, no una estimacion de efecto; no entra en la
familia BH ni en el TOST. Su valor es logico: ningun selector no-circular puede superar `upper_bound`.

**Report-before-prose:** esto toca la decision de nube y potencialmente el encuadre del hallazgo
central. Nada de A.3/LACCI tocado.

### Estado

Suite **82 → 118 tests**; 116 pasan, **1 falla a proposito** (`test_scored_arms_complete` sobre
exp18/NLI = criterio de aceptacion del fix), 1 skip. `verify_summer_offline.py` exit 0 y **cubriendo
exp18** (sus 625 celdas HHEM re-agregan exacto desde los probs crudos). `git diff` vs
`nota3-evidencia-2026-06-11`: **solo altas**. Snapshot de determinismo tomado para small y base:
al relanzar, `baseline_repro` debe salir identico; si difiere, **parar**.

---

## Entrada 22 — exp18 CERRADO: el generador si usa el contexto · retiro de deberta-large · defecto #7 (2026-08-04)

Cierra el analisis pre-registrado de exp18 (committeado en `f6816b4` sin entrada de ledger),
formaliza el retiro de `deberta-large`, y registra un septimo defecto silencioso de la misma
familia. **Report-before-prose: nada de A.3/LACCI tocado.**

### Verificaciones previas al analisis, ambas verdes

- **Determinismo:** `baseline_repro` re-puntuado sale **bit-identico** al anterior en small y base,
  en filas, probs crudos y `claims_extraction.json`. Dos corridas independientes separadas por dias.
- **Criterio de aceptacion:** `tests/test_scored_arms_complete.py` pasa (el fix de `pass_N` de la
  entrada 21). Los cuatro brazos puntuados en los tres verificadores.

### Resultado por brazo (HHEM primario; ancla 0,4638, dentro de la guarda 0,40-0,55)

| Brazo | delta fidelidad | d_z | p_BH | Veredicto |
|---|---|---|---|---|
| `evidence_swapped` | **-0,3185** | -0,857 (grande) | **0,000** | **significativo en los 3 verificadores** |
| `oracle_evidence` | +0,0217 | 0,060 | 0,288 | n.s.; **TOST EQUIVALENTE** |
| `final_top_k_10` | +0,0161 | 0,050 | 0,477 | n.s. |

**`evidence_swapped` — el generador SI usa la evidencia.** Con el contexto de otra consulta:
declinacion por prefijo 45,9 % -> 88,3 %, `answered` 38,7 % -> **1,7 % (1 de 60)**, claims genuinos
10,6 -> 4,8, jaccard-5grama 0,036 y **reaparicion de claims del baseline 0,0009**. No solo puntua bajo:
**diverge casi por completo y se calla en vez de fabricar**. Esto **descarta la fila 4** de la matriz
de la entrada 20 ("el modelo ignora el contexto"), que era la ultima via por la que exp18 podia
justificar gasto en nube.

**`oracle_evidence` — TOST EQUIVALENTE en los tres** (HHEM p=0,012 · small p=0,0009 · base p=0,023)
dentro de la banda pre-registrada de +-0,081. No es "no significativo": es afirmacion **positiva** de
equivalencia. Seleccionar optimo **por relevancia topica** con un oraculo independiente
(`bge-reranker-large`) **no compra fidelidad**. El claim-level da un positivo pequeno bajo GLMM que el
bootstrap de cluster **no confirma** (HHEM p=0,114) — mismo patron que exp17, el GLMM sobre-estima
potencia; manda el endpoint primario.

**`final_top_k_10` — plano en fidelidad, incluso sin truncar.** Split por truncacion observada
(74 truncadas / 120 no): estrato no truncado +0,0259, truncado +0,0005. Pero las guardas muestran otra
cosa: `answered` 38,7 % -> **67,0 %**, claims 10,6 -> **15,0**, y la tasa de respuestas que **no afirman
nada** 3,1 % -> **0,5 %**. **Mas evidencia compra COBERTURA, no tasa de anclaje.** Util para despliegue;
no es afirmacion de tesis. Costo medido: latencia p50 **49,0 s -> 79,0 s (+61 %)**, 90,4 s en el estrato
truncado, y **74/194 prompts (38 %) en el limite de 4 096**.

### Lectura conjunta: fila 3 de la matriz, y NO-GO de nube por la via de la seleccion

swapped diverge (usa el contexto) + oraculo equivalente + **cota muy por encima del baseline**
(0,5834 alcanzable vs 0,4552, +0,128 ~ 1,6x la banda) = **fila 3**: hay margen, el **ranking topico no
lo encuentra**, y lo que falta es un **selector guiado por anclaje**, que es un metodo **local y sin
gasto**.

**Nube: NO-GO por la via de la seleccion.** Caveat honesto y explicito: **exp18 no testea capacidad de
generacion directamente**. Un modelo mayor podria anclar mejor. Lo que exp18 establece es que el
siguiente paso de mayor valor esperado es local, no que la nube valga cero. Esa pregunta queda
**abierta**, y requiere OK y presupuesto de Enzo. Cero gasto ejecutado.

### Retractacion de una afirmacion propia (entrada 21)

La entrada 21 cerraba con: *"Su valor es logico: ningun selector no-circular puede superar
`upper_bound`"*. **Es falso, y se retira.** La cota mantiene la RESPUESTA FIJA y solo reordena chunks
bajo ella. Un selector real cambia la respuesta, y con ella numerador y denominador: el propio brazo de
oraculo —un cambio suave de seleccion— ya da **reaparicion de claims 0,0132**, o sea que ~99 % de lo
afirmado cambia. La cota **no acota** a un selector real, ni por arriba ni por abajo.

Consecuencia operativa: **"que fraccion de los +0,128 recupera exp19" no es una cantidad bien definida**
y no se reportara. Lo que la cota SI establece, y basta para motivar exp19: **el pool k=50 respalda el
58,3 % de los claims que el baseline realmente escribio, mientras los 5 chunks elegidos respaldan el
45,5 %**. El ranking topico deja anclaje sobre la mesa. Queda escrito en el propio artefacto
(`selection_bound.json::not_a_target`) para que no se vuelva a leer como objetivo.

**Decision de Enzo (2026-08-04):** la primaria de exp19 es delta fidelidad vs baseline en los 3
verificadores con la **misma banda TOST +-0,081** y familia BH declarada; la cota entra solo como
motivacion, fuera de BH y de TOST.

### Retiro formal de `deberta-large`

- Quedo en **11/12 configs** (falta `hibrido | qwen3.5-9b`); el `.partial` se conserva committeado como
  registro del intento.
- **Nunca se uso para ninguna cifra**: el runner solo promueve `nli_probs__large.json.gz` al completar
  las 12, asi que ningun consumidor lo leyo jamas.
- **El estandar de triangulacion de la fase es NLI-small + NLI-base + HHEM: tres verificadores, DOS
  familias** (dos NLI de entailment + un modelo de grounding ortogonal). La ortogonalidad de familia es
  lo que da valor a la triangulacion; un tercer NLI habria sido un voto correlacionado. **Nunca fue un
  "trio NLI"**, y esa expresion queda retirada del vocabulario de la fase.

**Impacto en cifras publicadas (S1 de la entrada 21, resuelto): NINGUNA cambia.**

| Cifra reportada | Usa los ensembles del "trio" | Veredicto |
|---|---|---|
| front-runner `E5_base_and_hhem` 0,0025 | No (base + hhem) | **intacta** — la seleccion se sostiene |
| **NLI 22 % falso-contradicted** | No (verificador unico) | **intacta** — la cifra mas citada de la fase |
| `hhem` 0,0325 · `base` 0,215 · `small` 0,2375 | No | intactas |
| `noisy_or` 0,55 RECHAZADO | No | intacta |
| kappa 0,30-0,36 (Tier 0) | No (acuerdo small-vs-base) | intacta |
| Tier A · exp16 · exp17 · exp18 | No | intactos: triangulan small + base + HHEM |
| **`E1_mean` 0,09** (entrada 9) | Si | **valor correcto, etiqueta imprecisa**: media de **2**, no de 3. La afirmacion que lo acompana ("mitad del NLI solo") sigue siendo cierta |
| `E2_vote` 0,1025 · `E3_conservative` 0,4125 | Si | computados con 2 miembros (**E2 = unanimidad, no mayoria**). **Nunca se reportaron** fuera del artefacto -> ninguna cifra publicada depende de ellos |

**Codigo:** `NLI_TRIO_EXPECTED = (small, base, large)` -> `NLI_MEMBERS_EXPECTED = (small, base)`, y la
etiqueta de `E2_vote` pasa a anclarse en `len(members) >= 3` (mayoria posible) **y no** en "degradado
respecto de lo esperado". Con la regla vieja, retirar `large` habria devuelto `E2_vote[2m=unanimity]` a
un `E2_vote` plano: el mismo estimador renombrado a una afirmacion que no sostiene.
**Re-corrido el sweep: 10/10 candidatos con valores IDENTICOS**, ranking identico, front-runner
identico. Solo cambian 3 etiquetas y la metadata.

### DEFECTO #7 — `pure_decline` no significa rechazo (misma familia que los seis anteriores)

`classify_response` etiqueta `pure_decline` cuando hay un marcador de rechazo en los **primeros 300
chars**. Es un test de **prefijo**. Medido sobre el baseline de exp18:

| clase | n | con claims genuinos | media claims | tasa no-soportables |
|---|---|---|---|---|
| `answered` | 75 | 75 | 16,6 | 0,330 |
| `hedged_partial` | 30 | 29 | 11,7 | 0,376 |
| **`pure_decline`** | **89** | **84** | **5,6** | **0,471** <- la peor |

Forma dominante, q078 textual: *"I cannot find sufficient information ... **However, I can outline
general steps that are typically involved**"* -> 11 claims, 3 137 chars. **No es declinar: es contestar
de memoria parametrica con prefijo de declinacion.** La tasa real de respuestas que **no afirman nada**
es **6/194 = 3,1 %**, no 45,9 %.

**Familia:** identica a los seis defectos previos — una lista de trabajo (aqui el denominador primario)
derivada de un **proxy comodo** (regex de prefijo) en vez de la **fuente de verdad** (afirma contenido o
no), y consumida aguas abajo como si significara otra cosa.

**Tres consecuencias.**

1. **El argumento de la config de encuestas se apoyaba en la etiqueta rota.** "El 45,9 % de declinacion
   hundiria la percepcion en SUS" es falso tal cual. El caso de k=10 se reconstruye sobre claims,
   cobertura y latencia. **La eleccion de config sigue siendo de Enzo.**
2. **La familia PRIMARIA publicada excluye justo esas filas.** Excluirlas sube el nivel **+0,0914
   (HHEM)** y **+0,0983 (small)** — casi la banda TOST entera. **Ninguna cifra publicada es erronea**:
   la familia la eligio Enzo el 2026-06-11 y `sens_c` esta publicada al lado. Pero la etiqueta puede
   enganar a un revisor de LACCI, asi que el gap queda **declarado en la salida**
   (`denominators::primary_vs_all_gap`), no implicito.
3. **Es la mejor pista viva para los 759 claims sin respaldo**, y sale gratis de artefactos ya
   persistidos: el orden de tasas (0,471 > 0,376 > 0,330) es consistente con **conocimiento
   parametrico**, aunque el grueso (54 %, 410 claims) viene de `answered`, asi que es contribuyente y
   no explicacion unica.

**Implementacion, con un ajuste declarado:** el token `pure_decline` **se conserva** en la
serializacion. Aparece en **7 archivos de evidencia firmada** (`exp12_matrix/faithfulness_metrics_v2..
v4*.json`, `exp13_expansion`, `exp14_h5_replicas`) y `verify_v4_offline.py` re-deriva cifras publicadas
desde ahi; renombrarlo desincronizaria el codigo de la evidencia. Se arregla lo que lee un humano:
`DISPLAY_LABELS` (`pure_decline` -> `decline_prefix`), docstrings, y un campo nuevo **`asserts_content`**
medido por contenido. Las guardas ganan `asserts_nothing_rate` y
`n_decline_prefix_that_still_answers`.

**Coherencia entre artefactos vecinos.** El propio `guards.md` emitia *"un brazo que sube
`pure_decline` si esta callandose"* — **falso**, contradecia la tabla de al lado, y queda retirado del
texto generado. Efecto lateral util: en exp16, la tasa medida por contenido sube de 6,7 % (baseline) a
**15,0 % (`strict_abstain`) y 16,7 % (`anchored_cite`)**, o sea que el negativo de exp16 se **refuerza**
al medirlo bien. Segunda nota: `arm_stats` de exp18 corre sobre **todas** las filas (n=191, 0,4638)
mientras la PRIMARIA de v4 es `primary_answered`; cada artefacto declara ahora su denominador.

### Persistencia de la matriz claim x pool

`compute_exp18_selection_bound.py` calculaba la matriz HHEM claim x chunk y **la descartaba** tras tomar
los agregados. Todo lo que pregunte algo mas fino que "cuantos claims superan tau" —la taxonomia de los
759, la sonda de selector de exp19, la muestra de gold— la necesita, y re-derivarla cuesta otro pase
completo de HHEM. Ahora se persiste por query (`selection_scores/` + indice con textos de claims y
`pool_ids` en orden), con **gate de completitud**: si hay agregados sin matriz, aborta nombrando las
queries. Verificado en smoke de 12 queries: agregados **identicos** a los committeados y re-derivacion
desde `.npy` **exacta** en 10/10 con matriz.

### Estado

Suite **130 pasan, 0 fallan, 0 omitidas** (el test NLI softmax que estaba omitido ahora corre).
`verify_summer_offline.py` **exit 0** cubriendo exp15..exp18. `git diff` contra
`nota3-evidencia-2026-06-11` sobre `experiments/results`: **127 altas, 0 modificaciones**. Guardas de
exp16/exp17/exp18 re-corridas **sin mover un solo valor previo**. Nada publicado: **43 commits siguen
solo en el disco de Enzo**, a la espera del OK de push.

---

## Entrada 23 — los 759 claims, el harness del gold, y tres defectos silenciosos mas (2026-08-04)

Bloque posterior a la entrada 22, mismo dia. Cierra F5 (offline) y F4, deja exp19a corriendo, y
registra los defectos #8, #9 y #10 de la misma familia. **Nada de A.3/LACCI tocado. Nada publicado.**

### F5 — por que 759 claims no los soporta ningun chunk del pool

Corrido sobre la matriz HHEM claim x chunk ya persistida (entrada 22), **cero generacion**.
188 queries, 2 053 claims genuinos.

**H1 v1 — RETIRADA SIN COMPUTAR.** La hipotesis pre-registrada era *"entre respuestas con prefijo de
declinacion, los claims escritos DESPUES del marcador estan menos anclados que los de antes"*. Es
**degenerada por construccion**: `pure_decline` se define como marcador dentro de los primeros 300
chars, asi que el estrato "antes" queda vacio. Medido para documentar la retirada: offset del
marcador **p50 = 0, max = 253 chars** sobre 84 respuestas. La guarda del propio script ("un split
desbalanceado tiene que ser visible") lo hizo fallar ruidoso en vez de emitir una cifra espuria.

**H1 v2 — declarada antes de mirar un solo dato de posicion, y NO APOYADA.**
Diferencia-en-diferencias sobre el gradiente intra-respuesta (2.ª mitad − 1.ª mitad de los claims),
con `answered` como brazo de control. El split intra-query controla dificultad de la consulta; el
control absorbe el efecto generico de "el modelo deriva al escribir". Solo la DIFERENCIA seria firma
de memoria parametrica.

| grupo | gradiente |
|---|---|
| `answered` (control) | 0,0834 |
| prefijo de declinacion | 0,1275 |
| **DiD** | **0,0442** (IC95 **−0,0694 a 0,1601**) |

49 respuestas con prefijo vs 72 `answered`; 38 excluidas por <4 claims. **El IC cruza 0.** La
hipotesis de memoria parametrica **no** queda apoyada por esta via.

**Hallazgo lateral, no buscado:** AMBOS grupos degradan al avanzar (0,128 y 0,083). El desanclaje por
posicion dentro de la respuesta existe y **no es propio de declinar**. Es una linea nueva, no una
conclusion.

**Lectura conservadora** de la diferencia entre-query ya conocida (tasa no-soportable 0,471 con
prefijo vs 0,330 en `answered`, entrada 22): compatible con que el modelo **declina PORQUE la
evidencia falta**, y entonces lo que diga no tiene con que anclarse. Consistente con
`evidence_swapped`, y no exige invocar un defecto de memoria parametrica.

**Estratos candidatos para el gold — NO son veredictos:**

| estrato | n | % | mejor score medio sobre el pool |
|---|---|---|---|
| `d_threshold_artifact` (best > tau−0,1) | 123 | 16,2 % | 0,4501 |
| `a_synthesis_cand` (>=2 chunks sobre 0,3) | 58 | 7,6 % | 0,3581 |
| `b_parametric_cand` (query con prefijo) | 178 | 23,4 % | 0,1651 |
| `c_unattributed_cand` (resto) | 400 | 52,7 % | 0,1893 |

Sintesis legitima **(a)** y alucinacion **(c)** no las puede separar un modelo de grounding — para eso
existe el gold humano. Muestra estratificada de 40 claims en
`output/audit/unsupported_claims_sample.csv`.

### F4 — el harness del gold, probado por fin

`analyze_gold_v4.py` nunca habia corrido. Ahora `--simulate 0.15` corre de punta a punta y no escribe
nada. El dia que Enzo termine de anotar, el analisis sale solo.

### DEFECTO #8 — el rename de la entrada 22 dejo huerfano al harness del gold

Retirar `deberta-large` renombro `NLI_TRIO` -> `NLI_MEMBERS`, y las **tres** referencias
`ens.NLI_TRIO` de `analyze_gold_v4.py` quedaron como lookups muertos. **La suite entera siguio verde
porque ningun test importaba el harness del gold.** Habria explotado con `AttributeError` el dia que
Enzo terminara de anotar los 150 claims — despues de 4-5 horas de su tiempo.

Familia: un simbolo cruzado entre modulos sin contrato, que cambia sin que nada registre quien
dependia de el. Guarda: escaneo estatico de **todo** `ens.<attr>` bajo `scripts/` resuelto por
`getattr` contra el modulo, parametrizado para que el fallo nombre archivo y atributo. Cubre la
familia entera de renames, no este caso.

### DEFECTO #9 — el selector de exp19a devolvia seleccion VACIA, y la compuerta leia FAIL

`select_by_claims` arrancaba `best_so_far` en `-inf`, asi que la ganancia de todo chunk era `+inf`,
`np.isfinite` la rechazaba y el bucle rompia **antes de elegir nada**. Recall salio **exactamente
0,0** y la compuerta dijo FAIL. **Un bug habria matado un experimento valido** — el error inverso al
habitual: un falso negativo que ahorra GPU y destruye el hallazgo.

Lo que lo delato no fue una excepcion sino una **cifra implausible**: 0,0 exacto, con IC [0,0].
Guarda: cinco casos con respuesta conocida por construccion, incluido el que separa seleccion por
claim de ranking por score (claim A servido por los chunks 0 y 1, claim B solo por el 2 y mas debil:
ordenar por score toma {0,1} y deja B sin nada; un selector por claim debe tomar {0,2}). A k=1 las
dos reglas no pueden diferir, asi que el caso necesita k=2 para ser test.

### DEFECTO #10 — la metrica primaria de exp19a SATURABA

Era recall@5 del conjunto de chunks que anclan. Pero la mayoria de chunks del pool anclan **algun**
claim (q001: **38 de 50**), asi que recall@5 vale ~5/|soportantes| para cualquier eleccion de cinco
chunks soportantes, y es **ciega** a si se cubrieron los claims correctos. Daba diferencia 0,0 con IC
[0,0] mientras las selecciones **si** diferian.

Primaria nueva: **cobertura de claims a respuesta fija**, con la cota alcanzable k=5 como techo
explicito. El recall de chunks queda como diagnostico con su motivo escrito.

Declarado ademas un hecho del harness necesario para leer la tabla: **el pool se guarda en orden ya
re-rankeado por produccion**, asi que reordenarlo por `(query, chunk)` devuelve los indices 0-4 = el
top-5 del baseline. `query_rank` es un **control del harness**, no un segundo comparador; el
contraste que importa es `claim_rank` vs baseline.

`frac_of_headroom_closed` queda definida **solo** dentro del mundo de respuesta fija, y el artefacto
lo dice: no es la fraccion de los +0,128 de exp18 que exp19b entregaria, porque un selector real
cambia la respuesta (entrada 22).

### Deuda de checkpoint, pagada

El entorno mato la sonda **dos veces** y las dos perdio todo: no tenia checkpoint, contra la regla de
la fase. Ahora hay checkpoint por query, verificado simulando una muerte a mitad (retoma y produce el
mismo resultado). Y la misma familia cerrada de raiz: `rows` ya no se acumula en memoria durante el
bucle, se **deriva del checkpoint en disco**, con gate que aborta si faltan queries. **Una compuerta
leida sobre una muestra parcial es peor que no tener compuerta.**

### Hueco de transferencia de exp18 a la config de encuestas, cuantificado

Antes se afirmaba sin medir. `SURVEY_DEPLOY` lleva `balance_cross_cloud_providers=True` y exp18
genero **sin** balanceo (verificado por grep sobre `run_exp18_ceiling.py`: cero referencias a
`coverage_balancer`; el routing de prompt si coincide, via `rgm.build_prompt`). El balanceo solo actua
sobre queries `cross_cloud`, que son **51 de 194 = 26,3 %** del set. No es despreciable. Segundo
hueco: exp18 reporta tiempo **total** y la UI hace streaming (`chat_page.py:189` -> `query_stream`),
asi que lo que percibe un encuestado es el **TTFT**.

### exp19a — pre-registrado, corriendo, sin resultado en esta entrada

Compuerta offline del selector de anclaje, **cero generacion**. Selector = ms-marco-MiniLM-L-12-v2
sobre `(claim, chunk)`, greedy por claim. **Ningun verificador de fidelidad entra en el bucle**, asi
que small, base y HHEM quedan los tres evaluadores limpios para exp19b; `bge-reranker-large` no se
toca y sigue siendo el oraculo independiente (Flag 17). Fijado por
`tests/test_selector_hygiene.py` (21 casos), que analiza **solo tokens ejecutables** — la prosa puede
nombrar legitimamente lo que el codigo se prohibe alcanzar, y una guarda que se dispara con su propia
documentacion ensena a debilitarla.

Compuerta declarada antes de correr y deliberadamente **asimetrica**: FAIL mata exp19b (si el
mecanismo no encuentra la evidencia en condiciones tan favorables, tampoco con un borrador real);
PASS solo prueba que no esta muerto, **no** predice ganancia de fidelidad. Sanity check obligatorio:
el rerank por query debe reproducir el top-5 real de exp18 (>=4/5) o no se lee nada — en todos los
smokes dio **5,0/5**.

**Preliminar sobre 25 queries, explicitamente NO el veredicto:** diferencia pareada +0,0008 (IC95
−0,020 a 0,023), 0,6 % del margen. Si la corrida completa lo confirma, el resultado tiene una lectura
de fondo: **la senal que encontraria los chunks que anclan es, por definicion, una senal de anclaje**,
y cualquier modelo bastante fuerte para hallarla es un verificador cuyo uso contamina la evaluacion.
La restriccion anti-circularidad y la potencia del selector estan en tension directa. Decision
pendiente de Enzo.

### exp19a — COMPUERTA: **PASS** (resultado, cerrado el mismo dia)

n=188, cero generacion. **Sanity check perfecto:** reordenar el pool por `(query, chunk)` reproduce
el top-5 real de exp18 en **188/188** (solape 5,0/5, coincidencia exacta 1,0), y el baseline
calculado aqui, **0,4552**, coincide con el de la cota al cuarto decimal.

| seleccion | cobertura de claims a respuesta fija |
|---|---|
| baseline (top-5 de exp18) | 0,4552 |
| rerank por query (control del harness) | 0,4552 |
| **rerank por claim** | **0,4853** |
| cota alcanzable k=5 (techo) | 0,5834 |

Diferencia pareada **+0,0300**, IC95 **[0,0156, 0,0448]** — excluye 0. Cierra el **23,4 %** del
margen disponible (0,1282). **59 queries mejoran, 22 empeoran, 107 quedan igual.**

**PASS con el significado declarado antes de correr y ningun otro:** el mecanismo **no esta
muerto**. **NO** predice ganancia de fidelidad, porque la respuesta cambia cuando cambia la
evidencia (entrada 22). `frac_of_headroom_closed` vive solo dentro del mundo de respuesta fija.

**EL PRELIMINAR ERA FALSO, Y SE REPORTO COMO PRELIMINAR ANTES DE SABERLO.** Sobre las primeras 25
queries por qid la diferencia era +0,0008 y la compuerta leia **FAIL**; sobre las 188 es +0,0300 y
lee **PASS**. Las primeras 25 no eran representativas. **El gate de completitud anadido horas antes
es lo que impidio publicar ese FAIL como veredicto**: se nego a emitir resultado con muestra
incompleta. El escenario que motivo la guarda ocurrio de verdad, el mismo dia, y en la direccion
cara — un falso negativo que habria matado un experimento valido y ahorrado GPU por el motivo
equivocado.

**Correccion de una lectura propia.** Con el FAIL preliminar se argumento que *"la senal que
encontraria los chunks que anclan es, por definicion, una senal de anclaje, y cualquier modelo
bastante fuerte para hallarla es un verificador que contamina la evaluacion"*. Con el resultado
completo esa tension queda **debilitada, no confirmada**: un reranker de **produccion**, sin ningun
verificador en el bucle, recupera casi una cuarta parte del margen. **La higiene de instrumento no
cuesta el experimento.** La opcion de seleccionar con HHEM (que contaminaria a HHEM como evaluador)
deja de ser necesaria para tener un exp19b con señal.

**Siguiente paso, y sigue requiriendo decision de Enzo:** exp19b, el brazo generativo
(borrador -> claims -> rerank por claim -> regenerar), con primaria = Δ fidelidad vs baseline en los
3 verificadores, familia BH declarada y TOST ±0,081. Es una corrida de generacion en la GPU de Enzo.

### Estado

Suite **158 rapida / 4 excluidas**. `verify_summer_offline.py` **exit 0** cubriendo exp15..exp18.
`git diff` contra `nota3-evidencia-2026-06-11` sobre `experiments/results`: **solo altas**.
**Nada publicado**: los commits siguen solo en el disco de Enzo, a la espera del OK de push.

---

## Entrada 24 — defecto #12, el push publicado, y la nube reencuadrada (2026-08-04)

### DEFECTO #12 — la encuesta iba a correr un modelo sin evidencia

```
SURVEY_DEPLOY.llm_model  =  llama3.1:8b-instruct-q4_K_M
evidencia de la fase     =  granite4.1:8b   (exp15 Tier A, exp16, exp17, exp18 — los cuatro)
```

La config sobre la que correrian las encuestas SUS/Likert —y sobre la que corre la demo Streamlit—
usaba un generador **para el que no existe una sola cifra de fidelidad**. El ancla HHEM 0,4638, el
efecto de balanceo de exp17 y los tres veredictos de exp18 son todos de granite.

**Origen, confirmado por Enzo:** residuo de la configuracion experimental previa. El paper sometido a
**LACCI describe Llama 3.1 8B Q4 sobre 200 queries**, y el despliegue **nunca se migro** cuando la fase
de verano cambio de generador. No fue un error de codigo: fue un cambio de fase que dejo atras una
herencia.

**Como aparecio:** mi propia corrida de latencia modifico `data/llm_cache/llama3.1_*_cache.json`. Es
decir, lo delato un **efecto lateral en disco**, no una excepcion ni un test. Consecuencia inmediata:
las latencias que habia reportado (TTFT 26,0 s a k=5) eran de llama3.1 y **no comparaban** con los
49,0 s p50 de exp18. Eso ademas **retira la hipotesis de contencion de VRAM** que habia ofrecido para
explicar la "discrepancia 2x": era innecesaria, son modelos y cuantizaciones distintos.

**Decision de Enzo: la encuesta corre granite4.1:8b.**

`PROPOSED_HYBRID` **NO se toca**, a proposito: es el registro del sistema sometido a LACCI, y
alinearlo con la encuesta seria hacer que coincidan **falsificando el registro** en vez de tomando una
decision. `get_config("hybrid")` sigue devolviendo el sistema medido del paper. Hay test que fija las
dos cosas.

**Guardas anadidas** (la divergencia no puede volver a ser invisible):
- el artefacto de latencia registra `llm_model_measured`, `llm_model_of_summer_evidence` —derivado de
  los `results.json`, no de una lista paralela— y `model_divergence` con el aviso escrito;
- `tests/test_deployment_generator.py`: la evidencia de verano debe ser de **un solo** generador, la
  config debe declarar el suyo, y la encuesta debe correr el de la evidencia;
- `test_survey_config_flips_exactly_two_knobs` se puso **rojo** al migrar, y tenia razon. Se actualiza
  a tres perillas (`name`, `prompt_routing`, `balance_cross_cloud_providers`, `llm_model`), sigue
  **cerrado**, y gana la asercion de que `PROPOSED_HYBRID` conserve el generador de LACCI.

**Latencia re-medida con granite** (ruta de despliegue, n=12 por config, estratificada y seeded):

| config | retrieval p50 | TTFT p50 | TTFT p90 | total p50 | palabras |
|---|---|---|---|---|---|
| k=5 | 5,1 s | **12,8 s** | 47,7 s | 179,8 s | 271 |
| k=10 | 5,0 s | **15,2 s** | 34,6 s | 132,7 s | 254 |

Frente a llama3.1 (supersedido, etiquetado, no borrado): granite tiene **mejor TTFT** (12,8 vs 26,0 s)
y **peor tiempo total** (179,8 vs 94,1 s) con longitud de respuesta parecida.

**Dos cosas que NO se afirman, a proposito.** Que k=10 sea mas rapido que k=5: el total sale menor
(132,7 vs 179,8 s) pero la dispersion es enorme (TTFT sd 15,8 s a k=5, rango 7,3-55,2 s) con n=12 —
es ruido hasta que se mida con potencia. Y que la subida de retrieval de ~1 a ~5 s sea causal:
hipotesis no verificada de competencia por VRAM entre granite residente y los modelos de recuperacion.

Lo que si sostiene la medicion: **la latencia del despliegue local es mala para una encuesta con
cualquier modelo y cualquier k.**

### PUSH PUBLICADO (autorizado por Enzo)

`git push -u origin summer/mejoras` — **63 commits**, rama nueva, `main` intacto (`origin/main` sigue
en `a524b0d`), sin PR y sin force. 13 dias de trabajo dejan de existir en un solo disco.

Verificaciones previas: suite **177 pasan**; `verify_summer_offline.py` **exit 0**;
`git diff --name-status nota3-evidencia-2026-06-11 -- experiments/results` = **322 altas, 0 M, 0 D**;
`.env` no versionado (ignorado en `.gitignore:47`); **0** coincidencias de credencial.

Nota de la compuerta: el escaneo dio **1 coincidencia de alta senal**, y se paro a inspeccionarla
antes de publicar. Era la **linea 29 de `output/audit/push_inventory_2026-08-04.md`**, o sea la prosa
del propio inventario listando los patrones que escanea. Mismo fenomeno que las guardas a nivel de
fuente que se disparan con su propia documentacion. Benigna, verificada, y se publica.

Entra tambien la **paridad de despliegue de UI** autorizada el 2026-08-03 y sin versionar desde
entonces: Chat y Evaluation resuelven `hybrid` a `SURVEY_DEPLOY` por un mapeo exclusivo de Streamlit,
y `query_stream()` pasa a respetar routing de prompts y balanceo cross-cloud igual que `query()` —
antes divergian, asi que la demo mostraba un sistema distinto del medido.

### La nube, reencuadrada: infraestructura de despliegue, no experimento de capacidad

El **NO-GO sigue vigente** para la nube como experimento de capacidad (entrada 22). Lo que Enzo
necesita ahora es otra cosa: **alojar el sistema ya medido** para que los participantes accedan en
remoto sin que la lentitud contamine el SUS/Likert.

**Causa del TTFT, medida** (log de Ollama de Tier A, no supuesta):

```
file type = Q4_K - Medium
load_tensors: offloaded 30/41 layers to GPU          <- 11 capas en CPU
llama_context: graph splits = 114 (with bs=512), 3 (with bs=1)
msg="vram-based default context" total_vram="6.0 GiB" default_num_ctx=4096
granite.context_length = 131072
```

El **prefill cruza la frontera CPU/GPU 114 veces**; el decode solo 3. El prefill *es* el TTFT.
**No hace falta un modelo mayor ni una A100: hace falta que el modelo quepa.**

Corolario: **el tope de 4096 es derivado de la VRAM, no del modelo** (granite declara 131072). Las
74/194 queries que truncan a k=10 son artefacto de la laptop de 6 GB.

**Diseno en `docs/CLOUD_DEPLOYMENT_SURVEY.md`** (documento nuevo; `CLOUD_EXPERIMENT_DESIGN.md` queda
intacto como el experimento de capacidad). Decisiones de Enzo incorporadas:

- **`num_ctx=4096` fijo.** Un solo cambio respecto a local: 41/41 capas en GPU en vez de 30/41.
  Beneficio no previsto al preguntarlo: **conserva el estrato truncado**, asi que la evidencia de
  fidelidad de exp18 para k=10 **transfiere tal cual** y la equivalencia mide un solo cambio.
- **Exposicion bajo demanda, por sesiones.** ~15-20 h de GPU en todo el estudio.
- **Si la equivalencia falla: parar y reportar.** No se despliega; la encuesta corre local a k=5; la
  no-equivalencia se reporta como hallazgo (el sistema es sensible al reparto GPU/CPU).

**GPU:** ~9 GiB de VRAM en total (granite 41/41 5,0 · KV 4096x4 slots 2,5 · bge-large 0,7 · ms-marco
0,15 · NLI small 0,6 — la UI verifica en vivo). **16 GB bastan, 24 GB da margen.** RTX 4090 recomendada.
**No hace falta A100 ni L40S.**

**Coste: USD 6-14 esperado, techo sugerido USD 30.** Precios por hora estimados, a verificar al
reservar. Dejar el pod encendido toda la ventana costaria USD 59-235 y es GPU ociosa.

**Compuerta `exp21_hosted_equivalence`, pre-registrada:** digest del modelo verificado contra el local
antes de generar; 194 queries sobre contextos congelados de exp18; sonda de determinismo 3x (H5);
**solo vuelven JSON, la puntuacion se queda en local** con los 3 verificadores; **TOST banda ±0,081**
en los tres; guarda del ancla HHEM en 0,40-0,55. **No se espera identidad bit a bit** —local corre
30/41 capas en GPU, el alojado 41/41— y por eso el criterio es equivalencia declarada, no igualdad.

**Cero gasto ejecutado.** El diseno no cuesta nada; la ejecucion requiere OK con el costo a la vista.

### Estado

Suite **177 pasan, 0 fallan, 0 omitidas**. `verify_summer_offline.py` exit 0. Evidencia firmada
**322 altas, 0 M, 0 D**. **Todo publicado en `origin/summer/mejoras`.**
