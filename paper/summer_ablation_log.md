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
