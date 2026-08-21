# Estado unificado del proyecto — hybrid-rag-system (CloudRAG)

> **Sección de [Kimi Work] — 2026-08-06 23:10 (hora local)**
> Documento de recopilación y diagnóstico. No modifica ningún archivo existente ni evidencia.
> Fuentes: CLAUDE.md, SUMMER_RESULTS.md, RESULTADOS_RESUMEN.md, NOTA3_NEXT_STEPS.md,
> paper/summer_ablation_log.md (entradas 1-24), docs/*, output/audit/*, git log/branch/status,
> y verificaciones ejecutadas en esta sesión (declaradas abajo).

---

## 1. Qué es el proyecto

CloudRAG: sistema RAG híbrido en Python para responder preguntas sobre documentación oficial de
AWS, Azure y GCP. Combina BM25 + embeddings BGE + fusión RRF + reranking cross-encoder, generación
local vía Ollama y medición de fidelidad con verificadores NLI/HHEM. Sustenta:

- la **tesis/curso Seminario de Investigación I** (ficha A.1: "Diseño y evaluación de un
  procedimiento RAG híbrido…", asesor Winston Lewis; H1 recuperación, H2 supuesto
  recuperación→fidelidad sometido a contraste), y
- el **paper LACCI 2026** (sometido; describe Llama 3.1 8B Q4 sobre 200 queries).

El A.1/A.3 son prosa del curso que se apoya en este código; aquí solo importan como consumidores
de cifras, no como fuente de verdad técnica.

## 2. Mapa de autoridad documental (qué archivo manda en qué)

| Tema | Autoridad | Nota |
|---|---|---|
| Evidencia Nota 3 (exp9..exp13) | `RESULTADOS_RESUMEN.md` + `experiments/results/` | cerrada, firmada (tags git) |
| Fase verano (exp15..exp18) + post-verano (exp19a) | `SUMMER_RESULTS.md` + `paper/summer_ablation_log.md` | vivo, al día al 2026-08-04 |
| Reglas de trabajo y restricciones | `CLAUDE.md` | **parcialmente desactualizado** (ver §7) |
| Reproducibilidad | `REPRODUCE.md` + `scripts/verify_*_offline.py` | falta `PYTHONUTF8=1` en la receta |
| Trazabilidad Nota 3 | `docs/TRACEABILITY_nota3.md` | contradicción de redacción con `docs/KNOB_MAP_summer.md` ya explicada (ambas correctas en su contexto; falta fijar texto canónico) |
| Despliegue encuestas (nube como infra) | `docs/CLOUD_DEPLOYMENT_SURVEY.md` | diseño aprobado, cero gasto ejecutado |
| Nube como experimento de capacidad | `docs/CLOUD_EXPERIMENT_DESIGN.md` | **NO-GO** tras exp18 |

## 3. Estado actual verificado (2026-08-06)

- Rama: `summer/mejoras`, worktree limpio. **Push publicado el 2026-08-04**: 63 commits en
  `origin/summer/mejoras`; `main` intacto. Incluye la paridad de despliegue de UI.
- Suite: **177 tests pasan, 0 fallos, 0 omitidos** (según ledger entrada 24; no re-ejecutada en
  esta sesión). Verificadores `verify_v4_offline.py` y `verify_summer_offline.py` en exit 0
  (requieren `PYTHONUTF8=1` en consola cp1252).
- Evidencia firmada intacta: 322 altas, 0 modificaciones, 0 borrados desde
  `nota3-evidencia-2026-06-11`.
- Generador de la evidencia de verano: **granite4.1:8b** (un solo generador, con test guarda).
  `PROPOSED_HYBRID` conserva llama3.1 a propósito (registro del sistema sometido a LACCI).
  `SURVEY_DEPLOY` = sistema medido + 3 perillas (`name`, `prompt_routing`,
  `balance_cross_cloud_providers`, `llm_model`).

## 4. Resultados clave (compacto)

| Hallazgo | Valor | Fuente |
|---|---|---|
| NDCG@5 híbrido (oráculo independiente) | 0,7405 (vs léxico 0,442) | exp11 |
| Fidelidad Granite lex/denso/híbrido (NLI small) | 0,235 / 0,247 / 0,299 | exp12 v4 |
| RAG-vs-RAG significativos | 0/12 (NLI) — **1/12 con HHEM** (granite híb>léx, p_BH 0,020) | Tier 3 |
| Nivel HHEM vs NLI | ~+0,31 (0,40-0,44 vs 0,23-0,30) | Tier 3 |
| NLI marca texto aleatorio como contradicted | 22 % → instrumento ruidoso | control negativo |
| Tier A (rerank/top-k/orden/lost-middle) | 0/4 brazos, robusto en 3 instrumentos | exp15_tierA |
| exp16 (prompt anclado/abstención) | 0/2, tiende abajo; sube "no afirma nada" 6,7→15-16,7 % | exp16 |
| exp17 (balanceo cobertura cross-cloud) | **positivo** en 3 instrumentos (HHEM +0,081; no sig a n=25; GLMM p=0,021 sin confirmar por bootstrap) | exp17 |
| exp18 (la compuerta) | generador SÍ usa contexto (−0,3185 con evidencia cambiada); oráculo tópico TOST-equivalente; k=10 compra cobertura, no anclaje (+61 % latencia) | exp18 |
| Cota de selección k=5 | baseline 0,4552 → alcanzable 0,5834 | exp18 |
| exp19a (sonda offline de selector) | **PASS**: rerank por claim sube cobertura 0,4552→0,4853 (23 % del margen), sin verificador en el bucle | exp19a |

**Síntesis de la fase:** la fidelidad responde a QUÉ evidencia entra (selección de contenido),
no al arreglo del contexto ni al prompt. Falta un *selector guiado por anclaje* (exp19b) y la
validación humana del nivel de fidelidad (gold).

## 5. Gates / cuellos de botella, ordenados por lo que desbloquean

| # | Gate | Esfuerzo | Qué desbloquea | Estado |
|---|---|---|---|---|
| **G1** | **Gold humano v4 — Etapa A (150 claims) + Etapa B (50)** | ~4-5 h + ~3,5 h | El único árbitro objetivo del nivel real de fidelidad (¿0,30 NLI u 0,55 HHEM?) y del verificador definitivo. Sin esto el paper queda sin validar | **0/150 y 0/50** — el pendiente más antiguo (v3 en 0/50 desde el 11/06 quedó reemplazado) |
| **G2** | **Congelar config de encuestas** (k=5 vs k=10 sobre la ruta de despliegue; TTFT ya medido con granite: 12,8 s k=5 / 15,2 s k=10) | decisión + corrida corta | Todo lo de encuestas. **Vence a mediados de agosto (≈1 semana)** — es el gate con fecha más próxima | pendiente de decisión de Enzo |
| **G3** | **Decisión nube como infraestructura** + compuerta `exp21_hosted_equivalence` (TOST ±0,081, 3 verificadores) | USD 6-14 (techo 30), ~2-4 h + corrida 194 q | Encuestas remotas sin contaminar SUS/Likert con la lentitud local. Si la equivalencia falla: encuesta local a k=5 | diseño listo; **cero gasto ejecutado; requiere OK explícito con costo a la vista** |
| **G4** | **Decisión exp19b** (brazo generativo: borrador → claims → rerank por claim → regenerar; primaria Δ fidelidad + TOST en 3 verificadores) | 1 corrida GPU local | El posible segundo positivo de la fase: selector guiado por anclaje, motivado por la cota 0,4552→0,5834 | pendiente de decisión de Enzo |
| **G5** | Encuestas SUS/Likert (ejecución) | semanas | Validación con usuarios; última milla de la tesis | no iniciado; depende de G2/G3 |
| **G6** | Taxonomía de los 759 claims sin respaldo (síntesis legítima vs memoria paramétrica vs alucinación vs fallo del verificador) | offline, incremental | Discusión del paper; interpreta el 54 % de claims no anclados | en curso (F5 cerrado offline; taxonomía abierta) |
| **G7** | Confirmatorio pre-registrado de exp17 (queries nuevas) | 1 corrida | Cruzar significancia del único positivo de selección | opcional; decisión de Enzo |

**Cadena crítica:** G1 → (verificador definitivo + nivel validado) → reescritura A.3/paper con
cifras validadas. En paralelo: G2+G3 → G5. G4 es independiente y local.

## 6. Gold humano v4 — estado y auditoría de esta sesión

Archivos (todos con `juicio_humano` y `comentario` **vacíos**):

- `output/audit/claim_audit_sample_v4.csv` — Etapa A: 150 claims, 1 chunk (~800 chars) + pregunta.
- `output/audit/claim_audit_sample_v4_stageB.csv` — Etapa B: 50 de esos mismos claims, 5 chunks
  (~1500 chars c/u, paridad exacta con lo que ve HHEM).
- `output/audit/claim_audit_sample_v4.md` / `_stageB.md` — versiones legibles para anotar.
- `output/audit/claim_audit_sample_v4_meta.json` — estratos y etiquetas de verificadores.
  **⚠ No abrir durante la anotación: revela los estratos y ancla el juicio.**
- `output/audit/claim_audit_sample_v3.csv` — 50 claims, 0/50 desde el 11/06. **Reemplazado por
  v4** (diseño de 2 etapas). Recomiendo no tocarlo (historial) pero ignorarlo operativamente.

**Auditoría de integridad ejecutada (9/9 OK):** v4 disjunto de v3; estratos = objetivo
(30/40/50/30); flags `stage_b` del meta coinciden exactamente con los `stage_a_idx` de B; claims de
B idénticos a los de A; sin duplicados (query, claim); idx contiguos 1..150 y 1..50; longitudes de
evidencia según diseño; distribución de configs razonable (12 combos modelo×escenario).

**Harness probado:** `scripts/analyze_gold_v4.py --simulate 0.15` corre de punta a punta (κ
ponderada Horvitz-Thompson con replay del muestreador, κ sin pesos sobre `random_anchor`, curva de
confiabilidad + ECE, corrección de sesgo por evidencia etapa B, familia BH declarada) y **no
escribe nada** en modo simulación. Es decir: cuando la anotación esté completa, el análisis no
será un cuello de botella. Candidatos previstos: small, base, hhem, E5_base_and_hhem, E1_mean.

**Plan de tandas sugerido** (criterio de la guía de anotación del repo: sesiones de 1-2 h con
descansos cada 30-40 min):

| Tanda | Contenido | Estimado |
|---|---|---|
| A1-A5 | Etapa A, idx 1-30, 31-60, 61-90, 91-120, 121-150 | ~50-60 min c/u |
| B1-B2 | Etapa B, idx 1-25 y 26-50 | ~1,5-2 h c/u |

Reglas del diseño: terminar **toda** la etapa A antes de abrir la B; no consultar lo respondido en
A al hacer B; juicio ciego (solo `correcto` / `incorrecto` / `dudoso`); comentario opcional.

## 7. Deuda técnica y documental detectada

**CLAUDE.md desactualizado** (las secciones vivas quedaron al 2026-08-03/04; verificado hoy):

- Dice que `SUMMER_RESULTS.md` marca exp18 como pendiente → **ya no**: lo declara CERRADA el
  2026-08-04 (línea 285). La "contradicción sin resolver" correspondiente también quedó obsoleta.
- Reporta suites de 121/123 tests → el ledger reporta **177** tras la paridad de despliegue.
- La entrada de Kimi Code del 2026-08-04 dice "paridad de despliegue SIN COMMIT" → ya está
  commiteada y publicada en `origin/summer/mejoras`.

**Contradicciones abiertas heredadas (sin cambios desde el 2026-08-04):**

- Versión de Python: badge README 3.14 / texto README 3.10+ / setup.py 3.11+ / REPRODUCE 3.14.
  Decisión registrada: separar soporte de paquete (3.11+) de entorno reproducible (3.14) —
  **pendiente de ejecución**.
- `requirements.txt` omite Streamlit, Plotly y pytest; `setup.py` omite parte del stack; sin
  lockfile. Manifiesto canónico por decidir.
- `REPRODUCE.md` no menciona `PYTHONUTF8=1` y la receta PowerShell de `verify_v4_offline.py`
  falla en consola cp1252 sin ella.
- Texto canónico de la contradicción de prompt routing (TRACEABILITY vs KNOB_MAP): ambas
  descripciones son correctas en su contexto; falta redactar la versión canónica.
- README describe solo Nota 3 (exp3..exp13); no cubre la fase de verano. Decisión pendiente.
- Test `test_nli_output_is_softmax_probabilities` se omite con `HF_HUB_OFFLINE=1` aunque el
  snapshot local existe (pasa sin el flag; aserción manual contra el snapshot OK). Deuda de
  cobertura, sin tocar.

**Sin CI versionada; sin issues/PRs en el remoto.** La protección depende de tests + verificadores
offline + ledgers. Riesgo conocido y aceptado hasta ahora.

## 8. Recomendaciones (priorizadas)

1. **Empezar G1 ya** (tanda A1). Es el gate más antiguo, el de mayor valor y el único que valida
   el nivel de fidelidad del paper. El harness está probado; el único recurso escaso son las ~8 h
   de anotador.
2. **Decidir G2 esta semana** (vence ~mediados de agosto): la medición de TTFT con granite ya
   existe; la decisión k=5 vs k=10 es de trade-off cobertura/latencia sobre la ruta de despliegue.
3. **Encadenar G3 detrás de G2**: si hay encuestas remotas, ejecutar `exp21_hosted_equivalence`
   con el costo a la vista (USD 6-14, techo 30) antes de cualquier despliegue.
4. **Decidir G4 (exp19b)** en la misma sentada que G2: usa la misma GPU local, no cuesta dinero,
   y su pre-registro ya está escrito.
5. **Actualizar las secciones vivas de CLAUDE.md** (ítems de §7) en la próxima sesión de
   mantenimiento de memoria; hoy solo lo registro, no lo reescribo.
6. **No gastar en nube como experimento de capacidad**: el NO-GO de exp18 sigue vigente; la nube
   solo como infraestructura de despliegue (G3).

## 9. Consultas abiertas para Enzo

1. ¿Autorizas que actualice las secciones vivas de CLAUDE.md (exp18 cerrada, 177 tests, paridad ya
   publicada) o prefieres que lo haga Claude Code/Kimi Code en su próxima sesión?
2. ¿G2: k=5 o k=10 para la config de encuestas? (Con los TTFT de granite ya medidos.)
3. ¿G4: exp19b sí/no? Si sí, ¿en qué ventana corro la generación?
4. ¿G7: confirmatorio pre-registrado de exp17, sí/no?
5. ¿Mantenemos v3 archivado como historial o lo marco explícitamente como "reemplazado por v4"
   dentro del propio CSV/MD para evitar que alguien lo retome?

---

*Fin de la sección de [Kimi Work]. Verificaciones de esta sesión: integridad del gold v4 (9/9),
smoke test del analizador (OK, sin escritura), git branch/status, lectura completa de CLAUDE.md,
SUMMER_RESULTS.md, NOTA3_NEXT_STEPS.md, cola del ledger y guías de anotación. No se modificó código,
evidencia ni documentos existentes.*
