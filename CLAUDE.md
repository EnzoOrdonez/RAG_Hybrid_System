# CLAUDE.md: hybrid-rag-system

Contexto permanente para Claude Code en este repo, tesis CloudRAG, Nota 3 y LACCI 2026.
Se carga automáticamente cada sesión; estas reglas no hace falta repetirlas.

## Qué es esto

CloudRAG es un sistema RAG híbrido en Python para responder preguntas sobre documentación de AWS,
Azure y GCP. Combina BM25, embeddings BGE, fusión RRF, reranking con cross-encoder, generación local
por Ollama y medición de fidelidad con verificadores NLI/HHEM. El repo contiene el pipeline, una UI
Streamlit y la evidencia experimental de una tesis y un paper.

## Estado actual

- La evidencia de Nota 3 está cerrada en `exp9..exp13`; `exp3..exp8` y `exp8b` son historia.
- La fase de verano tiene artefactos para `exp15..exp18`. El verificador offline del 2026-08-03
  descubre los cuatro experimentos y reproduce sus cifras desde probabilidades persistidas.
- Las rutas Streamlit de Chat y Evaluation resuelven el brazo `hybrid` a `SURVEY_DEPLOY`. El registro
  experimental conserva `get_config("hybrid") == PROPOSED_HYBRID`. `query_stream()` aplica el mismo
  routing de prompts y balanceo cross-cloud que `query()` cuando esas perillas están activas.
- `exp18` ya tiene resultados y análisis committeados en `f6816b4`, pero el resumen vivo
  `SUMMER_RESULTS.md` aún lo describe como pendiente. La autoridad documental sigue por confirmar.
- La rama activa observada el 2026-08-03 es `summer/mejoras`, con worktree limpio al iniciar esta
  exploración. No hay configuración de CI versionada bajo `.github/` ni ADRs formales. El remoto de
  GitHub no tiene issues ni PRs, abiertos o cerrados; el seguimiento real vive en documentos y ledgers.
- Línea base previa del 2026-08-03 con Python 3.14.3, modo offline y seed 42: suite rápida
  `115 passed, 3 deselected`; suite completa `117 passed, 1 skipped`. Tras la paridad de despliegue:
  `121 passed, 3 deselected` y `123 passed, 1 skipped`, sin fallos. No hay medición de coverage
  configurada.
- Verificación offline del 2026-08-03: Nota 3 v4 y fase de verano pasan con `PYTHONUTF8=1`; ninguna
  evidencia bajo `experiments/` cambió. La comprobación v4 creó
  `output/audit/v4_offline_check_2026-08-03.md`, aún sin versionar.
- Health check local aprobado: 24.481 chunks, FAISS y BM25 cargan 24.481 entradas, Ollama responde con
  Llama 3.1 disponible y el snapshot NLI local carga correctamente.

## Sensible / no tocar sin permiso

- `experiments/results/exp3..exp14` y `exp8b`: evidencia firmada. Solo se admite reanálisis offline en
  archivos `_vN` nuevos.
- `paper/audit_findings.md` y `paper/audit_outputs/exp8_stats_corrected.csv`: inmutables.
- `paper/overleaf_ready/main.tex` y la prosa A.3: cualquier corrección requiere aprobación explícita
  frase por frase.
- `experiments/results/exp15..exp18`: evidencia de verano ya committeada. No regenerar, sobrescribir
  ni reinterpretar sin fijar antes el alcance y revisar el ledger.
- `.env`: existe localmente, está ignorado y puede contener secretos. No leer, imprimir ni versionar.

## Problemas conocidos abiertos

- El gold humano de dos etapas sigue pendiente de anotación real.
- La decisión sobre experimentos de nube y gasto está bloqueada hasta interpretar `exp18` y recibir
  aprobación explícita.
- La documentación de estado no está sincronizada con el cierre de `exp18`.
- No hay CI versionada; la protección depende de pruebas y verificadores locales.
- La receta PowerShell de `verify_v4_offline.py` falla con `UnicodeEncodeError` bajo la consola
  `cp1252` al imprimir `≈`. El verificador pasa completo al fijar `PYTHONUTF8=1`, variable ausente de
  `REPRODUCE.md`.
- La instalación declarada no reproduce todos los flujos. `requirements.txt` omite Streamlit, Plotly
  y pytest; `setup.py` omite buena parte del stack de retrieval y evaluación. No existe lockfile.
- La prueba crítica `test_nli_output_is_softmax_probabilities` se omite en modo offline aunque el modelo
  local existe. La prueba carga el ID de Hugging Face directamente y no comparte el fallback local de
  `HallucinationDetector`. La aserción manual contra el snapshot local sí pasó: suma 1,0 y probabilidad
  de entailment 0,9982 frente al umbral 0,7.

## Contradicciones sin resolver

- `docs/TRACEABILITY_nota3.md` dice que `RAGPipeline.query()` no replica el ruteo de prompts de exp12.
  `docs/KNOB_MAP_summer.md`, re-verificado después, dice que esa afirmación era imprecisa y que el prompt
  sí coincide, aunque cambia el origen del contexto. Pregunta pendiente: cuál descripción debe quedar
  como canónica antes de tocar el pipeline o sus pruebas de paridad.
- `SUMMER_RESULTS.md` marca `exp18` como pendiente. Git, el ledger y
  `output/audit/summer_offline_check_2026-08-03.md` muestran artefactos completos y verificación offline
  aprobada. Pregunta pendiente: si se actualiza el resumen y qué texto se autoriza.
- El soporte de Python difiere: README dice 3.10+, `setup.py` exige 3.11+ y `REPRODUCE.md` fija el entorno
  experimental local en 3.14. Pregunta pendiente: distinguir soporte del paquete de entorno reproducible
  o unificar la versión declarada.
- README presenta 12 experimentos de `exp3..exp13` más `exp8b`, pero el repo ya contiene la fase
  `exp15..exp18`. Pregunta pendiente: si README debe describir solo Nota 3 o también el estado de verano.
- `REPRODUCE.md` afirma que los niveles 1 y 2 pasan y da comandos PowerShell, pero el comando v4 falla
  en la consola local por codificación salvo que se agregue `PYTHONUTF8=1`. Pregunta pendiente: corregir
  el script para salida tolerante, documentar la variable, o hacer ambas cosas.
- README indica instalar `requirements.txt` y luego lanzar Streamlit, pero Streamlit y Plotly no están
  declarados allí. `setup.py` expone otro conjunto más corto. Pregunta pendiente: cuál manifiesto es
  canónico y si UI, pruebas y ML deben separarse en extras o instalarse juntos.
- La suite completa se reporta verde con una omisión, pero la prueba omitida es una guarda NLI central y
  el modelo local sí está presente. Pregunta pendiente: si la prueba debe resolverse mediante el mismo
  cargador del detector o mediante una ruta local explícita.

## Próximos pasos

- Resolver con Enzo las contradicciones documentales que siguen abiertas antes de cambiar manifiestos,
  reproducibilidad o afirmaciones del paper.
- Mantener separada cualquier corrección futura del test NLI omitido; la paridad de UI ya tiene cobertura
  de configuración, routing y balanceo.

## Reglas de trabajo (permanentes)

1. **Sé crítico con tu propio trabajo.** No des nada por bueno solo porque corrió sin error.
   Audita y prueba antes y después de cada cambio de código.
2. **Sé proactivo.** Si encuentras una falla que nadie pidió revisar, repórtala.
3. **Ante ambigüedad, pregunta.** Cuando algo sea ambiguo, se contradiga, o no tengas certeza de la
   interpretación correcta, sobre todo si afecta una cifra ya entregada en A.3 o el paper de LACCI,
   pregunta a Enzo en vez de asumir.
4. **Da contexto suficiente en tus reportes** para que una decisión se pueda tomar sin ambigüedad.

## Restricciones inviolables (evidencia de la tesis)

- **NO modificar ni borrar** `experiments/results/exp3..exp14` ni `exp8b`. No existen exp1/exp2.
  Evidencia firmada con tags `nota3-evidencia-2026-06-11` y `nota3-N9-cierre-2026-07-02`.
  Todo recálculo crea archivos **`_vN` nuevos**: v3=N8, v4=N9 y así sucesivamente.
- **Generación LLM nueva solo bajo IDs `exp15+`**, fase de verano autorizada el 2026-07-22.
  Sobre `exp3..exp14`, solo reanálisis offline. Ledger: `paper/summer_ablation_log.md`.
  Perillas: `docs/KNOB_MAP_summer.md`. Para la paridad de despliegue se autorizó `summer/mejoras`.
- Entorno: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42`, seed=42. Intérprete con
  stack ML: `C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe`. No usar el Python
  3.11 del PATH para reproducir experimentos.
- **Compuerta antes de `git push`**: reportar qué se publicaría y esperar un OK explícito. Parar si
  aparece `.env` o un secreto.
- `paper/audit_findings.md` y `paper/audit_outputs/exp8_stats_corrected.csv` son inmutables.
- `paper/overleaf_ready/main.tex` y A.3 son prosa del paper. Corregir solo con OK explícito frase por
  frase.

## Punteros

- Ledger de auditoría: `paper/audit_findings_cc_addenda.md` (N1 a N9) y `paper/correction_log.md`.
- Trazabilidad: `docs/TRACEABILITY_nota3.md`. Resultados de Nota 3: `RESULTADOS_RESUMEN.md`.
- Fase de verano: `paper/summer_ablation_log.md`, `SUMMER_RESULTS.md` y
  `output/audit/summer_offline_check_2026-08-03.md`.

## Log de decisiones y sesiones

- 2026-08-03: exploración inicial sin cambios de código. Se identificó un proyecto Python de tesis con
  pipeline RAG, UI Streamlit y evidencia experimental versionada. Se confirmó worktree limpio en
  `summer/mejoras`, ausencia de CI versionada y cuatro contradicciones documentales pendientes. Se
  reorganizó esta memoria al protocolo de secciones vivas más log append-only.
- 2026-08-03: línea base confirmada con el intérprete experimental Python 3.14.3, PyTorch 2.10.0 con
  CUDA disponible y Transformers 5.2.0. Suite rápida: 115 aprobadas y 3 excluidas. Suite completa:
  117 aprobadas y 1 omitida. No hubo fallos; coverage no está configurado.
- 2026-08-03: se rastreó `SURVEY_DEPLOY` hasta sus llamadores. Solo aparece en configuración y tests;
  las páginas Chat y Evaluation cargan `hybrid`, con las dos perillas de verano apagadas. El camino
  streaming tampoco implementa routing ni balanceo. Se registró la decisión de producto como pendiente
  y no se tocó código central.
- 2026-08-03: `verify_v4_offline.py` falló primero por salida `cp1252` al imprimir `≈`; con
  `PYTHONUTF8=1` pasó completo y creó un reporte nuevo sin versionar. `verify_summer_offline.py` también
  pasó y cubrió exp15 a exp18. No cambió ningún archivo de evidencia.
- 2026-08-03: se contrastaron imports y manifiestos. El repo usa pip con `requirements.txt` y
  `setup.py`, sin lockfile; la UI y las pruebas dependen de paquetes no declarados en la receta de
  instalación. Se dejó abierta la política de empaquetado.
- 2026-08-03: el health check aprobó corpus, índices, Ollama y NLI. La única prueba omitida intenta el ID
  remoto del NLI sin el fallback local; su aserción softmax se ejecutó manualmente contra el snapshot y
  pasó. Quedó registrada como deuda de cobertura, sin tocar el test.
- 2026-08-03: se revisó el remoto `EnzoOrdonez/RAG_Hybrid_System` mediante el conector de GitHub. No hay
  issues ni PRs, abiertos o cerrados. Tampoco existe CI versionada en el checkout, por lo que los
  ledgers, tests y verificadores offline son las fuentes operativas de estado.
- 2026-08-03: Enzo autorizó implementar la paridad de despliegue en `summer/mejoras`: Chat y Evaluation
  usan `SURVEY_DEPLOY` mediante un mapeo exclusivo de Streamlit, mientras el registro experimental
  conserva `PROPOSED_HYBRID`. Se corrigió `query_stream()` para respetar routing de prompts y balanceo
  cross-cloud como `query()`, y Chat muestra `RAG Híbrido (despliegue encuestas)`. Se añadieron seis
  pruebas de comportamiento con TDD. Validación final: suite rápida 121 aprobadas y 3 excluidas; suite
  completa 123 aprobadas y 1 omitida; verificadores v4/verano y health check aprobados. No se tocó
  evidencia, resultados ni documentación del paper, y no hubo generación LLM.

[Kimi Code] — 2026-08-04 — Modo: Sondeo

Qué revisé: memoria existente (CLAUDE.md, única; no hay AGENTS.md), git log y worktree,
línea base de tests con el intérprete 3.14, contradicciones documentales abiertas contra el
código real, y re-auditoría del ruteo de prompts pedida por Enzo.

Hallazgos:
- Línea base: suite rápida 121 passed / 3 deselected (igual que la entrada de Claude Code del
  2026-08-03). Suite completa SIN forzar modo offline: 124 passed, 0 skipped — el test
  test_nli_output_is_softmax_probabilities corrió y pasó; solo se omite con HF_HUB_OFFLINE=1.
- La paridad de despliegue autorizada el 2026-08-03 sigue SIN COMMIT en summer/mejoras:
  CLAUDE.md, rag_pipeline.py, index_loader.py, chat_page.py modificados; test_ui_deployment_parity.py
  y v4_offline_check_2026-08-03.md sin rastrear. La memoria reorganizada existe solo en el worktree.
- Re-auditoría de prompts: no hay bug. Construcción del prompt idéntica entre
  rag_pipeline.py:328-340 y run_generation_matrix.py:110-115 dado el mismo query_type. Lo que
  cambia es el origen del query_type: prompt_routing (pipeline_config.py:49, default False) es
  opt-in; configs legacy sin esa perilla clasifican todo como default. TRACEABILITY describía la
  config legacy evaluada; KNOB_MAP describe el camino con prompt_routing=True. Ambas correctas
  en su contexto; la contradicción es de redacción.

Contradicciones encontradas: ninguna nueva más allá de las ya registradas por Claude Code.

Verificación de entradas previas:
- Confirmo requirements.txt sin Streamlit/Plotly/pytest y setup.py >=3.11 (Claude Code, 2026-08-03).
- Confirmo SUMMER_RESULTS.md:271 describe exp18 como pendiente pese a f6816b4 (sigue así).
- Confirmo README badge 3.14 / texto 3.10+ / setup.py 3.11+ / REPRODUCE 3.14 (sigue así).
- La afirmación "worktree limpio" del 2026-08-03 ya no aplica: la paridad de despliegue quedó
  sin commit al cerrar esa sesión.

Recomendación / siguiente paso: Enzo decidió (a) registrar exp18 sin tocar SUMMER_RESULTS.md
por ahora, (b) separar soporte de paquete (3.11+) de entorno reproducible (3.14) cuando se
autorice Ejecución sobre README/setup.py, (c) no commitear la paridad sin OK explícito.
Pendiente de Ejecución: fijar texto canónico de la contradicción de prompts y las ediciones
de versionado de Python.

[Kimi Work] — 2026-08-06 23:10 — Modo: Sondeo + recopilación

Qué hice: auditoría de solo lectura del estado del proyecto y del paquete gold v4; documento
unificado nuevo en `docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md`. No toqué código, evidencia
ni documentos existentes (solo esta entrada de log).

Verificaciones ejecutadas:
- Integridad del gold v4 (9/9 OK): v4 disjunto de v3; estratos = objetivo (30/40/50/30);
  flags stage_b del meta == stage_a_idx de la etapa B (50/50); claims B == claims A; sin
  duplicados; idx contiguos; longitudes según diseño. juicio_humano 0/150 (A) y 0/50 (B).
- `scripts/analyze_gold_v4.py --simulate 0.15` corre de punta a punta con el intérprete 3.14
  (PYTHONUTF8=1) y no escribe nada → el análisis post-anotación no será cuello de botella.
- git: rama `summer/mejoras`, worktree limpio; push del 2026-08-04 publicado (63 commits).

Secciones vivas de ESTE archivo que quedaron desactualizadas (registro, no reescritura —
pendiente de OK de Enzo para actualizarlas):
- "Estado actual" y "Contradicciones sin resolver" dicen que SUMMER_RESULTS.md marca exp18
  como pendiente: ya no, lo declara CERRADA el 2026-08-04.
- Línea base de tests 121/123: el ledger entrada 24 reporta 177 pasan, 0 fallos, 0 omitidas.
- La entrada [Kimi Code] del 2026-08-04 reporta la paridad de despliegue sin commit: ya está
  commiteada y publicada en origin/summer/mejoras.
- La verificación "SUMMER_RESULTS.md:271 describe exp18 como pendiente (sigue así)" de la
  entrada previa también quedó obsoleta.

Cuellos de botella vigentes (detalle en el documento unificado): G1 gold humano v4 A+B
(~8 h de anotador, el más antiguo); G2 congelar config de encuestas (vence ~mediados de
agosto); G3 decisión nube-infraestructura + compuerta exp21 (cero gasto ejecutado); G4
decisión exp19b; G6 taxonomía de 759 claims; G7 confirmatorio exp17 (opcional).

[Claude Code] — 2026-08-21 15:50 — Modo: Auditoría de ramas (solo lectura) + runner de exp19b (G4)

Rama: `summer/exp19b`, creada desde `summer/mejoras` en `89dc654`. Sin commit y sin push.

**Paquete 0 — auditoría de ramas (solo lectura, nada borrado).** 10 ramas locales, 7 remotas.
Totalmente fusionadas en `summer/mejoras` y sin contenido único: `codex/plan-a-thesis-safe`,
`fase-2.5-recompute-retrieval-stats`, `fix/phase-1-no-rerun`, `fix/phase-2-nli-and-seeds`,
`pre-corpus-rebuild-2026-05-21`, `summer/ablacion`. Con commits NO fusionados: solo dos.
`fase-3-regenerate-figures` (3 commits, publicados en origin, trae `generate_phase3_artifacts.py`
y 20 artefactos `_phase3` ausentes de `summer/mejoras`) y
`fase-3.5-nli-recompute-saved-answers` (9 commits, de los cuales **6 existen solo en este disco**:
`origin/…` está en `c119c99`, local en `e8d2e2e`; incluye el commit de anti-circularidad
Flag 17/142 y los scripts `recompute_nli_over_saved.py` y `build_annotation_pool.py`). Los 5 tags
están contenidos en `main`, `summer/ablacion` y `summer/mejoras`: ningún borrado de rama los
huerfanaría. `main` local va 2 commits por delante de `origin/main`. La eliminación queda a
decisión de Enzo, rama por rama.

**Paquete 1 — runner de exp19b, entregado sin corrida.** Diseño y pre-registro completos en la
entrada 25 de `paper/summer_ablation_log.md`. Cuatro decisiones de Enzo tomadas antes de escribir
código: borrador regenerado en sesión (el borrador ES el brazo `baseline_repro`), n=194, familia BH
= 1 contraste por verificador con `p_BH == p_raw` declarado, y separación en cuatro scripts para
que el selector no pueda alcanzar un verificador. Archivos nuevos: `scripts/run_exp19b_generation.py`,
`scripts/extract_exp19b_claims.py`, `scripts/select_exp19b_evidence.py`,
`scripts/compute_exp19b_stats.py`, `tests/test_exp19b_runner.py`. Modificado:
`tests/test_selector_hygiene.py` (las prohibiciones universales pasan a cubrir los dos selectores;
solo las dos aserciones propias de una sonda se estrecharon a exp19a).

**Validado.** Extracción de claims sobre las 194 respuestas reales del baseline de exp18:
2 053 claims genuinos y **6 queries sin ninguno** — exactamente las 6 de 194 que declara la entrada
21 del ledger. Selector: sanity check 5,0/5 contra el top-5 real de exp18, y los tres caminos
ejercitados (swap real, reorden puro, fallback). Estadística: primaria, bootstrap, d_z, BH y TOST
±0,081 corren de punta a punta (sobre filas sintéticas de scratchpad; las cifras no significan nada,
solo el plumbing). Defecto propio cazado antes de la GPU: el script de estadística volcaba el dict
crudo de `paired_comparison` y moría con `TypeError: Object of type bool is not JSON serializable`
—habría muerto al final de una corrida de 5 h con los brazos ya generados—; corregido extrayendo
escalares, como ya hacía `compute_tierA_arm_stats.py`.

**NO verificado.** Las dos etapas de generación (`--stage draft` y `--stage regen`) no se ejecutaron:
el servidor de Ollama no estaba escuchando en 11434 durante la sesión. Al buscar el binario se lanzó
sin querer el proceso `ollama app` (PID 17412), que no llegó a levantar el servidor; queda a decisión
de Enzo cerrarlo. Pendiente por tanto: el smoke de 3 queries de generación, y la corrida real.

**Estado de auditoría.** Suite **206 pasan, 0 fallan, 0 omitidas** (línea base 177 confirmada al
iniciar). `verify_summer_offline.py` **exit 0** con `PYTHONUTF8=1`. Evidencia firmada contra
`nota3-evidencia-2026-06-11`: **322 altas, 0 modificaciones, 0 borrados**; `git status` sobre
`experiments/` vacío. Nada bajo `experiments/results/` fue creado ni tocado: el directorio
`exp19b_anchored_selector` aún no existe y lo creará la primera corrida.

**Deuda detectada, no tocada (requiere decisión).** `src/utils/signed_evidence.py` protege
`exp3..exp14` y `exp8b`, pero **no** `exp15..exp19a`, que las reglas de trabajo declaran igualmente
intocables. Hoy esa protección depende solo de disciplina humana. No se modificó porque cambia una
guarda existente y su test.

[Claude Code] — 2026-08-21 02:30 — Modo: smoke de exp19b + guarda de evidencia + tag de rescate

Continuación de la entrada anterior. Rama `summer/exp19b`, ahora con **3 commits**. Sin push.

**Ollama.** El proceso colgado ya no era el PID 17412 (había muerto); el vivo era `ollama app`
**PID 26360**, cerrado con `taskkill`. Servidor levantado con la ruta completa
(`%LOCALAPPDATA%\Programs\Ollama\ollama.exe serve`), versión **0.22.1**, `granite4.1:8b`
presente. Ollama reporta actualización disponible a 0.32.15, **no aplicada**.

**Smoke de 3 queries: PASA las 4 etapas.** Determinismo 3x **bit-idéntico en ambos brazos**
(`all_arms_bit_deterministic: True`). Selector: sanity check **5,0/5**, `n_fallback=2`.
`results.json` con dos configs, cada una con `scenario`. Todo bajo
`experiments/results/exp19b_anchored_selector/_smoke/`, invisible al descubrimiento por forma.
Resume verificado: relanzar draft con el checkpoint completo no regeneró ninguna query.

**Hallazgo 1 — el borrador fresco NO reproduce las respuestas guardadas de exp18.** 1 de 3
idénticas; las otras dos divergen fuerte (jaccard-5grama **0,086** y **0,204**, ratio de
caracteres 0,27 y 0,36). Diagnóstico limpio: **`tokens_in` idéntico en 3/3** (1844 / 1925 / 790)
→ el prompt se reproduce **byte a byte**; `tokens_out` difiere (302→235, 211→134) → la
divergencia es **enteramente del runtime de generación** entre el 2026-07-31 y hoy. exp18 ya
registra `all_arms_bit_deterministic: False`: misma familia H5, compuerta relajada por diseño.
**Consecuencia para leer exp19b:** su brazo baseline no reproducirá las cifras de exp18, así que
solo vale el contraste pareado **dentro** de exp19b — que es exactamente lo que dice el
pre-registro. La guarda del ancla HHEM en 0,40-0,55 pasa a ser el control que importa. Y valida
la decisión Q1: releer el baseline de exp18 habría metido esta deriva dentro del pareado.

**Hallazgo 2 — defecto propio, cazado por el smoke.** `--stage regen` reconstruye `results.json`
y perdía `draft_vs_exp18_identical`, que solo calcula `--stage draft`: el sanity check corría, se
logueaba, y dejaba de existir donde alguien lo leería después. Corregido con `carry_forward()` y
dos tests; probado de punta a punta re-corriendo draft→regen sobre el artefacto real.

**Guarda de evidencia — registro invertido** (deuda de la entrada anterior, aprobada por Enzo).
Antes era una lista de lo-que-proteger, el mismo patrón que ya costó dos defectos a esta fase.
Ahora **todo dir `expN*` bajo `experiments/results/` está protegido contra sobrescritura salvo
los declarados en `LIVE_EXPERIMENTS`** (hoy `exp19b`). Una entrada LIVE obsoleta cuesta un
rechazo falso: ruidoso e inofensivo. La lista vieja costaba evidencia pisada: silenciosa y
permanente. `SIGNED_EXPERIMENTS` se conserva aparte y sigue significando *firmado*, porque el
mensaje de error cita el motivo real de cada dir (tag `nota3-evidencia` vs congelado por regla de
fase). **`guard_write` no cambia de semántica**: sigue lanzando solo si el archivo YA existe, así
que los runners de verano pueden seguir creando artefactos nuevos bajo exp15+; hay test propio
para esa propiedad. Verificado en vivo: exp19b escribible, exp18 y exp12 bloqueados con motivos
distintos y correctos.

**Limitación declarada:** los runners de verano no llaman a `guard_write`, así que la guarda no
los alcanza. Cubre los dos scripts que escriben en sitio y cualquier llamador futuro; ninguno de
los dos había escrito nunca bajo exp15..exp19a (cero archivos `faithfulness_metrics*` /
`retrieval_metrics*` allí), por lo que el cambio no altera ningún flujo existente. Proteger a los
runners exigiría decidir qué artefactos de verano son congelados y cuáles reescribibles: decisión
mayor, fuera de este alcance.

**Ramas.** Creado el tag `rescue/fase-3.5-pre-cleanup-2026-08-21` sobre `e8d2e2e`, punta de
`fase-3.5-nli-recompute-saved-answers`; rescata exactamente los **6 commits que solo existen en
este disco**. Sin push. **Ninguna rama borrada**: la tabla del Paquete 0 sigue esperando decisión
de Enzo rama por rama.

**Commits en `summer/exp19b`** (ninguno publicado):
1. `feat(verano): runner de exp19b — selector guiado por anclaje (G4)`
2. `ee9bd90 fix(guard): proteger exp15..exp19a invirtiendo el registro de evidencia`
3. esta entrada de log.

**Estado de auditoría.** Suite **218 pasan, 0 fallan, 0 omitidas** (177 → 206 → 208 → 218; +18
exp19b, +20 higiene de selector, +10 guarda). `verify_summer_offline.py` **exit 0**. Evidencia
firmada contra `nota3-evidencia-2026-06-11`: **322 altas, 0 modificaciones, 0 borrados**.

**Falta / no verificado.** (a) La corrida real de exp19b, ~5,3 h de GPU: la autoriza Enzo aparte,
y solo ella ejercita las etapas de puntuación (`--pass N`, HHEM, arm_stats, guards, diagnosis) que
aquí no se han corrido sobre datos reales. (b) El push: la compuerta sigue vigente. (c) El
directorio `experiments/results/exp19b_anchored_selector/_smoke/` queda **sin versionar** — son
respuestas desechables, no evidencia; borrarlo es decisión de Enzo. (d) Los 4 documentos sin
rastrear de Kimi Work siguen sin versionar, por decisión de Enzo. (e) El servidor de Ollama quedó
**corriendo**; ciérralo si no lo necesitas.
