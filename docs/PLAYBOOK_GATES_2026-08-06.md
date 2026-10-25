# Playbook de gates — cómo se cierra cada uno y quién hace qué

> **ARCHIVO HISTÓRICO (cerrado el 2026-08-30).** Conserva el plan operativo del
> 2026-08-06; no describe pendientes actuales. G1, G4 y G6 se ejecutaron, el repositorio
> tiene CI y lockfile, y el estado vigente está en `CLAUDE.md` y
> `docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md`.

> **Sección de [Kimi Work] — 2026-08-06 23:55 (hora local)**
> Documento operativo. Cada gate trae: diagnóstico, solución paso a paso, qué hace Enzo,
> qué se delega a un agente de código (Claude Code / Kimi Code / Codex) y el prompt listo
> para pegar cuando el gate toca código. Complementa a
> `docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md` (diagnóstico) — este es el plan de ejecución.
> Incluye al final la evaluación de las dos investigaciones externas (Gemini y Qwen)
> contra lo que el proyecto YA tiene hecho.

---

## G1 — Referencia humana piloto v4 (Etapa A: 150 · Etapa B: 50)

**Por qué es gate:** es el único árbitro independiente del nivel real de fidelidad
(¿0,30 NLI o 0,55 HHEM?) y de qué verificador queda como definitivo. Sin esto, ni el paper
ni el A.3 pueden afirmar un nivel de fidelidad validado.

**Quién lo cierra:** **Enzo (manual)**. No requiere tocar código. Si más adelante hace
falta un script de merge de tandas o un segundo anotador IA sellado, se delega (ver abajo).

**Qué tiene que hacer Enzo, paso a paso:** está en la guía dedicada
`docs/GUIA_ANOTACION_GOLD_V4.md` (leer antes de empezar la tanda A1).

**Qué se puede delegar (opcional, sin tocar el gold):**
1. **Script de merge de tandas → CSV** (si Enzo anota en un archivo plano en vez de
   directamente en el CSV). Media página de código; puede hacerlo cualquier agente o yo.
2. **Segundo juez IA sellado** (`claim_audit_sample_v4_kimi.csv`): yo anoto en un archivo
   SEPARADO y SELLADO que Enzo no abre hasta terminar sus 200 juicios. Desbloquea
   κ(Enzo, Kimi) gratis y detecta claims ambiguos. No contamina nada mientras permanezca
   sellado. **Requiere OK de Enzo.**
3. **MiniCheck-FT5 como candidato adicional** en `analyze_gold_v4.py` (sugerencia de las
   investigaciones externas; ver §Investigaciones). Solo tiene sentido **antes** de que
   exista la primera anotación humana analizada. Ver prompt P-MC.

**Criterio de cierre:** 150/150 + 50/50 con `juicio_humano` lleno, re-anotación de control
≥85 % de auto-acuerdo, y corrida real de `analyze_gold_v4.py` con reporte en
`output/audit/`.

---

## G2 — Congelar la configuración de encuestas (vence ~mediados de agosto)

**Por qué es gate:** todo lo de usuarios (G5) depende de qué sistema se pone frente al
participante. La decisión es **k=5 vs k=10** en la ruta de despliegue (`SURVEY_DEPLOY`),
más confirmar el modelo (ya decidido: granite4.1:8b) y la nube (G3).

**Estado verificado:** TTFT re-medido con granite en la ruta de despliegue (n=12/config):
k=5 → TTFT p50 12,8 s, total p50 179,8 s · k=10 → TTFT p50 15,2 s, total p50 132,7 s.
Dispersión enorme → la diferencia k=5/k=10 en latencia es ruido. Script existente:
`scripts/measure_survey_config_latency.py`. El hueco de transferencia exp18→despliegue
está cuantificado (balanceo cross-cloud actúa solo en 51/194 = 26,3 % de queries).

**Qué tiene que hacer Enzo (decisión, no código):**
1. Decidir el criterio: con nube (G3 GO) la latencia deja de ser limitante → **k=10** compra
   cobertura (answered 38,7 %→67,0 %, "no afirma nada" 3,1 %→0,5 %) sin costo de fidelidad
   (plana). Sin nube (G3 NO-GO) → **k=5** por latencia local.
2. Por tanto: **G2 se decide DESPUÉS de G3**, o se decide condicional: "k=10 si exp21 pasa,
   k=5 si no". Recomiendo la decisión condicional ya, y cerrar G3 cuanto antes.
3. Registrar la decisión en el ledger (entrada nueva) para que quede pre-registrada antes
   de las encuestas.

**Qué se delega:** nada por ahora. Si al congelar se quiere una re-medición con mayor n,
ejecutar `measure_survey_config_latency.py` (sin cambios de código); si el script necesita
retoques, se añade al prompt P-G3.

**Criterio de cierre:** decisión k=5/k=10 registrada en el ledger + `SURVEY_DEPLOY`
congelada (sin más cambios de perillas hasta terminar encuestas).

---

## G3 — Nube como infraestructura + compuerta `exp21_hosted_equivalence`

**Por qué es gate:** las encuestas remotas sin nube heredan un TTFT de 12,8-15,2 s (y
totales de ~2-3 min) que contaminaría el SUS/Likert. La causa está medida: el prefill
cruza la frontera CPU/GPU 114 veces (30/41 capas en GPU). Con 41/41 capas (~9 GiB VRAM)
una RTX 4090 de alquiler basta; costo estimado USD 6-14, techo USD 30.

**Estado verificado:** diseño completo en `docs/CLOUD_DEPLOYMENT_SURVEY.md`
(num_ctx=4096 fijo, exposición por sesiones, ~15-20 h GPU totales). La compuerta exp21 está
**pre-registrada en prosa pero NO tiene script**: no existe runner de equivalencia
(verificado 2026-08-06). **Cero gasto ejecutado** — nada se reserva sin OK de Enzo.

**Qué tiene que hacer Enzo:**
1. Dar el OK con el costo a la vista (o rechazar → encuesta local a k=5 y se documenta).
2. Elegir proveedor (RunPod u otro con RTX 4090) y crear la cuenta — esto es manual.
3. Supervisar el gasto: apagar el pod entre sesiones (dejarlo encendido = USD 59-235).
4. Decisión GO/NO-GO final leyendo el reporte de exp21.

**Qué se delega (código):** construir el harness de exp21 (script nuevo) +, si se quiere,
automatización del arranque/parada del pod. **Prompt P-G3 abajo.**

**Criterio de cierre:** exp21 ejecutada con TOST ±0,081 en los 3 verificadores + ancla
HHEM en 0,40-0,55 + digest del modelo verificado → veredicto escrito en el ledger.

### Prompt P-G3 (para Claude Code / Kimi Code / Codex)

```text
¿Puedes entender este proyecto entero antes de tocar nada? Lee en este orden: CLAUDE.md,
docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md, docs/CLOUD_DEPLOYMENT_SURVEY.md,
SUMMER_RESULTS.md (sección exp18 y "En curso / pendiente") y la entrada 24 de
paper/summer_ablation_log.md.

Tu tarea: construir el harness de la compuerta exp21_hosted_equivalence, que está
pre-registrada pero NO tiene script. Requisitos (del diseño pre-registrado, no los
redefinas):
1. Verificar el digest del modelo alojado contra el local ANTES de generar.
2. Correr las 194 queries sobre los contextos congelados de exp18
   (experiments/results/exp18_evidence_ceiling), apuntando al endpoint remoto de Ollama.
3. Sonda de determinismo 3x (patrón H5).
4. Devolver SOLO JSON; la puntuación queda en local con los 3 verificadores
   (NLI-small, NLI-base, HHEM), TOST bilateral banda ±0,081, familia BH declarada,
   ancla HHEM 0,40-0,55. NO se espera identidad bit a bit (local 30/41 capas, alojado
   41/41): el criterio es equivalencia declarada, no igualdad.
5. Reutiliza la infraestructura existente (run_exp18_ceiling.py, rescore_*, verify_*
   como referencia de patrones). ID de experimento nuevo: exp21_hosted_equivalence.
   Está PROHIBIDO escribir sobre experiments/results/exp3..exp18 ni re-puntuar artefactos
   firmados. Modo offline para los verificadores: HF_HUB_OFFLINE=1
   TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42, intérprete
   C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe.
6. El endpoint remoto debe ser configurable por variable de entorno; NUNCA hardcodees
   IPs, tokens ni URLs en el repo. Si necesitas secretos, documenta el nombre de la
   variable y para.

Reglas de trabajo (innegociables):
- Esta fase es primero exploración: antes de escribir código, repórtame qué archivos
  reutilizarás y tu diseño en una página, y espera mi OK.
- Cuando edites o crees archivos, firma tu sección con tu nombre, fecha y hora.
- Trabaja en una rama nueva (sugerencia: summer/exp21-harness) desde summer/mejoras.
- Sé incrédulo: audita tu propio trabajo antes y después (suite de tests +
  verify_summer_offline.py con PYTHONUTF8=1 deben seguir en verde; suite actual: 177
  pasan). No des nada por bueno solo porque corrió sin error.
- Ante ambigüedad que afecte cifras o el pre-registro, PREGÚNTAME en vez de asumir.
- No ejecutes ninguna corrida contra la nube ni gastes un centavo: el harness se entrega
  probado en local (con un mock del endpoint o contra el Ollama local) y la ejecución
  real la autorizo yo aparte.
- Al terminar, repórtame: qué se hizo, qué quedó pendiente, qué no pudiste verificar, y
  qué comando exacto debo correr yo cuando autorice el gasto.
```

---

## G4 — Decisión y ejecución de exp19b (selector guiado por anclaje)

**Por qué es gate:** es el candidato a segundo positivo de la fase. exp19a (offline, PASS)
mostró que reordenar el pool por (claim, chunk) con ms-marco-L12 sube la cobertura de
claims 0,4552→0,4853 (23 % del margen hasta la cota 0,5834) SIN verificador en el bucle.
exp19b es el brazo generativo: borrador → extraer claims → rerank por claim → regenerar.
Primaria pre-registrada: Δ fidelidad + TOST ±0,081 en los 3 verificadores, familia BH
declarada; la cota solo como motivación. Corre en la GPU local, costo cero.

**Estado verificado:** NO existe runner de exp19b (solo `compute_exp19a_selector_probe.py`
y el diseño en el ledger, entradas 19-24). Decisión de Enzo pendiente.

**Qué tiene que hacer Enzo:**
1. Decidir GO/NO-GO (recomendado GO: gratis, local, pre-registrado, y es la única palanca
   con señal después de exp17).
2. Reservar la ventana de GPU (una corrida de generación, ~1-2 noches según n).
3. Leer el resultado contra el pre-registro, no contra la esperanza: si TOST dice
   equivalente o peor, se reporta como nulo/negativo igual que exp16 — sin re-interpretar.

**Qué se delega (código):** el runner de exp19b. **Prompt P-G4 abajo.**

**Criterio de cierre:** artefactos de exp19b bajo `experiments/results/exp19b_*` (ID nuevo
exp15+), análisis con los 3 verificadores, entrada de ledger con veredicto, y
`verify_summer_offline.py` extendido o confirmado en verde.

### Prompt P-G4 (para Claude Code / Kimi Code / Codex)

```text
¿Puedes entender este proyecto entero antes de tocar nada? Lee en este orden: CLAUDE.md,
docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md, SUMMER_RESULTS.md (secciones exp18 y
exp19a completas) y las entradas 19 a 24 de paper/summer_ablation_log.md, que contienen
el pre-registro de exp19b.

Tu tarea: implementar el runner de exp19b (brazo generativo del selector guiado por
anclaje). Del pre-registro, sin redefinirlo:
1. Flujo del brazo: (a) generar borrador con el contexto baseline de exp18; (b) extraer
   los claims del borrador con el extractor de claims EXISTENTE del pipeline de
   fidelidad; (c) reordenar el pool k=50 por (claim, chunk) con ms-marco-L12 replicando
   el método de compute_exp19a_selector_probe.py; (d) regenerar con el nuevo top-5.
   Baseline = el brazo baseline de exp18, pareado within-session.
2. Primaria: Δ fidelidad vs baseline en los 3 verificadores (NLI-small, NLI-base, HHEM)
   + TOST bilateral ±0,081 + familia BH declarada antes de correr. La cota de selección
   (0,5834) se cita solo como motivación, NUNCA como denominador de "% recuperado".
3. Guardas anti-gaming obligatorias (mismo patrón que exp16/exp17): tasa de declinación
   con classify_response, asserts_nothing_rate, palabras, claims por respuesta, solape
   jaccard-5grama y reaparición de claims del borrador.
4. Determinismo: granite4.1:8b, temp 0, seed 42, modo offline (HF_HUB_OFFLINE=1
   TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42), intérprete
   C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe. Cache de LLM
   documentada (qué se reutiliza y qué NO para que el borrador y la regeneración sean
   comparables). Sonda de determinismo 3x al inicio.
5. ID nuevo: exp19b_* bajo experiments/results/. PROHIBIDO tocar exp3..exp19a. La
   generación nueva solo existe bajo IDs exp15+, según las restricciones del repo.
6. Checkpoint/resume por query como run_generation_matrix.py, porque la corrida es larga
   y puedo necesitar reanudar.

Reglas de trabajo (innegociables):
- Primero exploración: antes de escribir código, repórtame el diseño en una página
  (qué scripts reutilizas, qué artefactos produces, cuál es la familia BH exacta y el n)
  y espera mi OK.
- Firma tus secciones con nombre, fecha y hora en todo archivo que edites o crees.
- Trabaja en rama nueva (sugerencia: summer/exp19b) desde summer/mejoras.
- Sé incrédulo y audítate antes y después: suite de tests en verde (actual: 177 pasan),
  verify_summer_offline.py exit 0 con PYTHONUTF8=1, y verificación de que ningún archivo
  de evidencia firmada cambió (git diff --name-status sobre experiments/results).
- Ante ambigüedad que afecte cifras o el pre-registro, PREGÚNTAME. No asumas.
- La corrida de generación la lanzo yo cuando me entregues el runner validado con un
  smoke test de 2-3 queries. No la lances tú sin mi OK.
- Al terminar repórtame: qué se hizo, qué falta, qué no pudiste verificar, y el comando
  exacto de lanzamiento y de reanudación.
```

---

## G5 — Encuestas SUS/Likert (ejecución con usuarios)

**Por qué es gate:** es la validación con usuarios y la última milla de la tesis.
Depende de G2 (config congelada) y G3 (si hay remoto).

**Quién lo cierra:** Enzo. No hay código nuevo en el repo salvo imprevistos.

**Checklist de Enzo:**
1. Instrumento: SUS (10 ítems) + Likert de utilidad/confianza por respuesta. Definir
   cuántas respuestas evalúa cada participante y con qué queries (¿las 194? ¿submuestra
   estratificada? — decisión que conviene pre-registrar).
2. Reclutamiento: población objetivo del A.1 (profesionales DevOps/cloud, estudiantes,
   postulantes a certificaciones). n objetivo y criterio de inclusión.
3. Ética: consentimiento informado, datos mínimos, anonimización (revisar si la
   universidad exige formato específico — preguntar al asesor).
4. Piloto con 2-3 participantes antes del estudio completo (detecta fricción de la UI y
   del TTFT percibido).
5. Logística con nube (si G3 GO): ventanas de sesión, encendido/apagado del pod,
   contingencia si cae el servicio a mitad de encuesta.
6. Análisis: `scripts/analyze_user_sessions.py` ya existe — verificar que cubre SUS antes
   de las encuestas, no después.

**Criterio de cierre:** datos de encuestas recolectados, analizados y reportados.

---

## G6 — Taxonomía de los 759 claims sin respaldo

**Por qué es gate (menor):** interpreta el ~54 % de claims no anclados y alimenta la
discusión del paper (síntesis legítima vs memoria paramétrica vs alucinación vs fallo del
verificador). Ya hay muestra exportada (`output/audit/unsupported_claims_sample.csv`) y
script (`scripts/compute_exp18_unsupported_taxonomy.py`).

**Qué tiene que hacer Enzo:** decidir la profundidad (¿taxonomía completa o muestra
representativa?) y revisar/adjudicar las categorías ambiguas.

**Qué se delega:** iterar el análisis offline sobre artefactos persistidos. Prompt corto:

### Prompt P-G6

```text
¿Puedes entender este proyecto entero antes de tocar nada? Lee CLAUDE.md,
docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md y la sección F5 / "759 claims" de
paper/summer_ablation_log.md (entradas 22-23), más
experiments/results/exp18_evidence_ceiling/unsupported_taxonomy.md si existe.

Tu tarea: completar la taxonomía de los 759 claims sin respaldo usando SOLO artefactos
persistidos (cero generación nueva, modo offline). Parte de
scripts/compute_exp18_unsupported_taxonomy.py y de
output/audit/unsupported_claims_sample.csv: verifica primero qué produce hoy ese script,
si corre con el intérprete 3.14, y qué falta para llegar a las 4 categorías del diseño
(síntesis legítima / memoria paramétrica / alucinación / fallo del verificador) con
conteos, ejemplos representativos por categoría y un muestreo auditable (seed 42).

Reglas: primero repórtame qué encontraste en el script y tu plan en media página y espera
mi OK; firma tus secciones con nombre, fecha y hora; rama nueva si tocas código
(sugerencia: summer/taxonomia-759); PROHIBIDO modificar experiments/results/exp3..exp19a
(escribe solo archivos _vN nuevos si recalculas); audítate antes y después (suite en
verde, verify_summer_offline.py exit 0 con PYTHONUTF8=1); ante ambigüedad pregúntame;
al terminar repórtame qué se hizo, qué falta y qué no pudiste verificar.
```

---

## G7 — Confirmatorio pre-registrado de exp17 (opcional)

**Por qué es gate (opcional):** exp17 es el único positivo de selección pero con n=25 no
cruza significancia de forma robusta (GLMM p=0,021 una-cola; bootstrap por query no
confirma). Un confirmatorio con queries NUEVAS pre-registradas lo resolvería.

**Qué tiene que hacer Enzo:** decidir sí/no. Mi recomendación: **posponer hasta después
del gold (G1)** — si el gold cambia el verificador definitivo, el confirmatorio debe
correr con ese verificador, no con los actuales.

**Si se decide sí:** delegar con un prompt del mismo molde que P-G4 (queries nuevas
pre-registradas, MISMO diseño de brazos que exp17, n calculado por potencia sobre el
efecto HHEM +0,081, familia BH declarada antes de generar). Lo redacto completo cuando
lo autorices.

---

## Deuda documental y de empaquetado (transversal, no numerada)

**Qué es:** CLAUDE.md con secciones vivas desactualizadas (exp18, 177 tests, paridad
publicada); versiones de Python contradictorias (badge 3.14 / texto 3.10+ / setup.py
3.11+ / REPRODUCE 3.14) con decisión ya tomada (separar soporte de paquete de entorno
reproducible); `requirements.txt` sin Streamlit/Plotly/pytest y sin lockfile;
`REPRODUCE.md` sin `PYTHONUTF8=1`; README sin la fase de verano; texto canónico de la
contradicción de prompt routing; test NLI omitido offline pese a snapshot local.

**Qué tiene que hacer Enzo:** autorizar la ejecución (las decisiones ya están tomadas en
su mayoría; el resto las responde en 5 minutos).

### Prompt P-DOC

```text
¿Puedes entender este proyecto entero antes de tocar nada? Lee CLAUDE.md completo
(incluida la entrada [Kimi Work] del 2026-08-06 al final, que lista exactamente qué
secciones quedaron desactualizadas), docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md §7 y
la entrada de Kimi Code del 2026-08-04 en CLAUDE.md (decisiones de Enzo ya registradas).

Tu tarea es SOLO mantenimiento documental y de empaquetado, en rama nueva
(sugerencia: summer/mantenimiento-docs) desde summer/mejoras:
1. CLAUDE.md: actualiza las secciones vivas "Estado actual" y "Contradicciones sin
   resolver" que la entrada [Kimi Work] marca como obsoletas (exp18 cerrada, suite 177,
   paridad publicada en origin/summer/mejoras). Conserva el log append-only intacto.
2. Versiones de Python: ejecuta la decisión ya tomada — README/setup.py declaran soporte
   del paquete (3.11+) y REPRODUCE.md declara el entorno experimental reproducible
   (3.14); corrige el badge y el texto 3.10+ del README. NO elijas una versión nueva.
3. REPRODUCE.md: agrega PYTHONUTF8=1 a la receta PowerShell del verificador v4 (falla en
   consola cp1252 sin ella, documentado en CLAUDE.md).
4. Manifiestos: propón (antes de ejecutar) cuál es canónico entre requirements.txt y
   setup.py, con Streamlit/Plotly/pytest como extras o en requirements; genera lockfile
   solo si te lo autorizo tras ver tu propuesta.
5. Texto canónico de la contradicción de prompt routing (TRACEABILITY vs KNOB_MAP): la
   re-auditoría del 2026-08-04 concluyó que ambas son correctas en su contexto; redacta
   la versión canónica en AMBOS documentos citando esa conclusión, sin tocar código ni
   tests de paridad.
6. NO toques: paper/overleaf_ready/main.tex, prosa A.3, experiments/, ni el test NLI
   omitido (ese es un hilo separado, no lo mezcles aquí).

Reglas: repórtame primero el plan de ediciones archivo por archivo y espera mi OK; firma
tus secciones con nombre, fecha y hora; audítate antes y después (suite 177 en verde,
verify_v4_offline.py y verify_summer_offline.py exit 0 con PYTHONUTF8=1); ante ambigüedad
pregúntame en vez de asumir; al terminar repórtame qué se hizo, qué falta y qué no
pudiste verificar.
```

---

## Investigaciones externas (Gemini y Qwen) vs. lo que el proyecto YA tiene

> Evaluación escéptica de [Kimi Work]. Ambas investigaciones son serias y bien
> referenciadas (FActScore, MiniCheck, RAGChecker, TOST/SESOI, CRAG, RAPTOR/SVD-RAG,
> reproducibilidad GGUF/lockfile, OpenTelemetry), pero **no vieron el código** y por eso
> recomiendan varias cosas que ya existen. Lo honesto es separar:

**Ya cubierto por el proyecto (no rehacer):**
- Evaluación a nivel de claim (paradigma FActScore): ES la unidad de todo el pipeline de
  fidelidad desde exp12; el gold mismo es claim-level.
- Gold ciego de dos etapas: diseñado y construido (v4 A+B, seed 42, estratos ocultos).
  Falta solo la anotación (G1).
- Circularidad del oráculo: resuelta con bge-reranker-large independiente (exp11) y
  re-validada en exp18 (cota con discrepancia 0,0000 en 188 queries).
- Lost-in-the-middle / edge-placement: TESTEADO y descartado (Tier A: 0/3 instrumentos;
  orden y posición no mueven la fidelidad). La recomendación de edge-placement de Gemini
  ya tiene respuesta empírica negativa en este sistema.
- Filtrado de respuestas vacías / abstención: tratado con denominadores decline-aware y
  `asserts_nothing_rate` (defecto #7: la tasa real de "no afirma nada" es 3,1 %, no 45,9 %).
- TOST con banda ±0,081: ya es el estándar de exp18/exp21, con la banda declarada antes
  de correr.
- Selección de contexto por cobertura/diversidad: exp17 (positivo) y exp18/19a (cota y
  rerank por claim) son exactamente esta línea.
- Determinismo: temp 0 + seed 42 + hashseed documentados; no-determinismo de gemma4/mistral
  medido y declarado en MODELS.md; caché congela la muestra.

**Genuinamente nuevo y accionable (en orden de valor):**
1. **MiniCheck-FT5 (770M) como candidato adicional de verificador** (ambas lo proponen).
   Encaja natural en `analyze_gold_v4.py` como un candidato más de la familia BH.
   **Decisión con fecha: solo sirve si se integra ANTES de analizar el gold** (después
   sería elegir candidatos con el resultado a la vista). Bajo costo, local. Ver P-MC.
2. **Segundo/tercer anotador e índice de acuerdo inter-anotador** (ambas: κ/α entre
   humanos). El diseño actual es un solo anotador (Enzo). Mitigaciones baratas:
   re-anotación intra-anotador (ya incluida en la guía) y el segundo juez IA sellado
   (oferta abierta, G1). Un segundo humano sería lo ideal si hay alguien disponible.
3. **Justificación formal del SESOI ±0,081** (Gemini): hoy viene del piloto exp17. Un memo
   corto en el ledger derivándolo (bootstrap del piloto + criterio de efecto mínimo de
   interés) cierra la crítica sin tocar experimentos. Se puede añadir a P-DOC como ítem.
4. **Lockfile y CI** (ambas): ya es deuda registrada; entra en P-DOC (lockfile) y queda
   CI como mejora post-encuestas.
5. **OpenTelemetry / observabilidad** (Qwen): útil pero NO bloquea nada; anotar como
   trabajo futuro, no antes de las encuestas.
6. **RAPTOR/SVD-RAG, reconstrucción del corpus, KG de servicios** (ambas): largo plazo,
   post-tesis. El desbalance del corpus (AWS 26 %/Azure 39 %/GCP 35 %) conviene
   DECLARARLO como limitación en la discusión; re-balancearlo ahora invalidaría la
   comparabilidad con toda la evidencia firmada.

**Lo que NO adoptaría:** búsqueda web correctiva de CRAG (fuera de dominio cerrado y de la
pregunta de investigación), LLM-juez grande como verificador principal (introduce un juez
no calibrado nuevo justo cuando el gold está por calibrar los actuales) y re-chunkear el
corpus (rompe la cadena de evidencia).

### Prompt P-MC (opcional, SOLO si se decide antes de analizar el gold)

```text
¿Puedes entender este proyecto entero antes de tocar nada? Lee CLAUDE.md,
docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md, la cabecera de
scripts/analyze_gold_v4.py (contrato de candidatos y familia BH) y
scripts/compute_exp15_ensemble_sweep.py (label_one compartido).

Tu tarea: agregar MiniCheck-FT5 (Tang et al. 2024, 770M) como candidato adicional de
verificador en el análisis del gold v4, ANTES de que exista cualquier análisis con
anotaciones humanas. Requisitos:
1. Puntuar los 150 claims de la etapa A y los 50 de la etapa B con MiniCheck sobre las
   MISMAS premisas que ven los verificadores actuales (1 chunk en A, 5 chunks max en B),
   en modo offline, seed 42, intérprete 3.14. Persistir probabilidades como artefacto
   nuevo bajo experiments/results/exp15_ablation_nli/ (archivos _vN nuevos, nada se
   sobrescribe).
2. Integrarlo a la lista de candidatos de analyze_gold_v4.py con la misma disciplina que
   los demás: candidato declarado, familia BH una por candidato, y si el modelo no está
   disponible que el gap se REPORTE, no que desaparezca en silencio (patrón ya escrito
   en la cabecera del script).
3. Verificación: smoke test con --simulate y corrida real SOLO cuando el gold esté
   anotado (esa corrida la autorizo yo).

Reglas: primero repórtame el diseño en media página y espera mi OK; firma tus secciones
con nombre, fecha y hora; rama nueva (sugerencia: summer/minicheck-candidate); audítate
antes y después (suite 177 en verde, verificadores offline exit 0 con PYTHONUTF8=1);
ante ambigüedad pregúntame; al terminar repórtame qué se hizo, qué falta y qué no
pudiste verificar.
```

---

## Orden sugerido de ejecución (resumen)

| Semana | Acción | Quién |
|---|---|---|
| Ahora | Empezar G1 tanda A1 · decidir G3 (OK de gasto) y G4 (GO/NO-GO) | Enzo |
| Esta semana | Lanzar P-G4 (runner exp19b) y P-G3 (harness exp21) en paralelo, son agentes distintos o secuenciales | Agente(s) de código |
| Esta semana | Lanzar P-DOC (mantenimiento documental) | Agente de código |
| Próxima semana | Ejecutar exp21 si el harness está listo y el gasto autorizado → decisión k=5/k=10 (G2) | Enzo + agente |
| Continuo | Tandas A2-A5 y B1-B2 del gold | Enzo |
| Tras gold | Corrida real de analyze_gold_v4.py → verificador definitivo → reescritura A.3/paper | Enzo + agente |
| Post-encuestas | CI, observabilidad, corpus (largo plazo) | — |

*Fin de la sección de [Kimi Work].*
