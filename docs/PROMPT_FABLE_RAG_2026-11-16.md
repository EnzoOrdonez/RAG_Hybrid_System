# Prompt maestro: auditoría y endurecimiento del código RAG (Fable 5.2)

Preparado el 2026-11-16 a pedido de Enzo. El texto entre los marcadores es el prompt que se pega
en la sesión de Fable. Las tres decisiones D1 a D3 traen el valor recomendado. Enzo puede
cambiarlas antes de pegar el prompt.

## Cómo lanzarlo (para Enzo)

1. En la app, en la pestaña Code, abre una sesión nueva en la carpeta
   `C:\Users\enziz\Projects\hybrid-rag-system\.worktrees\fable-rag-auditoria`. Ábrela tal cual,
   sin que la app cree otro worktree.
   Esa carpeta ya está lista:
   - rama `fable/rag-auditoria`, cortada de `1574d22`;
   - uniones `.venv-app`, `data/embeddings`, `data/indices` y `data/models`, a los mismos destinos
     que usa el resto del proyecto.
2. Elige el modelo Fable 5.2 y el esfuerzo más alto que acepte tu cuota.
3. Pega el bloque de abajo, desde «INICIO DEL PROMPT» hasta «FIN DEL PROMPT».

---- INICIO DEL PROMPT ----

# Tarea: auditar y endurecer el código RAG de CloudRAG, sin tocar el despliegue

## Decisiones de Enzo (vigentes para toda la sesión)

- **D1, ejecución de modelos:** no se ejecuta ningún modelo. Eso incluye no cargar Ollama, BGE,
  el cross-encoder, NLI ni HHEM, y no descargar nada. Solo valen tests de CPU con dobles (fakes) y
  la relectura de salidas firmadas. (Alternativa, si Enzo la escribe aquí: se permiten los
  encoders locales sin Ollama, con 20 consultas fijas como máximo, solo para pruebas de
  equivalencia.)
- **D2, cambios que alteran resultados (clase B):** solo se proponen y nunca se aplican.
- **D3, publicación:** sin push, sin merge y sin PR. Todo queda en commits locales de
  `fable/rag-auditoria`.

## Objetivo

Dejar el código RAG (`src/`, salvo lo excluido abajo) en su mejor versión verificable:
- encontrar defectos reales de corrección, con evidencia;
- corregir con TDD los que no cambian ningún resultado;
- documentar los demás para que Enzo decida.

La calidad pesa más que la cantidad. Un hallazgo vale si se demuestra con un test o con una traza
de código concreta. No reportes nitpicks de estilo.

## Dónde trabajas y por qué hay una rama aparte

- **Tu carpeta:** `C:\Users\enziz\Projects\hybrid-rag-system\.worktrees\fable-rag-auditoria`, en la
  rama `fable/rag-auditoria` con base `1574d22`. Al empezar, comprueba que
  `git rev-parse --abbrev-ref HEAD` da `fable/rag-auditoria`. Si no, detente y avisa.
- **Por qué una rama aparte:** en paralelo, otra sesión opera la fase humana del despliegue I7.
  - El ensayo P998 corre sobre una VM cuya imagen es `b5e54f1`.
  - El operador `i7` corre desde el worktree fijo `.worktrees/i7-human-phase-603fa49`.
  - `verify_tool_source` rechaza cualquier checkout cuyo diff contra `b5e54f1` salga de `docs/`
    y `output/audit/`.

  Tu rama no llega a la VM ni a ese worktree. Lo que hagas solo se integra después, cuando Enzo lo
  decida. Llevarlo a la web exigiría otra Fase B pagada.

## Límites duros (no negociables)

1. **Otros worktrees y ramas:** no toques `.worktrees/i7-human-phase-603fa49`,
   `.worktrees/i7-human-ledger-ready`, `.worktrees/interview-readiness`,
   `.worktrees/overnight-cleanup` ni el checkout principal.
   - No uses `git worktree add`, `git worktree remove` ni `git worktree prune`.
   - No cambies de rama en tu carpeta.
   - No uses `git stash` sin nombre: la pila de stash es compartida.
   - Los commits van solo a `fable/rag-auditoria`. No hay rebase, tags, ramas borradas ni push.
2. **El entorno es compartido.** `.venv-app` es una unión al intérprete que también usa el operador
   I7.
   - Nada de `pip install`, `uninstall` ni `upgrade`, y nada de escribir dentro de `.venv-app`.
   - Si falta una dependencia (por ejemplo `pytest-cov`), anótalo y sigue sin ella.
   - `data/embeddings`, `data/indices` y `data/models` son de solo lectura.
3. **Nada de nube ni de operación:**
   - no uses `gcloud` ni Chrome, y no toques tareas programadas;
   - no escribas en `C:\CloudRAG\` ni en `D:\CloudRAG_archivo\`. `init.sh` lee el runtime del
     navegador en `C:\CloudRAG\i7-browser-runtime`, y eso sí está permitido.
4. **Fuera de alcance** (despliegue y estudio):
   - `scripts/study_operator/**`, `scripts/*i7*`, `scripts/study_*`, `scripts/*gate*` y sus tests;
   - `src/ui/**`, `src/utils/deployment_artifacts.py`, `src/utils/checkout_boundary.py` y `config/`;
   - las tareas F-005, F-016, F-020 y F-021;
   - F-009 (Ruff), que ya avanza en la rama `chore/f009-ruff-scanner`. No hagas limpiezas de Ruff.
5. **Evidencia y registros inmutables:** no escribas en `experiments/results/**`, `paper/**` ni en
   los archivos existentes de `output/audit/**`.
   - No debilites `src/utils/signed_evidence.py`: `LIVE_EXPERIMENTS` sigue vacío.
   - No edites `.harness/progress.md`, `.harness/feature_list.json`, `docs/decisions/**` ni
     `CLAUDE.md`. La sesión de despliegue los edita en paralelo y el merge chocaría. Lo que haría
     falta allí va como texto propuesto en tu informe.
6. **Registros de configuración congelados:** no cambies valores por defecto ni el comportamiento
   de:
   - `PROPOSED_HYBRID` (registro del paper LACCI, ADR-0005);
   - `SURVEY_DEPLOY` (lo desplegado);
   - `BASELINE_LEXICAL`, `BASELINE_SEMANTIC` y `LLM_ONLY_NO_RAG`;
   - cualquier configuración que hayan usado `exp3..exp19b`.

   La lógica RAG está congelada (ADR-0004). Solo se aplica lo que demuestres equivalente (clase A).
7. **Privacidad y secretos:** el repo es público. Nada de IPs, hosts, correos, tokens, IDs de
   proyecto ni datos de participantes.
   - No leas ni imprimas `.env`.
   - Entorno de cada comando Python: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
     PYTHONUTF8=1`, con el intérprete `.venv-app/Scripts/python.exe` (3.14, no el 3.11 del PATH).
8. **Commits:** sin líneas de atribución, en formato `tipo: descripción` en español, uno por cambio.

## Dos clases de cambio

- **Clase A, se aplica:** en todas las entradas que hoy funcionan, el resultado sigue idéntico bit
  a bit, en cualquier configuración usada por la evidencia o por el despliegue. Eso incluye:
  - ids, orden y puntajes recuperados;
  - el texto del prompt;
  - el parseo de la respuesta;
  - las métricas calculadas sobre las salidas firmadas.

  Entran aquí:
  - tests nuevos;
  - manejo de errores en entradas que hoy lanzan una excepción (documenta el caso);
  - tipos, docstrings, mensajes y logs;
  - código muerto, con prueba de que nadie lo importa;
  - rendimiento sin cambio de salida.
- **Clase B, solo se propone:** cualquier cambio en salidas, rankings, prompts, umbrales, patrones,
  métricas o cifras publicadas. Cada uno entrega cuatro cosas:
  1. un test que muestra el defecto, marcado
     `@pytest.mark.xfail(strict=True, reason="FABLE-B<n>: pendiente de decisión de Enzo")`, con
     commit;
  2. el parche, sin aplicar, en `docs/reports/fable-rag-2026-11/patches/B<n>.diff`;
  3. el impacto: qué configuraciones, experimentos y cifras toca. Estímalo releyendo las salidas
     firmadas, sin escribir en ellas;
  4. la corrida nueva que haría falta, con un ID `exp20+`.
- **Ante la duda:** el cambio es de clase B.

## Fases (en orden; un commit y una actualización del informe por cada paso)

Tu cuota es finita. Después de cada hallazgo o arreglo, haz commit y actualiza el informe, para que
un corte deje un estado útil. Trabaja por prioridad, no por orden de archivo.

**F0. Línea base, sin cambios de código.**
1. Copia sin cambios
   `C:\Users\enziz\Projects\hybrid-rag-system\.worktrees\i7-human-ledger-ready\docs\PROMPT_FABLE_RAG_2026-11-16.md`
   a `docs/` de tu rama y haz el primer commit. Es la única lectura permitida en otro worktree.
2. Lee `CLAUDE.md`, `docs/guides/guardrails.md`, `docs/architecture/system-design.md`,
   `docs/architecture/models.md`, ADR-0004, ADR-0005 y ADR-0009 en `docs/decisions/adr-log.md`, y
   `docs/KNOB_MAP_summer.md`.
3. Lee en solo lectura `paper/audit_findings_cc_addenda.md` (N1 a N9) y `paper/correction_log.md`.
   Un flag ya registrado o cerrado no es un hallazgo nuevo. Ejemplo: el bug de mayúsculas de
   `DECLINE_PATTERNS` ya está corregido.
4. Ejecuta `bash init.sh` desde Git Bash. Anota los exit de las tres capas y los conteos.
5. Ejecuta `scripts/verify_v4_offline.py` y `scripts/verify_summer_offline.py`. Antes, lee cada uno
   y confirma que no carga modelos. Guarda en el informe el SHA-256 de su salida: es la referencia
   dorada.

**F1. Auditoría, de solo lectura, por prioridad.** Escribe cada hallazgo en el informe con: id,
severidad, clase, `archivo:línea`, evidencia y estado.
- **P1, cadena que produce las cifras del paper:**
  - `src/evaluation/`: `hallucination_metrics`, `generation_metrics`, `retrieval_metrics`,
    `statistical_analysis`, `decline_classifier`, `study_statistics`, `study_analysis`,
    `benchmark_runner` y `results_exporter`;
  - `src/generation/hallucination_detector.py`: agregación NLI/HHEM, denominadores, casos vacuos
    (N9) y exclusiones por `method`.
- **P2, camino de respuesta de `SURVEY_DEPLOY`:**
  - `src/pipeline/` y `src/retrieval/`: fusión RRF, empates, duplicados, `k` y
    `coverage_balancer`;
  - `src/reranking/`;
  - `src/generation/`: `prompt_templates`, `response_formatter` (citas y declinación) y
    `llm_manager` (timeouts, reintentos y errores).
- **P3, cadena de construcción del índice:** `src/chunking/`, `src/embedding/`,
  `src/preprocessing/` y `src/ingestion/`. Aquí cualquier cambio de salida es de clase B, porque
  exige reindexar.
- **P4, utilidades:** `src/utils/reproducibility.py`, `src/utils/local_models.py` y
  `src/utils/signed_evidence.py`. Solo se refuerzan, nunca se relajan.

Severidad:
- **CRÍTICO:** cifra o evidencia incorrecta, o riesgo para la evidencia firmada.
- **ALTO:** comportamiento incorrecto en el camino desplegado.
- **MEDIO** y **BAJO:** el resto.

**F2. Arreglos de clase A, con TDD.**
1. Escribe el test y confirma que falla (rojo).
2. Aplica el cambio mínimo y confirma que pasa (verde).
3. Corre los tests del módulo.
4. Haz el commit.

Para cambios en P1, vuelve a correr los dos verificadores offline: el SHA-256 de su salida debe
seguir igual a la línea base.

**F3. Propuestas de clase B**, con los cuatro entregables descritos arriba.

**F4. Huecos de tests** en los caminos P1 y P2 que no tengan tests de comportamiento. Mide la
cobertura con `pytest-cov` solo si ya está instalado.

**F5. Cierre.**
- `bash init.sh` con exit 0 en las tres capas. Si sale distinto de 0, se arregla la causa: nunca
  se desactiva un check.
- Los verificadores offline coinciden con la línea base.
- `git diff --stat 1574d22..HEAD -- experiments paper config src/ui scripts/study_operator .harness`
  sale vacío.
- `python scripts/scan_secrets.py --baseline secrets_baseline.json` da exit 0.

## Entregables

1. El informe `docs/reports/fable-rag-auditoria-2026-11.md`. Empieza con la cabecera de telemetría
   de `CLAUDE.md` y contiene:
   - la línea base de F0;
   - la tabla de hallazgos;
   - los commits de clase A, cada uno con su prueba de equivalencia;
   - las propuestas de clase B;
   - las preguntas abiertas para Enzo;
   - el texto propuesto, sin número, de un ADR y de una entrada de `feature_list.json`;
   - lo que no se verificó y por qué.
2. Commits locales en `fable/rag-auditoria`, sin push.
3. Un mensaje final con:
   - la cabecera de telemetría;
   - un resumen de 10 líneas como máximo;
   - la lista de decisiones que debe tomar Enzo, una por cada propuesta B;
   - el comando para revisar: `git log --oneline 1574d22..fable/rag-auditoria`.

## Cuándo preguntar

Usa AskUserQuestion ante algo ambiguo o contradictorio, o ante cualquier cosa que roce un límite
duro. Enzo puede no estar presente. En ese caso, no te bloquees: anota la pregunta en el informe,
trata el caso como clase B y sigue con el siguiente punto.

---- FIN DEL PROMPT ----
