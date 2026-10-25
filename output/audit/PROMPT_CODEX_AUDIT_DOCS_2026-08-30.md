# PROMPT MAESTRO — Codex — Auditoría exhaustiva de documentación, SOLO LECTURA (2026-08-30)

Auditoría documental exhaustiva del repo `C:\Users\enziz\projects\hybrid-rag-system`.
El paper fue aceptado en IEEE LACCI 2026 y el repo es su artefacto público; la
documentación debe quedar sin cabos sueltos.

## REGLAS DURAS (incumplir cualquiera = abortar y reportar)

- **MODO SOLO LECTURA.** No modifiques, muevas, borres ni crees archivos, EXCEPTO tu
  único artefacto de salida: `output/audit/docs_audit_codex_2026-08-30.md` (nuevo).
- No hagas commits, no hagas push, no cambies de rama, no crees ramas ni tags.
  Rama actual esperada: `summer/taxonomia-759` o `main`; verifícala con
  `git branch --show-current` y NO la cambies.
- No ejecutes experimentos, modelos, GPU, Ollama, NLI, HHEM ni redes externas.
- No abras archivos `.env`, credenciales ni claves.
- No leas los CSV del gold (`output/audit/claim_audit_sample_v4*.csv`,
  `unsupported_claims_sample_v2.csv`) — cegamiento ya consumado, no hace falta.
- Puedes correr tests SOLO si es estrictamente necesario para verificar una afirmación
  documental (usa el intérprete `C:\Users\enziz\AppData\Local\Python\pythoncore-3.14-64\python.exe`,
  `PYTHONUTF8=1`, marcadores `not slow and not gpu`). En principio NO hace falta.

## Pre-flight (reportar en el artefacto)

1. `git branch --show-current`
2. `git status --short --untracked-files=all`
3. `git log --oneline -3`

## Tarea: auditoría exhaustiva de TODA la documentación

Archivos a auditar (todos los .md de raíz y docs/, más CITATION.cff):

- Raíz: `README.md`, `CLAUDE.md`, `MODELS.md`, `REPRODUCE.md`, `SUMMER_RESULTS.md`,
  `RESULTADOS_RESUMEN.md`, `NOTA3_NEXT_STEPS.md`, `CITATION.cff`
- `docs/`: ESTADO_PROYECTO_UNIFICADO_2026-08-06.md, FICHA_EXPERIMENTAL.md,
  SECCION_VALIDACION_HUMANA.md, LIMITACIONES_Y_TRABAJO_FUTURO.md, TRACEABILITY_nota3.md,
  APP_VS_EXPERIMENTO.md, BRANCH_INVENTORY_2026-08-23.md, MANIFESTS_PROPOSAL_2026-08-21.md,
  KNOB_MAP_summer.md, GUIA_ANOTACION_GOLD_V4.md, PLAYBOOK_GATES_2026-08-06.md,
  CLOUD_DEPLOYMENT_SURVEY.md, CLOUD_EXPERIMENT_DESIGN.md
- `experiments/probes/runtime_noise/README.md`

### Chequeos (todos, sin excepción)

1. **Enlaces internos**: verifica que CADA ruta relativa enlazada en esos MD exista en
   disco (scripts, docs, outputs, datos). Lista cada enlace roto con archivo:línea.
2. **Conteos y cifras**: cualquier conteo de experimentos distinto de 19 carpetas
   exp3..exp19b+exp8b, o de tests distinto de 341+, o cifras que contradigan:
   corpus 24.481 chunks (AWS 6.366/Azure 9.606/GCP 8.509), NDCG@5 0,740/0,995,
   194 queries, 4 LLMs, gold v4 (150 claims / 200 juicios / κ=0,30 / κ₂=0,754 /
   0,17–0,20 / 55 %), exp19b Δ=+0,0451 IC95 [0,0076; 0,0817] TOST ±0,081.
   Cuando un número no se pueda verificar sin ejecutar nada, márcalo "no verificable
   en solo lectura" — no lo des por bueno ni por malo.
3. **Afirmaciones de pendiente ya cumplidas**: "future work", "pendiente", "TODO",
   "WIP", "por hacer" que ya estén hechas (gold completado, exp19b cerrado,
   CI creado, paper aceptado, eXpress PASS).
4. **Rutas/ramas/tags inexistentes**: menciones a ramas borradas
   (`fase-2.5-recompute-retrieval-stats`, `fix/phase-1-no-rerun`,
   `fix/phase-2-nli-and-seeds` fueron eliminadas el 2026-08-23), a tags que no existan
   (`git tag --list`), o a scripts inexistentes.
5. **Contradicciones entre documentos**: misma métrica con valores distintos en dos
   archivos sin explicación; unidades de análisis distintas para el mismo experimento.
6. **CLAUDE.md específicamente**: ¿sigue describiendo comandos, rutas o flujos que ya no
   existen? ¿Menciona el estado pre-gold como si fuera actual?
7. **CITATION.cff**: YAML válido, título/autores/licencia coherentes con el paper
   aceptado (LACCI 2026, Ordoñez Flores + Lewis Fuentes, GPL-3.0).

## Artefacto de salida (único archivo que puedes crear)

`output/audit/docs_audit_codex_2026-08-30.md` con:

1. Pre-flight (los 3 comandos y su salida).
2. Tabla por archivo: OK / DESACTUALIZADO / CONTRADICE / ENLACES ROTOS, con hallazgos
   citados por archivo:línea, evidencia y corrección concreta sugerida.
3. Sección "No verificable en solo lectura" (lo que requeriría ejecutar algo).
4. Lista priorizada de correcciones (P1 bloquea credibilidad / P2 conviene / P3 menor).
5. Declaración final: qué NO hiciste (modificaciones, tests, etc.).

## Cierre

Además del artefacto, responde en chat con: **Hecho / Auditoría / Bloqueos / Falta**
(el formato habitual). Si el pre-flight muestra algo inesperado (rama distinta,
archivos raros), PARA y repórtalo sin seguir.
