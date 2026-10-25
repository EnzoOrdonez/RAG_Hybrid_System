# CLAUDE.md: hybrid-rag-system

CloudRAG: un sistema RAG híbrido (BM25 + BGE + RRF + cross-encoder + Ollama + verificación NLI/HHEM)
sobre documentación de AWS, Azure y GCP, con la evidencia de una tesis y del paper IEEE LACCI 2026.
Este archivo es un **índice**. El detalle vive en los archivos enlazados y se lee solo cuando hace falta.

## Arranque en frío: cinco preguntas, cinco archivos

| Pregunta | Dónde está la respuesta |
|---|---|
| 1. ¿Qué es y cómo está construido? | [docs/architecture/system-design.md](docs/architecture/system-design.md) (ISO 42010) y [models.md](docs/architecture/models.md) |
| 2. ¿En qué estado está y qué sigue? | [.harness/progress.md](.harness/progress.md) y [.harness/feature_list.json](.harness/feature_list.json) |
| 3. ¿Qué no puedo tocar? | Resumen abajo; texto completo en [docs/guides/guardrails.md](docs/guides/guardrails.md) |
| 4. ¿Cómo ejecuto y verifico? | `init.sh` (abajo) y [docs/guides/reproduce.md](docs/guides/reproduce.md) |
| 5. ¿Por qué está así y qué pasó antes? | [docs/decisions/adr-log.md](docs/decisions/adr-log.md), [session-log.md](docs/decisions/session-log.md) y el mapa de rutas en [docs/README.md](docs/README.md) |

## Ciclo obligatorio para toda tarea

1. **Plan:** elegir una tarea de `feature_list.json` (o crearla con un `id` nuevo), marcarla `active` y escribir sus pasos. Si hay ambigüedad, preguntar antes de escribir código.
2. **Escribe:** el cambio mínimo, dentro del alcance de la tarea.
3. **Verifica:** `bash init.sh`, que debe dar exit 0 en las tres capas. Si el cambio toca cifras o evidencia, ejecutar también los verificadores offline.
4. **Corrige:** si una capa falla, se arregla la causa. No se desactiva el check ni se marca `passing` sin evidencia.
5. **Traspasa:**
   - actualizar el estado y los pasos de la tarea, con `passes` en true solo si está `passing`;
   - reescribir `.harness/progress.md`;
   - registrar un ADR si se tomó una decisión;
   - abrir el reporte con la cabecera de telemetría.

## Exit code 0 en tres capas (`init.sh`)

| Capa | Comando | Garantiza |
|---|---|---|
| 1 Secretos | `python scripts/scan_secrets.py --baseline secrets_baseline.json` | Nada sensible en el repo público |
| 2 Tests CPU | `python -m pytest tests/ -q -m "not slow and not gpu"` | Código y guardas de evidencia intactos |
| 3 Docs/harness | `python scripts/check_docs.py` | Enlaces, esquema de tareas, `CLAUDE.md` < 100 líneas, ninguna línea del CLAUDE.md anterior perdida |

Ejecutar `bash init.sh` desde Git Bash en Windows (no WSL), o en Linux o CI. Usa `.venv-app` si
existe (lo exigen los tests de I7); `PYTHON=<ruta>` lo cambia. Si una capa sale distinta de 0,
la tarea no está terminada.

## Cabecera de telemetría (obligatoria al inicio de cada reporte o traspaso)

```
[Agente] | AAAA-MM-DD HH:MM | rama `<rama>` @ `<hash>` | tarea F-NNN | modo: sondeo o ejecución
Capas: 1 secretos=exit N | 2 tests=exit N (P pasan, F fallan, O omitidas) | 3 docs=exit N
Cambios: <archivos o commits> | Sin verificar: <lo que no se ejecutó y por qué>
```

## Preguntar con AskUserQuestion es obligatorio

No adivinar. Se usa la herramienta de preguntas interactivas (AskUserQuestion o su equivalente) cuando:

- algo es ambiguo o contradictorio;
- hay que mover, renombrar o borrar archivos;
- una decisión toca una cifra ya entregada (A.3, ficha, paper LACCI) o la evidencia;
- antes de cualquier `git push`, gasto en la nube o corrida de modelos.

Hay que dar contexto suficiente para decidir sin ambigüedad y recomendar una opción.

## Guardrails esenciales (completos en docs/guides/guardrails.md)

- **Evidencia firmada inmutable:** `experiments/results/exp3..exp14`, `exp8b` y la fase de verano `exp15..exp19b`. Los recálculos se escriben en archivos `_vN` nuevos (ADR-0001, ADR-0009).
- **Generación LLM nueva solo bajo IDs `exp15+`.** La lógica RAG está congelada y no se ejecutan modelos sin autorización (ADR-0004).
- **Inmutables:** `paper/audit_findings.md`, `paper/audit_outputs/exp8_stats_corrected.csv` y los ledgers append-only. La prosa del paper y del A.3 solo se cambia con OK frase por frase.
- **`.env`:** no leer, no imprimir, no versionar. El repo es **público**: sin IPs, hosts, correos, tokens ni datos de participantes.
- **Compuerta de push (ADR-0013):** reportar qué se publicaría y esperar OK explícito. Sin push a `main` ni force push.
- **Entorno:** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUTF8=1`, seed 42, intérprete 3.14 (no el 3.11 del PATH).
- **No borrar:** se mueve y se declara (ADR-0011). Los originales se archivan y nunca se sobrescriben.

## Punteros

- Ledgers: `paper/audit_findings_cc_addenda.md` (N1 a N9), `paper/correction_log.md`, `paper/summer_ablation_log.md`.
- Trazabilidad: `docs/TRACEABILITY_nota3.md`. Resultados: [results-summary.md](docs/reports/results-summary.md) y [summer-benchmarks.md](docs/reports/summer-benchmarks.md).
- Estudio con usuarios y despliegue: `docs/INTERVIEW_READINESS.md`, `docs/STUDY_I7_*`, `docs/CLOUD_DEPLOYMENT.md`.
- Registros de ejecución fuera del repo: índice en `C:\CloudRAG\LEEME_CLOUDRAG.md`; lo cerrado vive en `D:\CloudRAG_archivo\`.
- Copia del CLAUDE.md anterior: [docs/archive/CLAUDE_legacy_backup.md](docs/archive/CLAUDE_legacy_backup.md).
