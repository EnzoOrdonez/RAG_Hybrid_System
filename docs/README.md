# Documentación de hybrid-rag-system

Organizada según [Diátaxis](https://diataxis.fr/): cada carpeta responde a una necesidad distinta.
El punto de entrada para agentes es `CLAUDE.md` (raíz); el estado operativo vive en `.harness/`.

| Carpeta | Tipo Diátaxis | Para qué sirve | Contenido |
|---|---|---|---|
| `architecture/` | Explicación | Entender cómo está construido el sistema | [system-design.md](architecture/system-design.md) (ISO/IEC/IEEE 42010), [models.md](architecture/models.md) |
| `decisions/` | Explicación | Saber por qué se decidió algo | [adr-log.md](decisions/adr-log.md) (MADR), [session-log.md](decisions/session-log.md) (log literal) |
| `guides/` | Guía práctica | Ejecutar una tarea concreta | [reproduce.md](guides/reproduce.md), [guardrails.md](guides/guardrails.md) |
| `specs/` | Referencia | Especificaciones de trabajo delegables | [showcase-spec.md](specs/showcase-spec.md), [animations-spec.md](specs/animations-spec.md) |
| `reports/` | Referencia | Resultados medidos y su procedencia | [results-summary.md](reports/results-summary.md), [summer-benchmarks.md](reports/summer-benchmarks.md) |
| `archive/` | Historial | Material inmutable que ya no es vigente | [README](archive/README.md) |

Los demás archivos de `docs/` (estudio con usuarios, despliegue, preregistros, paper) se mantienen
en su sitio: tienen enlaces entrantes desde código, ledgers y documentos firmados, y moverlos no
aporta trazabilidad.

## Mapa de rutas anteriores (migración del 2026-10-25)

| Ruta anterior (raíz) | Ruta actual |
|---|---|
| `MODELS.md` | `docs/architecture/models.md` |
| `REPRODUCE.md` | `docs/guides/reproduce.md` |
| `RESULTADOS_RESUMEN.md` | `docs/reports/results-summary.md` |
| `SUMMER_RESULTS.md` | `docs/reports/summer-benchmarks.md` |
| `NOTA3_NEXT_STEPS.md` | `docs/archive/nota3-next-steps-2026-06.md` |
| `PROMPT_MAESTRO_SHOWCASE.md` | `docs/specs/showcase-spec.md` |
| `PROMPT_MAESTRO_ANIMACIONES.md` | `docs/specs/animations-spec.md` |
| `CLAUDE.md` (versión de 543 líneas) | `docs/archive/CLAUDE_legacy_backup.md`; contenido redistribuido según [ADR-0014](decisions/adr-log.md#adr-0014) |

Estos documentos fechados o append-only conservan las rutas anteriores a propósito, porque
reescribirlos alteraría un registro histórico. Se leen con el mapa de arriba:

- `docs/PLAYBOOK_GATES_2026-08-06.md`
- `docs/ESTADO_PROYECTO_UNIFICADO_2026-08-06.md`
- `docs/MANIFESTS_PROPOSAL_2026-08-21.md`
- `paper/summer_ablation_log.md`
- `paper/audit_findings_cc_addenda.md`
- `output/audit/*`
- `docs/archive/*`
