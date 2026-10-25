# docs/archive: material histórico inmutable

Archivos que ya no describen el estado vigente y se conservan sin cambios como procedencia.
No se editan; si algo de aquí vuelve a ser relevante, se cita desde el documento vivo.

| Archivo | Qué es | Procedencia |
|---|---|---|
| `CLAUDE_legacy_backup.md` | Copia exacta del `CLAUDE.md` anterior a la migración del 2026-10-25 (543 líneas, 39 192 bytes, SHA-256 `545d93526235db6feec84396b9b7891fbaba9f8ad91720dbd103efecea4fe7d4`, finales de línea LF). | Commit `504b2a3` de `chore/docs-harness`: `fix/interview-readiness` (`3b52c4b`) + fusión de `summer/taxonomia-759` (`93bd029`) + 5 líneas de punteros que estaban sin commitear. |
| `nota3-next-steps-2026-06.md` | Runbook de Nota 3 (antes `NOTA3_NEXT_STEPS.md` en la raíz). Sus TODO son historial: el propio archivo lo declara desde el 2026-08-30. | Movido con `git mv`; el historial se sigue con `git log --follow`. |

## Versiones previas de CLAUDE.md que no están en la copia

La fusión resolvió a favor de `summer/taxonomia-759` algunas líneas que esa rama había cerrado o
reescrito (problemas conocidos ya resueltos, contradicciones cerradas). Esas líneas no se perdieron:
siguen en los commits originales.

- `git show 3b52c4b:CLAUDE.md` (versión de `fix/interview-readiness`).
- `git show 93bd029:CLAUDE.md` (versión de `summer/taxonomia-759`).
- Las 5 líneas sin commitear de ambos checkouts quedaron en el commit `504b2a3`.
