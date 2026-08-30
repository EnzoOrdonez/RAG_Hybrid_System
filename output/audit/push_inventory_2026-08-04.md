# Inventario de push — compuerta (2026-08-04)

**Estado: ESPERANDO OK EXPLÍCITO DE ENZO. Nada publicado.**

Este documento es el reporte de la compuerta de `git push` exigida por `CLAUDE.md`. Lista qué se
publicaría, el resultado del escaneo de secretos, y las verificaciones de integridad de la evidencia.

## Qué se publicaría

| | |
|---|---|
| Rama local | `summer/mejoras` |
| Upstream | **ninguno** — la rama no existe en el remoto |
| Comando | `git push -u origin summer/mejoras` |
| Efecto sobre `main` | **ninguno**. `origin/main` es ancestro de `HEAD`; no hay PR, no hay rebase, no hay force |
| Commits | **45** |
| Archivos | 195 (336 434 inserciones, 22 supresiones) |
| Antigüedad | del 2026-07-22 al 2026-08-04 — **13 días de trabajo verificado solo en el disco de Enzo** |

Reparto por directorio: `experiments/results` 119 · `output/audit` 16 · `scripts` 20 · `tests` 8 ·
`src` 5 · `docs`/raíz 27.

## Escaneo de secretos

| Comprobación | Resultado |
|---|---|
| `.env` versionado | **NO** — ignorado en `.gitignore:47` |
| Nombres de archivo sospechosos (`*secret*`, `*credential*`, `*.pem`, `*.key`, `*token*`) | **0** |
| Claves de alta señal en el diff (`AKIA…`, `sk-…`, `ghp_…`, `xox[baprs]-`, `BEGIN … PRIVATE KEY`, `aws_secret_access_key=`) | **0 coincidencias** |
| `password` / `api_key` / `token` asignados en código o config | **0 coincidencias** |

Nota: un escaneo ingenuo de la palabra `password` sí produce ruido, pero todo el ruido está en
**texto de documentación de AWS/Azure/GCP** dentro de los artefactos de resultados (respuestas del
modelo y chunks del corpus). No hay credenciales.

## Integridad de la evidencia firmada

```
git diff --name-status nota3-evidencia-2026-06-11..HEAD -- experiments/results
  → 127 A  (altas)
  → 0 M, 0 D
```

Ninguna modificación ni borrado sobre `exp3..exp14` ni `exp8b`. Todo lo nuevo entra como archivo
nuevo, que es la vía sancionada. Desde el 2026-08-03 esto además está protegido en código por
`src/utils/signed_evidence.py::guard_write`.

## Verificaciones de estado en el momento del inventario

- `pytest tests/` → **130 pasan, 0 fallan, 0 omitidas**
- `scripts/verify_summer_offline.py` → **exit 0**, cubriendo exp15..exp18
- Sweep de ensembles re-corrido tras el retiro de `deberta-large` → **10/10 candidatos idénticos**
- Guardas de exp16/exp17/exp18 re-corridas → **ningún valor previo movido**

## Dos cosas que conviene que Enzo decida a la vez

1. **`data/llm_cache/` está versionado** (6,1 MB, 5 archivos). Es deliberado —permite reproducir
   generaciones sin GPU— pero conviene confirmarlo antes de que el repo remoto lo herede para
   siempre. No es un secreto ni un bloqueante.
2. **Trabajo sin committear que NO entra en este push.** El working tree tiene cambios de la sesión
   del 2026-08-03 (paridad de despliegue de UI) que **no son míos** y no he tocado:
   `src/pipeline/rag_pipeline.py`, `src/ui/components/index_loader.py`, `src/ui/pages/chat_page.py`,
   `tests/test_ui_deployment_parity.py` (227 líneas, sin versionar), `CLAUDE.md`, y
   `output/audit/v4_offline_check_2026-08-03.md`. Sus 6 tests pasan dentro de los 130.
   **¿Se committean y entran en el push, o se quedan fuera?**
