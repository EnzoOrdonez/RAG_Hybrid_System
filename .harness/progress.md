# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-25 por Claude, Turno 6.
**Estado: Clean State** (F-006 cerrada: PR #5 fusionado en `main`, CI real verde, tres capas en exit 0 sobre `main`).
Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar, no se acumula: el
historial va en los commits y, si hubo una decisión, en un ADR nuevo.

## Dónde estoy

- **Rama activa:** `main` en el **checkout principal** (`C:\Users\enziz\Projects\hybrid-rag-system`), igual a `origin/main` en `2f2d6d4`
  (Merge Commit del PR #5, sin squash; su segundo padre es `5495b6f`, el cierre del Turno 5). Los SHAs citados en ADRs y `.harness/` siguen vivos.
- **CI real (primer run de esta línea de trabajo):** run `40227713051`, GitHub Actions, ubuntu-latest, Python 3.14.8, success en 1 min 3 s.
  Bloqueantes en verde: instalación con `--require-hashes`, secretos, `check_docs`, pytest CPU (1813 pasan, 57 omitidas, 0 fallos). Ruff sigue informativo (150 hallazgos, F-009).
- **Verificación local en `main`:** `bash ./init.sh` (Git Bash) exit 0 en las tres capas (1866 pasan, 0 fallan, 4 omitidas).
- **Ramas (ninguna eliminada, decisión de Enzo):** `chore/docs-harness`, `summer/taxonomia-759` y `fix/interview-readiness` siguen locales y en `origin`.
  `fix/interview-readiness` (`3b52c4b`) y `summer/taxonomia-759` (`93bd029`) son ancestros de `main`, y `chore/docs-harness` (`5495b6f`) es su segundo padre: borrarlas después es seguro.
- **`.worktrees/interview-readiness`** se conserva: su `.venv-app` es el único entorno (36 571 archivos) y el checkout principal lo usa por una unión (`.venv-app`).

## Qué se hizo en la última sesión

1. Draft PR #5 de `chore/docs-harness` a `main`; el CI real pasó a la primera, sin commits de corrección.
2. Compuerta con AskUserQuestion: Enzo autorizó Ready + Merge Commit y no eliminar ramas.
3. Merge Commit `2f2d6d4`; checkout principal a `main` (`pull --ff-only`); `init.sh` exit 0 en las tres capas.
4. F-006 cerrada en `feature_list.json` con la evidencia anterior.

## Qué sigue (orden sugerido)

Pendientes para sesiones dedicadas futuras: **F-005**, **F-009**, **F-010** y **F-016**.

1. **Ramas obsoletas:** decidir cuándo borrar las tres ramas (con `git merge-base --is-ancestor` previo). Para `fix/interview-readiness` hay que desacoplar antes el worktree (sin tocar su `.venv-app`).
2. **F-005:** I7 Fase B, no antes del 2026-11-01 y con el gasto aprobado.
3. **F-009:** hallazgos de Ruff (hoy informativos en CI); volverlo bloqueante al llegar a 0.
4. **F-010:** repetir las celdas de Qwen 3.5 y Gemma 4 sin modo de razonamiento (bloqueada).
5. **F-016:** investigar la intermitencia de `test_short_calls_always_check_host_telemetry` (no reapareció en las corridas de `main`).

## Riesgos y bloqueos vigentes

- **Cobertura del CI:** en CI se omiten 57 tests (frente a 4 en local) porque no hay playwright ni el Chromium fijado de I7, y `CLOUDRAG_REQUIRE_I7_BROWSER` no se exporta. La capa de navegador de I7 solo se ejercita en local.
- **Intermitencia (F-016):** si reaparece el fallo del supervisor, repetir esa prueba antes de culpar al cambio en curso.
- Dependencia: la unión `.venv-app` del checkout principal apunta a `.worktrees/interview-readiness/.venv-app`. No quitar ese worktree sin crear antes un entorno propio. Además, `fix/interview-readiness` sigue en checkout en ese worktree: `git branch -d` la rechazará hasta desacoplarlo (`git checkout --detach` allí).
- `init.sh` exporta solo el Chromium fijado de I7 si existe en `C:/CloudRAG/i7-browser-runtime`; en otra máquina sin él, los 8 tests de navegador de I7 fallan con `PINNED_BROWSER_RUNTIME_REQUIRED` (instalación en `docs/STUDY_I7_BROWSER_VALIDATION.md`).
- `scan_secrets.py` en modo git no lee lo ignorado por `.gitignore`; para auditarlo, `python scripts/scan_secrets.py --root .` (lento con `data/`).
- `.venv-app` aparece como no versionado (`??`) en `git status`: no está cubierto por `.gitignore` (la unión no casa con el patrón de directorio). Inofensivo hoy; no usar `git add -A`.
- Entrevistas en **NO-GO** hasta validar el despliegue real (`docs/INTERVIEW_READINESS.md`).
- Sin participantes reales hasta la aprobación ética; la lógica RAG está congelada.
- `paper/summer_ablation_log.md` y los documentos fechados conservan las rutas antiguas a propósito
  (mapa en `docs/README.md`).

## Cómo verificar en frío

```bash
bash init.sh   # Git Bash en Windows; usa .venv-app si existe y exporta solo el Chromium fijado de I7 si existe en C:/CloudRAG; objetivo: exit 0 en las tres capas
```
