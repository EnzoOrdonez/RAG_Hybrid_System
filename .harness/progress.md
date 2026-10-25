# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-25 por Claude (Cowork), segunda sesión.
Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar, no se acumula: el
historial va en los commits y, si hubo una decisión, en un ADR nuevo.

## Dónde estoy

- **Rama de trabajo:** `chore/docs-harness`, creada desde `fix/interview-readiness` (`3b52c4b`), en el
  worktree `.worktrees/docs-harness`. Commits locales, **sin publicar**.
- **Estado de las ramas:**
  - `main` (`670f8e5`) está detrás de `fix/interview-readiness` (214 commits).
  - `summer/taxonomia-759` (9 commits de documentación del 2026-08-30) ya está integrada en esta rama.
- **Checkout principal** (`C:\Users\enziz\Projects\hybrid-rag-system`, en `summer/taxonomia-759`):
  - Conserva sin tocar sus cambios sin commitear. Su copia quedó integrada aquí (commit `504b2a3`).
  - F-011 parcial: `PROMPT_MAESTRO_*` y `Claude outputs/` archivados en `D:/CloudRAG_archivo/hybrid-rag-system-checkout-20261025` (MANIFEST con SHA-256). Siguen sin versionar `showcase/`, 39 archivos de `output/audit/` y otros (ver F-011).

## Qué se hizo en la última sesión (cierre de F-001)

1. **F-012:** el Chromium 1208 ya estaba instalado en `C:/CloudRAG/i7-browser-runtime` (SHA-256 igual al pin). Faltaba `PLAYWRIGHT_BROWSERS_PATH` al correr `init.sh`; no se descargó nada.
2. **F-013:** `tests/test_study_task_evidence.py`: marcador `needs_artifacts` y fixture `require_chunk_map` que omite con motivo si falta `data/indices` (3 tests). Sin cambios sin commitear fuera de este archivo y `.harness/`.
3. `bash init.sh` con la variable definida: capa 1 exit 0, capa 2 exit 0 (1848 pasan, 0 fallan, 6 omitidas), capa 3 exit 0.
4. F-001, F-012 y F-013 pasan a `passing`.

## Qué sigue (orden sugerido)

1. El cambio de F-012/F-013 está commiteado en local (`c43e61f`), sin publicar.
2. **F-011 (resto):** Enzo decide `showcase/`, `output/audit/PROMPT_*` y los cambios sin commitear del checkout principal. `.worktrees/interview-readiness` NO se quita: no es prunable y es el destino de las uniones `.venv-app` y `data/indices`.
3. **F-006:** Enzo revisa los commits de `chore/docs-harness` y decide push y PR (compuerta ADR-0013).
4. **F-007:** corregir la contradicción del README sobre el modelo de la UI.
5. **F-005:** I7 Fase B, no antes del 2026-11-01 y con el gasto aprobado.

## Riesgos y bloqueos vigentes

- Verificación del 2026-10-25: las tres capas dan exit 0 con `PLAYWRIGHT_BROWSERS_PATH=C:/CloudRAG/i7-browser-runtime`. Sin esa variable los 8 tests de navegador de I7 fallan con `PINNED_BROWSER_RUNTIME_REQUIRED` (F-012).
- El worktree tiene uniones (junctions) ignoradas por git: `.venv-app` y `data/indices` apuntan a
  `.worktrees/interview-readiness`. Se crearon para ejecutar la suite y no se versionan.
- Entrevistas en **NO-GO** hasta validar el despliegue real (`docs/INTERVIEW_READINESS.md`).
- Sin participantes reales hasta la aprobación ética; la lógica RAG está congelada.
- `paper/summer_ablation_log.md` y los documentos fechados conservan las rutas antiguas a propósito
  (mapa en `docs/README.md`).

## Cómo verificar en frío

```bash
export PLAYWRIGHT_BROWSERS_PATH=C:/CloudRAG/i7-browser-runtime   # obligatorio para los tests de I7
bash init.sh   # Git Bash en Windows; usa .venv-app si existe; objetivo: exit 0 en las tres capas
```
