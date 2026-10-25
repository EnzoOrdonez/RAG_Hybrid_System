# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-25 por Claude (Cowork).
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
  - También conserva los originales `PROMPT_MAESTRO_*` y `showcase/` sin versionar (F-011).

## Qué se hizo en la última sesión (F-001)

1. Integración de las dos ramas con los conflictos resueltos sin perder texto (`4f2d637`), y del trabajo sin commitear (`504b2a3`).
2. Copia inmutable del `CLAUDE.md` anterior (`0c4ad41`).
3. Documentos de la raíz movidos a `docs/` según Diátaxis, con las referencias vivas actualizadas (`bd84023`).
4. Arquitectura según ISO 42010, ADRs en formato MADR, log literal y guardrails (`3702683`).
5. `.harness/`, `init.sh`, `scripts/check_docs.py` con sus tests, paso en CI y `CLAUDE.md` nuevo.

## Qué sigue (orden sugerido)

1. **F-012:** la capa 2 sale con exit 1 por 8 tests de navegador de I7, que requieren el Chromium fijado (falla igual en la línea base). Enzo decide si se instala o se excluye. Mientras tanto, **F-001** sigue `active`.
2. **F-006:** Enzo revisa los commits de `chore/docs-harness` y decide el push y el PR (compuerta ADR-0013).
3. **F-007:** corregir la contradicción del README sobre el modelo de la UI.
4. **F-005:** I7 Fase B, no antes del 2026-11-01 y con el gasto aprobado.

## Riesgos y bloqueos vigentes

- Verificación del 2026-10-25: capa 1 exit 0, capa 3 exit 0, capa 2 exit 1, con el mismo conjunto de fallos
  que `fix/interview-readiness` (8 de navegador). Hay 3 tests más que necesitan `data/indices` sin marcador (F-013).
- El worktree tiene uniones (junctions) ignoradas por git: `.venv-app` y `data/indices` apuntan a
  `.worktrees/interview-readiness`. Se crearon para ejecutar la suite y no se versionan.
- Entrevistas en **NO-GO** hasta validar el despliegue real (`docs/INTERVIEW_READINESS.md`).
- Sin participantes reales hasta la aprobación ética; la lógica RAG está congelada.
- `paper/summer_ablation_log.md` y los documentos fechados conservan las rutas antiguas a propósito
  (mapa en `docs/README.md`).

## Cómo verificar en frío

```bash
bash init.sh   # Git Bash en Windows; usa .venv-app si existe; objetivo: exit 0 en las tres capas
```
