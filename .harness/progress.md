# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-25 por Claude (Cowork), segunda sesión.
Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar, no se acumula: el
historial va en los commits y, si hubo una decisión, en un ADR nuevo.

## Dónde estoy

- **Rama de trabajo:** `chore/docs-harness`, creada desde `fix/interview-readiness` (`3b52c4b`) e integrada con
  `summer/taxonomia-759`. Commits locales, **sin publicar**. Está activa en el **checkout principal**
  (`C:\Users\enziz\Projects\hybrid-rag-system`); `.worktrees/docs-harness` queda en HEAD desacoplado en el mismo commit y ya no hace falta.
- **Estado de las ramas:**
  - `main` (`670f8e5`) está detrás de `fix/interview-readiness` (214 commits).
  - `summer/taxonomia-759` (9 commits de documentación del 2026-08-30) ya está integrada en `chore/docs-harness`; su punta (`93bd029`) sigue intacta.
- **Checkout principal:** limpio. Lo que estaba sin versionar quedó archivado en
  `D:/CloudRAG_archivo/hybrid-rag-system-checkout-20261025` (MANIFEST con SHA-256), incluidas las copias de los `CLAUDE.md` y
  `paper/summer_ablation_log.md` modificados (su contenido ya estaba en esta rama). `showcase/` queda ignorada por decisión de Enzo.
- **`.worktrees/interview-readiness`** se conserva: es el destino de las uniones `.venv-app` y `data/indices`.

## Qué se hizo en la última sesión

1. **F-001, F-012, F-013** cerrados en `passing` (commits `c43e61f`, `4569de5`): el Chromium 1208 ya estaba instalado; 3 tests de `task_evidence` omiten con motivo sin `data/indices`.
2. **`init.sh`** exporta `PLAYWRIGHT_BROWSERS_PATH=C:/CloudRAG/i7-browser-runtime` solo si no está definida y el directorio existe; CI no cambia. Verificado sin exportarla: exit 0 en las tres capas (1848 pasan, 0 fallan, 6 omitidas).
3. **F-011** resuelto: archivado con SHA-256 (68 archivos de la tanda 2 más los de la tanda 1), `showcase/` y `.worktrees/` en `.gitignore`.

## Qué sigue (orden sugerido)

1. **F-006:** push y PR de `chore/docs-harness` solo con OK explícito de Enzo (ADR-0013); sigue sin publicar. Un PR a `main` arrastra el historial de `fix/interview-readiness`.
2. **F-007:** corregir la contradicción del README sobre el modelo de la UI.
3. **F-005:** I7 Fase B, no antes del 2026-11-01 y con el gasto aprobado.

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
bash init.sh   # Git Bash en Windows; usa .venv-app si existe y exporta solo el Chromium fijado de I7 si existe en C:/CloudRAG; objetivo: exit 0 en las tres capas
```
