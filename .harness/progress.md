# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-25 por Claude (Cowork), segunda sesión.
Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar, no se acumula: el
historial va en los commits y, si hubo una decisión, en un ADR nuevo.

## Dónde estoy

- **Rama de trabajo:** `chore/docs-harness` (creada desde `fix/interview-readiness`, integrada con `summer/taxonomia-759`),
  activa en el **checkout principal** (`C:\Users\enziz\Projects\hybrid-rag-system`). Commits locales, **sin publicar**.
  El worktree `.worktrees/docs-harness` se retiró (era redundante).
- **Estado de las ramas:**
  - `main` (`670f8e5`) está detrás de `fix/interview-readiness` (214 commits).
  - `summer/taxonomia-759` ya está integrada en `chore/docs-harness`; su punta (`93bd029`) sigue intacta.
- **Checkout principal:** limpio y autónomo. `bash ./init.sh` da exit 0 en las tres capas (1861 pasan, 0 fallan, 4 omitidas, ~6 min).
  Lo que estaba sin versionar quedó archivado en `D:/CloudRAG_archivo/hybrid-rag-system-checkout-20261025` (MANIFEST con SHA-256); `showcase/` queda ignorada.
- **`.worktrees/interview-readiness`** se conserva: su `.venv-app` es el único entorno (36 571 archivos) y el checkout principal lo usa por una unión ignorada (`.venv-app`).

## Qué se hizo en la última sesión

1. **F-014:** el límite del checkout deja de ser `ROOT.parent.parent` (`src/utils/checkout_boundary.py`); la evidencia externa sigue protegida en ambos layouts.
2. **F-015:** `scan_secrets.py` enumera con `git ls-files` sin `--root` (0,4 s en vez de no terminar); con `--root` recorre el disco.
3. Verificado con `bash ./init.sh` desde el checkout principal, sin variables exportadas a mano. Commit `640fcab`.
4. Worktree `docs-harness` retirado: se quitaron sus dos uniones con `rmdir` y se comprobó que `interview-readiness` quedó intacto.

## Qué sigue (orden sugerido)

1. **F-006:** push y PR de `chore/docs-harness` solo con OK explícito de Enzo (ADR-0013); sigue sin publicar. Un PR a `main` arrastra el historial de `fix/interview-readiness`.
2. **F-007:** corregir la contradicción del README sobre el modelo de la UI.
3. **F-005:** I7 Fase B, no antes del 2026-11-01 y con el gasto aprobado.

## Riesgos y bloqueos vigentes

- Verificación del 2026-10-25 desde el checkout principal: las tres capas dan exit 0. `init.sh` exporta solo el Chromium fijado de I7 si existe en `C:/CloudRAG/i7-browser-runtime`; en otra máquina sin él, los 8 tests de navegador de I7 fallan con `PINNED_BROWSER_RUNTIME_REQUIRED` (instalación en `docs/STUDY_I7_BROWSER_VALIDATION.md`).
- Dependencia: la unión `.venv-app` del checkout principal apunta a `.worktrees/interview-readiness/.venv-app`. No quitar ese worktree sin crear antes un entorno propio.
- `scan_secrets.py` en modo git no lee lo ignorado por `.gitignore`; para auditarlo, `python scripts/scan_secrets.py --root .` (lento con `data/`).
- Entrevistas en **NO-GO** hasta validar el despliegue real (`docs/INTERVIEW_READINESS.md`).
- Sin participantes reales hasta la aprobación ética; la lógica RAG está congelada.
- `paper/summer_ablation_log.md` y los documentos fechados conservan las rutas antiguas a propósito
  (mapa en `docs/README.md`).

## Cómo verificar en frío

```bash
bash init.sh   # Git Bash en Windows; usa .venv-app si existe y exporta solo el Chromium fijado de I7 si existe en C:/CloudRAG; objetivo: exit 0 en las tres capas
```
