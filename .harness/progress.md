# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-25 por Claude, Turno 5.
**Estado: Clean State** (árbol limpio, rama publicada, tres capas en exit 0).
Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar, no se acumula: el
historial va en los commits y, si hubo una decisión, en un ADR nuevo.

## Dónde estoy

- **Rama de trabajo:** `chore/docs-harness` (creada desde `fix/interview-readiness`, integrada con `summer/taxonomia-759`),
  activa en el **checkout principal** (`C:\Users\enziz\Projects\hybrid-rag-system`). Publicada en `origin` el 2026-10-25 (OK explícito, ADR-0013) y sincronizada con `origin/main` (0 commits detrás). **No se ha abierto PR a `main`.**
  El worktree `.worktrees/docs-harness` se retiró (era redundante).
- **Estado de las ramas:**
  - `main` (`670f8e5`) está detrás de `fix/interview-readiness` (214 commits).
  - `summer/taxonomia-759` ya está integrada en `chore/docs-harness`; su punta (`93bd029`) sigue intacta.
- **Checkout principal:** limpio y autónomo. `bash ./init.sh` da exit 0 en las tres capas (1866 pasan, 0 fallan, 4 omitidas; 16 de 16 tests de navegador de I7 ejecutados).
  Lo que estaba sin versionar quedó archivado en `D:/CloudRAG_archivo/hybrid-rag-system-checkout-20261025` (MANIFEST con SHA-256); `showcase/` queda ignorada.
- **`.worktrees/interview-readiness`** se conserva: su `.venv-app` es el único entorno (36 571 archivos) y el checkout principal lo usa por una unión ignorada (`.venv-app`).

## Qué se hizo en la última sesión

1. **Blindaje de CI (I7):** los tests de navegador se omiten con `pytest.skip` si falta `playwright` o el Chromium fijado (`tests/i7_browser_guard.py`). Con el runtime detectado, `init.sh` exporta `CLOUDRAG_REQUIRE_I7_BROWSER=1` y omitir pasa a ser fallo. Emulado sin playwright: 11 omitidos, 0 fallos.
2. **F-008:** `reranking` y las banderas `multidimensional_scoring`/`terminology_normalization` documentadas como inertes/reservadas (`config/config.yaml`, `system-design.md` §8). `PROPOSED_HYBRID` intacto (ADR-0005).
3. **F-006:** `origin/main` (PR #4) integrada sin conflictos; `bash ./init.sh` exit 0 en las tres capas y push de `chore/docs-harness`.
4. **F-016 (nuevo):** un test del supervisor falló una vez en la suite completa y pasó al repetir (ver Riesgos).

## Qué sigue (orden sugerido)

Pendientes para sesiones dedicadas futuras: **F-005**, **F-009** y **F-010**.

1. **PR a `main`:** decisión de Enzo. Traería el historial de `fix/interview-readiness` (214 commits) y dispararía el CI por primera vez sobre esta rama.
2. **F-005:** I7 Fase B, no antes del 2026-11-01 y con el gasto aprobado.
3. **F-009:** hallazgos de Ruff (hoy informativos en CI).
4. **F-010:** repetir las celdas de Qwen 3.5 y Gemma 4 sin modo de razonamiento (bloqueada).
5. **F-016:** investigar la intermitencia del test del supervisor.

## Riesgos y bloqueos vigentes

- **CI sin verificar:** el job "Run CPU-only test suite" de GitHub Actions (Linux, Python 3.14, `uv pip sync --require-hashes`) no se ha ejecutado sobre esta rama; solo se emuló bloqueando `playwright`.
- **Intermitencia (F-016):** de dos corridas completas de `init.sh` hoy, una falló en `test_short_calls_always_check_host_telemetry`; la repetición pasó. Si reaparece, repetir esa prueba antes de culpar al cambio en curso.

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
