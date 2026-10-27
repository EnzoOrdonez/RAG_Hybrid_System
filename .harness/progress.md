# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-27 por Claude (tarea F-020, fase A cerrada).
**Estado:** F-020 `active`; fase A hecha, fase B (pre-vuelo, USD 0) siguiente. F-017 `passing`.
Este archivo dice dónde está el trabajo **hoy**. Se reescribe al traspasar, no se acumula: el
historial va en los commits y, si hubo una decisión, en un ADR nuevo.

## Dónde estoy

- **Rama de esta documentación:** `docs/f020-despliegue`, creada desde `main` en `8fdaa65`, en el worktree `.worktrees/f020-docs`. Sin push: espera el OK de Enzo (ADR-0013).
- **Rama de la Fase B:** `fix/interview-readiness` (`3b52c4b`, worktree `.worktrees/interview-readiness`). El código del operador clona esa rama y exige que su punta remota coincida con HEAD, así que toda la ejecución de la Fase B y el anexo I7 se hacen allí.
- **`main`:** `8fdaa65` = `origin/main`. `bash init.sh` da exit 0 en las tres capas (1866 pasan, 0 fallan, 4 omitidas; 2026-10-27).
- **Worktree `.worktrees/interview-readiness`:** tiene `CLAUDE.md` modificado sin commit (5 líneas sobre los registros en C:\CloudRAG), anterior a esta sesión. No se toca ni se incluye en commits.
- **Registros privados de hoy:** `C:\CloudRAG\iteration7-faseB-prep-20261027\` (logs, `SHA256SUMS.txt` y la tabla A3).

## Qué se hizo en la última sesión

1. Se aprobó el plan de F-020: de la Fase B de I7 a la primera sesión, con topes de USD 20 + 3 (Fase B) y USD 100 (estudio). Ver ADR-0015.
2. Origen de la fecha mínima del 1/11: es una decisión del dueño (clase b), fijada en el código y sin razón documentada. Se decidió adelantarla al 2026-10-28 UTC, con respaldo al 1/11 (ADR-0016).
3. Conciliación del checklist del 1/10 con el runbook I7. Enzo decidió que la ruta humana es la nube con navegador propio y que la cohorte NLI no es precondición (ADR-0017).
4. `verify-draw`: el sello del 1/10 (`C:/CloudRAG/study-config`) falla con el código vigente. La config vigente es `study-config-reviewed-v2` (protocolo v2 con UEQ-S, misma asignación), así que no hace falta un sello nuevo.
5. F-017: `docs/guides/project-map.md` y `scripts/check_project_map.py`, con su prueba.

## Qué sigue (orden)

1. **F-020 fase B** (USD 0), en `fix/interview-readiness`:
   - adelantar el date guard con TDD (`i7.py`, `i7_phase_b.py`) y pedir OK de push;
   - regenerar el recibo de preparación: Docker Desktop, suites de Windows, Linux e I7 y `verify_local` (19/19 y 3/3);
   - preflight de Chromium 1208, catálogos de precios, `i7_phase_b --check`, lecturas GCP y plan de energía;
   - investigar la ruta humana sobre la imagen I7 (`i7.py:136-138` solo admite propósitos técnicos).
2. **Parada humana:** el bloque "Fase B" en una pregunta (gasto, región, retención a 24 h y ventana de disponibilidad de Enzo).
3. **F-020 fases C a E:** el tramo pagado de 24 h como máximo, según `docs/STUDY_I7_RUNBOOK_FASE_B.md`.
4. Pendientes que no son de F-020: F-009, F-010 y F-016. Las ramas obsoletas siguen sin borrar (decisión de Enzo).

## Riesgos y bloqueos vigentes

- **Ruta humana sin procedimiento:** el operador I7 rechaza rehearsal, pilot y study. La CLI de I5 los acepta, pero está fijada a otra raíz. Se resuelve en la fase B, antes del GO.
- **Respaldo de la fecha:** si las suites regeneradas fallan por una causa distinta de F-016, rige el 2026-11-01T00:00Z (31/10, 19:00 en Lima).
- **Ventana de 24 h:** cuenta desde el primer efecto pagado (`i7_costs.py:120-130`). Enzo debe poder responder durante ese tramo (OK del push del anexo y retención).
- **Ausencia de Enzo del 3 al 6/11:** no se hace ningún gasto que exija respuesta inmediata.
- **Capacidad L4:** el runbook I7 pide 2 rondas separadas por 30 min y un recibo humano. El código de I5 usa 3 rondas de 45 min y no aplica a I7.
- **Intermitencia (F-016):** si reaparece, repetir la prueba antes de culpar al cambio en curso.
- Entrevistas en NO-GO hasta el GO de la compuerta y la aprobación ética escrita.

## Cómo verificar en frío

```bash
bash init.sh   # Git Bash en Windows; en un worktree sin .venv-app: PYTHON=<ruta a .venv-app/Scripts/python.exe>
python scripts/check_project_map.py   # F-017: exit 0 si todas las rutas citadas existen
```
