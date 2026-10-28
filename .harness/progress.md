# Revisi?n actual: F-020, preparaci?n offline I7

[VERIFICADO] Rama fix/i7-human-ledger-ready; base d8954cb + cherry-picks 9dbba79 y 5ae784f.
[VERIFICADO] L?nea base: init.sh secretos=0, CPU=0 (1896 pasan, 11 omitidas, 5 excluidas), docs=0.
[VERIFICADO] F1=77a19aa, F3=3660a10, F4=f30f4cf, F5=32622b5, F6=c06a919, F9=cc01ac9.
[VERIFICADO] 146 tests focalizados pasan; seis checks documentales pasan. Mutantes detectados y restaurados.
[DECLARADO] F2 y F7 pendientes de aclaraci?n del due?o: cantidades humanas por SKU y referencia de versi?n B.4.
[DECLARADO] Falta el init.sh final tras esas correcciones; no hay autorizaci?n de push, gasto ni Fase B. Ver ADR-0019.

## Traspaso heredado de d8954cb (hist?rico; no es estado de esta rama)

# Progreso: estado para arranque en frío

**Última actualización:** 2026-10-27 por Claude (tarea F-020, fases A y B cerradas salvo energía).
**Estado:** F-020 `active`; parada humana del bloque "Fase B" (gasto, push y disponibilidad). F-017 `passing`.
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

1. **Fase B hecha** en `fix/interview-readiness` (sin push), con `ab267dd` (piso de pago) y `d0ba95a` (fase humana, ADR-0018).
   - Recibo vigente: `C:\CloudRAG\iteration7-preflight-20261027b\local-readiness.json`.
   - Spec privado: `C:\CloudRAG\i7-preparacion-20261027\fase-b-spec.json`.
   - Falta verificar la energía de la laptop (`powercfg` dio acceso denegado).
2. **Parada humana:** OK de push de `fix/interview-readiness`, bloque de gasto de la Fase B (USD 20 + 3) y ventana de 24 h de Enzo. Después: `i7_config` y el paso 1 del runbook.
3. **F-020 fases C a E:** el tramo pagado de 24 h como máximo, según `docs/STUDY_I7_RUNBOOK_FASE_B.md`.
4. Pendientes que no son de F-020: F-009, F-010 y F-016. Las ramas obsoletas siguen sin borrar (decisión de Enzo).

## Riesgos y bloqueos vigentes

- **Presupuesto humano:** el ledger de USD 70 (ADR-0018) alcanza para unos 12 días con 2 sesiones por boot [ESTIMADO]. Con 1 sesión por boot no alcanza.
- **Memoria de la laptop:** el arnés detuvo una regeneración por memoria baja (recuperada con `docker logs`). Cerrar programas antes de las corridas largas.
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
