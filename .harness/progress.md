[Codex] | 2026-10-29 12:29 America/Lima | rama `fix/interview-readiness` @ `a32c9dc901c3d6334bec36a11c534ae8748a5c46` | tarea F-020 | modo: cierre verificado de la base de ronda 2
Capas de a32c9dc: 1 secretos=exit 0 | 2 tests=exit 0 (2056 pasan, 0 fallan, 4 omitidos POSIX, 5 excluidos, 9 subtests) | 3 docs=exit 0 | init.sh=exit 0
Evidencia: log i7-gpu-restore-01-a32c9dc-init-20261029T122432.log, SHA-256 f33457591e9c8d0bdca60dd4ea44800ee43151d5e8080a476f7acf1ae1c4866a | Sin verificar: Linux, nube, GO y ética reales

# Ronda 2 sobre a32c9dc

[VERIFICADO] Esta cabecera identifica el árbol base cuyo init.sh ya terminó; no es el hash autorreferente del commit que contiene este archivo. El nuevo cierre se entrega externamente con el hash exacto tras los cuatro commits de ronda 2.
[VERIFICADO] H-A=547ac1a: aceptar solo truncado o half-up SDK en heredados; conservar .999 al desbordar. Nuevas reservas a segundos: igualdad exacta. Cuatro rojos antes del arreglo.
[VERIFICADO] H-B=b2f6b36: fake decimal half-up derivado del SDK, con casos .968532/.968499/.9997 y Z/-05:00. Dos rojos antes del arreglo del fake.
[VERIFICADO] M1=20 fallos; MX=3; control de fake truncador=2. Todos restaurados byte por byte; después pasan 104 pruebas focalizadas.
[DECLARADO] F-020 sigue active, passes=false. Sin corrección de estado privado, gcloud con efecto, push ni gasto. Informe: [auditoría y ronda 2](../docs/reports/i7-gpu-restore-01-audit.md).

## Preparación del cierre anterior (histórico, completado sobre a32c9dc)

# Incidente gpu-restore-01 sobre 468eb25

[VERIFICADO] 6bfbec9: reserva UTC a segundos y comparación a precisión SDK; UNKNOWN existente se recupera sin otro CREATE. Ocho regresiones rojas antes; 133 tests verdes después.
[VERIFICADO] 1f3b5ff: beta local requerido antes de reserve/admit_paid y entradas pagadas; consulta components list --only-local-state. Cleanup y parada segura CPU conservados.
[VERIFICADO] 5763124: 15 familias/95 flags, 94 visibles en --help y listen-on-stdin oculto pero registrado en el parser; sin corrección de flags. e0d8b3e: auditoría de fechas adicionales, sin otros arreglos.
[VERIFICADO] 7f309a9: ADR-0021, hashes del estado histórico y referencia al force push autorizado previo. Ningún cambio al estado privado.
[VERIFICADO] Siete tests fijan imagen/commit del GO y admisión de fuente. Las siete mutaciones de tiempo, beta y GO se detectaron y restauraron.
[VERIFICADO] Todos los test_study_i7*.py más policy: 314 pasan, 8 omitidos. Preflight local real: GCLOUD_BETA_INSTALLED, versión 2026.09.25.
[DECLARADO] El sexto commit incluye pruebas de identidad, informe y este traspaso. Después se ejecuta init.sh completo con máquina quieta; el cierre se reporta con hash exacto, 0/0/0 y git diff --stat 468eb25 HEAD.
[DECLARADO] Cambiar HEAD con código nuevo bloquea el operador sobre una instalación antigua; un GO futuro debe corresponder al nuevo build. Sin retag ni migración privada, gcloud con efecto, push o gasto.
[DECLARADO] F-020 continúa active, passes=false. Informe: [auditoría del incidente](../docs/reports/i7-gpu-restore-01-audit.md).

## Cierre anterior sobre e2335ed (histórico)

# F-020, ajuste final offline sobre e2335ed

[VERIFICADO] H3 liga approved_b4_version a nombre@fingerprint del sello verificado.
Dos mutantes detectados: solo nombre (2 fallos) y comparación retirada (5); bytes restaurados y 87 tests verdes.
[VERIFICADO] H9 fecha el código y ambos mensajes CLI del piso como 2026-10-28 UTC; 92 tests pasan.
[VERIFICADO] Positivo integrado start('study') con aprobación sintética, GO, fase humana y reserva; política y startup reales, nube simulada.
[VERIFICADO] Suite focalizada final: ledger, human, operator, phase_b, policy y sessions; 159 pasan.
[DECLARADO] ADR-0020 complementa ADR-0019 con límites inferiores según REPORTE_REVISION2.md y backlog H1, H2, H4–H8 y P7, sin corregirlos.
[DECLARADO] El cuarto commit contiene el cierre documental; después se ejecuta init.sh sin otras corridas concurrentes y se reportan hash exacto, 0/0/0 y git diff --stat e2335ed HEAD.
[DECLARADO] F-020 sigue active, passes=false: publicación, recibos vigentes, expediente ético y Fase B real son pasos del dueño. Sin gcloud, push ni gasto.

## Traspaso heredado de d8954cb (histórico; no es estado de esta rama)

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
