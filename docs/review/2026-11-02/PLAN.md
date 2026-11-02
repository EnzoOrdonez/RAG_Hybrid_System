# Plan de auditoría 2026-11-02 (base `fix/interview-readiness` @ b470d3e)

## Fase 0 [VERIFICADO]
- `git rev-parse origin/fix/interview-readiness` = b470d3e76fe569fd63ee98b1000c6d75817cc964 (= HEAD esperado).
- `git merge-base HEAD origin/fix/interview-readiness` = b470d3e… (rama `claude/review-2026-11-02`).
- Existen `docs/decisions/ADR-0025-i7-reserva-ttl.md` y `ADR-0026-i7-rechazo-capacidad-reserva.md`.
- `scripts/study_operator/i7.py:279` contiene `.replace(microsecond=0)` dentro de `reserve()`.

## Crítica previa (§0.2)
Mayor riesgo (orden de pasadas):
1. Dinero y estado persistido de I7: `i7.py` (reserve/guard/reconcile/start), `i7_cpu_pipeline.py`
   (gpu_restore, cleanup_cpu), `i7_phase_b.py`/`i7_workflows.py` (gate, close), `i7_costs.py`.
2. Recuperación y cierre: caminos de error que dejan `installation.json` o `STATE.json` en un estado
   del que `close` no sale (sospecha `period_id` tras smoke).
3. Invariantes duplicadas (rama, P999, horas, presupuesto).
4. Seguridad de `src/ui/**` y del puente guest.
5. Calidad de tests del operador.

Supuestos ocultos del plan:
- Los tests usan dobles de `Cloud`; el comportamiento real de gcloud (formato de `deleteAtTime`,
  consistencia eventual de `list`) es [SUPUESTO] salvo recibos del repo.
- Python 3.14.6 de uv en Linux, no el 3.14.3 de Windows: los tests solo-Windows quedan OMITIDOS.
- No hay acceso a la nube, Docker, GPU ni a `C:\CloudRAG`: todo lo que dependa de eso es NO EJECUTADO.

Sin cubrir a propósito: scripts fuera de `scripts/study_operator/**` y `src/ui/**`, salvo para
re-validar R-03..R-06 (que viven ahí).

## Pasadas y subagentes
- P1 (yo): `period_id` tras smoke, R-01, R-02 residual, invariantes de rama/P999.
- P2 (subagente A): dinero, TTL, presupuesto y cuotas con reloj simulado.
- P3 (subagente B): recuperación, idempotencia y estados intermedios en close/gate/series/gpu_restore.
- P4 (subagente C): seguridad en study_operator y src/ui.
- P5 (yo): re-validación R-03, R-04, R-05, R-06, R-09, R-15, R-16.
- P6: refutación adversarial independiente de cada S0/S1 (subagentes nuevos sin mi razonamiento).
