# Parches de la auditoría 2026-11-02

**No aplicados, decisión del dueño: se evita un commit nuevo en scripts/ antes del intento 8.**

Son documentación. Cada uno se probó solo en un worktree desechable de la base b470d3e, nunca en
esta rama ni en `fix/interview-readiness`.

| Parche | Hallazgo | Archivos | Orden | Verificación en el worktree |
|---|---|---|---|---|
| `001-cleanup-cpu-build-period.patch` | F-01 (S1) | `scripts/study_operator/i7_cpu_pipeline.py` | 1 | El reproductor F-01 pasa sin xfail. Suite completa sin regresiones. |
| `003-cleanup-cpu-resumable.patch` | B-3 (S1) | `scripts/study_operator/i7_cpu_pipeline.py` | 2 (encima de 001) | La caracterización `test_b3_*` se invierte. |
| `002-guard-unallocated-gpu-reservation.patch` | B-4 (S1) | `scripts/study_operator/i7.py` | 3 | `close` devuelve `BLOCKED_UNKNOWN_RESOURCE` con `RESERVATION_DELETED_VERIFIED`. |
| `004-r15-tornado-lock.patch` | R-15 (tornado) | `requirements-lock.txt`, `config/i7-build-recipe-pin.json` | independiente | Suite completa sin regresiones. Imagen NO reconstruida. |
| `005-r05-exp17-deterministic-bootstrap.patch` | R-05 | `scripts/compute_exp17_powered.py` | independiente | `tests/test_review_2026_11_02_backlog.py` pasa a XPASS(strict). |

Al aplicar cualquiera de 001 a 005:
- hay que quitar el `xfail` o invertir la aserción del test de caracterización correspondiente
  (`tests/test_review_2026_11_02_*.py`);
- se siguen las consecuencias de ADR-0025/0026: commit nuevo, suites Windows/Linux/I7,
  `local-readiness.json` nuevo y raíz nueva. Para 001 a 004 también cpu-build nuevo.
